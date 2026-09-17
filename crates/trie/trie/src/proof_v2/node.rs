use crate::proof_v2::DeferredValueEncoder;
use alloy_rlp::Encodable;
use reth_execution_errors::trie::StateProofError;
use reth_trie_common::{
    BranchNodeMasks, BranchNodeV2, ExtensionNodeRef, LeafNode, LeafNodeRef, Nibbles,
    ProofTrieNodeV2, RlpNode, TrieMask, TrieNodeV2,
};

/// A trie node which is the child of a branch in the trie.
#[derive(Debug)]
pub(crate) enum ProofTrieBranchChild<RF> {
    /// A leaf node whose value has yet to be calculated and encoded.
    Leaf {
        /// The short key of the leaf.
        short_key: Nibbles,
        /// The [`DeferredValueEncoder`] which will encode the leaf's value.
        value: RF,
    },
    /// A branch node whose children have already been flattened into [`RlpNode`]s.
    Branch {
        /// The node itself, for use during RLP encoding.
        node: BranchNodeV2,
        /// Bitmasks carried over from cached `BranchNodeCompact` values, if any.
        masks: Option<BranchNodeMasks>,
    },
    /// A node whose type is not known, as it has already been converted to an [`RlpNode`].
    RlpNode {
        /// The RLP-encoded node.
        node: RlpNode,
        /// The path from the parent branch's child nibble to the encoded node. This field can only
        /// be set if `node` was sourced from the hashes of a cached branch, and therefore we know
        /// that it is a blinded branch.
        short_key: Nibbles,
        /// Whether this node contributes to its parent's hash mask when it is a direct child.
        hash_mask_bit: bool,
        /// Whether this node contributes to its parent's tree mask.
        tree_mask_bit: bool,
    },
}

/// Mask metadata preserved while a child is consumed by its existing encoding path.
pub(crate) enum EncodedChildMasks {
    /// Leaf, extension and already-encoded nodes retain their existing mask bits.
    Fixed(bool, bool),
    /// A direct branch contributes a hash exactly when its encoded node is hashed.
    DirectBranch { tree: bool },
}

impl EncodedChildMasks {
    pub(crate) fn resolve(self, node: &RlpNode) -> (bool, bool) {
        match self {
            Self::Fixed(hash, tree) => (hash, tree),
            Self::DirectBranch { tree } => (node.is_hash(), tree),
        }
    }
}

impl<RF: DeferredValueEncoder> ProofTrieBranchChild<RF> {
    /// Converts this child into its RLP node representation.
    ///
    /// This potentially also returns an `RlpNode` buffer which can be re-used for other
    /// [`ProofTrieBranchChild`]s.
    pub(crate) fn into_rlp(
        self,
        buf: &mut Vec<u8>,
    ) -> Result<(RlpNode, Option<Vec<RlpNode>>), StateProofError> {
        match self {
            Self::Leaf { short_key, value } => {
                // RLP encode the value itself
                value.encode(buf)?;
                let value_enc_len = buf.len();

                // Determine the required buffer size for the encoded leaf
                let leaf_enc_len = LeafNodeRef::new(&short_key, buf).length();

                // We want to re-use buf for the encoding of the leaf node as well. To do this we
                // will keep appending to it, leaving the already encoded value in-place. First we
                // must ensure the buffer is big enough, then we'll split.
                buf.resize(value_enc_len + leaf_enc_len, 0);

                // SAFETY we have just resized the above to be greater than `value_enc_len`, so it
                // must be in-bounds.
                let (value_buf, mut leaf_buf) =
                    unsafe { buf.split_at_mut_unchecked(value_enc_len) };

                // Encode the leaf into the right side of the split buffer, and return the RlpNode.
                LeafNodeRef::new(&short_key, value_buf).encode(&mut leaf_buf);
                Ok((RlpNode::from_rlp(&buf[value_enc_len..]), None))
            }
            Self::Branch { node: branch_node, .. } => {
                branch_node.encode(buf);
                Ok((RlpNode::from_rlp(buf), Some(branch_node.stack)))
            }
            Self::RlpNode { node, short_key, hash_mask_bit, .. } => {
                if short_key.is_empty() {
                    return Ok((node, None))
                }

                // Only branch hashes sourced from a cached hash mask can have an external short
                // key. Other committed nodes already encode their key internally.
                debug_assert!(hash_mask_bit);
                ExtensionNodeRef::new(&short_key, node.as_slice()).encode(buf);
                Ok((RlpNode::from_rlp(buf), None))
            }
        }
    }

    /// Converts this child into a [`ProofTrieNodeV2`] having the given path.
    ///
    /// # Errors
    ///
    /// Returns [`StateProofError::TrieInconsistency`] if called on a [`Self::RlpNode`].
    pub(crate) fn into_proof_trie_node(
        self,
        path: Nibbles,
        buf: &mut Vec<u8>,
    ) -> Result<ProofTrieNodeV2, StateProofError> {
        let (node, masks) = match self {
            Self::Leaf { short_key, value } => {
                value.encode(buf)?;
                // Counter-intuitively a clone is better here than a `core::mem::take`. If we take
                // the buffer then future RLP-encodes will need to re-allocate a new one, and
                // RLP-encodes after those may need a bigger buffer and therefore re-alloc again.
                //
                // By cloning here we do a single allocation of exactly the size we need to take
                // this value, and the passed in buffer can remain with whatever large capacity it
                // already has.
                let rlp_val = buf.clone();
                (TrieNodeV2::Leaf(LeafNode::new(short_key, rlp_val)), None)
            }
            Self::Branch { node, masks } => (TrieNodeV2::Branch(node), masks),
            // Cached hashes cannot be retained as proof nodes: targeted children are recalculated,
            // while untargeted children are either combined into a branch or discarded. Reaching
            // this arm means inconsistent cached trie data left a blinded node as the local root.
            Self::RlpNode { .. } => {
                return Err(StateProofError::TrieInconsistency(
                    "cannot convert RLP node to proof node".to_string(),
                ))
            }
        };

        Ok(ProofTrieNodeV2 { node, path, masks })
    }

    /// Returns the child's short key.
    pub(crate) const fn short_key(&self) -> &Nibbles {
        match self {
            Self::Leaf { short_key, .. } |
            Self::Branch { node: BranchNodeV2 { key: short_key, .. }, .. } |
            Self::RlpNode { short_key, .. } => short_key,
        }
    }

    /// Preserves mask metadata; resolve it after the child's existing RLP conversion.
    /// Direct branches reuse that conversion's hash decision instead of scanning their length.
    pub(crate) fn mask_bits(&self) -> EncodedChildMasks {
        match self {
            Self::Leaf { .. } => EncodedChildMasks::Fixed(false, false),
            Self::Branch { node, masks } => {
                let tree = masks.is_some_and(|masks| !masks.is_empty());
                if node.key.is_empty() {
                    EncodedChildMasks::DirectBranch { tree }
                } else {
                    EncodedChildMasks::Fixed(false, tree)
                }
            }
            Self::RlpNode { short_key, hash_mask_bit, tree_mask_bit, .. } => {
                EncodedChildMasks::Fixed(*hash_mask_bit && short_key.is_empty(), *tree_mask_bit)
            }
        }
    }

    /// Trims the given number of nibbles off the head of the short key.
    ///
    /// # Panics
    ///
    /// - If the given len is longer than the short key
    pub(crate) fn trim_short_key_prefix(&mut self, len: usize) {
        match self {
            Self::Leaf { short_key, .. } | Self::RlpNode { short_key, .. } => {
                *short_key = trim_nibbles_prefix(short_key, len);
            }
            Self::Branch { node: BranchNodeV2 { key, branch_rlp_node, .. }, .. } => {
                *key = trim_nibbles_prefix(key, len);
                if key.is_empty() {
                    *branch_rlp_node = None;
                }
            }
        }
    }
}

/// A single branch in the trie which is under construction. The actual child nodes of the branch
/// will be tracked as [`ProofTrieBranchChild`]s on a stack.
#[derive(Debug)]
pub(crate) struct ProofTrieBranch {
    /// The length of the parent extension node's short key. If zero then the branch's parent is
    /// not an extension but instead another branch.
    pub(crate) ext_len: u8,
    /// A mask tracking which child nibbles are set on the branch so far. There will be a single
    /// child on the stack for each set bit.
    pub(crate) state_mask: TrieMask,
}

/// Trims the first `len` nibbles from the head of the given `Nibbles`.
///
/// # Panics
///
/// Panics if the given `len` is greater than the length of the `Nibbles`.
pub(crate) fn trim_nibbles_prefix(n: &Nibbles, len: usize) -> Nibbles {
    debug_assert!(n.len() >= len);
    n.slice_unchecked(len, n.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct TestValue;

    impl DeferredValueEncoder for TestValue {
        fn encode(self, buf: &mut Vec<u8>) -> Result<(), StateProofError> {
            buf.push(1);
            Ok(())
        }
    }

    #[test]
    fn encoded_branch_masks_preserve_inline_hash_boundary_and_extensions() {
        for (sizes, expected_len) in [([4, 5], 31), ([5, 5], 32), ([6, 6], 34)] {
            let children = sizes.map(|size| {
                let mut encoded = Vec::new();
                LeafNode::new(Nibbles::new(), vec![1; size]).encode(&mut encoded);
                RlpNode::from_rlp(&encoded)
            });
            for masks in [
                None,
                Some(BranchNodeMasks::default()),
                Some(BranchNodeMasks {
                    hash_mask: TrieMask::new(1),
                    tree_mask: TrieMask::default(),
                }),
            ] {
                for extension in [false, true] {
                    let mut branch = BranchNodeV2::new(
                        Nibbles::new(),
                        children.to_vec(),
                        TrieMask::new(3),
                        None,
                    );
                    let mut branch_bytes = Vec::new();
                    branch.encode(&mut branch_bytes);
                    assert_eq!(branch_bytes.len(), expected_len);
                    if extension {
                        branch.key = Nibbles::from_nibbles([1]);
                        branch.branch_rlp_node = Some(RlpNode::from_rlp(&branch_bytes));
                    }
                    let expected =
                        (!extension && expected_len >= 32, masks.is_some_and(|m| !m.is_empty()));
                    let child =
                        ProofTrieBranchChild::<TestValue>::Branch { node: branch.clone(), masks };
                    let pending = child.mask_bits();
                    let mut encoded = Vec::new();
                    let (actual, returned_stack) = child.into_rlp(&mut encoded).unwrap();
                    assert_eq!(pending.resolve(&actual), expected);
                    let mut oracle = Vec::new();
                    branch.encode(&mut oracle);
                    assert_eq!(encoded, oracle);
                    assert_eq!(returned_stack.unwrap(), children);
                    // The retained-proof conversion used by commit_last_child must agree too.
                    let child = ProofTrieBranchChild::<TestValue>::Branch { node: branch, masks };
                    let pending = child.mask_bits();
                    let proof =
                        child.into_proof_trie_node(Nibbles::new(), &mut Vec::new()).unwrap();
                    let mut encoded = Vec::new();
                    proof.node.encode(&mut encoded);
                    assert_eq!(pending.resolve(&RlpNode::from_rlp(&encoded)), expected);
                    assert_eq!(proof.masks, masks);
                }
            }
        }
    }

    #[test]
    fn encoded_masks_preserve_leaf_cached_flags_and_errors() {
        for short_key in [Nibbles::new(), Nibbles::from_nibbles([1])] {
            for hash_mask_bit in [false, true] {
                for tree_mask_bit in [false, true] {
                    let child = ProofTrieBranchChild::<TestValue>::RlpNode {
                        node: RlpNode::word_rlp(&alloy_primitives::B256::ZERO),
                        short_key,
                        hash_mask_bit,
                        tree_mask_bit,
                    };
                    let pending = child.mask_bits();
                    // Fixed cached metadata is not inferred from encoded size.
                    assert_eq!(
                        pending.resolve(&RlpNode::default()),
                        (hash_mask_bit && short_key.is_empty(), tree_mask_bit)
                    );
                }
            }
        }
        let child = ProofTrieBranchChild::Leaf { short_key: Nibbles::new(), value: TestValue };
        let pending = child.mask_bits();
        let (encoded, _) = child.into_rlp(&mut Vec::new()).unwrap();
        assert_eq!(pending.resolve(&encoded), (false, false));
        struct FailingValue;
        impl DeferredValueEncoder for FailingValue {
            fn encode(self, _: &mut Vec<u8>) -> Result<(), StateProofError> {
                Err(StateProofError::TrieInconsistency("sentinel".into()))
            }
        }
        let child = ProofTrieBranchChild::Leaf { short_key: Nibbles::new(), value: FailingValue };
        let _pending = child.mask_bits();
        assert!(
            matches!(child.into_rlp(&mut Vec::new()), Err(StateProofError::TrieInconsistency(s)) if s == "sentinel")
        );
    }

    #[test]
    fn test_trim_nibbles_prefix_basic() {
        // Create nibbles [1, 2, 3, 4, 5, 6]
        let nibbles = Nibbles::from_nibbles([1, 2, 3, 4, 5, 6]);

        // Trim first 2 nibbles
        let trimmed = trim_nibbles_prefix(&nibbles, 2);
        assert_eq!(trimmed.len(), 4);

        // Verify the remaining nibbles are [3, 4, 5, 6]
        assert_eq!(trimmed.get(0), Some(3));
        assert_eq!(trimmed.get(1), Some(4));
        assert_eq!(trimmed.get(2), Some(5));
        assert_eq!(trimmed.get(3), Some(6));
    }

    #[test]
    fn test_trim_nibbles_prefix_zero() {
        // Create nibbles [10, 11, 12, 13]
        let nibbles = Nibbles::from_nibbles([10, 11, 12, 13]);

        // Trim zero nibbles - should return identical nibbles
        let trimmed = trim_nibbles_prefix(&nibbles, 0);
        assert_eq!(trimmed, nibbles);
    }

    #[test]
    fn test_trim_nibbles_prefix_all() {
        // Create nibbles [1, 2, 3, 4]
        let nibbles = Nibbles::from_nibbles([1, 2, 3, 4]);

        // Trim all nibbles - should return empty
        let trimmed = trim_nibbles_prefix(&nibbles, 4);
        assert!(trimmed.is_empty());
    }

    #[test]
    fn test_trim_nibbles_prefix_empty() {
        // Create empty nibbles
        let nibbles = Nibbles::new();

        // Trim zero from empty - should return empty
        let trimmed = trim_nibbles_prefix(&nibbles, 0);
        assert!(trimmed.is_empty());
    }
}
