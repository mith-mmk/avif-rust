use crate::av1::syntax::{BlockSize, Partition};
use crate::av1::tile_decode::partition_syntax::first_partition_child_size;

#[test]
fn first_partition_child_matches_decode_order() {
    assert_eq!(
        first_partition_child_size(BlockSize::Block128x128, Partition::Vertical).unwrap(),
        BlockSize::Block64x128
    );
    assert_eq!(
        first_partition_child_size(BlockSize::Block128x128, Partition::HorizontalA).unwrap(),
        BlockSize::Block64x64
    );
    assert_eq!(
        first_partition_child_size(BlockSize::Block128x128, Partition::VerticalB).unwrap(),
        BlockSize::Block64x128
    );
}

#[test]
fn first_partition_child_rejects_invalid_edge_shape() {
    assert!(first_partition_child_size(BlockSize::Block4x4, Partition::Vertical).is_err());
}
