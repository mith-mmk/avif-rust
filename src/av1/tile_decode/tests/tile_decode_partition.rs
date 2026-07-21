use crate::av1::sequence::{ChromaSamplePosition, ColorConfig, ColorRange, SequenceHeader};
use crate::av1::syntax::{BlockSize, Partition};
use crate::av1::tile_decode::partition_syntax::{first_partition_child_size, root_block_size};

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

#[test]
fn root_block_size_tracks_superblock_flag() {
    let mut sequence = sample_sequence_header();
    sequence.use_128x128_superblock = false;
    assert_eq!(root_block_size(&sequence), BlockSize::Block64x64);

    sequence.use_128x128_superblock = true;
    assert_eq!(root_block_size(&sequence), BlockSize::Block128x128);
}

fn sample_sequence_header() -> SequenceHeader {
    SequenceHeader {
        seq_profile: 1,
        still_picture: false,
        reduced_still_picture_header: false,
        seq_level_idx_0: 0,
        frame_width_bits: 6,
        frame_height_bits: 6,
        max_frame_width: 64,
        max_frame_height: 64,
        frame_id_numbers_present: false,
        use_128x128_superblock: false,
        enable_filter_intra: true,
        enable_intra_edge_filter: true,
        enable_order_hint: true,
        enable_warped_motion: false,
        order_hint_bits: 7,
        seq_force_screen_content_tools: 2,
        seq_force_integer_mv: 2,
        enable_ref_frame_mvs: false,
        enable_superres: false,
        enable_cdef: false,
        enable_restoration: false,
        color_config: ColorConfig {
            high_bitdepth: false,
            twelve_bit: false,
            bit_depth: 8,
            monochrome: false,
            color_description: None,
            color_range: ColorRange::Full,
            subsampling_x: false,
            subsampling_y: false,
            chroma_sample_position: Some(ChromaSamplePosition::Unknown),
            separate_uv_delta_q: false,
        },
        film_grain_params_present: false,
    }
}
