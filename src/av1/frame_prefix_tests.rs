use super::super::sequence::{ColorConfig, ColorRange, SequenceHeaderMetadata};
use super::{
    FrameType, InterpolationFilter, NO_REFERENCES, SequenceHeader, finish_frame_header,
    parse_frame_header_with_references_and_metadata, parse_frame_prefix,
};
use crate::DecoderError;

fn deterministic_prefix_sequence() -> SequenceHeader {
    SequenceHeader {
        seq_profile: 0,
        still_picture: true,
        reduced_still_picture_header: false,
        seq_level_idx_0: 0,
        frame_width_bits: 5,
        frame_height_bits: 5,
        max_frame_width: 17,
        max_frame_height: 19,
        frame_id_numbers_present: false,
        frame_id_length: 0,
        delta_frame_id_length: 0,
        use_128x128_superblock: false,
        enable_filter_intra: false,
        enable_intra_edge_filter: false,
        enable_dual_filter: false,
        enable_masked_compound: false,
        enable_interintra_compound: false,
        enable_dist_wtd_comp: false,
        enable_order_hint: false,
        enable_warped_motion: false,
        order_hint_bits: 0,
        seq_force_screen_content_tools: 0,
        seq_force_integer_mv: 0,
        enable_ref_frame_mvs: false,
        enable_superres: false,
        enable_cdef: false,
        enable_restoration: false,
        color_config: ColorConfig {
            high_bitdepth: false,
            twelve_bit: false,
            bit_depth: 8,
            monochrome: true,
            color_description: None,
            color_range: ColorRange::Full,
            subsampling_x: true,
            subsampling_y: true,
            chroma_sample_position: None,
            separate_uv_delta_q: false,
        },
        film_grain_params_present: false,
    }
}

fn parse_prefix(data: &[u8]) -> super::FramePrefix<'_, 'static> {
    parse_frame_prefix(
        data,
        &deterministic_prefix_sequence(),
        &SequenceHeaderMetadata::default(),
        &NO_REFERENCES,
    )
    .expect("synthetic normal frame prefix should parse")
}

#[test]
fn normal_prefix_retains_all_pre_tile_fields_and_exact_reader_offset() {
    // show_existing=0, frame_type=KEY, show_frame=1,
    // disable_cdf_update=0, frame_size_override=0,
    // render_and_frame_size_different=0, disable_frame_end_update_cdf=1.
    let data = [0b0001_0001];
    let prefix = parse_prefix(&data);

    assert_eq!(prefix.reader.bit_position(), 8);
    assert_eq!(prefix.frame_type, FrameType::Key);
    assert!(!prefix.show_existing_frame);
    assert!(prefix.show_frame);
    assert!(!prefix.showable_frame);
    assert!(prefix.error_resilient_mode);
    assert!(!prefix.disable_cdf_update);
    assert!(!prefix.allow_screen_content_tools);
    assert_eq!(prefix.force_integer_mv, 2);
    assert!(!prefix.frame_size_override_flag);
    assert_eq!(prefix.order_hint, 0);
    assert_eq!(prefix.primary_ref_frame, 7);
    assert_eq!(prefix.refresh_frame_flags, 0xff);
    assert_eq!(prefix.reference_frame_indices, [0; 7]);
    assert_eq!(prefix.reference_order_hints, [None; 7]);
    assert!(!prefix.frame_refs_short_signaling);
    assert_eq!(prefix.frame_id, None);
    assert!(!prefix.allow_high_precision_mv);
    assert!(!prefix.is_filter_switchable);
    assert_eq!(prefix.interpolation_filter, InterpolationFilter::Regular);
    assert!(!prefix.is_motion_mode_switchable);
    assert!(!prefix.use_ref_frame_mvs);
    assert_eq!(
        (
            prefix.frame_size.width,
            prefix.frame_size.height,
            prefix.frame_size.upscaled_width
        ),
        (17, 19, 17)
    );
    assert_eq!(
        (prefix.render_size.width, prefix.render_size.height),
        (17, 19)
    );
    assert!(!prefix.allow_intrabc);
    assert!(prefix.disable_frame_end_update_cdf);
    assert!(prefix.frame_is_intra);
}

#[test]
fn normal_prefix_and_finish_share_the_same_reader_and_fields() {
    let sequence = deterministic_prefix_sequence();
    let data = [0x11, 0, 0];
    let prefix = parse_frame_prefix(
        &data,
        &sequence,
        &SequenceHeaderMetadata::default(),
        &NO_REFERENCES,
    )
    .expect("synthetic normal frame prefix should parse");
    assert_eq!(prefix.reader.bit_position(), 8);
    let header = finish_frame_header(prefix).expect("synthetic normal frame should finish");
    let wrapped = parse_frame_header_with_references_and_metadata(
        &data,
        &sequence,
        &SequenceHeaderMetadata::default(),
        &NO_REFERENCES,
    )
    .expect("existing full-header wrapper should finish identically");

    assert_eq!(header, wrapped);
    assert_eq!(header.frame_type, FrameType::Key);
    assert!(header.show_frame);
    assert!(!header.show_existing_frame);
    assert_eq!((header.frame_width, header.frame_height), (17, 19));
    assert_eq!(header.upscaled_width, 17);
    assert_eq!((header.render_width, header.render_height), (17, 19));
    assert_eq!(header.uncompressed_header_bits, 21);
    assert_eq!(header.payload_after_header_offset, 3);
}

#[test]
fn prefix_reports_truncation_before_tile_or_trailing_syntax() {
    let sequence = deterministic_prefix_sequence();
    let error = match parse_frame_prefix(
        &[],
        &sequence,
        &SequenceHeaderMetadata::default(),
        &NO_REFERENCES,
    ) {
        Ok(_) => panic!("truncated prefix unexpectedly parsed"),
        Err(error) => error,
    };
    assert!(
        matches!(error, DecoderError::NotEnoughData(message) if message.contains("show_existing_frame"))
    );
}

#[test]
fn finished_header_reports_truncation_after_prefix() {
    let sequence = deterministic_prefix_sequence();
    let data = [0x11, 0];
    let prefix = parse_frame_prefix(
        &data,
        &sequence,
        &SequenceHeaderMetadata::default(),
        &NO_REFERENCES,
    )
    .expect("prefix should finish before the tile syntax");
    let direct_error = finish_frame_header(prefix).expect_err("tile/trailing syntax is truncated");
    let wrapped_error = parse_frame_header_with_references_and_metadata(
        &data,
        &sequence,
        &SequenceHeaderMetadata::default(),
        &NO_REFERENCES,
    )
    .expect_err("the full-header wrapper should report the same truncation");

    assert_eq!(direct_error, wrapped_error);
    assert!(matches!(direct_error, DecoderError::NotEnoughData(_)));
}
