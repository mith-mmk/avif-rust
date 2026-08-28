use super::frame::{
    append_native_alpha_plane, validate_native_frame_prefix_limits, validate_native_sequence_limits,
};
use crate::container::parse_avif;
use crate::limits::NativeDecodeLimits;
use crate::obu::{ObuType, find_obu_payload};

fn sample_path(relative: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(relative)
}

fn limits(input_len: usize) -> NativeDecodeLimits {
    NativeDecodeLimits::new(
        input_len.saturating_mul(2),
        8192,
        8192,
        1 << 28,
        1 << 30,
        1 << 24,
        1 << 24,
        128,
        256,
        256,
        8,
        8,
    )
}

fn hide_first_frame(payload: &mut [u8]) {
    let frame = find_obu_payload(payload, ObuType::Frame)
        .expect("frame OBU lookup must succeed")
        .or_else(|| find_obu_payload(payload, ObuType::FrameHeader).unwrap())
        .expect("frame OBU must be present");
    let offset = frame.as_ptr() as usize - payload.as_ptr() as usize;
    assert!(
        payload[offset] & 0x80 == 0,
        "fixture must not be show-existing"
    );
    payload[offset] &= !0x10;
}

#[test]
fn native_primary_prefix_rejects_hidden_frame_before_finish() {
    let data = std::fs::read(sample_path("samples/WML2Viewer.avif"))
        .expect("native hookup fixture must be present");
    let mut info = parse_avif(&data).expect("native hookup fixture must parse");
    hide_first_frame(&mut info.primary_item_payload);
    let resource_limits = limits(data.len());
    let sequence = validate_native_sequence_limits(&info, &resource_limits).unwrap();
    let error = match validate_native_frame_prefix_limits(&info, &sequence, &resource_limits) {
        Ok(_) => panic!("hidden native primary must be rejected before finish"),
        Err(error) => error,
    };
    assert!(
        matches!(error, crate::DecoderError::Unsupported(message) if message.contains("displayed"))
    );
}

#[test]
fn native_alpha_prefix_is_checked_before_primary_decode() {
    let data = std::fs::read(sample_path(
        "test/images/external/avif/unsupported/plum-blossom-small.profile1.8bpc.yuv444.alpha-full.avif",
    ))
    .expect("native alpha hookup fixture must be present");
    let info = parse_avif(&data).expect("native alpha hookup fixture must parse");
    let resource_limits = limits(data.len());
    let prefix = super::still::validate_alpha_auxiliary_header_limits(
        &info,
        &resource_limits,
        info.width.expect("master width must be present") as usize,
        info.height.expect("master height must be present") as usize,
        8,
    )
    .expect("native alpha prefix must pass before primary decode");
    assert!(prefix.prefix.show_frame());
}

#[test]
fn native_alpha_attachment_moves_the_decoded_plane_owner() {
    let data = std::fs::read(sample_path(
        "test/images/external/avif/unsupported/plum-blossom-small.profile1.8bpc.yuv444.alpha-full.avif",
    ))
    .expect("native alpha hookup fixture must be present");
    let info = parse_avif(&data).expect("native alpha hookup fixture must parse");
    let resource_limits = limits(data.len());

    let sequence = validate_native_sequence_limits(&info, &resource_limits).unwrap();
    let master_prefix = validate_native_frame_prefix_limits(&info, &sequence, &resource_limits)
        .expect("native master prefix must pass");
    let alpha_header = super::still::validate_alpha_auxiliary_header_limits(
        &info,
        &resource_limits,
        info.width.expect("master width must be present") as usize,
        info.height.expect("master height must be present") as usize,
        sequence.color_config.bit_depth,
    )
    .expect("native alpha prefix must pass");

    let mut master = {
        let headers = super::parse_av1_headers_with_frame_prefix(&info, master_prefix)
            .expect("native master headers must parse");
        super::decode_still_frame(&headers, Some(&info)).expect("native master must decode")
    };
    let alpha = super::still::decode_alpha_auxiliary_frame_with_prefix(
        &info,
        alpha_header.parts,
        alpha_header.prefix,
    )
    .expect("native alpha must decode");
    let source_plane = alpha
        .buffers
        .planes
        .first()
        .expect("native alpha must contain one plane");
    let source_ptr = source_plane.samples.as_ptr();
    let source_len = source_plane.samples.len();
    let source_capacity = source_plane.samples.capacity();
    let source_samples = source_plane.samples.clone();

    append_native_alpha_plane(&mut master, alpha).expect("native alpha attachment must pass");

    let attached = master
        .buffers
        .planes
        .iter()
        .find(|plane| plane.layout.plane == 3)
        .expect("native alpha plane must be attached as plane 3");
    assert_eq!(attached.samples.as_ptr(), source_ptr);
    assert_eq!(attached.samples.len(), source_len);
    assert_eq!(attached.samples.capacity(), source_capacity);
    assert_eq!(attached.samples.as_slice(), source_samples.as_slice());
}
