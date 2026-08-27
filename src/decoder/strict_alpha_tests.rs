use super::frame::validate_strict_alpha;
use super::*;

fn frame(width: usize, height: usize, config: ColorConfig, plane_count: usize) -> DecodedFrame {
    let planes = (0..plane_count)
        .map(|plane| PlaneBuffer {
            layout: PlaneLayout {
                plane: plane as u8,
                width,
                height,
                subsampling_x: 0,
                subsampling_y: 0,
                sample_count: width * height,
            },
            samples: vec![0; width * height],
        })
        .collect();
    DecodedFrame {
        width,
        height,
        render_width: width,
        render_height: height,
        bit_depth: config.bit_depth,
        color_config: config,
        color_information: None,
        alpha_premultiplied: false,
        buffers: FrameBuffers {
            width,
            height,
            planes,
        },
    }
}

fn alpha_config() -> ColorConfig {
    ColorConfig {
        high_bitdepth: false,
        twelve_bit: false,
        bit_depth: 8,
        monochrome: true,
        color_description: None,
        color_range: ColorRange::Full,
        subsampling_x: false,
        subsampling_y: false,
        chroma_sample_position: None,
        separate_uv_delta_q: false,
    }
}

#[test]
fn strict_alpha_accepts_matching_mono_full_range_frame() {
    let master = frame(3, 5, alpha_config(), 1);
    let alpha = frame(3, 5, alpha_config(), 1);
    validate_strict_alpha(&master, &alpha).unwrap();
}

#[test]
fn strict_alpha_rejects_dimensions_depth_mono_and_range() {
    let master = frame(3, 5, alpha_config(), 1);

    let mut mismatch = frame(4, 5, alpha_config(), 1);
    assert!(matches!(
        validate_strict_alpha(&master, &mismatch),
        Err(DecoderError::Bitstream(message)) if message.contains("dimensions")
    ));

    mismatch = frame(3, 5, alpha_config(), 1);
    mismatch.color_config.bit_depth = 10;
    mismatch.bit_depth = 10;
    assert!(matches!(
        validate_strict_alpha(&master, &mismatch),
        Err(DecoderError::Unsupported(message)) if message.contains("bit depth")
    ));

    mismatch = frame(3, 5, alpha_config(), 3);
    assert!(matches!(
        validate_strict_alpha(&master, &mismatch),
        Err(DecoderError::Unsupported(message)) if message.contains("monochrome")
    ));

    mismatch = frame(3, 5, alpha_config(), 1);
    mismatch.color_config.monochrome = false;
    assert!(matches!(
        validate_strict_alpha(&master, &mismatch),
        Err(DecoderError::Unsupported(message)) if message.contains("monochrome")
    ));

    mismatch = frame(3, 5, alpha_config(), 1);
    mismatch.color_config.color_range = ColorRange::Studio;
    assert!(matches!(
        validate_strict_alpha(&master, &mismatch),
        Err(DecoderError::Unsupported(message)) if message.contains("full range")
    ));
}

#[test]
fn strict_alpha_rejects_plane_header_geometry_mismatch() {
    let master = frame(3, 5, alpha_config(), 1);
    let mut alpha = frame(3, 5, alpha_config(), 1);
    alpha.buffers.planes[0].layout.width = 2;
    assert!(matches!(
        validate_strict_alpha(&master, &alpha),
        Err(DecoderError::Bitstream(message)) if message.contains("plane geometry")
    ));
}
