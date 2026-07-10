use std::io;
use std::path::Path;
use std::process::Command;

mod support;

use support::{
    ExpectedPlane, assert_exact_decoded_planes, assert_rgba8_max_error, assert_rgba16_max_error,
    sample_path,
};

const SAMPLE_WIDTH: usize = 900;
const SAMPLE_HEIGHT: usize = 900;
const SAMPLE_PIXELS: usize = SAMPLE_WIDTH * SAMPLE_HEIGHT;
const SAMPLE_RGBA_LEN: usize = SAMPLE_PIXELS * 4;

#[derive(Debug)]
struct DiffMetrics {
    average_rgb_abs: f64,
    max_rgb_abs: u8,
}

fn ffmpeg_decode_rgba(path: &Path) -> Option<Vec<u8>> {
    let output = match Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin"])
        .arg("-i")
        .arg(path)
        .args(["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgba", "-"])
        .output()
    {
        Ok(output) => output,
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            eprintln!("ffmpeg is not available; skipping AVIF oracle comparison");
            return None;
        }
        Err(err) => panic!("failed to execute ffmpeg: {err}"),
    };

    assert!(
        output.status.success(),
        "ffmpeg failed for {}: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout.len(), SAMPLE_RGBA_LEN);
    Some(output.stdout)
}

fn diff_rgb(left: &[u8], right: &[u8]) -> DiffMetrics {
    assert_eq!(left.len(), right.len());
    assert_eq!(left.len(), SAMPLE_RGBA_LEN);

    let mut sum = 0u64;
    let mut max = 0u8;
    for index in 0..left.len() {
        if index % 4 == 3 {
            continue;
        }
        let diff = left[index].abs_diff(right[index]);
        sum += u64::from(diff);
        max = max.max(diff);
    }

    DiffMetrics {
        average_rgb_abs: sum as f64 / (SAMPLE_PIXELS * 3) as f64,
        max_rgb_abs: max,
    }
}

#[test]
fn ffmpeg_avif_decode_is_close_to_original_png() {
    let Some(avif_rgba) = ffmpeg_decode_rgba(&sample_path("WML2Viewer.avif")) else {
        return;
    };
    let Some(png_rgba) = ffmpeg_decode_rgba(&sample_path("WML2Viewer.png")) else {
        return;
    };
    let metrics = diff_rgb(&avif_rgba, &png_rgba);
    assert!(
        metrics.average_rgb_abs <= 0.5,
        "average RGB absolute error was {}",
        metrics.average_rgb_abs
    );
    assert!(
        metrics.max_rgb_abs <= 40,
        "max RGB absolute error was {}",
        metrics.max_rgb_abs
    );
}

#[test]
fn layered_conformance_helpers_compare_planes_and_rgba_max_error() {
    let layout = avif_rust::av1::PlaneLayout {
        plane: 0,
        width: 2,
        height: 1,
        subsampling_x: 0,
        subsampling_y: 0,
        sample_count: 2,
    };
    let frame = avif_rust::DecodedFrame {
        width: 2,
        height: 1,
        render_width: 2,
        render_height: 1,
        bit_depth: 8,
        color_config: avif_rust::av1::ColorConfig {
            high_bitdepth: false,
            twelve_bit: false,
            bit_depth: 8,
            monochrome: false,
            color_description: Some(avif_rust::av1::ColorDescription {
                color_primaries: 1,
                transfer_characteristics: 13,
                matrix_coefficients: 0,
            }),
            color_range: avif_rust::av1::ColorRange::Full,
            subsampling_x: false,
            subsampling_y: false,
            chroma_sample_position: None,
            separate_uv_delta_q: false,
        },
        color_information: None,
        alpha_premultiplied: false,
        buffers: avif_rust::av1::FrameBuffers {
            width: 2,
            height: 1,
            planes: vec![
                avif_rust::av1::PlaneBuffer {
                    layout,
                    samples: vec![20, 21],
                },
                avif_rust::av1::PlaneBuffer {
                    layout: avif_rust::av1::PlaneLayout { plane: 1, ..layout },
                    samples: vec![30, 31],
                },
                avif_rust::av1::PlaneBuffer {
                    layout: avif_rust::av1::PlaneLayout { plane: 2, ..layout },
                    samples: vec![40, 41],
                },
            ],
        },
    };

    assert_exact_decoded_planes(
        &frame,
        &[
            ExpectedPlane {
                width: 2,
                height: 1,
                stride: 2,
                samples: &[20, 21],
            },
            ExpectedPlane {
                width: 2,
                height: 1,
                stride: 2,
                samples: &[30, 31],
            },
            ExpectedPlane {
                width: 2,
                height: 1,
                stride: 2,
                samples: &[40, 41],
            },
        ],
    );

    let rgba = frame.to_rgba8().expect("identity GBR should convert");
    assert_rgba8_max_error(
        &rgba.rgba,
        &[40, 20, 30, 255, 41, 21, 31, 255],
        0,
        "synthetic identity GBR",
    );

    let rgba16 = frame.to_rgba16().expect("identity GBR should convert");
    assert_rgba16_max_error(
        &rgba16.rgba,
        &[10280, 5140, 7710, u16::MAX, 10537, 5397, 7967, u16::MAX],
        1,
        "synthetic identity GBR",
    );
}

#[test]
fn pure_rust_decode_returns_sample_rgba_dimensions() {
    let avif_data =
        std::fs::read(sample_path("WML2Viewer.avif")).expect("sample AVIF should exist");
    let decoded = avif_rust::image_from_bytes(&avif_data).expect("AVIF should decode");

    assert_eq!(decoded.width, SAMPLE_WIDTH);
    assert_eq!(decoded.height, SAMPLE_HEIGHT);
    assert_eq!(decoded.rgba.len(), SAMPLE_RGBA_LEN);
    assert!(decoded.rgba.chunks_exact(4).all(|pixel| pixel[3] == 255));
    assert!(
        decoded
            .rgba
            .chunks_exact(4)
            .any(|pixel| pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0)
    );
}

#[test]
fn pure_rust_decode_exposes_source_planes_for_oracle_tests() {
    let avif_data =
        std::fs::read(sample_path("WML2Viewer.avif")).expect("sample AVIF should exist");
    let decoded = avif_rust::decode_frame_bytes(&avif_data).expect("AVIF should decode");

    assert_eq!(decoded.width, SAMPLE_WIDTH);
    assert_eq!(decoded.height, SAMPLE_HEIGHT);
    assert_eq!(decoded.render_width, SAMPLE_WIDTH);
    assert_eq!(decoded.render_height, SAMPLE_HEIGHT);
    assert_eq!(decoded.bit_depth, 8);
    assert_eq!(decoded.buffers.planes.len(), 3);

    let parsed_info = avif_rust::container::parse_avif(&avif_data).expect("AVIF should parse");
    assert_eq!(decoded.color_information, parsed_info.color_information);
    assert_eq!(decoded.alpha_premultiplied, parsed_info.alpha_premultiplied);

    for (plane_index, plane) in decoded.buffers.planes.iter().enumerate() {
        assert_eq!(usize::from(plane.layout.plane), plane_index);
        assert_eq!(plane.layout.width, SAMPLE_WIDTH);
        assert_eq!(plane.layout.height, SAMPLE_HEIGHT);
        assert_eq!(plane.layout.stride(), SAMPLE_WIDTH);
        assert_eq!(plane.layout.sample_count, SAMPLE_PIXELS);
        assert_eq!(plane.samples.len(), SAMPLE_PIXELS);
    }

    let rgba8 = decoded
        .to_rgba8()
        .expect("identity GBR should convert to RGBA8");
    assert_eq!(rgba8.width, SAMPLE_WIDTH);
    assert_eq!(rgba8.height, SAMPLE_HEIGHT);
    assert_eq!(rgba8.rgba.len(), SAMPLE_RGBA_LEN);

    let rgba16 = decoded
        .to_rgba16()
        .expect("identity GBR should convert to RGBA16");
    assert_eq!(rgba16.width, SAMPLE_WIDTH);
    assert_eq!(rgba16.height, SAMPLE_HEIGHT);
    assert_eq!(rgba16.rgba.len(), SAMPLE_RGBA_LEN);
    assert!(
        rgba16
            .rgba
            .chunks_exact(4)
            .all(|pixel| pixel[3] == u16::MAX)
    );
}

#[test]
fn decoded_frame_rejects_icc_rgba_conversion_until_colour_management_exists() {
    let layout = avif_rust::av1::PlaneLayout {
        plane: 0,
        width: 1,
        height: 1,
        subsampling_x: 0,
        subsampling_y: 0,
        sample_count: 1,
    };
    let frame = avif_rust::DecodedFrame {
        width: 1,
        height: 1,
        render_width: 1,
        render_height: 1,
        bit_depth: 8,
        color_config: avif_rust::av1::ColorConfig {
            high_bitdepth: false,
            twelve_bit: false,
            bit_depth: 8,
            monochrome: false,
            color_description: Some(avif_rust::av1::ColorDescription {
                color_primaries: 1,
                transfer_characteristics: 13,
                matrix_coefficients: 0,
            }),
            color_range: avif_rust::av1::ColorRange::Full,
            subsampling_x: false,
            subsampling_y: false,
            chroma_sample_position: None,
            separate_uv_delta_q: false,
        },
        color_information: Some(avif_rust::ColorInformation {
            color_type: *b"prof",
            payload: vec![1, 2, 3],
        }),
        alpha_premultiplied: false,
        buffers: avif_rust::av1::FrameBuffers {
            width: 1,
            height: 1,
            planes: vec![
                avif_rust::av1::PlaneBuffer {
                    layout,
                    samples: vec![10],
                },
                avif_rust::av1::PlaneBuffer {
                    layout: avif_rust::av1::PlaneLayout { plane: 1, ..layout },
                    samples: vec![20],
                },
                avif_rust::av1::PlaneBuffer {
                    layout: avif_rust::av1::PlaneLayout { plane: 2, ..layout },
                    samples: vec![30],
                },
            ],
        },
    };

    let err = frame.to_rgba8().unwrap_err();

    assert!(
        matches!(err, avif_rust::DecoderError::Unsupported(message) if message.contains("ICC"))
    );
}

#[test]
#[ignore = "single-sample RGB error is diagnostic until plane-level conformance fixtures exist"]
fn report_current_wml2viewer_rgb_error() {
    let avif_data =
        std::fs::read(sample_path("WML2Viewer.avif")).expect("sample AVIF should exist");
    let decoded = avif_rust::image_from_bytes(&avif_data).expect("AVIF should decode");
    let Some(ffmpeg_rgba) = ffmpeg_decode_rgba(&sample_path("WML2Viewer.avif")) else {
        return;
    };

    let metrics = diff_rgb(&decoded.rgba, &ffmpeg_rgba);
    eprintln!("average RGB absolute error: {}", metrics.average_rgb_abs);
}

#[test]
#[ignore = "requires AOM_PREFILTER_ORACLE from a decoder build with post-filters disabled"]
fn report_current_wml2viewer_prefilter_plane_error() {
    let Some(oracle_path) = std::env::var_os("AOM_PREFILTER_ORACLE") else {
        eprintln!("AOM_PREFILTER_ORACLE is not set; skipping pre-filter plane diagnostic");
        return;
    };
    let oracle = std::fs::read(&oracle_path).unwrap_or_else(|err| {
        panic!(
            "failed to read pre-filter oracle {}: {err}",
            Path::new(&oracle_path).display()
        )
    });
    assert_eq!(oracle.len(), SAMPLE_PIXELS * 3);

    let avif_data =
        std::fs::read(sample_path("WML2Viewer.avif")).expect("sample AVIF should exist");
    let decoded = avif_rust::decode_frame_bytes(&avif_data).expect("AVIF should decode");
    assert_eq!(decoded.buffers.planes.len(), 3);

    for (plane_index, plane) in decoded.buffers.planes.iter().enumerate() {
        assert_eq!(plane.layout.width, SAMPLE_WIDTH);
        assert_eq!(plane.layout.height, SAMPLE_HEIGHT);
        let expected = &oracle[plane_index * SAMPLE_PIXELS..(plane_index + 1) * SAMPLE_PIXELS];
        let first_mismatch = plane
            .samples
            .iter()
            .zip(expected)
            .position(|(&actual, &expected)| actual != u16::from(expected));
        let mismatches = plane
            .samples
            .iter()
            .zip(expected)
            .filter(|(actual, expected)| **actual != u16::from(**expected))
            .count();
        eprintln!(
            "pre-filter plane {plane_index}: first mismatch={first_mismatch:?}, mismatches={mismatches}"
        );
        if let Some(index) = first_mismatch {
            let row_start = index / SAMPLE_WIDTH * SAMPLE_WIDTH;
            let start = index.saturating_sub(4).max(row_start);
            let end = (index + 12).min(row_start + SAMPLE_WIDTH);
            eprintln!(
                "pre-filter plane {plane_index} window {start}..{end}: actual={:?} expected={:?}",
                &plane.samples[start..end],
                &expected[start..end]
            );
        }
    }
}

#[test]
#[ignore = "pure Rust output does not yet meet the AV1 conformance threshold"]
fn pure_rust_decode_matches_ffmpeg_oracle_and_original_png() {
    let avif_data =
        std::fs::read(sample_path("WML2Viewer.avif")).expect("sample AVIF should exist");
    let decoded = avif_rust::image_from_bytes(&avif_data).expect("AVIF should decode");
    assert_eq!(decoded.width, SAMPLE_WIDTH);
    assert_eq!(decoded.height, SAMPLE_HEIGHT);
    assert_eq!(decoded.rgba.len(), SAMPLE_RGBA_LEN);

    let Some(ffmpeg_rgba) = ffmpeg_decode_rgba(&sample_path("WML2Viewer.avif")) else {
        return;
    };
    let Some(png_rgba) = ffmpeg_decode_rgba(&sample_path("WML2Viewer.png")) else {
        return;
    };
    let ffmpeg_metrics = diff_rgb(&decoded.rgba, &ffmpeg_rgba);
    assert!(
        ffmpeg_metrics.average_rgb_abs <= 0.5,
        "average RGB absolute error against ffmpeg was {}",
        ffmpeg_metrics.average_rgb_abs
    );
    assert!(
        ffmpeg_metrics.max_rgb_abs <= 4,
        "max RGB absolute error against ffmpeg was {}",
        ffmpeg_metrics.max_rgb_abs
    );

    let png_metrics = diff_rgb(&decoded.rgba, &png_rgba);
    assert!(
        png_metrics.average_rgb_abs <= 0.5,
        "average RGB absolute error against original PNG was {}",
        png_metrics.average_rgb_abs
    );
    assert!(
        png_metrics.max_rgb_abs <= 40,
        "max RGB absolute error against original PNG was {}",
        png_metrics.max_rgb_abs
    );
}
