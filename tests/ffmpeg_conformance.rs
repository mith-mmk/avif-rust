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

fn ffmpeg_decode_rgba_dynamic(path: &Path, width: usize, height: usize) -> Option<Vec<u8>> {
    let executable = std::env::var_os("AVIF_FFMPEG")
        .map(std::path::PathBuf::from)
        .filter(|path| path.is_file())
        .or_else(|| {
            let bundled = Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("workspace root should exist")
                .join("test/images/external/plugins/ffmpeg/ffmpeg.exe");
            bundled.is_file().then_some(bundled)
        })
        .unwrap_or_else(|| std::path::PathBuf::from("ffmpeg"));
    let output = match Command::new(executable)
        .args(["-v", "error", "-nostdin"])
        .arg("-i")
        .arg(path)
        .args(["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgba", "-"])
        .output()
    {
        Ok(output) => output,
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            eprintln!("ffmpeg is not available; skipping external subsampling oracle");
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
    assert_eq!(output.stdout.len(), width * height * 4);
    Some(output.stdout)
}

fn ffmpeg_decode_raw(path: &Path, pixel_format: &str) -> Option<Vec<u8>> {
    let executable = std::env::var_os("AVIF_FFMPEG")
        .map(std::path::PathBuf::from)
        .filter(|path| path.is_file())
        .or_else(|| {
            let bundled = Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("workspace root should exist")
                .join("test/images/external/plugins/ffmpeg/ffmpeg.exe");
            bundled.is_file().then_some(bundled)
        })
        .unwrap_or_else(|| std::path::PathBuf::from("ffmpeg"));
    let output = match Command::new(executable)
        .args(["-v", "error", "-nostdin"])
        .arg("-i")
        .arg(path)
        .args([
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            pixel_format,
            "-",
        ])
        .output()
    {
        Ok(output) => output,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return None,
        Err(err) => panic!("failed to execute ffmpeg: {err}"),
    };
    assert!(
        output.status.success(),
        "ffmpeg failed for {}: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    Some(output.stdout)
}

fn ffmpeg_decode_alpha_plane(path: &Path, width: usize, height: usize) -> Option<Vec<u8>> {
    let executable = std::env::var_os("AVIF_FFMPEG")
        .map(std::path::PathBuf::from)
        .filter(|path| path.is_file())
        .or_else(|| {
            let bundled = Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("workspace root should exist")
                .join("test/images/external/plugins/ffmpeg/ffmpeg.exe");
            bundled.is_file().then_some(bundled)
        })
        .unwrap_or_else(|| std::path::PathBuf::from("ffmpeg"));
    let output = match Command::new(executable)
        .args(["-v", "error", "-nostdin"])
        .arg("-i")
        .arg(path)
        .args([
            "-map",
            "0:1",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ])
        .output()
    {
        Ok(output) => output,
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            eprintln!("ffmpeg is not available; skipping alpha oracle");
            return None;
        }
        Err(err) => panic!("failed to execute ffmpeg: {err}"),
    };
    assert!(
        output.status.success(),
        "ffmpeg alpha decode failed for {}: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout.len(), width * height);
    Some(output.stdout)
}

fn diff_rgb_dynamic(left: &[u8], right: &[u8]) -> DiffMetrics {
    assert_eq!(left.len(), right.len());
    assert_eq!(left.len() % 4, 0);
    let mut sum = 0u64;
    let mut max = 0u8;
    let mut channels = 0usize;
    for (index, (actual, expected)) in left.iter().zip(right.iter()).enumerate() {
        if index % 4 == 3 {
            continue;
        }
        sum += u64::from(actual.abs_diff(*expected));
        max = max.max(actual.abs_diff(*expected));
        channels += 1;
    }
    DiffMetrics {
        average_rgb_abs: sum as f64 / channels as f64,
        max_rgb_abs: max,
    }
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
fn pure_rust_decode_displays_sample_with_filters() {
    let avif_data =
        std::fs::read(sample_path("WML2Viewer.avif")).expect("sample AVIF should exist");
    let image = avif_rust::image_from_bytes(&avif_data).expect("AVIF sample should decode");
    assert_eq!((image.width, image.height), (SAMPLE_WIDTH, SAMPLE_HEIGHT));
    assert_eq!(image.rgba.len(), SAMPLE_RGBA_LEN);
}

#[test]
fn pure_rust_decode_exposes_filtered_source_planes() {
    let avif_data =
        std::fs::read(sample_path("WML2Viewer.avif")).expect("sample AVIF should exist");
    let frame = avif_rust::decode_frame_bytes(&avif_data).expect("AVIF sample should decode");
    assert_eq!((frame.width, frame.height), (SAMPLE_WIDTH, SAMPLE_HEIGHT));
    assert_eq!(frame.buffers.planes.len(), 3);
    for plane in &frame.buffers.planes {
        assert_eq!(
            (plane.layout.width, plane.layout.height),
            (SAMPLE_WIDTH, SAMPLE_HEIGHT)
        );
    }
}

#[test]
fn public_subsampling_samples_match_ffmpeg_rgba_when_present() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root should exist");
    let cases = [
        ("avif/unsupported/fox.profile0.8bpc.yuv420.avif", 1204, 800),
        (
            "avif/unsupported/fox.profile0.8bpc.yuv420.monochrome.avif",
            1204,
            800,
        ),
        (
            "avif/supported/fox.profile0.8bpc.yuv420.odd-width.avif",
            1203,
            800,
        ),
        (
            "avif/supported/fox.profile0.8bpc.yuv420.odd-height.avif",
            1204,
            799,
        ),
        (
            "avif/supported/fox.profile0.8bpc.yuv420.monochrome.odd-width.odd-height.avif",
            1203,
            799,
        ),
        ("avif/unsupported/fox.profile2.8bpc.yuv422.avif", 1204, 800),
        (
            "avif/supported/fox.profile2.8bpc.yuv422.odd-width.avif",
            1203,
            800,
        ),
        (
            "avif/supported/fox.profile2.8bpc.yuv422.odd-height.avif",
            1204,
            799,
        ),
        (
            "avif/supported/fox.profile2.8bpc.yuv422.odd-width.odd-height.avif",
            1203,
            799,
        ),
    ];
    if cases
        .iter()
        .map(|(relative, _, _)| root.join("test/images/external").join(relative))
        .any(|path| !path.is_file())
    {
        eprintln!("external subsampling samples are unavailable; skipping oracle");
        return;
    }
    for (relative, width, height) in cases {
        let path = root.join("test/images/external").join(relative);
        let data = std::fs::read(&path).expect("external AVIF sample should be readable");
        let actual = avif_rust::image_from_bytes(&data).expect("subsampling sample should decode");
        assert_eq!((actual.width, actual.height), (width, height), "{relative}");
        let Some(expected) = ffmpeg_decode_rgba_dynamic(&path, width, height) else {
            return;
        };
        let metrics = diff_rgb_dynamic(&actual.rgba, &expected);
        eprintln!(
            "{relative}: average RGB absolute error={}, max={}",
            metrics.average_rgb_abs, metrics.max_rgb_abs
        );
        assert!(
            metrics.average_rgb_abs <= 2.0 && metrics.max_rgb_abs <= 32,
            "{relative}: FFmpeg RGBA error average={} max={}",
            metrics.average_rgb_abs,
            metrics.max_rgb_abs
        );
    }
}

#[test]
fn public_subsampling_planes_match_ffmpeg_when_present() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root should exist");
    let cases = [
        (
            "avif/unsupported/fox.profile0.8bpc.yuv420.avif",
            "yuv420p",
            [1204 * 800, 602 * 400, 602 * 400],
        ),
        (
            "avif/unsupported/fox.profile0.8bpc.yuv420.monochrome.avif",
            "gray",
            [1204 * 800, 0, 0],
        ),
        (
            "avif/unsupported/fox.profile2.8bpc.yuv422.avif",
            "yuv422p",
            [1204 * 800, 602 * 800, 602 * 800],
        ),
    ];
    for (relative, pixel_format, plane_lengths) in cases {
        let path = root.join("test/images/external").join(relative);
        if !path.is_file() {
            eprintln!("external subsampling sample is unavailable; skipping {relative}");
            return;
        }
        let data = std::fs::read(&path).expect("external AVIF sample should be readable");
        let actual =
            avif_rust::decode_frame_bytes(&data).expect("subsampling sample should decode");
        let Some(expected) = ffmpeg_decode_raw(&path, pixel_format) else {
            return;
        };
        let expected_len = plane_lengths.iter().sum::<usize>();
        assert_eq!(expected.len(), expected_len, "{relative}");
        let mut offset = 0;
        for (plane_index, &plane_len) in plane_lengths.iter().enumerate() {
            if plane_len == 0 {
                continue;
            }
            let expected_plane = &expected[offset..offset + plane_len];
            let actual_plane = &actual.buffers.planes[plane_index].samples;
            assert_eq!(
                actual_plane.len(),
                plane_len,
                "{relative} plane {plane_index}"
            );
            let errors = actual_plane
                .iter()
                .zip(expected_plane)
                .map(|(actual, expected)| u8::try_from(*actual).unwrap().abs_diff(*expected));
            let max_error = errors.clone().max().unwrap_or(0);
            let average_error = errors.map(u64::from).sum::<u64>() as f64 / plane_len as f64;
            assert!(
                average_error <= 2.0 && max_error <= 32,
                "{relative} plane {plane_index}: average={average_error} max={max_error}"
            );
            offset += plane_len;
        }
    }
}

#[test]
fn public_10bit_sample_matches_ffmpeg_when_present() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root should exist")
        .join("test/images/external/avif/unsupported/fox.profile1.10bpc.yuv444.avif");
    if !path.is_file() {
        eprintln!("external 10-bit sample is unavailable; skipping oracle");
        return;
    }
    let data = std::fs::read(&path).expect("external 10-bit AVIF should be readable");
    let public_image =
        avif_rust::image_from_bytes(&data).expect("10-bit public decode should succeed");
    assert_eq!((public_image.width, public_image.height), (1204, 800));
    let frame = avif_rust::decode_frame_bytes(&data).expect("10-bit sample should decode");
    assert_eq!((frame.width, frame.height), (1204, 800));
    assert_eq!(frame.buffers.planes.len(), 3);
    let Some(expected_bytes) = ffmpeg_decode_raw(&path, "yuv444p10le") else {
        return;
    };
    assert_eq!(expected_bytes.len(), 1204 * 800 * 3 * 2);
    let expected: Vec<u16> = expected_bytes
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]) & 0x03ff)
        .collect();
    let plane_samples = 1204 * 800;
    for plane_index in 0..3 {
        let actual = &frame.buffers.planes[plane_index].samples;
        let expected = &expected[plane_index * plane_samples..(plane_index + 1) * plane_samples];
        let max_error = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| actual.abs_diff(*expected))
            .max()
            .unwrap_or(0);
        let average_error = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| f64::from(actual.abs_diff(*expected)))
            .sum::<f64>()
            / actual.len() as f64;
        eprintln!("10-bit plane {plane_index}: average={average_error} max={max_error}");
        assert!(
            average_error <= 1.0 && max_error <= 16,
            "10-bit plane {plane_index}: average={average_error} max={max_error}"
        );
    }
    let actual = frame
        .to_rgba8()
        .expect("10-bit RGBA conversion should succeed");
    let rgba16 = frame
        .to_rgba16()
        .expect("10-bit RGBA16 conversion should succeed");
    assert_eq!(rgba16.rgba.len(), 1204 * 800 * 4);
    let expected = ffmpeg_decode_rgba_dynamic(&path, 1204, 800).unwrap();
    let metrics = diff_rgb_dynamic(&actual.rgba, &expected);
    eprintln!(
        "10-bit RGBA: average={} max={}",
        metrics.average_rgb_abs, metrics.max_rgb_abs
    );
    assert!(metrics.average_rgb_abs <= 2.0 && metrics.max_rgb_abs <= 32);
}

#[test]
fn public_grid_sample_matches_ffmpeg_when_present() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root should exist")
        .join("test/images/external/avif/unsupported/sofa_grid1x5_420.avif");
    if !path.is_file() {
        eprintln!("external grid sample is unavailable; skipping oracle");
        return;
    }
    let data = std::fs::read(&path).expect("external grid AVIF should be readable");
    let actual = avif_rust::image_from_bytes(&data).expect("grid public decode should succeed");
    assert_eq!((actual.width, actual.height), (1024, 770));
    let Some(expected) = ffmpeg_decode_rgba_dynamic(&path, 1024, 770) else {
        return;
    };
    let metrics = diff_rgb_dynamic(&actual.rgba, &expected);
    eprintln!(
        "grid sofa_grid1x5_420: average RGB absolute error={}, max={}",
        metrics.average_rgb_abs, metrics.max_rgb_abs
    );
    assert!(
        metrics.average_rgb_abs <= 2.0 && metrics.max_rgb_abs <= 48,
        "grid FFmpeg RGBA error average={} max={}",
        metrics.average_rgb_abs,
        metrics.max_rgb_abs
    );
}

#[test]
fn public_12bit_sample_rejects_active_film_grain_when_present() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root should exist")
        .join("test/images/external/avif/unsupported/fox.profile2.12bpc.yuv444.avif");
    if !path.is_file() {
        eprintln!("external 12-bit sample is unavailable; skipping unsupported-feature check");
        return;
    }
    let data = std::fs::read(&path).expect("external 12-bit AVIF should be readable");
    let error =
        avif_rust::image_from_bytes(&data).expect_err("active film grain must remain fail-closed");
    assert!(
        error
            .to_string()
            .contains("AV1 film grain is not supported by public decode yet"),
        "unexpected 12-bit error: {error}"
    );
}

#[test]
fn public_alpha_sample_matches_ffmpeg_rgba_when_present() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root should exist")
        .join("test/images/external/avif/unsupported/plum-blossom-small.profile1.8bpc.yuv444.alpha-full.avif");
    if !path.is_file() {
        eprintln!("external alpha sample is unavailable; skipping oracle");
        return;
    }
    let data = std::fs::read(&path).expect("external alpha AVIF should be readable");
    let actual = avif_rust::image_from_bytes(&data).expect("alpha sample should decode");
    assert_eq!((actual.width, actual.height), (128, 128));
    let Some(expected_rgb) = ffmpeg_decode_rgba_dynamic(&path, 128, 128) else {
        return;
    };
    let Some(expected_alpha) = ffmpeg_decode_alpha_plane(&path, 128, 128) else {
        return;
    };
    let metrics = diff_rgb_dynamic(&actual.rgba, &expected_rgb);
    let alpha_max = actual
        .rgba
        .chunks_exact(4)
        .zip(expected_alpha)
        .map(|(actual, expected)| actual[3].abs_diff(expected))
        .max()
        .unwrap_or(0);
    eprintln!(
        "{}: average RGB absolute error={}, max={}, alpha_max={}",
        path.display(),
        metrics.average_rgb_abs,
        metrics.max_rgb_abs,
        alpha_max
    );
    assert!(metrics.average_rgb_abs <= 2.0 && metrics.max_rgb_abs <= 32);
    assert!(alpha_max <= 1, "alpha channel max error was {alpha_max}");
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

#[test]
fn public_transform_samples_cover_crop_and_mirror() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root should exist");
    let sample = |name: &str| {
        root.join("test/images/external/avif/unsupported")
            .join(name)
    };

    for (name, expected_width, expected_height) in [
        ("kimono.crop.avif", 385, 330),
        ("kimono.mirror-horizontal.avif", 722, 1024),
    ] {
        let path = sample(name);
        if !path.is_file() {
            eprintln!(
                "external transform sample missing; skipping {}",
                path.display()
            );
            return;
        }
        let image = avif_rust::image_from_bytes(
            &std::fs::read(&path).expect("transform sample should be readable"),
        )
        .expect("crop/mirror transform should decode");
        assert_eq!(
            (image.width, image.height),
            (expected_width, expected_height)
        );
    }
}
