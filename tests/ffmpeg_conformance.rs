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
fn generated_two_tile_sample_matches_ffmpeg_when_encoder_present() {
    let root = std::env::temp_dir().join(format!(".test-avif-multitile-{}", std::process::id()));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary AVIF sample directory: {err}");
    }
    let output_path = root.join("two-tile.avif");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .arg("-i")
        .arg(sample_path("WML2Viewer.png"))
        .args([
            "-frames:v",
            "1",
            "-c:v",
            "libaom-av1",
            "-still-picture",
            "1",
            "-crf",
            "30",
            "-cpu-used",
            "8",
            "-aom-params",
            "tile-columns=1:tile-rows=0:enable-cdef=0:enable-restoration=0",
            "-f",
            "avif",
        ])
        .arg(&output_path)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping generated two-tile sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!("libaom encoder is unavailable; skipping generated two-tile sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    }
    let data = std::fs::read(&output_path).expect("generated two-tile AVIF should be readable");
    let decoded = avif_rust::image_from_bytes(&data).expect("two-tile AVIF should decode");
    assert_eq!(
        (decoded.width, decoded.height),
        (SAMPLE_WIDTH, SAMPLE_HEIGHT)
    );
    if let Some(expected) = ffmpeg_decode_rgba_dynamic(&output_path, SAMPLE_WIDTH, SAMPLE_HEIGHT) {
        let metrics = diff_rgb_dynamic(&decoded.rgba, &expected);
        assert!(
            metrics.average_rgb_abs <= 2.0 && metrics.max_rgb_abs <= 128,
            "two-tile FFmpeg RGB error average={} max={}",
            metrics.average_rgb_abs,
            metrics.max_rgb_abs
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn generated_12bit_sample_decodes_without_post_filter_overflow() {
    let root = std::env::temp_dir().join(format!(".test-avif-12bit-{}", std::process::id()));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary AVIF sample directory: {err}");
    }
    let output_path = root.join("generated-12bit.avif");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .arg("-i")
        .arg(sample_path("WML2Viewer.png"))
        .args([
            "-frames:v",
            "1",
            "-c:v",
            "libaom-av1",
            "-still-picture",
            "1",
            "-crf",
            "30",
            "-cpu-used",
            "8",
            "-pix_fmt",
            "yuv444p12le",
            "-aom-params",
            "enable-cdef=0:enable-restoration=0",
            "-f",
            "avif",
        ])
        .arg(&output_path)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping generated 12-bit sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!("libaom 12-bit encoder is unavailable; skipping generated sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    }
    let data = std::fs::read(&output_path).expect("generated 12-bit AVIF should be readable");
    let frame = avif_rust::decode_frame_bytes(&data).expect("12-bit native decode should succeed");
    assert_eq!(frame.bit_depth, 12);
    assert_eq!((frame.width, frame.height), (SAMPLE_WIDTH, SAMPLE_HEIGHT));
    let image = avif_rust::image_from_bytes(&data).expect("12-bit public decode should succeed");
    assert_eq!((image.width, image.height), (SAMPLE_WIDTH, SAMPLE_HEIGHT));
    if let Some(expected) = ffmpeg_decode_rgba(&output_path) {
        let metrics = diff_rgb(&image.rgba, &expected);
        assert!(
            metrics.average_rgb_abs <= 100.0,
            "12-bit FFmpeg RGB error average={} max={}",
            metrics.average_rgb_abs,
            metrics.max_rgb_abs
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn generated_12bit_rect_partition_sample_matches_ffmpeg_when_encoder_present() {
    let root = std::env::temp_dir().join(format!(".test-avif-12bit-rect-{}", std::process::id()));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary AVIF sample directory: {err}");
    }
    let output_path = root.join("generated-12bit-rect.avif");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i", "testsrc2=size=1204x800:rate=1"])
        .args([
            "-frames:v",
            "1",
            "-c:v",
            "libaom-av1",
            "-still-picture",
            "1",
        ])
        .args([
            "-crf",
            "18",
            "-cpu-used",
            "8",
            "-pix_fmt",
            "yuv444p12le",
            "-aom-params",
            "enable-rect-partitions=1:enable-1to4-partitions=1:enable-ab-partitions=1:enable-cdef=0:enable-restoration=0",
            "-f",
            "avif",
        ])
        .arg(&output_path)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping generated 12-bit rectangular sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!("libaom 12-bit rectangular encoder is unavailable; skipping generated sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    }
    let data =
        std::fs::read(&output_path).expect("generated 12-bit rectangular AVIF should be readable");
    let frame =
        avif_rust::decode_frame_bytes(&data).expect("12-bit rectangular AVIF should decode");
    assert_eq!(frame.bit_depth, 12);
    assert_eq!((frame.width, frame.height), (1204, 800));
    let image = avif_rust::image_from_bytes(&data)
        .expect("12-bit rectangular public decode should succeed");
    if let Some(expected) = ffmpeg_decode_rgba_dynamic(&output_path, 1204, 800) {
        let metrics = diff_rgb_dynamic(&image.rgba, &expected);
        assert!(
            metrics.average_rgb_abs <= 100.0,
            "12-bit rectangular FFmpeg RGB error average={} max={}",
            metrics.average_rgb_abs,
            metrics.max_rgb_abs
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn generated_12bit_128_superblock_sample_matches_ffmpeg_when_encoder_present() {
    let root = std::env::temp_dir().join(format!(".test-avif-12bit-sb128-{}", std::process::id()));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary AVIF sample directory: {err}");
    }
    let output_path = root.join("generated-12bit-sb128.avif");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i", "testsrc2=size=1204x800:rate=1"])
        .args([
            "-frames:v",
            "1",
            "-c:v",
            "libaom-av1",
            "-still-picture",
            "1",
            "-crf",
            "18",
            "-cpu-used",
            "8",
            "-pix_fmt",
            "yuv444p12le",
            "-aom-params",
            "sb-size=128:enable-rect-partitions=1:enable-1to4-partitions=1:enable-ab-partitions=1:enable-cdef=0:enable-restoration=0",
            "-f",
            "avif",
        ])
        .arg(&output_path)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping generated 12-bit 128-superblock sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!("libaom 12-bit 128-superblock encoder is unavailable; skipping generated sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    }
    let data = std::fs::read(&output_path)
        .expect("generated 12-bit 128-superblock AVIF should be readable");
    let frame =
        avif_rust::decode_frame_bytes(&data).expect("12-bit 128-superblock AVIF should decode");
    assert_eq!(frame.bit_depth, 12);
    assert_eq!((frame.width, frame.height), (1204, 800));
    let image = avif_rust::image_from_bytes(&data)
        .expect("12-bit 128-superblock public decode should succeed");
    if let Some(expected) = ffmpeg_decode_rgba_dynamic(&output_path, 1204, 800) {
        let metrics = diff_rgb_dynamic(&image.rgba, &expected);
        assert!(
            metrics.average_rgb_abs <= 100.0,
            "12-bit 128-superblock FFmpeg RGB error average={} max={}",
            metrics.average_rgb_abs,
            metrics.max_rgb_abs
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn generated_lossless_sample_matches_ffmpeg_when_encoder_present() {
    let root = std::env::temp_dir().join(format!(".test-avif-lossless-{}", std::process::id()));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary AVIF sample directory: {err}");
    }
    let output_path = root.join("lossless.avif");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .arg("-i")
        .arg(sample_path("WML2Viewer.png"))
        .args([
            "-frames:v",
            "1",
            "-c:v",
            "libaom-av1",
            "-still-picture",
            "1",
            "-lossless",
            "1",
            "-cpu-used",
            "8",
            "-pix_fmt",
            "yuv444p",
            "-aom-params",
            "enable-cdef=0:enable-restoration=0",
            "-f",
            "avif",
        ])
        .arg(&output_path)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping generated lossless sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!("libaom lossless encoder is unavailable; skipping generated sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    }
    let data = std::fs::read(&output_path).expect("generated lossless AVIF should be readable");
    let actual = avif_rust::image_from_bytes(&data).expect("lossless AVIF should decode");
    assert_eq!((actual.width, actual.height), (SAMPLE_WIDTH, SAMPLE_HEIGHT));
    if let Some(expected) = ffmpeg_decode_rgba(&output_path) {
        let metrics = diff_rgb(&actual.rgba, &expected);
        assert!(
            metrics.average_rgb_abs <= 2.0 && metrics.max_rgb_abs <= 32,
            "lossless FFmpeg RGB error average={} max={}",
            metrics.average_rgb_abs,
            metrics.max_rgb_abs
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn generated_level_zero_qmatrix_sample_matches_ffmpeg_when_encoder_present() {
    let root = std::env::temp_dir().join(format!(".test-avif-qmatrix-{}", std::process::id()));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary AVIF sample directory: {err}");
    }
    let output_path = root.join("level-zero-qmatrix.avif");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .arg("-i")
        .arg(sample_path("WML2Viewer.png"))
        .args([
            "-frames:v",
            "1",
            "-c:v",
            "libaom-av1",
            "-still-picture",
            "1",
            "-crf",
            "30",
            "-cpu-used",
            "8",
            "-aq-mode",
            "0",
            "-pix_fmt",
            "yuv444p",
            "-aom-params",
            "enable-qm=1:qm-min=0:qm-max=0",
            "-f",
            "avif",
        ])
        .arg(&output_path)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping generated level-zero qmatrix sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!("libaom encoder is unavailable; skipping generated level-zero qmatrix sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    }
    let data = std::fs::read(&output_path).expect("generated qmatrix AVIF should be readable");
    let actual = avif_rust::image_from_bytes(&data).expect("level-zero qmatrix AVIF should decode");
    assert_eq!((actual.width, actual.height), (SAMPLE_WIDTH, SAMPLE_HEIGHT));
    if let Some(expected) = ffmpeg_decode_rgba(&output_path) {
        let metrics = diff_rgb(&actual.rgba, &expected);
        assert!(
            metrics.average_rgb_abs <= 2.0 && metrics.max_rgb_abs <= 32,
            "identity qmatrix FFmpeg RGB error average={} max={}",
            metrics.average_rgb_abs,
            metrics.max_rgb_abs
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn generated_non_identity_qmatrix_sample_matches_ffmpeg_when_encoder_present() {
    let root =
        std::env::temp_dir().join(format!(".test-avif-qmatrix-level1-{}", std::process::id()));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary AVIF sample directory: {err}");
    }
    let output_path = root.join("non-identity-qmatrix.avif");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .arg("-i")
        .arg(sample_path("WML2Viewer.png"))
        .args([
            "-frames:v",
            "1",
            "-c:v",
            "libaom-av1",
            "-still-picture",
            "1",
            "-crf",
            "30",
            "-cpu-used",
            "8",
            "-aq-mode",
            "0",
            "-pix_fmt",
            "yuv444p",
            "-aom-params",
            "enable-qm=1:qm-min=1:qm-max=1",
            "-f",
            "avif",
        ])
        .arg(&output_path)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping level-1 qmatrix sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!("libaom encoder is unavailable; skipping level-1 qmatrix sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    }
    let data = std::fs::read(&output_path).expect("generated qmatrix AVIF should be readable");
    let actual = avif_rust::image_from_bytes(&data).expect("level-1 qmatrix AVIF should decode");
    assert_eq!((actual.width, actual.height), (SAMPLE_WIDTH, SAMPLE_HEIGHT));
    if let Some(expected) = ffmpeg_decode_rgba(&output_path) {
        let metrics = diff_rgb(&actual.rgba, &expected);
        assert!(
            metrics.average_rgb_abs <= 2.0 && metrics.max_rgb_abs <= 32,
            "level-1 qmatrix FFmpeg RGB error average={} max={}",
            metrics.average_rgb_abs,
            metrics.max_rgb_abs
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn generated_smpte240m_matrix_sample_matches_ffmpeg_when_encoder_present() {
    let root = std::env::temp_dir().join(format!(".test-avif-smpte240m-{}", std::process::id()));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary AVIF sample directory: {err}");
    }
    let output_path = root.join("smpte240m.avif");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .arg("-i")
        .arg(sample_path("WML2Viewer.png"))
        .args([
            "-frames:v",
            "1",
            "-c:v",
            "libaom-av1",
            "-still-picture",
            "1",
            "-crf",
            "30",
            "-cpu-used",
            "8",
            "-pix_fmt",
            "yuv444p",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "iec61966-2-1",
            "-colorspace",
            "smpte240m",
            "-f",
            "avif",
        ])
        .arg(&output_path)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping SMPTE 240M matrix sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!("libaom encoder is unavailable; skipping SMPTE 240M matrix sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    }
    let data = std::fs::read(&output_path).expect("generated SMPTE 240M AVIF should be readable");
    let actual = avif_rust::image_from_bytes(&data).expect("SMPTE 240M AVIF should decode");
    assert_eq!((actual.width, actual.height), (SAMPLE_WIDTH, SAMPLE_HEIGHT));
    if let Some(expected) = ffmpeg_decode_rgba(&output_path) {
        let metrics = diff_rgb(&actual.rgba, &expected);
        assert!(
            metrics.average_rgb_abs <= 2.0 && metrics.max_rgb_abs <= 32,
            "SMPTE 240M FFmpeg RGB error average={} max={}",
            metrics.average_rgb_abs,
            metrics.max_rgb_abs
        );
    }
    let _ = std::fs::remove_dir_all(&root);
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
fn public_sequence_primary_item_decodes_first_frame_when_present() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root should exist")
        .join("test/images/external/avif/unsupported/star-8bpc.avifs");
    if !path.is_file() {
        eprintln!("external sequence sample is unavailable; skipping oracle");
        return;
    }
    let data = std::fs::read(&path).expect("external sequence AVIF should be readable");
    let actual = avif_rust::image_from_bytes(&data)
        .expect("the primary item of the sequence should decode as its first frame");
    assert_eq!((actual.width, actual.height), (159, 159));
    let Some(expected) = ffmpeg_decode_rgba_dynamic(&path, 159, 159) else {
        return;
    };
    let metrics = diff_rgb_dynamic(&actual.rgba, &expected);
    eprintln!(
        "sequence primary item: average RGB absolute error={}, max={}",
        metrics.average_rgb_abs, metrics.max_rgb_abs
    );
    assert!(metrics.average_rgb_abs <= 2.0 && metrics.max_rgb_abs <= 48);
}

#[test]
fn public_12bit_sample_matches_ffmpeg_when_present() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root should exist")
        .join("test/images/external/avif/unsupported/fox.profile2.12bpc.yuv444.avif");
    if !path.is_file() {
        eprintln!("external 12-bit sample is unavailable; skipping decode oracle");
        return;
    }
    let data = std::fs::read(&path).expect("external 12-bit AVIF should be readable");
    let actual = avif_rust::image_from_bytes(&data).expect("12-bit AVIF should decode");
    assert_eq!((actual.width, actual.height), (1204, 800));
    let Some(expected) = ffmpeg_decode_rgba_dynamic(&path, 1204, 800) else {
        return;
    };
    let metrics = diff_rgb_dynamic(&actual.rgba, &expected);
    eprintln!(
        "fox.profile2.12bpc.yuv444.avif: average RGB absolute error={}, max={}",
        metrics.average_rgb_abs, metrics.max_rgb_abs
    );
    assert!(
        metrics.average_rgb_abs <= 2.0 && metrics.max_rgb_abs <= 32,
        "12-bit FFmpeg RGB error average={} max={}",
        metrics.average_rgb_abs,
        metrics.max_rgb_abs
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
fn decoded_frame_rejects_unsupported_icc_profile() {
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
fn public_icc_matrix_shaper_sample_applies_profile_when_present() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root should exist")
        .join(
            "test/images/external/avif/unsupported/red-at-12-oclock-with-color-profile-8bpc.avif",
        );
    if !path.is_file() {
        eprintln!("external ICC sample is unavailable; skipping profile check");
        return;
    }
    let data = std::fs::read(&path).expect("ICC AVIF sample should be readable");
    let actual = avif_rust::image_from_bytes(&data).expect("ICC AVIF sample should decode");
    assert_eq!((actual.width, actual.height), (800, 800));
    let raw = avif_rust::decode_frame_bytes(&data).expect("ICC frame should decode");
    let source = avif_rust::av1::frame_buffers_to_rgba_8(&raw.buffers, &raw.color_config)
        .expect("source RGB conversion should succeed");
    let changed_pixels = actual
        .rgba
        .chunks_exact(4)
        .zip(source.rgba.chunks_exact(4))
        .filter(|(actual, source)| actual[..3] != source[..3])
        .count();
    assert!(
        changed_pixels > 0,
        "ICC profile must affect at least one RGB pixel"
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
fn public_transform_samples_cover_crop_mirror_and_rotate() {
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
        ("kimono.rotate270.avif", 722, 1024),
        ("kimono.mirror-vertical.rotate270.avif", 722, 1024),
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
        .expect("crop/mirror/rotate transform should decode");
        assert_eq!(
            (image.width, image.height),
            (expected_width, expected_height)
        );
    }
}

#[test]
fn public_rotate_transform_samples_match_ffmpeg_when_present() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root should exist");
    for (name, width, height) in [
        ("kimono.rotate90.avif", 722, 1024),
        ("kimono.rotate270.avif", 722, 1024),
        ("kimono.mirror-vertical.rotate270.avif", 722, 1024),
    ] {
        let path = root
            .join("test/images/external/avif/unsupported")
            .join(name);
        if !path.is_file() {
            eprintln!(
                "external rotate sample missing; skipping {}",
                path.display()
            );
            return;
        }
        let data = std::fs::read(&path).expect("rotate sample should be readable");
        let actual = avif_rust::image_from_bytes(&data).expect("rotate sample should decode");
        assert_eq!((actual.width, actual.height), (width, height), "{name}");
        let Some(expected) = ffmpeg_decode_rgba_dynamic(&path, width, height) else {
            return;
        };
        let metrics = diff_rgb_dynamic(&actual.rgba, &expected);
        assert!(
            metrics.average_rgb_abs <= 2.0 && metrics.max_rgb_abs <= 48,
            "{name} FFmpeg RGB error average={} max={}",
            metrics.average_rgb_abs,
            metrics.max_rgb_abs
        );
    }
}

#[test]
fn public_irot_alpha_sample_matches_ffmpeg_when_present() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root should exist")
        .join("test/images/external/avif/unsupported/abc_color_irot_alpha_irot.avif");
    if !path.is_file() {
        eprintln!("official irot+alpha sample is unavailable; skipping oracle");
        return;
    }
    let data = std::fs::read(&path).expect("irot+alpha AVIF should be readable");
    let actual = avif_rust::image_from_bytes(&data).expect("irot+alpha sample should decode");
    assert_eq!((actual.width, actual.height), (256, 512));
    let Some(expected_rgb) = ffmpeg_decode_rgba_dynamic(&path, 256, 512) else {
        return;
    };
    let metrics = diff_rgb_dynamic(&actual.rgba, &expected_rgb);
    assert!(
        metrics.average_rgb_abs <= 2.0 && metrics.max_rgb_abs <= 48,
        "irot+alpha FFmpeg RGB error average={} max={}",
        metrics.average_rgb_abs,
        metrics.max_rgb_abs
    );
    let Some(expected_alpha) = ffmpeg_decode_alpha_plane(&path, 256, 512) else {
        return;
    };
    let max_alpha = actual
        .rgba
        .chunks_exact(4)
        .zip(expected_alpha)
        .map(|(pixel, expected)| pixel[3].abs_diff(expected))
        .max()
        .unwrap_or(0);
    assert!(max_alpha <= 8, "irot+alpha alpha error max={max_alpha}");
}
