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

fn ffmpeg_decode_rgba_stream_frame(
    path: &Path,
    stream_index: usize,
    frame_index: usize,
    width: usize,
    height: usize,
) -> Option<Vec<u8>> {
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
        .args(["-map", &format!("0:{stream_index}")])
        .args([
            "-frames:v",
            &(frame_index.saturating_add(1)).to_string(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
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
        "ffmpeg sequence decode failed for {}: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    let frame_len = width * height * 4;
    let start = frame_index
        .checked_mul(frame_len)
        .expect("ffmpeg frame offset should not overflow");
    let end = start
        .checked_add(frame_len)
        .expect("ffmpeg frame end should not overflow");
    assert!(
        output.stdout.len() >= end,
        "ffmpeg stream {} has no frame {}",
        stream_index,
        frame_index
    );
    Some(output.stdout[start..end].to_vec())
}

fn ffmpeg_decode_rgba_with_filter(
    path: &Path,
    width: usize,
    height: usize,
    filter: &str,
) -> Option<Vec<u8>> {
    let output = match Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin"])
        .arg("-i")
        .arg(path)
        .args([
            "-vf",
            filter,
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
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
        "ffmpeg filtered decode failed for {}: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout.len(), width * height * 4);
    Some(output.stdout)
}

fn imagemagick_decode_rgba(path: &Path, width: usize, height: usize) -> Option<Vec<u8>> {
    let output = match Command::new("magick")
        .arg(path)
        .args(["-depth", "8", "rgba:-"])
        .output()
    {
        Ok(output) => output,
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            eprintln!("ImageMagick is not available; skipping AVIF oracle comparison");
            return None;
        }
        Err(err) => panic!("failed to execute ImageMagick: {err}"),
    };
    assert!(
        output.status.success(),
        "ImageMagick failed for {}: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout.len(), width * height * 4);
    Some(output.stdout)
}

fn ffmpeg_decode_raw(path: &Path, pixel_format: &str) -> Option<Vec<u8>> {
    ffmpeg_decode_raw_stream(path, None, pixel_format)
}

fn ffmpeg_decode_raw_stream(
    path: &Path,
    stream_index: Option<usize>,
    pixel_format: &str,
) -> Option<Vec<u8>> {
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
    let mut command = Command::new(executable);
    command.args(["-v", "error", "-nostdin"]);
    command.arg("-i").arg(path);
    if let Some(stream_index) = stream_index {
        command.args(["-map", &format!("0:{stream_index}")]);
    }
    let output = match command
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
fn generated_sequence_sample_decodes_first_frame_when_encoder_present() {
    let root = std::env::temp_dir().join(format!(".test-avif-sequence-{}", std::process::id()));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary AVIF sequence directory: {err}");
    }
    let output_path = root.join("sequence.avifs");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i", "testsrc2=size=64x64:rate=1"])
        .args([
            "-t",
            "2",
            "-c:v",
            "libaom-av1",
            "-cpu-used",
            "8",
            "-crf",
            "0",
            "-g",
            "1",
            "-aom-params",
            "enable-cdef=0:enable-restoration=0",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-color_range",
            "tv",
            "-f",
            "avif",
        ])
        .arg(&output_path)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping generated sequence sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!("libaom sequence encoder is unavailable; skipping generated sequence sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    }
    let data = std::fs::read(&output_path).expect("generated AVIF sequence should be readable");
    let decoded = avif_rust::image_from_bytes(&data)
        .expect("the first frame of an AVIF sequence should decode");
    assert_eq!((decoded.width, decoded.height), (64, 64));
    if let Some(expected) = ffmpeg_decode_rgba_with_filter(
        &output_path,
        64,
        64,
        "zscale=matrixin=709:transferin=709:primariesin=709:rangein=limited:matrix=709:transfer=709:primaries=709:range=full,format=rgba",
    ) {
        let metrics = diff_rgb_dynamic(&decoded.rgba, &expected);
        eprintln!(
            "generated sequence first frame: average RGB absolute error={} max={}",
            metrics.average_rgb_abs, metrics.max_rgb_abs
        );
        assert!(
            metrics.average_rgb_abs <= 8.0 && metrics.max_rgb_abs <= 224,
            "sequence first-frame FFmpeg RGB error average={} max={}",
            metrics.average_rgb_abs,
            metrics.max_rgb_abs
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn generated_inter_sequence_sample_is_classified_for_decoder_gate_when_encoder_present() {
    let root =
        std::env::temp_dir().join(format!(".test-avif-sequence-inter-{}", std::process::id()));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary AVIF sequence directory: {err}");
    }
    let output_path = root.join("sequence.avifs");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i", "testsrc2=size=64x64:rate=1"])
        .args([
            "-t",
            "2",
            "-c:v",
            "libaom-av1",
            "-cpu-used",
            "8",
            "-crf",
            "0",
        ])
        .args(["-g", "30", "-frame-parallel", "1"])
        .args(["-aom-params", "enable-cdef=0:enable-restoration=0"])
        .args([
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-color_range",
            "tv",
            "-f",
            "avif",
        ])
        .arg(&output_path)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping generated inter sequence sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!("libaom sequence encoder is unavailable; skipping generated inter sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    }
    let data = std::fs::read(&output_path).expect("generated AVIF sequence should be readable");
    let info =
        avif_rust::container::parse_avif(&data).expect("generated AVIF sequence should parse");
    let inter_sample = info
        .sequence_sample_payloads
        .get(1)
        .expect("generated sequence should contain a second sample");
    assert_eq!(
        avif_rust::classify_av1_sequence_sample(inter_sample).unwrap(),
        Some(avif_rust::AvifSequenceSampleKind::Inter)
    );
    let decoded = avif_rust::decode_sequence_frame_bytes(&data, 1)
        .expect("generated inter sample should decode without partial output");
    assert_eq!((decoded.width, decoded.height), (64, 64));
    assert_eq!(decoded.buffers.planes[0].samples.len(), 64 * 64);
    if let Some(expected) = ffmpeg_decode_rgba_stream_frame(&output_path, 1, 1, 64, 64) {
        let actual = decoded
            .to_rgba8()
            .expect("generated inter sample should convert to RGBA8")
            .rgba;
        let metrics = diff_rgb_dynamic(&actual, &expected);
        eprintln!(
            "generated inter frame: average RGB absolute error={}, max={}",
            metrics.average_rgb_abs, metrics.max_rgb_abs
        );
        assert!(
            metrics.average_rgb_abs <= 48.0,
            "generated inter FFmpeg RGB error average={} max={}",
            metrics.average_rgb_abs,
            metrics.max_rgb_abs
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn generated_reduced_inter_transform_sequence_decodes_when_encoder_present() {
    let root = std::env::temp_dir().join(format!(
        ".test-avif-sequence-inter-idtx-{}",
        std::process::id()
    ));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary reduced-transform AVIF directory: {err}");
    }
    let output_path = root.join("sequence.avifs");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i", "testsrc2=size=128x128:rate=1"])
        .args([
            "-t",
            "2",
            "-c:v",
            "libaom-av1",
            "-cpu-used",
            "8",
            "-crf",
            "20",
            "-g",
            "30",
            "-frame-parallel",
            "1",
            "-enable-flip-idtx",
            "1",
            "-use-inter-dct-only",
            "0",
            "-reduced-tx-type-set",
            "0",
            "-enable-tx64",
            "0",
            "-aom-params",
            "enable-cdef=0:enable-restoration=0",
            "-f",
            "avif",
        ])
        .arg(&output_path)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping reduced-transform AVIF sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!("libaom reduced-transform encoder is unavailable; skipping sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    }
    let data = std::fs::read(&output_path)
        .expect("generated reduced-transform AVIF sequence should be readable");
    let info = avif_rust::container::parse_avif(&data)
        .expect("generated reduced-transform AVIF sequence should parse");
    assert!(info.sequence_sample_payloads.len() >= 2);
    assert_eq!(
        avif_rust::classify_av1_sequence_sample(&info.sequence_sample_payloads[1]).unwrap(),
        Some(avif_rust::AvifSequenceSampleKind::Inter)
    );
    let frames = avif_rust::decode_sequence_frames_bytes(&data)
        .expect("reduced inter transform sequence should decode fully");
    assert_eq!(frames.len(), info.sequence_sample_payloads.len());
    assert!(
        frames
            .iter()
            .all(|frame| (frame.width, frame.height) == (128, 128))
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn generated_10bit_inter_sequence_sample_decodes_when_encoder_present() {
    run_generated_high_bit_inter_sequence_sample("10bit", "yuv420p10le", 10);
}

#[test]
fn generated_12bit_inter_sequence_sample_decodes_when_encoder_present() {
    run_generated_high_bit_inter_sequence_sample("12bit", "yuv420p12le", 12);
}

fn run_generated_high_bit_inter_sequence_sample(label: &str, pixel_format: &str, bit_depth: u8) {
    let root = std::env::temp_dir().join(format!(
        ".test-avif-sequence-inter-{label}-{}",
        std::process::id()
    ));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary {bit_depth}-bit AVIF sequence directory: {err}");
    }
    let output_path = root.join(format!("sequence-{label}.avifs"));
    let format_filter = format!("format={pixel_format}");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i", "testsrc2=size=64x64:rate=1"])
        .args([
            "-vf",
            format_filter.as_str(),
            "-t",
            "2",
            "-c:v",
            "libaom-av1",
            "-cpu-used",
            "8",
            "-crf",
            "0",
            "-g",
            "30",
            "-frame-parallel",
            "1",
            "-aom-params",
            "enable-cdef=0:enable-restoration=0",
            "-pix_fmt",
            pixel_format,
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-color_range",
            "tv",
            "-f",
            "avif",
        ])
        .arg(&output_path)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping generated {bit_depth}-bit inter sequence");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!("libaom {bit_depth}-bit inter encoder is unavailable; skipping generated sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    }
    let data = std::fs::read(&output_path)
        .unwrap_or_else(|err| panic!("generated {bit_depth}-bit AVIS should be readable: {err}"));
    let info = avif_rust::container::parse_avif(&data)
        .unwrap_or_else(|err| panic!("generated {bit_depth}-bit AVIS should parse: {err}"));
    let inter_sample = info.sequence_sample_payloads.get(1).unwrap_or_else(|| {
        panic!("generated {bit_depth}-bit sequence should contain a second sample")
    });
    assert_eq!(
        avif_rust::classify_av1_sequence_sample(inter_sample).unwrap(),
        Some(avif_rust::AvifSequenceSampleKind::Inter)
    );
    let decoded = avif_rust::decode_sequence_frame_bytes(&data, 1).unwrap_or_else(|err| {
        panic!("generated {bit_depth}-bit inter sample should decode without partial output: {err}")
    });
    assert_eq!((decoded.width, decoded.height), (64, 64));
    assert_eq!(decoded.bit_depth, bit_depth);
    assert_eq!(decoded.buffers.planes.len(), 3);
    assert_eq!(decoded.buffers.planes[0].samples.len(), 64 * 64);
    assert_eq!(decoded.buffers.planes[1].samples.len(), 32 * 32);
    assert_eq!(decoded.buffers.planes[2].samples.len(), 32 * 32);
    for (plane_index, plane) in decoded.buffers.planes.iter().enumerate() {
        assert!(
            plane
                .samples
                .iter()
                .all(|sample| *sample <= ((1_u16 << bit_depth) - 1)),
            "{bit_depth}-bit inter plane {plane_index} exceeds the declared range"
        );
    }
    if let Some(expected) = ffmpeg_decode_rgba_stream_frame(&output_path, 1, 1, 64, 64) {
        let actual = decoded
            .to_rgba8()
            .unwrap_or_else(|err| {
                panic!("generated {bit_depth}-bit inter sample should convert to RGBA8: {err}")
            })
            .rgba;
        let metrics = diff_rgb_dynamic(&actual, &expected);
        eprintln!(
            "generated {bit_depth}-bit inter frame: average RGB absolute error={}, max={}",
            metrics.average_rgb_abs, metrics.max_rgb_abs
        );
        assert!(
            metrics.average_rgb_abs <= 64.0,
            "generated {bit_depth}-bit inter FFmpeg RGB error average={} max={}",
            metrics.average_rgb_abs,
            metrics.max_rgb_abs
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn generated_local_warp_sample_matches_ffmpeg_when_encoder_present() {
    run_generated_local_warp_sample("localwarp", "testsrc2=size=128x128:rate=1", None);
}

#[test]
fn generated_local_warp_yuv444_sample_matches_ffmpeg_when_encoder_present() {
    run_generated_local_warp_sample(
        "localwarp-yuv444",
        "testsrc=size=128x128:rate=1,format=yuv444p",
        Some("yuv444p"),
    );
}

#[test]
fn generated_local_warp_12bit_sample_matches_ffmpeg_when_encoder_present() {
    run_generated_local_warp_sample(
        "localwarp-12bit",
        "testsrc2=size=128x128:rate=1",
        Some("yuv420p12le"),
    );
}

fn run_generated_local_warp_sample(label: &str, input: &str, pixel_format: Option<&str>) {
    let root = std::env::temp_dir().join(format!(".test-avif-{label}-{}", std::process::id()));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary AVIF local-warp directory: {err}");
    }
    let output_path = root.join(format!("{label}.avifs"));
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i", input])
        .args([
            "-t",
            "4",
            "-c:v",
            "libaom-av1",
            "-cpu-used",
            "6",
            "-crf",
            "25",
        ])
        .args(
            pixel_format
                .into_iter()
                .flat_map(|format| ["-pix_fmt", format]),
        )
        .args(["-g", "30", "-frame-parallel", "0"])
        .args([
            "-aom-params",
            "enable-cdef=0:enable-restoration=0:enable-obmc=0:enable-warped-motion=1",
        ])
        .args(["-f", "avif"])
        .arg(&output_path)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping generated {label} sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!("libaom {label} encoder option is unavailable; skipping sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    }
    let data = std::fs::read(&output_path).expect("generated local-warp AVIF should be readable");
    let decoded = avif_rust::decode_sequence_frame_bytes(&data, 1)
        .unwrap_or_else(|err| panic!("generated {label} inter sample should decode: {err}"));
    assert_eq!((decoded.width, decoded.height), (128, 128));
    let max_sample = ((1_u32 << u32::from(decoded.bit_depth.min(16))) - 1) as u16;
    for (plane_index, plane) in decoded.buffers.planes.iter().enumerate() {
        assert!(
            plane.samples.iter().all(|sample| *sample <= max_sample),
            "generated {label} plane {plane_index} exceeds {}-bit range",
            decoded.bit_depth
        );
    }
    if let Some(expected) = ffmpeg_decode_rgba_stream_frame(&output_path, 1, 1, 128, 128) {
        let metrics = diff_rgb_dynamic(
            &decoded
                .to_rgba8()
                .expect("generated local-warp sample should convert to RGBA8")
                .rgba,
            &expected,
        );
        eprintln!(
            "generated {label} frame: average RGB absolute error={}, max={}",
            metrics.average_rgb_abs, metrics.max_rgb_abs
        );
        assert!(
            metrics.average_rgb_abs <= 64.0,
            "generated {label} FFmpeg RGB error average={} max={}",
            metrics.average_rgb_abs,
            metrics.max_rgb_abs
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn generated_obmc_sample_matches_ffmpeg_when_encoder_present() {
    run_generated_obmc_sample(
        "obmc",
        "enable-cdef=0:enable-restoration=0:enable-obmc=1:enable-warped-motion=0",
    );
}

#[test]
fn generated_obmc_dual_filter_sample_matches_ffmpeg_when_encoder_present() {
    run_generated_obmc_sample(
        "obmc-dual-filter",
        "enable-cdef=0:enable-restoration=0:enable-obmc=1:enable-warped-motion=0:enable-dual-filter=1",
    );
}

fn run_generated_obmc_sample(label: &str, aom_params: &str) {
    let root = std::env::temp_dir().join(format!(".test-avif-{label}-{}", std::process::id()));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary AVIF OBMC directory: {err}");
    }
    let output_path = root.join(format!("{label}.avifs"));
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i", "testsrc2=size=128x128:rate=1"])
        .args([
            "-t",
            "4",
            "-c:v",
            "libaom-av1",
            "-cpu-used",
            "6",
            "-crf",
            "25",
        ])
        .args(["-g", "30", "-frame-parallel", "0"])
        .args(["-aom-params", aom_params])
        .args(["-f", "avif"])
        .arg(&output_path)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping generated {label} sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!("libaom {label} encoder option is unavailable; skipping sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    }
    let data = std::fs::read(&output_path).expect("generated OBMC AVIF should be readable");
    let decoded = avif_rust::decode_sequence_frame_bytes(&data, 1)
        .unwrap_or_else(|err| panic!("generated {label} inter sample should decode: {err}"));
    assert_eq!((decoded.width, decoded.height), (128, 128));
    if let Some(expected) = ffmpeg_decode_rgba_stream_frame(&output_path, 1, 1, 128, 128) {
        let metrics = diff_rgb_dynamic(
            &decoded
                .to_rgba8()
                .expect("generated OBMC sample should convert to RGBA8")
                .rgba,
            &expected,
        );
        eprintln!(
            "generated {label} frame: average RGB absolute error={}, max={}",
            metrics.average_rgb_abs, metrics.max_rgb_abs
        );
        assert!(
            metrics.average_rgb_abs <= 64.0,
            "generated {label} FFmpeg RGB error average={} max={}",
            metrics.average_rgb_abs,
            metrics.max_rgb_abs
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn generated_global_motion_sample_matches_ffmpeg_when_encoder_present() {
    let root =
        std::env::temp_dir().join(format!(".test-avif-global-motion-{}", std::process::id()));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary AVIF global-motion directory: {err}");
    }
    let output_path = root.join("global-motion.avifs");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i", "testsrc2=size=256x128:rate=1"])
        .args(["-vf", "crop=128:128:2*n:0", "-t", "4"])
        .args(["-c:v", "libaom-av1", "-cpu-used", "6", "-crf", "25"])
        .args([
            "-g",
            "30",
            "-frame-parallel",
            "0",
            "-enable-global-motion",
            "1",
        ])
        .args([
            "-aom-params",
            "enable-cdef=0:enable-restoration=0:enable-obmc=0",
        ])
        .args(["-f", "avif"])
        .arg(&output_path)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping generated global-motion sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!("libaom global-motion encoder options are unavailable; skipping sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    }
    let data =
        std::fs::read(&output_path).expect("generated global-motion AVIS should be readable");
    let info =
        avif_rust::container::parse_avif(&data).expect("generated global-motion AVIS should parse");
    assert!(info.sequence_sample_payloads.len() >= 2);
    assert_eq!(
        avif_rust::classify_av1_sequence_sample(&info.sequence_sample_payloads[1]).unwrap(),
        Some(avif_rust::AvifSequenceSampleKind::Inter)
    );
    for frame_index in [1] {
        let decoded = avif_rust::decode_sequence_frame_bytes(&data, frame_index)
            .expect("generated global-motion inter sample should decode");
        assert_eq!((decoded.width, decoded.height), (128, 128));
        if let Some(expected) =
            ffmpeg_decode_rgba_stream_frame(&output_path, 1, frame_index, 128, 128)
        {
            let actual = decoded
                .to_rgba8()
                .expect("generated global-motion sample should convert to RGBA8")
                .rgba;
            let metrics = diff_rgb_dynamic(&actual, &expected);
            eprintln!(
                "generated global-motion frame {}: average RGB absolute error={} max={}",
                frame_index, metrics.average_rgb_abs, metrics.max_rgb_abs
            );
            assert!(
                metrics.average_rgb_abs <= 64.0,
                "generated global-motion FFmpeg RGB error average={} max={}",
                metrics.average_rgb_abs,
                metrics.max_rgb_abs
            );
        }
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn generated_affine_global_motion_sample_matches_ffmpeg_when_encoder_present() {
    let root =
        std::env::temp_dir().join(format!(".test-avif-global-affine-{}", std::process::id()));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary AVIF affine-global directory: {err}");
    }
    let output_path = root.join("global-affine.avifs");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i", "testsrc2=size=256x256:rate=1"])
        .args(["-vf", "rotate=0.03*n:fillcolor=black", "-t", "4"])
        .args(["-c:v", "libaom-av1", "-cpu-used", "6", "-crf", "25"])
        .args([
            "-g",
            "30",
            "-frame-parallel",
            "0",
            "-enable-global-motion",
            "1",
        ])
        .args([
            "-aom-params",
            "enable-cdef=0:enable-restoration=0:enable-obmc=0",
        ])
        .args(["-f", "avif"])
        .arg(&output_path)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping generated affine-global sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!("libaom affine global-motion encoder options are unavailable; skipping sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    }
    let data =
        std::fs::read(&output_path).expect("generated affine-global AVIS should be readable");
    let info =
        avif_rust::container::parse_avif(&data).expect("generated affine-global AVIS should parse");
    assert!(info.sequence_sample_payloads.len() >= 2);
    assert_eq!(
        avif_rust::classify_av1_sequence_sample(&info.sequence_sample_payloads[1]).unwrap(),
        Some(avif_rust::AvifSequenceSampleKind::Inter)
    );
    let decoded = avif_rust::decode_sequence_frame_bytes(&data, 1)
        .expect("generated affine-global inter sample should decode");
    assert_eq!((decoded.width, decoded.height), (256, 256));
    if let Some(expected) = ffmpeg_decode_rgba_stream_frame(&output_path, 1, 1, 256, 256) {
        let metrics = diff_rgb_dynamic(
            &decoded
                .to_rgba8()
                .expect("generated affine-global sample should convert to RGBA8")
                .rgba,
            &expected,
        );
        eprintln!(
            "generated affine-global frame: average RGB absolute error={}, max={}",
            metrics.average_rgb_abs, metrics.max_rgb_abs
        );
        assert!(
            metrics.average_rgb_abs <= 64.0,
            "generated affine-global FFmpeg RGB error average={} max={}",
            metrics.average_rgb_abs,
            metrics.max_rgb_abs
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn generated_difference_weighted_compound_sample_matches_ffmpeg_when_encoder_present() {
    let root =
        std::env::temp_dir().join(format!(".test-avif-diff-weighted-{}", std::process::id()));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary AVIF difference-weighted directory: {err}");
    }
    let output_path = root.join("diff-weighted.avifs");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i", "testsrc2=size=128x128:rate=1"])
        .args(["-t", "4", "-c:v", "libaom-av1", "-cpu-used", "6", "-crf", "25"])
        .args(["-g", "30", "-frame-parallel", "0"])
        .args(["-aom-params", "enable-cdef=0:enable-restoration=0:enable-masked-comp=1:enable-diff-wtd-comp=1:enable-interinter-wedge=0:enable-dist-wtd-comp=0:enable-obmc=0"])
        .args(["-f", "avif"])
        .arg(&output_path)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping generated diff-weighted sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!("libaom diff-weighted encoder options are unavailable; skipping sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    }
    let data =
        std::fs::read(&output_path).expect("generated diff-weighted AVIS should be readable");
    let decoded = avif_rust::decode_sequence_frame_bytes(&data, 1)
        .expect("generated diff-weighted inter sample should decode");
    assert_eq!((decoded.width, decoded.height), (128, 128));
    if let Some(expected) = ffmpeg_decode_rgba_stream_frame(&output_path, 1, 1, 128, 128) {
        let metrics = diff_rgb_dynamic(
            &decoded
                .to_rgba8()
                .expect("generated diff-weighted sample should convert to RGBA8")
                .rgba,
            &expected,
        );
        eprintln!(
            "generated diff-weighted frame: average RGB absolute error={}, max={}",
            metrics.average_rgb_abs, metrics.max_rgb_abs
        );
        assert!(
            metrics.average_rgb_abs <= 64.0,
            "generated diff-weighted FFmpeg RGB error average={} max={}",
            metrics.average_rgb_abs,
            metrics.max_rgb_abs
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn generated_wedge_compound_sample_matches_ffmpeg_when_encoder_present() {
    let root = std::env::temp_dir().join(format!(".test-avif-wedge-{}", std::process::id()));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary AVIF wedge directory: {err}");
    }
    let output_path = root.join("wedge.avifs");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i", "testsrc2=size=128x128:rate=1"])
        .args(["-t", "4", "-c:v", "libaom-av1", "-cpu-used", "6", "-crf", "25"])
        .args(["-g", "30", "-frame-parallel", "0"])
        .args(["-aom-params", "enable-cdef=0:enable-restoration=0:enable-masked-comp=1:enable-interinter-wedge=1:enable-diff-wtd-comp=0:enable-dist-wtd-comp=0:enable-obmc=0"])
        .args(["-f", "avif"])
        .arg(&output_path)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping generated wedge sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!("libaom wedge encoder options are unavailable; skipping sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    }
    let data = std::fs::read(&output_path).expect("generated wedge AVIS should be readable");
    let decoded = avif_rust::decode_sequence_frame_bytes(&data, 1)
        .expect("generated wedge inter sample should decode");
    assert_eq!((decoded.width, decoded.height), (128, 128));
    if let Some(expected) = ffmpeg_decode_rgba_stream_frame(&output_path, 1, 1, 128, 128) {
        let metrics = diff_rgb_dynamic(
            &decoded
                .to_rgba8()
                .expect("generated wedge sample should convert to RGBA8")
                .rgba,
            &expected,
        );
        eprintln!(
            "generated wedge frame: average RGB absolute error={}, max={}",
            metrics.average_rgb_abs, metrics.max_rgb_abs
        );
        assert!(
            metrics.average_rgb_abs <= 64.0,
            "generated wedge FFmpeg RGB error average={} max={}",
            metrics.average_rgb_abs,
            metrics.max_rgb_abs
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn generated_interintra_sample_matches_ffmpeg_when_encoder_present() {
    let root = std::env::temp_dir().join(format!(".test-avif-interintra-{}", std::process::id()));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary AVIF inter-intra directory: {err}");
    }
    let output_path = root.join("interintra.avifs");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i", "testsrc2=size=128x128:rate=1"])
        .args(["-t", "4", "-c:v", "libaom-av1", "-cpu-used", "6", "-crf", "25"])
        .args(["-g", "30", "-frame-parallel", "0"])
        .args(["-aom-params", "enable-cdef=0:enable-restoration=0:enable-interintra-comp=1:enable-interintra-wedge=1:enable-smooth-interintra=1:enable-obmc=0"])
        .args(["-f", "avif"])
        .arg(&output_path)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping generated inter-intra sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!("libaom inter-intra encoder options are unavailable; skipping sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    }
    let data = std::fs::read(&output_path).expect("generated inter-intra AVIS should be readable");
    let decoded = avif_rust::decode_sequence_frame_bytes(&data, 1)
        .expect("generated inter-intra inter sample should decode");
    assert_eq!((decoded.width, decoded.height), (128, 128));
    if let Some(expected) = ffmpeg_decode_rgba_stream_frame(&output_path, 1, 1, 128, 128) {
        let metrics = diff_rgb_dynamic(
            &decoded
                .to_rgba8()
                .expect("generated inter-intra sample should convert to RGBA8")
                .rgba,
            &expected,
        );
        eprintln!(
            "generated inter-intra frame: average RGB absolute error={}, max={}",
            metrics.average_rgb_abs, metrics.max_rgb_abs
        );
        assert!(
            metrics.average_rgb_abs <= 64.0,
            "generated inter-intra FFmpeg RGB error average={} max={}",
            metrics.average_rgb_abs,
            metrics.max_rgb_abs
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn generated_filter_intra_sample_matches_ffmpeg_when_encoder_present() {
    generated_filter_intra_sample_matches_ffmpeg_impl("yuv444p", "yuv444p", 256, 8);
}

#[test]
fn generated_10bit_filter_intra_sample_matches_ffmpeg_when_encoder_present() {
    generated_filter_intra_sample_matches_ffmpeg_impl("yuv444p10le", "yuv444p10le", 128, 10);
}

#[test]
fn generated_12bit_filter_intra_sample_matches_ffmpeg_when_encoder_present() {
    generated_filter_intra_sample_matches_ffmpeg_impl("yuv444p12le", "yuv444p12le", 96, 12);
}

fn generated_filter_intra_sample_matches_ffmpeg_impl(
    pixel_format: &str,
    expected_format: &str,
    dimension: usize,
    bit_depth: u8,
) {
    let root = std::env::temp_dir().join(format!(
        ".test-avif-filter-intra-{}-{}",
        bit_depth,
        std::process::id()
    ));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary AVIF filter-intra directory: {err}");
    }
    let output_path = root.join("filter-intra.avif");
    let source = format!("testsrc2=size={dimension}x{dimension}:rate=1");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i", &source])
        .args([
            "-frames:v",
            "1",
            "-c:v",
            "libaom-av1",
            "-still-picture",
            "1",
            "-cpu-used",
            "8",
            "-crf",
            if bit_depth >= 12 { "0" } else { "20" },
            "-pix_fmt",
            pixel_format,
            "-enable-filter-intra",
            "1",
            "-enable-intra-edge-filter",
            "1",
            "-f",
            "avif",
        ])
        .arg(&output_path)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping generated filter-intra sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!("libaom filter-intra encoder is unavailable; skipping generated sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    }
    let data = std::fs::read(&output_path).expect("generated filter-intra AVIF should be readable");
    let decoded = avif_rust::image_from_bytes(&data).expect("filter-intra AVIF should decode");
    assert_eq!((decoded.width, decoded.height), (dimension, dimension));
    let frame =
        avif_rust::decode_frame_bytes(&data).expect("filter-intra native frame should decode");
    assert_eq!(frame.bit_depth, bit_depth);
    if let Some(expected) = ffmpeg_decode_raw(&output_path, expected_format) {
        let plane_size = dimension * dimension;
        if bit_depth == 8 {
            assert_eq!(expected.len(), plane_size * 3);
            for plane_index in 0..3 {
                let expected_plane =
                    &expected[plane_index * plane_size..(plane_index + 1) * plane_size];
                let max_error = frame.buffers.planes[plane_index]
                    .samples
                    .iter()
                    .zip(expected_plane)
                    .map(|(actual, expected)| u8::try_from(*actual).unwrap().abs_diff(*expected))
                    .max()
                    .unwrap_or(0);
                assert!(
                    max_error <= 2,
                    "filter-intra native plane {plane_index} max error was {max_error}"
                );
            }
        } else {
            assert_eq!(expected.len(), plane_size * 3 * 2);
            let expected: Vec<u16> = expected
                .chunks_exact(2)
                .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]) & ((1 << bit_depth) - 1))
                .collect();
            for plane_index in 0..3 {
                let expected_plane =
                    &expected[plane_index * plane_size..(plane_index + 1) * plane_size];
                let max_error = frame.buffers.planes[plane_index]
                    .samples
                    .iter()
                    .zip(expected_plane)
                    .map(|(actual, expected)| actual.abs_diff(*expected))
                    .max()
                    .unwrap_or(0);
                let average_error = frame.buffers.planes[plane_index]
                    .samples
                    .iter()
                    .zip(expected_plane)
                    .map(|(actual, expected)| f64::from(actual.abs_diff(*expected)))
                    .sum::<f64>()
                    / plane_size as f64;
                eprintln!(
                    "filter-intra {bit_depth}-bit native plane {plane_index}: average={average_error} max={max_error}"
                );
                let max_error_limit = if bit_depth >= 12 { 2 } else { 16 };
                assert!(
                    average_error <= 2.0 && max_error <= max_error_limit,
                    "filter-intra {bit_depth}-bit native plane {plane_index}: average={average_error} max={max_error} limit={max_error_limit}"
                );
            }
        }
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn generated_cdef_sample_matches_explicit_ffmpeg_yuv_oracle_when_encoder_present() {
    let root = std::env::temp_dir().join(format!(".test-avif-cdef-{}", std::process::id()));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary AVIF sample directory: {err}");
    }
    let output_path = root.join("cdef.avif");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i", "testsrc2=size=256x256:rate=1"])
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
            "enable-cdef=1:enable-restoration=0",
            "-f",
            "avif",
        ])
        .arg(&output_path)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping generated CDEF sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!("libaom CDEF encoder is unavailable; skipping generated sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    }

    let data = std::fs::read(&output_path).expect("generated CDEF AVIF should be readable");
    let actual = avif_rust::image_from_bytes(&data).expect("CDEF AVIF should decode");
    let Some(expected) = ffmpeg_decode_rgba_with_filter(
        &output_path,
        256,
        256,
        "zscale=matrixin=709:transferin=709:primariesin=709:rangein=limited:matrix=709:transfer=709:primaries=709:range=full,format=rgba",
    ) else {
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    let metrics = diff_rgb_dynamic(&actual.rgba, &expected);
    eprintln!(
        "CDEF FFmpeg RGB error average={} max={}",
        metrics.average_rgb_abs, metrics.max_rgb_abs
    );
    assert!(metrics.average_rgb_abs <= 0.5 && metrics.max_rgb_abs <= 16);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn generated_cdef_420_sample_matches_ffmpeg_when_encoder_present() {
    let root = std::env::temp_dir().join(format!(".test-avif-cdef420-{}", std::process::id()));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary AVIF sample directory: {err}");
    }
    let output_path = root.join("cdef-420.avif");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i", "testsrc2=size=256x256:rate=1"])
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
            "yuv420p",
            "-aom-params",
            "enable-cdef=1:enable-restoration=0",
            "-f",
            "avif",
        ])
        .arg(&output_path)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping generated 4:2:0 CDEF sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!("libaom 4:2:0 CDEF encoder is unavailable; skipping generated sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    }

    let data = std::fs::read(&output_path).expect("generated 4:2:0 CDEF AVIF should be readable");
    let actual = avif_rust::image_from_bytes(&data).expect("4:2:0 CDEF AVIF should decode");
    let actual_frame =
        avif_rust::decode_frame_bytes(&data).expect("4:2:0 CDEF frame should decode");
    eprintln!(
        "4:2:0 CDEF chroma position={:?}",
        actual_frame.color_config.chroma_sample_position
    );
    if let Some(expected) = ffmpeg_decode_raw(&output_path, "yuv420p") {
        let plane_lengths = [256 * 256, 128 * 128, 128 * 128];
        let mut offset = 0;
        for (plane_index, &plane_len) in plane_lengths.iter().enumerate() {
            let expected_plane = &expected[offset..offset + plane_len];
            let actual_plane = &actual_frame.buffers.planes[plane_index].samples;
            let errors = actual_plane
                .iter()
                .zip(expected_plane)
                .map(|(actual, expected)| u8::try_from(*actual).unwrap().abs_diff(*expected));
            let max_error = errors.clone().max().unwrap_or(0);
            let average_error = errors.map(u64::from).sum::<u64>() as f64 / plane_len as f64;
            eprintln!(
                "4:2:0 CDEF plane {plane_index} error average={average_error} max={max_error}"
            );
            assert!(
                average_error <= 2.0 && max_error <= 32,
                "4:2:0 CDEF plane {plane_index}: average={average_error} max={max_error}"
            );
            offset += plane_len;
        }
    }
    let Some(expected) = ffmpeg_decode_rgba_with_filter(
        &output_path,
        256,
        256,
        "zscale=matrixin=709:transferin=709:primariesin=709:rangein=limited:matrix=709:transfer=709:primaries=709:range=full,format=rgba",
    ) else {
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    let metrics = diff_rgb_dynamic(&actual.rgba, &expected);
    eprintln!(
        "4:2:0 CDEF FFmpeg RGB error average={} max={}",
        metrics.average_rgb_abs, metrics.max_rgb_abs
    );
    // Chroma siting differences at a handful of edge pixels can be large in
    // RGB even when the filtered YUV planes remain within the average gate.
    let within_oracle = metrics.average_rgb_abs <= 2.0 && metrics.max_rgb_abs <= 192;
    let _ = std::fs::remove_dir_all(&root);
    assert!(within_oracle);
}

#[test]
fn generated_delta_q_sample_matches_ffmpeg_when_encoder_present() {
    let root = std::env::temp_dir().join(format!(".test-avif-deltaq-{}", std::process::id()));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary AVIF sample directory: {err}");
    }
    let output_path = root.join("delta-q.avif");
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
            "6",
            "-aom-params",
            "deltaq-mode=1:enable-chroma-deltaq=1",
            "-f",
            "avif",
        ])
        .arg(&output_path)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping generated delta-q sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!("libaom encoder is unavailable; skipping generated delta-q sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    }
    let data = std::fs::read(&output_path).expect("generated delta-q AVIF should be readable");
    let decoded = avif_rust::image_from_bytes(&data).expect("delta-q AVIF should decode");
    assert_eq!(
        (decoded.width, decoded.height),
        (SAMPLE_WIDTH, SAMPLE_HEIGHT)
    );
    if let Some(expected) = ffmpeg_decode_rgba_dynamic(&output_path, SAMPLE_WIDTH, SAMPLE_HEIGHT) {
        let metrics = diff_rgb_dynamic(&decoded.rgba, &expected);
        assert!(
            metrics.average_rgb_abs <= 2.0 && metrics.max_rgb_abs <= 128,
            "delta-q FFmpeg RGB error average={} max={}",
            metrics.average_rgb_abs,
            metrics.max_rgb_abs
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn generated_film_grain_sample_matches_ffmpeg_when_encoder_present() {
    let root = std::env::temp_dir().join(format!(".test-avif-film-grain-{}", std::process::id()));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary AVIF sample directory: {err}");
    }
    let output_path = root.join("film-grain.avif");
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
            "6",
            "-pix_fmt",
            "yuv420p",
            "-aom-params",
            "film-grain-test=1",
            "-f",
            "avif",
        ])
        .arg(&output_path)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping generated film-grain sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!("libaom film-grain encoder is unavailable; skipping generated sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    }
    let data = std::fs::read(&output_path).expect("generated film-grain AVIF should be readable");
    let decoded = avif_rust::image_from_bytes(&data).expect("film-grain AVIF should decode");
    assert_eq!(
        (decoded.width, decoded.height),
        (SAMPLE_WIDTH, SAMPLE_HEIGHT)
    );
    if let Some(expected) = ffmpeg_decode_rgba_dynamic(&output_path, SAMPLE_WIDTH, SAMPLE_HEIGHT) {
        let metrics = diff_rgb_dynamic(&decoded.rgba, &expected);
        eprintln!(
            "film-grain FFmpeg RGB error average={} max={}",
            metrics.average_rgb_abs, metrics.max_rgb_abs
        );
        assert!(
            metrics.average_rgb_abs <= 4.0 && metrics.max_rgb_abs <= 128,
            "film-grain FFmpeg RGB error average={} max={}",
            metrics.average_rgb_abs,
            metrics.max_rgb_abs
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn generated_adaptive_quantization_sample_matches_ffmpeg_when_encoder_present() {
    let root = std::env::temp_dir().join(format!(".test-avif-aq-{}", std::process::id()));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary AVIF sample directory: {err}");
    }
    let output_path = root.join("adaptive-quantization.avif");
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
            "6",
            "-aq-mode",
            "3",
            "-pix_fmt",
            "yuv444p",
            "-f",
            "avif",
        ])
        .arg(&output_path)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping generated adaptive-quantization sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!("libaom adaptive-quantization encoder is unavailable; skipping generated sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    }
    let data = std::fs::read(&output_path)
        .expect("generated adaptive-quantization AVIF should be readable");
    let decoded =
        avif_rust::image_from_bytes(&data).expect("adaptive-quantization AVIF should decode");
    assert_eq!(
        (decoded.width, decoded.height),
        (SAMPLE_WIDTH, SAMPLE_HEIGHT)
    );
    if let Some(expected) = ffmpeg_decode_rgba_dynamic(&output_path, SAMPLE_WIDTH, SAMPLE_HEIGHT) {
        let metrics = diff_rgb_dynamic(&decoded.rgba, &expected);
        assert!(
            metrics.average_rgb_abs <= 2.0 && metrics.max_rgb_abs <= 128,
            "adaptive-quantization FFmpeg RGB error average={} max={}",
            metrics.average_rgb_abs,
            metrics.max_rgb_abs
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn generated_ictcp_sample_matches_ffmpeg_when_encoder_present() {
    let root = std::env::temp_dir().join(format!(".test-avif-ictcp-{}", std::process::id()));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary AVIF sample directory: {err}");
    }
    let output_path = root.join("ictcp.avif");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .arg("-i")
        .arg(sample_path("WML2Viewer.png"))
        .args([
            "-frames:v",
            "1",
            "-vf",
            "zscale=primaries=2020:transfer=smpte2084:matrix=ictcp,format=yuv444p",
            "-c:v",
            "libaom-av1",
            "-still-picture",
            "1",
            "-crf",
            "30",
            "-cpu-used",
            "8",
            "-f",
            "avif",
        ])
        .arg(&output_path)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping generated ICTCP sample");
        return;
    };
    if !status.success() {
        eprintln!("libaom ICTCP encoder is unavailable; skipping generated ICTCP sample");
        return;
    }

    let data = std::fs::read(&output_path).expect("generated ICTCP AVIF should be readable");
    let actual = avif_rust::image_from_bytes(&data).expect("ICTCP AVIF should decode");
    let Some(expected) = ffmpeg_decode_rgba_with_filter(
        &output_path,
        SAMPLE_WIDTH,
        SAMPLE_HEIGHT,
        "zscale=matrix=bt709:transfer=bt709:primaries=2020,format=gbrp,format=rgba",
    ) else {
        return;
    };
    assert_eq!(actual.width, SAMPLE_WIDTH);
    assert_eq!(actual.height, SAMPLE_HEIGHT);
    let metrics = diff_rgb(&actual.rgba, &expected);
    eprintln!(
        "ICTCP FFmpeg RGB error average={} max={}",
        metrics.average_rgb_abs, metrics.max_rgb_abs
    );
    assert!(metrics.average_rgb_abs <= 10.0);
    assert!(metrics.max_rgb_abs <= 64);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn generated_delta_lf_sample_matches_ffmpeg_when_encoder_present() {
    let root = std::env::temp_dir().join(format!(".test-avif-delta-lf-{}", std::process::id()));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary AVIF sample directory: {err}");
    }
    let output_path = root.join("delta-lf.avif");
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
            "6",
            "-aom-params",
            "deltaq-mode=3:delta-lf-mode=1:enable-chroma-deltaq=1",
            "-f",
            "avif",
        ])
        .arg(&output_path)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping generated delta-lf sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!("libaom encoder is unavailable; skipping generated delta-lf sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    }
    let data = std::fs::read(&output_path).expect("generated delta-lf AVIF should be readable");
    let decoded = avif_rust::image_from_bytes(&data).expect("delta-lf AVIF should decode");
    assert!(
        avif_rust::image_from_bytes(&data[..data.len().saturating_sub(1)]).is_err(),
        "truncated delta-lf AVIF must fail closed"
    );
    assert_eq!(
        (decoded.width, decoded.height),
        (SAMPLE_WIDTH, SAMPLE_HEIGHT)
    );
    if let Some(expected) = ffmpeg_decode_rgba_dynamic(&output_path, SAMPLE_WIDTH, SAMPLE_HEIGHT) {
        let metrics = diff_rgb_dynamic(&decoded.rgba, &expected);
        assert!(
            metrics.average_rgb_abs <= 2.0 && metrics.max_rgb_abs <= 128,
            "delta-lf FFmpeg RGB error average={} max={}",
            metrics.average_rgb_abs,
            metrics.max_rgb_abs
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn generated_gamma22_transfer_sample_matches_ffmpeg_when_encoder_present() {
    generated_transfer_sample_matches_ffmpeg(4, "gamma22", "zscale=tin=4:t=13,format=rgba");
}

#[test]
fn generated_smpte428_transfer_sample_matches_ffmpeg_when_encoder_present() {
    generated_transfer_sample_matches_ffmpeg(17, "smpte428", "zscale=tin=17:t=13,format=rgba");
}

#[test]
fn generated_bt1361_extended_transfer_sample_matches_ffmpeg_when_encoder_present() {
    generated_transfer_sample_matches_ffmpeg_with_tolerance(
        12,
        "bt1361-extended",
        "zscale=tin=12:t=13,format=rgba",
        8.0,
        32,
    );
}

#[test]
fn generated_iec61966_2_4_transfer_sample_matches_ffmpeg_when_encoder_present() {
    generated_transfer_sample_matches_ffmpeg_with_tolerance(
        11,
        "iec61966-2-4",
        "zscale=tin=11:t=13,format=rgba",
        8.0,
        32,
    );
}

#[test]
fn generated_log_transfer_sample_matches_ffmpeg_when_encoder_present() {
    generated_transfer_sample_matches_ffmpeg(9, "log", "zscale=tin=9:t=13,format=rgba");
}

#[test]
fn generated_log_sqrt_transfer_sample_matches_ffmpeg_when_encoder_present() {
    generated_transfer_sample_matches_ffmpeg(10, "log-sqrt", "zscale=tin=10:t=13,format=rgba");
}

#[test]
fn generated_10bit_yuv420_sample_matches_ffmpeg_when_encoder_present() {
    let root = std::env::temp_dir().join(format!(".test-avif-10bit420-{}", std::process::id()));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary 10-bit AVIF sample directory: {err}");
    }
    let output_path = root.join("yuv420-10bit.avif");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .arg("-i")
        .arg(sample_path("WML2Viewer.png"))
        .args([
            "-vf",
            "scale=96:80:flags=neighbor,format=yuv420p10le",
            "-frames:v",
            "1",
            "-c:v",
            "libaom-av1",
            "-still-picture",
            "1",
            "-crf",
            "0",
            "-cpu-used",
            "8",
            "-pix_fmt",
            "yuv420p10le",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-color_range",
            "tv",
            "-f",
            "avif",
        ])
        .arg(&output_path)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping generated 10-bit 4:2:0 sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!("libaom 10-bit 4:2:0 encoder is unavailable; skipping generated sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    }

    let data = std::fs::read(&output_path).expect("generated 10-bit 4:2:0 AVIF should be readable");
    let frame =
        avif_rust::decode_frame_bytes(&data).expect("generated 10-bit 4:2:0 frame should decode");
    assert_eq!((frame.width, frame.height), (96, 80));
    assert_eq!(frame.bit_depth, 10);
    assert_eq!(frame.buffers.planes.len(), 3);
    assert_eq!(
        frame.buffers.planes[0].samples.len(),
        96 * 80,
        "10-bit 4:2:0 luma plane"
    );
    assert_eq!(
        frame.buffers.planes[1].samples.len(),
        48 * 40,
        "10-bit 4:2:0 U plane"
    );
    assert_eq!(
        frame.buffers.planes[2].samples.len(),
        48 * 40,
        "10-bit 4:2:0 V plane"
    );
    let actual = frame
        .to_rgba8()
        .expect("generated 10-bit 4:2:0 RGBA conversion should succeed");
    assert_eq!((actual.width, actual.height), (96, 80));
    if let Some(expected) = ffmpeg_decode_rgba_with_filter(
        &output_path,
        96,
        80,
        "zscale=matrixin=709:transferin=709:primariesin=709:rangein=limited:matrix=709:transfer=709:primaries=709:range=full,format=rgba",
    ) {
        let metrics = diff_rgb_dynamic(&actual.rgba, &expected);
        eprintln!(
            "generated 10-bit 4:2:0: average RGB absolute error={} max={}",
            metrics.average_rgb_abs, metrics.max_rgb_abs
        );
        assert!(
            metrics.average_rgb_abs <= 2.0 && metrics.max_rgb_abs <= 48,
            "generated 10-bit 4:2:0 FFmpeg RGB error average={} max={}",
            metrics.average_rgb_abs,
            metrics.max_rgb_abs
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn generated_10bit_identity_gbr_sample_matches_ffmpeg_when_encoder_present() {
    let root = std::env::temp_dir().join(format!(
        ".test-avif-10bit-identity-gbr-{}",
        std::process::id()
    ));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary 10-bit identity AVIF directory: {err}");
    }
    let output_path = root.join("identity-gbr-10bit.avif");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .arg("-i")
        .arg(sample_path("WML2Viewer.png"))
        .args([
            "-vf",
            "scale=512:512:flags=neighbor,format=gbrp10le",
            "-frames:v",
            "1",
            "-c:v",
            "libaom-av1",
            "-still-picture",
            "1",
            "-crf",
            "0",
            "-cpu-used",
            "8",
            "-pix_fmt",
            "gbrp10le",
            "-colorspace",
            "rgb",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "iec61966-2-1",
            "-color_range",
            "pc",
            "-f",
            "avif",
        ])
        .arg(&output_path)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping generated 10-bit identity GBR sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!("libaom 10-bit identity GBR encoder is unavailable; skipping generated sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    }

    let data =
        std::fs::read(&output_path).expect("generated 10-bit identity GBR AVIF should be readable");
    let frame = avif_rust::decode_frame_bytes(&data)
        .expect("generated 10-bit identity GBR frame should decode");
    assert_eq!((frame.width, frame.height), (512, 512));
    assert_eq!(frame.bit_depth, 10);
    assert_eq!(frame.buffers.planes.len(), 3);
    assert!(
        frame
            .color_config
            .color_description
            .is_some_and(|description| matches!(description.matrix_coefficients, 0 | 3))
    );
    let actual = frame
        .to_rgba8()
        .expect("generated 10-bit identity GBR RGBA conversion should succeed");
    if let Some(expected) = ffmpeg_decode_rgba_dynamic(&output_path, 512, 512) {
        let metrics = diff_rgb_dynamic(&actual.rgba, &expected);
        eprintln!(
            "generated 10-bit identity GBR: average RGB absolute error={} max={}",
            metrics.average_rgb_abs, metrics.max_rgb_abs
        );
        assert!(
            metrics.average_rgb_abs <= 2.0 && metrics.max_rgb_abs <= 8,
            "generated 10-bit identity GBR FFmpeg RGB error average={} max={}",
            metrics.average_rgb_abs,
            metrics.max_rgb_abs
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn generated_chroma_sample_positions_match_ffmpeg_when_encoder_present() {
    let root =
        std::env::temp_dir().join(format!(".test-avif-chroma-position-{}", std::process::id()));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary AVIF sample directory: {err}");
    }
    for (position, label) in [(0u8, "unknown"), (1, "vertical"), (2, "colocated")] {
        let output_path = root.join(format!("{label}.avif"));
        let status = Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error"])
            .arg("-i")
            .arg(sample_path("WML2Viewer.png"))
            .args([
                "-vf",
                "scale=64:64:flags=neighbor,format=yuv420p",
                "-frames:v",
                "1",
                "-c:v",
                "libaom-av1",
                "-still-picture",
                "1",
                "-cpu-used",
                "8",
                "-crf",
                "0",
                "-bsf:v",
            ])
            .arg(format!("av1_metadata=chroma_sample_position={position}"))
            .args(["-f", "avif"])
            .arg(&output_path)
            .status();
        let Ok(status) = status else {
            eprintln!("ffmpeg is not available; skipping generated {label} chroma sample");
            let _ = std::fs::remove_dir_all(&root);
            return;
        };
        if !status.success() {
            eprintln!("libaom chroma-position encoder is unavailable; skipping generated sample");
            let _ = std::fs::remove_dir_all(&root);
            return;
        }
        let data = std::fs::read(&output_path).expect("generated chroma AVIF should be readable");
        let actual =
            avif_rust::image_from_bytes(&data).expect("chroma-position AVIF should decode");
        assert_eq!((actual.width, actual.height), (64, 64));
        if let Some(expected) = ffmpeg_decode_rgba_dynamic(&output_path, 64, 64) {
            let metrics = diff_rgb_dynamic(&actual.rgba, &expected);
            eprintln!(
                "{label} chroma position: average RGB absolute error={} max={}",
                metrics.average_rgb_abs, metrics.max_rgb_abs
            );
            assert!(
                metrics.average_rgb_abs <= 2.0 && metrics.max_rgb_abs <= 64,
                "{label} chroma-position FFmpeg RGB error average={} max={}",
                metrics.average_rgb_abs,
                metrics.max_rgb_abs
            );
        }
    }
    let _ = std::fs::remove_dir_all(&root);
}

fn generated_transfer_sample_matches_ffmpeg(transfer: u8, label: &str, oracle_filter: &str) {
    generated_transfer_sample_matches_ffmpeg_with_tolerance(
        transfer,
        label,
        oracle_filter,
        2.0,
        32,
    );
}

fn generated_transfer_sample_matches_ffmpeg_with_tolerance(
    transfer: u8,
    label: &str,
    oracle_filter: &str,
    average_limit: f64,
    max_limit: u8,
) {
    let root = std::env::temp_dir().join(format!(
        ".test-avif-transfer-{label}-{}",
        std::process::id()
    ));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary AVIF sample directory: {err}");
    }
    let output_path = root.join(format!("{label}.avif"));
    let setparams = format!(
        "scale=4:4:flags=neighbor,setparams=colorspace=bt709:color_primaries=bt709:color_trc={transfer},format=yuv444p"
    );
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .arg("-i")
        .arg(sample_path("WML2Viewer.png"))
        .args([
            "-vf",
            &setparams,
            "-frames:v",
            "1",
            "-c:v",
            "libaom-av1",
            "-still-picture",
            "1",
            "-crf",
            "0",
            "-f",
            "avif",
        ])
        .arg(&output_path)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping generated {label} sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!("libaom {label} encoder is unavailable; skipping generated sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    }
    let data = std::fs::read(&output_path).expect("generated transfer AVIF should be readable");
    let actual = avif_rust::image_from_bytes(&data).expect("transfer AVIF should decode");
    assert_eq!((actual.width, actual.height), (4, 4));
    if let Some(expected) = ffmpeg_decode_rgba_with_filter(&output_path, 4, 4, oracle_filter) {
        let metrics = diff_rgb_dynamic(&actual.rgba, &expected);
        eprintln!(
            "{label} transfer: average RGB absolute error={} max={}",
            metrics.average_rgb_abs, metrics.max_rgb_abs
        );
        assert!(
            metrics.average_rgb_abs <= average_limit
                && f64::from(metrics.max_rgb_abs) <= f64::from(max_limit),
            "{label} FFmpeg RGB error average={} max={}",
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
fn generated_10bit_yuv420_sample_matches_ffmpeg_when_encoder_is_present() {
    generated_subsampled_sample_matches_ffmpeg("yuv420p10le", "10bit-yuv420", 10, Some((1, 1)));
}

#[test]
fn generated_10bit_yuv422_sample_matches_ffmpeg_when_encoder_is_present() {
    generated_subsampled_sample_matches_ffmpeg("yuv422p10le", "10bit-yuv422", 10, Some((1, 0)));
}

#[test]
fn generated_12bit_yuv420_sample_matches_ffmpeg_when_encoder_is_present() {
    generated_subsampled_sample_matches_ffmpeg("yuv420p12le", "12bit-yuv420", 12, Some((1, 1)));
}

#[test]
fn generated_12bit_yuv422_sample_matches_ffmpeg_when_encoder_is_present() {
    generated_subsampled_sample_matches_ffmpeg("yuv422p12le", "12bit-yuv422", 12, Some((1, 0)));
}

#[test]
fn generated_12bit_monochrome_sample_matches_ffmpeg_when_encoder_is_present() {
    generated_subsampled_sample_matches_ffmpeg("gray12le", "12bit-monochrome", 12, None);
}

fn generated_subsampled_sample_matches_ffmpeg(
    pixel_format: &str,
    label: &str,
    bit_depth: u8,
    expected_chroma_subsampling: Option<(u8, u8)>,
) {
    let root = std::env::temp_dir().join(format!(".test-avif-{label}-{}", std::process::id()));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary AVIF sample directory: {err}");
    }
    let output_path = root.join(format!("generated-{label}.avif"));
    let filter = format!("scale=64:64:flags=neighbor,format={pixel_format}");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .arg("-i")
        .arg(sample_path("WML2Viewer.png"))
        .args(["-vf"])
        .arg(&filter)
        .args([
            "-frames:v",
            "1",
            "-c:v",
            "libaom-av1",
            "-still-picture",
            "1",
            "-cpu-used",
            "8",
            "-crf",
            "0",
            "-pix_fmt",
        ])
        .arg(pixel_format)
        .args(["-f", "avif"])
        .arg(&output_path)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping generated {label} sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!(
            "libaom {bit_depth}-bit {label} encoder is unavailable; skipping generated sample"
        );
        let _ = std::fs::remove_dir_all(&root);
        return;
    }
    let data = std::fs::read(&output_path).expect("generated subsampled AVIF should be readable");
    let frame =
        avif_rust::decode_frame_bytes(&data).expect("generated subsampled AVIF should decode");
    assert_eq!(frame.bit_depth, bit_depth, "{label} bit depth");
    if let Some((subsampling_x, subsampling_y)) = expected_chroma_subsampling {
        for plane in frame.buffers.planes.get(1..3).unwrap_or_default() {
            assert_eq!(
                (plane.layout.subsampling_x, plane.layout.subsampling_y),
                (subsampling_x, subsampling_y),
                "{label} native chroma subsampling"
            );
        }
    }
    let actual =
        avif_rust::image_from_bytes(&data).expect("generated subsampled RGBA should decode");
    assert_eq!((actual.width, actual.height), (64, 64));
    if let Some(expected) = ffmpeg_decode_rgba_dynamic(&output_path, 64, 64) {
        let metrics = diff_rgb_dynamic(&actual.rgba, &expected);
        eprintln!(
            "{label}: average RGB absolute error={} max={}",
            metrics.average_rgb_abs, metrics.max_rgb_abs
        );
        assert!(
            metrics.average_rgb_abs <= 2.0 && metrics.max_rgb_abs <= 48,
            "{label} FFmpeg RGB error average={} max={}",
            metrics.average_rgb_abs,
            metrics.max_rgb_abs
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn generated_10bit_alpha_sample_decodes_native_and_rgba_when_encoder_present() {
    let root = std::env::temp_dir().join(format!(".test-avif-alpha10-{}", std::process::id()));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary AVIF alpha directory: {err}");
    }
    let output_path = root.join("generated-alpha10.avif");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i", "testsrc2=size=128x128:rate=1"])
        .args(["-f", "lavfi", "-i", "color=gray:size=128x128:rate=1"])
        .args([
            "-filter_complex",
            "[0:v]format=yuv444p10le[color];[1:v]format=gray10le[alpha]",
            "-map",
            "[color]",
            "-map",
            "[alpha]",
            "-frames:v",
            "1",
            "-c:v",
            "libaom-av1",
            "-still-picture",
            "1",
            "-crf",
            "0",
            "-b:v",
            "0",
            "-pix_fmt:v:0",
            "yuv444p10le",
            "-pix_fmt:v:1",
            "gray10le",
            "-f",
            "avif",
        ])
        .arg(&output_path)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping generated 10-bit alpha sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!("libaom 10-bit alpha encoder is unavailable; skipping generated sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    }
    let data = std::fs::read(&output_path).expect("generated 10-bit alpha AVIF should be readable");
    let frame = avif_rust::decode_frame_bytes(&data).expect("10-bit alpha frame should decode");
    assert_eq!((frame.width, frame.height), (128, 128));
    assert_eq!(frame.bit_depth, 10);
    assert_eq!(frame.buffers.planes.len(), 4);
    assert_eq!(frame.buffers.planes[3].layout.plane, 3);
    assert_eq!(frame.buffers.planes[3].samples.len(), 128 * 128);
    let image =
        avif_rust::image_from_bytes(&data).expect("10-bit alpha public decode should succeed");
    assert_eq!((image.width, image.height), (128, 128));
    if let Some(expected_planes) = ffmpeg_decode_raw_stream(&output_path, Some(0), "yuv444p10le") {
        let expected: Vec<u16> = expected_planes
            .chunks_exact(2)
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]) & 0x03ff)
            .collect();
        let plane_len = 128 * 128;
        for plane_index in 0..3 {
            let max_error = frame.buffers.planes[plane_index]
                .samples
                .iter()
                .zip(&expected[plane_index * plane_len..(plane_index + 1) * plane_len])
                .map(|(actual, expected)| actual.abs_diff(*expected))
                .max()
                .unwrap_or(0);
            assert!(
                max_error <= 16,
                "10-bit alpha color plane {plane_index} max error was {max_error}"
            );
        }
    }
    if let Some(expected_alpha_bytes) = ffmpeg_decode_raw_stream(&output_path, Some(1), "gray10le")
    {
        let expected_alpha: Vec<u16> = expected_alpha_bytes
            .chunks_exact(2)
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]) & 0x03ff)
            .collect();
        let max_native_error = frame.buffers.planes[3]
            .samples
            .iter()
            .zip(&expected_alpha)
            .map(|(actual, expected)| actual.abs_diff(*expected))
            .max()
            .unwrap_or(0);
        assert!(
            max_native_error <= 16,
            "10-bit alpha native max error was {max_native_error}"
        );
        let expected_alpha8 = expected_alpha
            .iter()
            .map(|sample| u8::try_from((*sample + 2) >> 2).unwrap())
            .collect::<Vec<_>>();
        let max_error = image
            .rgba
            .chunks_exact(4)
            .map(|rgba| rgba[3])
            .zip(expected_alpha8)
            .map(|(actual, expected)| actual.abs_diff(expected))
            .max()
            .unwrap_or(0);
        assert!(
            max_error <= 1,
            "10-bit alpha public max error was {max_error}"
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

fn generated_intrabc_sample_matches_ffmpeg(
    pixel_format: &str,
    expected_format: &str,
    plane_lengths: [usize; 3],
    enable_cdef: bool,
) {
    generated_intrabc_sample_matches_ffmpeg_sized(
        pixel_format,
        expected_format,
        plane_lengths,
        enable_cdef,
        128,
        128,
    );
}

fn generated_intrabc_sample_matches_ffmpeg_sized(
    pixel_format: &str,
    expected_format: &str,
    plane_lengths: [usize; 3],
    enable_cdef: bool,
    width: usize,
    height: usize,
) {
    let root = std::env::temp_dir().join(format!(
        ".test-avif-intrabc-{}-{width}x{height}-{pixel_format}-cdef{}",
        std::process::id(),
        u8::from(enable_cdef)
    ));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary IntrABC sample directory: {err}");
    }
    let output_path = root.join("intrabc.avif");
    let aom_params = format!(
        "sb-size=128:enable-intrabc=1:enable-cdef={}:enable-restoration=0",
        u8::from(enable_cdef)
    );
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i"])
        .arg(format!("testsrc2=size={width}x{height}:rate=1"))
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
            pixel_format,
            "-aom-params",
        ])
        .arg(&aom_params)
        .args(["-f", "avif"])
        .arg(&output_path)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping generated IntrABC sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!("libaom IntrABC encoder is unavailable; skipping generated sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    }
    let data = std::fs::read(&output_path).expect("generated IntrABC AVIF should be readable");
    let frame = avif_rust::decode_frame_bytes(&data).expect("IntrABC AVIF should decode");
    assert_eq!((frame.width, frame.height), (width, height));
    let image = avif_rust::image_from_bytes(&data).expect("IntrABC public decode should succeed");
    assert_eq!((image.width, image.height), (width, height));
    if let Some(expected) = ffmpeg_decode_raw(&output_path, expected_format) {
        assert_eq!(expected.len(), plane_lengths.iter().sum());
        let mut plane_start = 0;
        for (plane_index, actual_plane) in frame.buffers.planes.iter().take(3).enumerate() {
            let plane_len = plane_lengths[plane_index];
            let expected_plane = &expected[plane_start..plane_start + plane_len];
            assert_eq!(actual_plane.samples.len(), plane_len);
            let max_error = actual_plane
                .samples
                .iter()
                .zip(expected_plane)
                .map(|(actual, expected)| u8::try_from(*actual).unwrap().abs_diff(*expected))
                .max()
                .unwrap_or(0);
            let average_error = actual_plane
                .samples
                .iter()
                .zip(expected_plane)
                .map(|(actual, expected)| {
                    f64::from(u8::try_from(*actual).unwrap().abs_diff(*expected))
                })
                .sum::<f64>()
                / plane_len as f64;
            eprintln!("IntrABC plane {plane_index}: average={average_error} max={max_error}");
            assert!(
                average_error <= 2.0 && max_error <= 32,
                "IntrABC plane {plane_index}: average={average_error} max={max_error}"
            );
            plane_start += plane_len;
        }
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn generated_large_intrabc_sample_matches_ffmpeg_when_encoder_is_present() {
    generated_intrabc_sample_matches_ffmpeg_sized(
        "yuv444p",
        "yuv444p",
        [256 * 256; 3],
        false,
        256,
        256,
    );
}

#[test]
fn generated_intrabc_yuv444_sample_matches_ffmpeg_when_encoder_present() {
    generated_intrabc_sample_matches_ffmpeg("yuv444p", "yuv444p", [128 * 128; 3], false);
}

#[test]
fn generated_intrabc_yuv420_sample_matches_ffmpeg_when_encoder_present() {
    generated_intrabc_sample_matches_ffmpeg(
        "yuv420p",
        "yuv420p",
        [128 * 128, 64 * 64, 64 * 64],
        false,
    );
}

#[test]
fn generated_intrabc_yuv422_sample_matches_ffmpeg_when_encoder_present() {
    generated_intrabc_sample_matches_ffmpeg(
        "yuv422p",
        "yuv422p",
        [128 * 128, 64 * 128, 64 * 128],
        false,
    );
}

#[test]
fn generated_intrabc_with_cdef_when_encoder_present() {
    generated_intrabc_sample_matches_ffmpeg("yuv444p", "yuv444p", [128 * 128; 3], true);
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
    generated_matrix_sample_matches_ffmpeg("smpte240m", "SMPTE 240M", "bt709");
}

#[test]
fn generated_bt709_matrix_sample_matches_ffmpeg_when_encoder_present() {
    generated_matrix_sample_matches_ffmpeg("bt709", "BT.709", "bt709");
}

#[test]
fn generated_bt470bg_matrix_sample_matches_ffmpeg_when_encoder_present() {
    generated_matrix_sample_matches_ffmpeg("bt470bg", "BT.470 BG", "bt470bg");
}

#[test]
fn generated_bt2020_constant_luminance_matrix_sample_matches_ffmpeg_when_encoder_present() {
    generated_matrix_sample_matches_ffmpeg("bt2020c", "BT.2020 constant-luminance", "bt2020");
}

#[test]
fn generated_smpte2085_matrix_sample_decodes_when_encoder_present() {
    generated_matrix_sample_matches_ffmpeg("smpte2085", "SMPTE ST 2085", "bt2020");
}

#[test]
fn generated_ycgco_matrix_sample_matches_ffmpeg_when_encoder_present() {
    generated_matrix_sample_matches_ffmpeg("ycgco", "YCgCo", "bt709");
}

#[test]
fn generated_chroma_derived_ncl_matrix_sample_decodes_when_encoder_present() {
    generated_matrix_sample_matches_ffmpeg(
        "chroma-derived-nc",
        "chroma-derived non-constant-luminance",
        "bt2020",
    );
}

#[test]
fn generated_chroma_derived_cl_matrix_sample_decodes_when_encoder_present() {
    generated_matrix_sample_matches_ffmpeg(
        "chroma-derived-c",
        "chroma-derived constant-luminance",
        "bt2020",
    );
}

fn generated_matrix_sample_matches_ffmpeg(colorspace: &str, label: &str, primaries: &str) {
    let root = std::env::temp_dir().join(format!(
        ".test-avif-matrix-{colorspace}-{}",
        std::process::id()
    ));
    if let Err(err) = std::fs::create_dir_all(&root) {
        panic!("failed to create temporary AVIF sample directory: {err}");
    }
    let output_path = root.join(format!("{colorspace}.avif"));
    let mut command = Command::new("ffmpeg");
    command
        .args(["-y", "-loglevel", "error"])
        .arg("-i")
        .arg(sample_path("WML2Viewer.png"));
    if matches!(
        colorspace,
        "bt2020c" | "smpte2085" | "ycgco" | "chroma-derived-nc" | "chroma-derived-c"
    ) {
        // libaom cannot infer these non-default matrix conversions directly
        // from RGB input; mark the already formatted YUV frame instead.
        command.arg("-vf").arg(format!(
            "format=yuv444p,setparams=colorspace={colorspace}:color_primaries={primaries}"
        ));
    }
    let status = command
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
            primaries,
            "-color_trc",
            "iec61966-2-1",
            "-colorspace",
            colorspace,
            "-f",
            "avif",
        ])
        .arg(&output_path)
        .status();
    let Ok(status) = status else {
        eprintln!("ffmpeg is not available; skipping {label} matrix sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    };
    if !status.success() {
        eprintln!("libaom encoder is unavailable; skipping {label} matrix sample");
        let _ = std::fs::remove_dir_all(&root);
        return;
    }
    let data = std::fs::read(&output_path).expect("generated matrix AVIF should be readable");
    let actual = avif_rust::image_from_bytes(&data).expect("generated matrix AVIF should decode");
    assert_eq!((actual.width, actual.height), (SAMPLE_WIDTH, SAMPLE_HEIGHT));
    if colorspace == "ycgco" {
        if let Some(expected) = ffmpeg_decode_raw(&output_path, "yuv444p") {
            let frame =
                avif_rust::decode_frame_bytes(&data).expect("YCgCo native planes should decode");
            assert_eq!(expected.len(), SAMPLE_PIXELS * 3);
            for plane_index in 0..3 {
                let expected_plane =
                    &expected[plane_index * SAMPLE_PIXELS..(plane_index + 1) * SAMPLE_PIXELS];
                let max_error = frame.buffers.planes[plane_index]
                    .samples
                    .iter()
                    .zip(expected_plane)
                    .map(|(actual, expected)| u8::try_from(*actual).unwrap().abs_diff(*expected))
                    .max()
                    .unwrap_or(0);
                assert!(
                    max_error <= 2,
                    "YCgCo native plane {plane_index} max error was {max_error}"
                );
            }
        }
        let _ = std::fs::remove_dir_all(&root);
        return;
    }
    if colorspace == "bt2020c" {
        // This FFmpeg build can encode matrix 10 but cannot convert its
        // decoded frames to RGBA in the generic output path. ImageMagick's
        // AVIF decoder provides an independent pixel oracle here.
        if let Some(expected) = imagemagick_decode_rgba(&output_path, SAMPLE_WIDTH, SAMPLE_HEIGHT) {
            let metrics = diff_rgb(&actual.rgba, &expected);
            assert!(
                // ImageMagick and this decoder apply slightly different
                // gamut handling around the BT.2020 CL branch points.
                metrics.average_rgb_abs <= 6.0 && metrics.max_rgb_abs <= 128,
                "{label} ImageMagick RGB error average={} max={}",
                metrics.average_rgb_abs,
                metrics.max_rgb_abs
            );
        }
    } else if !matches!(
        colorspace,
        "smpte2085" | "chroma-derived-nc" | "chroma-derived-c"
    ) {
        if let Some(expected) = ffmpeg_decode_rgba(&output_path) {
            let metrics = diff_rgb(&actual.rgba, &expected);
            assert!(
                metrics.average_rgb_abs <= 2.0 && metrics.max_rgb_abs <= 32,
                "{label} FFmpeg RGB error average={} max={}",
                metrics.average_rgb_abs,
                metrics.max_rgb_abs
            );
        }
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
        ("avif/supported/fox.profile1.8bpc.yuv444.avif", 1204, 800),
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
            "avif/supported/fox.profile1.8bpc.yuv444.avif",
            "yuv444p",
            [1204 * 800, 1204 * 800, 1204 * 800],
        ),
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
    let scalar_rgba: Vec<u8> = rgba16
        .rgba
        .iter()
        .map(|sample| ((u32::from(*sample) * 255 + 32_767) / 65_535) as u8)
        .collect();
    let scalar_max_error = actual
        .rgba
        .iter()
        .zip(&scalar_rgba)
        .map(|(actual, scalar)| actual.abs_diff(*scalar))
        .max()
        .unwrap_or(0);
    assert!(
        scalar_max_error <= 1,
        "10-bit direct RGBA8 conversion differs from scalar RGBA16 path: max={scalar_max_error}"
    );
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
fn public_grid_sample_exposes_native_planes_when_present() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root should exist")
        .join("test/images/external/avif/unsupported/sofa_grid1x5_420.avif");
    if !path.is_file() {
        eprintln!("external grid sample is unavailable; skipping native-plane oracle");
        return;
    }
    let data = std::fs::read(&path).expect("grid AVIF should be readable");
    let frame =
        avif_rust::decode_frame_bytes(&data).expect("grid native-plane composition should succeed");
    assert_eq!((frame.width, frame.height), (1024, 770));
    assert_eq!(frame.buffers.planes.len(), 3);
    let Some(expected) = ffmpeg_decode_raw(&path, "yuv420p") else {
        return;
    };
    let expected_lengths = [1024 * 770, 512 * 385, 512 * 385];
    assert_eq!(expected.len(), expected_lengths.iter().sum());
    let mut offset = 0;
    for (plane_index, &plane_length) in expected_lengths.iter().enumerate() {
        let expected_plane = &expected[offset..offset + plane_length];
        let actual_plane = &frame.buffers.planes[plane_index].samples;
        assert_eq!(actual_plane.len(), plane_length, "grid plane {plane_index}");
        let max_error = actual_plane
            .iter()
            .zip(expected_plane)
            .map(|(actual, expected)| u8::try_from(*actual).unwrap().abs_diff(*expected))
            .max()
            .unwrap_or(0);
        let average_error = actual_plane
            .iter()
            .zip(expected_plane)
            .map(|(actual, expected)| f64::from(u8::try_from(*actual).unwrap().abs_diff(*expected)))
            .sum::<f64>()
            / plane_length as f64;
        eprintln!("grid plane {plane_index}: average={average_error} max={max_error}");
        assert!(
            average_error <= 2.0 && max_error <= 32,
            "grid plane {plane_index}: average={average_error} max={max_error}"
        );
        offset += plane_length;
    }
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
fn public_sequence_inter_sample_decodes_when_present() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root should exist")
        .join("test/images/external/avif/unsupported/star-8bpc.avifs");
    if !path.is_file() {
        eprintln!("external sequence sample is unavailable; skipping inter oracle");
        return;
    }
    let data = std::fs::read(&path).expect("external sequence AVIS should be readable");
    let frame = avif_rust::decode_sequence_frame_bytes(&data, 2)
        .expect("the inter sample should decode as the third AVIS sample");
    assert_eq!((frame.width, frame.height), (159, 159));
    let Some(expected) = ffmpeg_decode_rgba_stream_frame(&path, 1, 1, 159, 159) else {
        return;
    };
    let metrics = diff_rgb_dynamic(&frame.to_rgba8().unwrap().rgba, &expected);
    eprintln!(
        "sequence inter frame: average RGB absolute error={}, max={}",
        metrics.average_rgb_abs, metrics.max_rgb_abs
    );
    // The external sequence contains compound/warped blocks, so this remains a
    // broad quality gate while generated simple-inter coverage uses the tighter
    // threshold above.
    assert!(metrics.average_rgb_abs <= 64.0);
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
    let frame = avif_rust::decode_frame_bytes(&data).expect("12-bit frame should decode");
    let rgba16 = frame.to_rgba16().expect("12-bit RGBA16 should decode");
    let scalar_rgba: Vec<u8> = rgba16
        .rgba
        .iter()
        .map(|sample| ((u32::from(*sample) * 255 + 32_767) / 65_535) as u8)
        .collect();
    let scalar_max_error = actual
        .rgba
        .iter()
        .zip(&scalar_rgba)
        .map(|(actual, scalar)| actual.abs_diff(*scalar))
        .max()
        .unwrap_or(0);
    assert!(
        scalar_max_error <= 1,
        "12-bit direct RGBA8 max error={scalar_max_error}"
    );
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
fn public_alpha_sample_exposes_native_alpha_plane_when_present() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root should exist")
        .join("test/images/external/avif/unsupported/plum-blossom-small.profile1.8bpc.yuv444.alpha-full.avif");
    if !path.is_file() {
        eprintln!("external alpha sample is unavailable; skipping native alpha oracle");
        return;
    }
    let data = std::fs::read(&path).expect("external alpha AVIF should be readable");
    let frame = avif_rust::decode_frame_bytes(&data).expect("alpha frame should decode");
    assert_eq!((frame.width, frame.height), (128, 128));
    let alpha = frame
        .buffers
        .planes
        .get(3)
        .expect("native decoded frame should expose plane 3 as alpha");
    assert_eq!(alpha.layout.plane, 3);
    assert_eq!(alpha.samples.len(), 128 * 128);
    let Some(expected) = ffmpeg_decode_alpha_plane(&path, 128, 128) else {
        return;
    };
    let max_error = alpha
        .samples
        .iter()
        .zip(expected)
        .map(|(actual, expected)| u8::try_from(*actual).unwrap().abs_diff(expected))
        .max()
        .unwrap_or(0);
    assert!(
        max_error <= 1,
        "native alpha plane max error was {max_error}"
    );
    let converted = frame
        .to_rgba8()
        .expect("native alpha plane should feed RGBA conversion");
    let expected = ffmpeg_decode_alpha_plane(&path, 128, 128).expect("alpha oracle should exist");
    let converted_max_error = converted
        .rgba
        .chunks_exact(4)
        .zip(expected)
        .map(|(pixel, expected)| pixel[3].abs_diff(expected))
        .max()
        .unwrap_or(0);
    assert!(
        converted_max_error <= 1,
        "converted native alpha max error was {converted_max_error}"
    );
}

#[test]
fn public_alpha_noispe_sample_decodes_with_skip_tx_size_signaling() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root should exist")
        .join("test/images/external/avif/unsupported/alpha_noispe.avif");
    if !path.is_file() {
        eprintln!("external alpha_noispe sample is unavailable; skipping boundary check");
        return;
    }
    let data = std::fs::read(&path).expect("alpha_noispe AVIF should be readable");
    let image = avif_rust::image_from_bytes(&data).expect("alpha_noispe should decode fully");
    assert_eq!((image.width, image.height), (80, 80));
}

#[test]
fn public_alpha_noispe_sample_matches_ffmpeg_when_present() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root should exist")
        .join("test/images/external/avif/unsupported/alpha_noispe.avif");
    if !path.is_file() {
        eprintln!("external alpha_noispe sample is unavailable; skipping pixel oracle");
        return;
    }
    let data = std::fs::read(&path).expect("alpha_noispe AVIF should be readable");
    let actual = avif_rust::image_from_bytes(&data).expect("alpha_noispe should decode fully");
    let Some(expected) = ffmpeg_decode_rgba_dynamic(&path, 80, 80) else {
        return;
    };
    let metrics = diff_rgb_dynamic(&actual.rgba, &expected);
    eprintln!(
        "alpha_noispe: average RGB absolute error={}, max={}",
        metrics.average_rgb_abs, metrics.max_rgb_abs
    );
    assert!(
        metrics.average_rgb_abs <= 2.0 && metrics.max_rgb_abs <= 48,
        "alpha_noispe FFmpeg RGB error average={} max={}",
        metrics.average_rgb_abs,
        metrics.max_rgb_abs
    );
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
fn public_icc_device_class_samples_keep_profile_conversion() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root should exist")
        .join(
            "test/images/external/avif/unsupported/red-at-12-oclock-with-color-profile-8bpc.avif",
        );
    if !path.is_file() {
        eprintln!("external ICC sample is unavailable; skipping device-class check");
        return;
    }
    let original = std::fs::read(&path).expect("ICC AVIF sample should be readable");
    let baseline = avif_rust::image_from_bytes(&original).expect("ICC AVIF sample should decode");
    let class_offset = original
        .windows(4)
        .enumerate()
        .find_map(|(offset, bytes)| (bytes == b"mntr").then_some(offset))
        .expect("ICC sample should contain a profile device class");
    for class in [b"spac", b"scnr"] {
        let mut mutated = original.clone();
        mutated[class_offset..class_offset + 4].copy_from_slice(class);
        let converted = avif_rust::image_from_bytes(&mutated)
            .expect("ICC input/color-space profile class should decode");
        assert_eq!(converted, baseline, "ICC class {:?}", class);
    }
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
        ("kimono.rotate90.avif", 722, 1024),
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

#[test]
fn public_nonrotated_alpha_sample_matches_ffmpeg_when_present() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root should exist")
        .join("test/images/external/avif/unsupported/abc_color_irot_alpha_NOirot.avif");
    if !path.is_file() {
        eprintln!("official non-rotated alpha sample is unavailable; skipping oracle");
        return;
    }
    let data = std::fs::read(&path).expect("non-rotated alpha AVIF should be readable");
    let actual =
        avif_rust::image_from_bytes(&data).expect("non-rotated alpha sample should decode");
    assert_eq!((actual.width, actual.height), (256, 512));
    let Some(expected_alpha) = ffmpeg_decode_alpha_plane(&path, 512, 256) else {
        return;
    };
    let mut rotated_alpha = vec![0; 256 * 512];
    for y in 0..256 {
        for x in 0..512 {
            let destination_x = y;
            let destination_y = 512 - 1 - x;
            rotated_alpha[destination_y * 256 + destination_x] = expected_alpha[y * 512 + x];
        }
    }
    let max_alpha = actual
        .rgba
        .chunks_exact(4)
        .zip(rotated_alpha)
        .map(|(pixel, expected)| pixel[3].abs_diff(expected))
        .max()
        .unwrap_or(0);
    assert!(max_alpha <= 8, "non-rotated alpha error max={max_alpha}");
}

#[test]
fn public_additional_official_samples_match_ffmpeg_when_present() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root should exist")
        .join("test/images/external/avif/unsupported");
    let cases = [
        ("abc_color_irot_alpha_NOirot.avif", 256, 512, true),
        ("sofa_grid1x5_420_dimg_repeat.avif", 1024, 770, true),
        ("sofa_grid1x5_420_reversed_dimg_order.avif", 1024, 770, true),
        ("draw_points_idat.avif", 33, 11, true),
        // FFmpeg's current AVIF demuxer rejects progressive idat input; the
        // ImageMagick decoder provides the independent pixel oracle below.
        ("draw_points_idat_progressive.avif", 33, 11, false),
        ("draw_points_idat_progressive_metasize0.avif", 33, 11, false),
        ("draw_points_idat_metasize0.avif", 33, 11, true),
        ("extended_pixi.avif", 4, 4, true),
        ("clap_irot_imir_non_essential.avif", 10, 8, true),
        ("clop_irot_imor.avif", 34, 12, true),
    ];
    for (name, width, height, has_ffmpeg_oracle) in cases {
        let path = root.join(name);
        if !path.is_file() {
            eprintln!("additional official sample is unavailable; skipping {name}");
            continue;
        }
        let data = std::fs::read(&path).expect("official sample should be readable");
        let actual = avif_rust::image_from_bytes(&data)
            .unwrap_or_else(|err| panic!("{name} should decode: {err}"));
        assert_eq!((actual.width, actual.height), (width, height), "{name}");
        if name == "extended_pixi.avif" {
            let frame = avif_rust::decode_frame_bytes(&data).expect("extended pixi frame");
            let Some(expected) = ffmpeg_decode_raw(&path, "yuv420p") else {
                continue;
            };
            let expected_lengths = [16, 4, 4];
            assert_eq!(expected.len(), expected_lengths.iter().sum());
            let mut offset = 0;
            for (plane_index, &plane_length) in expected_lengths.iter().enumerate() {
                let actual_plane = &frame.buffers.planes[plane_index].samples;
                assert_eq!(
                    actual_plane.len(),
                    plane_length,
                    "{name} plane {plane_index}"
                );
                let expected_plane = &expected[offset..offset + plane_length];
                assert_eq!(
                    actual_plane
                        .iter()
                        .map(|sample| u8::try_from(*sample).unwrap())
                        .collect::<Vec<_>>(),
                    expected_plane,
                    "{name} plane {plane_index}"
                );
                offset += plane_length;
            }
            continue;
        }
        if name == "draw_points_idat_progressive_metasize0.avif" {
            // ImageMagick rejects a progressive idat stream whose meta box
            // uses the unspecified-size form. The decoder still has a strict
            // dimension/complete-output gate through the external manifest.
            continue;
        }
        if name == "draw_points_idat_progressive.avif" {
            let Some(expected) = imagemagick_decode_rgba(&path, width, height) else {
                continue;
            };
            let metrics = diff_rgb_dynamic(&actual.rgba, &expected);
            eprintln!(
                "{name}: ImageMagick RGB error average={} max={}",
                metrics.average_rgb_abs, metrics.max_rgb_abs
            );
            assert!(
                metrics.average_rgb_abs <= 2.0 && metrics.max_rgb_abs <= 48,
                "{name}: ImageMagick RGB error average={} max={}",
                metrics.average_rgb_abs,
                metrics.max_rgb_abs
            );
            continue;
        }
        if !has_ffmpeg_oracle {
            continue;
        }
        let Some(expected) = ffmpeg_decode_rgba_dynamic(&path, width, height) else {
            continue;
        };
        let metrics = diff_rgb_dynamic(&actual.rgba, &expected);
        eprintln!(
            "{name}: average RGB absolute error={} max={}",
            metrics.average_rgb_abs, metrics.max_rgb_abs
        );
        assert!(
            metrics.average_rgb_abs <= 2.0 && metrics.max_rgb_abs <= 48,
            "{name}: FFmpeg RGB error average={} max={}",
            metrics.average_rgb_abs,
            metrics.max_rgb_abs
        );
    }
}

#[test]
fn all_official_unsupported_samples_decode_without_partial_output() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root should exist")
        .join("test/images/external/avif/unsupported");
    if !root.is_dir() {
        eprintln!("official unsupported sample directory is unavailable; skipping audit");
        return;
    }
    let cases = [
        ("abc_color_irot_alpha_irot.avif", 256, 512),
        ("abc_color_irot_alpha_NOirot.avif", 256, 512),
        ("alpha_noispe.avif", 80, 80),
        ("clap_irot_imir_non_essential.avif", 10, 8),
        ("clop_irot_imor.avif", 34, 12),
        ("colors-animated-8bpc.avif", 150, 150),
        ("colors-animated-12bpc-keyframes-0-2-3.avif", 64, 64),
        ("colors_text_hdr_p3.avif", 200, 200),
        ("colors_text_hdr_rec2020.avif", 200, 200),
        ("colors_text_hdr_srgb.avif", 200, 200),
        ("colors_text_sdr_srgb.avif", 200, 200),
        ("colors_text_wcg_hdr_rec2020.avif", 200, 200),
        ("colors_text_wcg_sdr_rec2020.avif", 200, 200),
        ("colors_wcg_hdr_rec2020.avif", 200, 200),
        ("draw_points_idat.avif", 33, 11),
        ("draw_points_idat_metasize0.avif", 33, 11),
        ("draw_points_idat_progressive.avif", 33, 11),
        ("extended_pixi.avif", 4, 4),
        ("fox.profile0.8bpc.yuv420.avif", 1204, 800),
        ("fox.profile0.8bpc.yuv420.monochrome.avif", 1204, 800),
        ("fox.profile1.10bpc.yuv444.avif", 1204, 800),
        ("fox.profile2.12bpc.yuv444.avif", 1204, 800),
        ("fox.profile2.8bpc.yuv422.avif", 1204, 800),
        ("kimono.crop.avif", 385, 330),
        ("kimono.mirror-horizontal.avif", 722, 1024),
        ("kimono.mirror-vertical.rotate270.avif", 722, 1024),
        ("kimono.rotate270.avif", 722, 1024),
        ("kimono.rotate90.avif", 722, 1024),
        (
            "plum-blossom-small.profile1.8bpc.yuv444.alpha-full.avif",
            128,
            128,
        ),
        ("paris_icc_exif_xmp.avif", 403, 302),
        ("red-at-12-oclock-with-color-profile-8bpc.avif", 800, 800),
        ("sofa_grid1x5_420.avif", 1024, 770),
        ("sofa_grid1x5_420_dimg_repeat.avif", 1024, 770),
        ("sofa_grid1x5_420_reversed_dimg_order.avif", 1024, 770),
        ("star-8bpc.avifs", 159, 159),
    ];
    assert_eq!(cases.len(), 35);
    for (name, width, height) in cases {
        let path = root.join(name);
        assert!(
            path.is_file(),
            "official unsupported sample is missing: {name}"
        );
        let data = std::fs::read(&path).expect("official unsupported sample should be readable");
        let image = avif_rust::image_from_bytes(&data)
            .unwrap_or_else(|err| panic!("{name} should produce complete RGBA output: {err}"));
        assert_eq!(
            (image.width, image.height),
            (width, height),
            "{name} RGBA dimensions"
        );
        assert_eq!(
            image.rgba.len(),
            width * height * 4,
            "{name} partial RGBA output"
        );
    }
}

#[test]
fn every_external_unsupported_sample_decodes_completely() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root should exist")
        .join("test/images/external/avif/unsupported");
    let mut paths = Vec::new();
    collect_external_unsupported_samples(&root, &mut paths);
    if paths.is_empty() {
        eprintln!("external unsupported sample directory is unavailable; skipping dynamic audit");
        return;
    }
    paths.sort();
    for path in paths {
        let data = std::fs::read(&path).expect("external unsupported sample should be readable");
        if path.file_name().and_then(|name| name.to_str()) == Some("poc_b_506387278.avif") {
            assert!(matches!(
                avif_rust::image_from_bytes(&data),
                Err(avif_rust::DecoderError::Bitstream(message))
                    if message.contains("nclx range does not match")
            ));
            continue;
        }
        let image = avif_rust::image_from_bytes(&data)
            .unwrap_or_else(|error| panic!("{} should decode completely: {error}", path.display()));
        assert!(
            image.width > 0 && image.height > 0,
            "{} has empty dimensions",
            path.display()
        );
        assert_eq!(
            image.rgba.len(),
            image.width * image.height * 4,
            "{} has partial RGBA output",
            path.display()
        );
    }
}

fn collect_external_unsupported_samples(root: &Path, output: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries {
        let entry = entry.expect("external unsupported sample directory should be readable");
        let path = entry.path();
        if path.is_dir() {
            collect_external_unsupported_samples(&path, output);
            continue;
        }
        let is_avif = path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| {
                matches!(extension.to_ascii_lowercase().as_str(), "avif" | "avifs")
            });
        if is_avif {
            output.push(path);
        }
    }
}

#[test]
fn external_sato_12bit_to_16bit_sample_decodes() {
    let path = std::env::var_os("AVIF_SATO_SAMPLE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("workspace root should exist")
                .join("test/images/external/avif/unsupported/weld_sato_12B_8B_q0.avif")
        });
    if !path.is_file() {
        eprintln!("external sato sample is unavailable; skipping Sample Transform oracle");
        return;
    }
    let data = std::fs::read(&path).expect("sato sample should be readable");
    let frame = avif_rust::decode_frame_bytes(&data).expect("sato sample should decode");
    assert_eq!((frame.width, frame.height), (1024, 684));
    assert_eq!(frame.bit_depth, 16);
    let image = avif_rust::image_from_bytes(&data).expect("sato RGBA conversion should decode");
    assert_eq!((image.width, image.height), (1024, 684));
    assert_eq!(image.rgba.len(), 1024 * 684 * 4);
}

#[test]
fn gain_map_frame_api_is_absent_without_tmap() {
    let path = sample_path("WML2Viewer.avif");
    if !path.is_file() {
        eprintln!("WML2Viewer sample is unavailable; skipping gain-map API smoke test");
        return;
    }
    let data = std::fs::read(path).expect("WML2Viewer sample should be readable");
    assert!(
        avif_rust::decode_gain_map_frame_bytes(&data)
            .expect("ordinary AVIF should parse without a gain-map item")
            .is_none()
    );
}

#[test]
fn official_hdr_and_sample_transform_samples_keep_declared_native_range() {
    let root = std::env::var_os("AVIF_HDR_SAMPLE_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("workspace root should exist")
                .join("test/images/external/avif/unsupported")
        });
    let cases = [
        (
            "arc_triomphe_extent1000_nullbyte_extent1310.avif",
            64,
            64,
            8,
        ),
        ("colors_hdr_p3.avif", 200, 200, 10),
        ("colors_hdr_rec2020.avif", 200, 200, 10),
        ("colors_hdr_srgb.avif", 200, 200, 10),
        ("colors_sdr_srgb.avif", 200, 200, 8),
        // The 12B_8B sample-transform input is normalized to the item's
        // declared 16-bit output depth by the ISO transform metadata.
        ("weld_sato_12B_8B_q0.avif", 1024, 684, 16),
    ];
    for (name, width, height, bit_depth) in cases {
        let path = root.join(name);
        if !path.is_file() {
            eprintln!("official HDR/sample-transform sample is unavailable; skipping {name}");
            continue;
        }
        let data = std::fs::read(&path).expect("official HDR sample should be readable");
        let frame = avif_rust::decode_frame_bytes(&data)
            .unwrap_or_else(|error| panic!("{name} should decode completely: {error}"));
        assert_eq!((frame.width, frame.height), (width, height), "{name}");
        assert_eq!(frame.bit_depth, bit_depth, "{name} bit depth");
        let max_sample = (1u32 << bit_depth) - 1;
        assert!(
            frame
                .buffers
                .planes
                .iter()
                .flat_map(|plane| plane.samples.iter())
                .all(|&sample| u32::from(sample) <= max_sample),
            "{name} contains a sample outside its declared range"
        );
        let rgba = frame
            .to_rgba16()
            .unwrap_or_else(|error| panic!("{name} RGBA16 conversion should succeed: {error}"));
        assert_eq!(rgba.rgba.len(), width * height * 4, "{name} RGBA16 length");
    }
}

#[test]
fn external_gainmap_samples_keep_complete_base_decode() {
    let root = std::env::var_os("AVIF_GAINMAP_SAMPLE_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("workspace root should exist")
                .join("test/images/external/avif/gainmap")
        });
    let cases = [
        ("seine_hdr_srgb.avif", 400, 300),
        ("seine_hdr_rec2020.avif", 400, 300),
        ("seine_hdr_gainmap_srgb.avif", 400, 300),
        ("seine_hdr_gainmap_small_srgb.avif", 400, 300),
        ("seine_sdr_gainmap_big_srgb.avif", 400, 300),
        ("seine_sdr_gainmap_srgb.avif", 400, 300),
        ("seine_sdr_gainmap_notmapbrand.avif", 400, 300),
        ("seine_sdr_gainmap_gammazero.avif", 400, 300),
        ("seine_hdr_gainmap_wrongaltr.avif", 400, 300),
        ("unsupported_gainmap_version.avif", 100, 100),
        ("unsupported_gainmap_minimum_version.avif", 100, 100),
        (
            "unsupported_gainmap_writer_version_with_extra_bytes.avif",
            100,
            100,
        ),
        (
            "supported_gainmap_writer_version_with_extra_bytes.avif",
            100,
            100,
        ),
    ];
    if !cases.iter().any(|(name, _, _)| root.join(name).is_file()) {
        eprintln!("external gainmap samples are unavailable; skipping base decode audit");
        return;
    }
    for (name, width, height) in cases {
        let path = root.join(name);
        if !path.is_file() {
            eprintln!("external gainmap sample is unavailable; skipping {name}");
            continue;
        }
        let data = std::fs::read(&path).expect("gainmap sample should be readable");
        let image = avif_rust::image_from_bytes(&data)
            .unwrap_or_else(|error| panic!("{name} should keep a complete base decode: {error}"));
        assert_eq!(
            (image.width, image.height),
            (width, height),
            "{name} dimensions"
        );
        assert_eq!(image.rgba.len(), width * height * 4, "{name} complete RGBA");

        if name == "seine_hdr_srgb.avif" || name == "seine_hdr_rec2020.avif" {
            let frame = avif_rust::decode_frame_bytes(&data)
                .unwrap_or_else(|error| panic!("{name} base frame should decode: {error}"));
            assert_eq!(frame.bit_depth, 10, "{name} native HDR bit depth");
        }

        let gain_map_metadata = avif_rust::parse_gain_map_metadata(&data);
        if (name.starts_with("seine_")
            && name != "seine_hdr_srgb.avif"
            && name != "seine_hdr_rec2020.avif"
            && name != "seine_sdr_gainmap_gammazero.avif"
            && name != "seine_hdr_gainmap_wrongaltr.avif")
            || name == "unsupported_gainmap_writer_version_with_extra_bytes.avif"
        {
            assert!(
                gain_map_metadata
                    .expect("supported gain-map metadata should parse")
                    .is_some(),
                "{name} should expose a tmap descriptor"
            );
            let gain_map = avif_rust::decode_gain_map_frame_bytes(&data)
                .unwrap_or_else(|error| panic!("{name} gain-map item should decode: {error}"))
                .expect("supported gain-map sample should expose its AV1 map item");
            assert!(gain_map.frame.width > 0 && gain_map.frame.height > 0);
            assert!(matches!(gain_map.metadata.channel_count(), 1 | 3));
            let base_frame = avif_rust::decode_frame_bytes(&data)
                .unwrap_or_else(|error| panic!("{name} base frame should decode: {error}"));
            let composed = base_frame
                .to_rgba16_with_gain_map(&gain_map, 1.0)
                .unwrap_or_else(|error| {
                    panic!("{name} gain-map composition should decode: {error}")
                });
            assert_eq!((composed.width, composed.height), (width, height));
            assert_eq!(composed.rgba.len(), width * height * 4);
        } else if name == "seine_sdr_gainmap_gammazero.avif" {
            assert!(
                matches!(
                    gain_map_metadata,
                    Err(avif_rust::DecoderError::Bitstream(_))
                ),
                "{name} should reject its zero gamma"
            );
        } else if name == "seine_hdr_gainmap_wrongaltr.avif" {
            assert!(
                gain_map_metadata
                    .expect("wrong altr metadata should parse without selecting a map")
                    .is_none(),
                "{name} should ignore a non-preferred gain map"
            );
            assert!(
                avif_rust::decode_gain_map_frame_bytes(&data)
                    .expect("wrong altr gain-map API should stay available")
                    .is_none(),
                "{name} should not expose a non-preferred gain map"
            );
        } else if name == "unsupported_gainmap_version.avif"
            || name == "unsupported_gainmap_minimum_version.avif"
        {
            assert!(
                matches!(
                    gain_map_metadata,
                    Err(avif_rust::DecoderError::Unsupported(_))
                ),
                "{name} should fail closed on its unsupported descriptor version"
            );
        } else if name == "supported_gainmap_writer_version_with_extra_bytes.avif" {
            assert!(
                matches!(
                    gain_map_metadata,
                    Err(avif_rust::DecoderError::Bitstream(_))
                ),
                "{name} should reject trailing bytes for a supported writer version"
            );
        }
    }
}

#[test]
fn external_gainmap_icc_association_decodes_when_present() {
    let root = std::env::var_os("AVIF_GAINMAP_SAMPLE_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("workspace root should exist")
                .join("test/images/external/avif/gainmap")
        });
    let path = root.join("seine_sdr_gainmap_srgb_icc.avif");
    if !path.is_file() {
        eprintln!("gain-map ICC sample is unavailable; skipping ICC association audit");
        return;
    }
    let data = std::fs::read(path).expect("gain-map ICC sample should be readable");
    let image = avif_rust::image_from_bytes(&data)
        .expect("gain-map ICC sample should produce complete RGBA output");
    assert_eq!((image.width, image.height), (400, 300));
    let metadata = avif_rust::parse_gain_map_metadata(&data)
        .expect("gain-map ICC metadata should parse")
        .expect("gain-map ICC metadata should be present");
    let gain_map = avif_rust::decode_gain_map_frame_bytes(&data)
        .expect("gain-map ICC item should decode")
        .expect("gain-map ICC item should be exposed");
    let base = avif_rust::decode_frame_bytes(&data).expect("gain-map ICC base should decode");
    let composed = base
        .to_rgba16_with_gain_map(&gain_map, 1.0)
        .expect("gain-map ICC composition should decode");
    assert_eq!((composed.width, composed.height), (400, 300));
    assert_eq!(metadata.channel_count(), 3);
}

#[test]
fn external_paris_icc_and_nclx_sample_matches_ffmpeg_when_present() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root should exist")
        .join("test/images/external/avif/unsupported/paris_icc_exif_xmp.avif");
    if !path.is_file() {
        eprintln!("Paris ICC+nclx sample is unavailable; skipping colour oracle");
        return;
    }
    let data = std::fs::read(&path).expect("Paris ICC+nclx sample should be readable");
    let actual = avif_rust::image_from_bytes(&data)
        .expect("Paris ICC+nclx sample should produce complete RGBA output");
    let Some(expected) = ffmpeg_decode_rgba_dynamic(&path, 403, 302) else {
        return;
    };
    let metrics = diff_rgb_dynamic(&actual.rgba, &expected);
    eprintln!(
        "Paris ICC+nclx sample: average RGB absolute error={}, max={}",
        metrics.average_rgb_abs, metrics.max_rgb_abs
    );
    assert!(
        metrics.average_rgb_abs <= 2.0 && metrics.max_rgb_abs <= 48,
        "Paris ICC+nclx sample RGB error average={} max={}",
        metrics.average_rgb_abs,
        metrics.max_rgb_abs
    );
}

#[test]
fn external_gainmap_grid_samples_compose_when_present() {
    let root = std::env::var_os("AVIF_GAINMAP_SAMPLE_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("workspace root should exist")
                .join("test/images/external/avif/gainmap")
        });
    let names = [
        "color_nogrid_alpha_nogrid_gainmap_grid.avif",
        "color_grid_alpha_grid_gainmap_nogrid.avif",
        "color_grid_gainmap_different_grid.avif",
    ];
    let mut found = false;
    for name in names {
        let path = root.join(name);
        if !path.is_file() {
            continue;
        }
        found = true;
        let data = std::fs::read(&path).expect("gain-map grid sample should be readable");
        let base = avif_rust::decode_frame_bytes(&data)
            .unwrap_or_else(|error| panic!("{name} base frame should decode: {error}"));
        let gain_map = avif_rust::decode_gain_map_frame_bytes(&data)
            .unwrap_or_else(|error| panic!("{name} gain-map grid should decode: {error}"))
            .expect("gain-map grid sample should expose a tmap item");
        let composed = base
            .to_rgba16_with_gain_map(&gain_map, 1.0)
            .unwrap_or_else(|error| panic!("{name} gain-map grid should compose: {error}"));
        assert_eq!((composed.width, composed.height), (base.width, base.height));
    }
    if !found {
        eprintln!("external gain-map grid samples are unavailable; skipping grid audit");
    }
}

#[test]
fn external_official_grid_variants_decode_when_present() {
    let root = std::env::var_os("AVIF_OFFICIAL_GRID_SAMPLE_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("workspace root should exist")
                .join("test/images/external/avif/unsupported")
        });
    let names = [
        "color_grid_alpha_nogrid.avif",
        "color_grid_alpha_grid_tile_shared_in_dimg.avif",
    ];
    let mut found = false;
    for name in names {
        let path = root.join(name);
        if !path.is_file() {
            continue;
        }
        found = true;
        let data = std::fs::read(&path).expect("official grid sample should be readable");
        let image = avif_rust::image_from_bytes(&data)
            .unwrap_or_else(|error| panic!("{name} should decode completely: {error}"));
        assert!(image.width > 0 && image.height > 0);
        assert_eq!(image.rgba.len(), image.width * image.height * 4);
        if let Some(expected) = ffmpeg_decode_rgba_dynamic(&path, image.width, image.height) {
            let metrics = diff_rgb_dynamic(&image.rgba, &expected);
            assert!(
                metrics.average_rgb_abs <= 2.0 && metrics.max_rgb_abs <= 48,
                "{name} FFmpeg RGBA error average={} max={}",
                metrics.average_rgb_abs,
                metrics.max_rgb_abs
            );
        }
        let frame = avif_rust::decode_frame_bytes(&data)
            .unwrap_or_else(|error| panic!("{name} native planes should decode: {error}"));
        assert_eq!((frame.width, frame.height), (image.width, image.height));
    }
    if !found {
        eprintln!("official grid variants are unavailable; skipping grid variant audit");
    }
}
