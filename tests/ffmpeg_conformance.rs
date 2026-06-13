use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

const SAMPLE_WIDTH: usize = 900;
const SAMPLE_HEIGHT: usize = 900;
const SAMPLE_PIXELS: usize = SAMPLE_WIDTH * SAMPLE_HEIGHT;
const SAMPLE_RGBA_LEN: usize = SAMPLE_PIXELS * 4;

#[derive(Debug)]
struct DiffMetrics {
    average_rgb_abs: f64,
    max_rgb_abs: u8,
}

fn sample_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root should exist")
        .join("samples")
        .join(name)
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
#[ignore = "enable once the pure Rust AV1 image decoder returns pixels"]
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
