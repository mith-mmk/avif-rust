use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn median(values: &mut [Duration]) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}

fn main() {
    let sample = std::env::var_os("AVIF_BENCH_SAMPLE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("samples")
                .join("WML2Viewer.avif")
        });
    let iterations = std::env::var("AVIF_BENCH_ITERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value >= 3)
        .unwrap_or(10);
    let data = std::fs::read(&sample)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", sample.display()));
    for _ in 0..2 {
        black_box(avif_rust::decode_frame_bytes(&data).unwrap());
        black_box(avif_rust::image_from_bytes(&data).unwrap());
    }
    let mut decode_times = Vec::with_capacity(iterations);
    let mut rgba_times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        black_box(avif_rust::decode_frame_bytes(&data).unwrap());
        decode_times.push(start.elapsed());

        let start = Instant::now();
        black_box(avif_rust::image_from_bytes(&data).unwrap());
        rgba_times.push(start.elapsed());
    }
    let decode_median = median(&mut decode_times);
    let rgba_median = median(&mut rgba_times);
    println!(
        "sample={} iterations={} decode_frame_median_ms={} image_rgba_median_ms={}",
        sample.display(),
        iterations,
        decode_median.as_secs_f64() * 1000.0,
        rgba_median.as_secs_f64() * 1000.0
    );
}
