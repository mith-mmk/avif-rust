use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn median(values: &mut [Duration]) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}

fn benchmark_sample(sample: &PathBuf, iterations: usize) {
    let data = std::fs::read(sample)
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

    let Ok(animation) = avif_rust::parse_avif_animation(&data) else {
        return;
    };
    if animation.sequence.color_samples.len() <= 1 {
        return;
    }
    for _ in 0..2 {
        black_box(decode_sequence(&data, false));
        black_box(decode_sequence(&data, true));
    }
    let mut sequence_times = Vec::with_capacity(iterations);
    let mut sequence_rgba_times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        black_box(decode_sequence(&data, false));
        sequence_times.push(start.elapsed());

        let start = Instant::now();
        black_box(decode_sequence(&data, true));
        sequence_rgba_times.push(start.elapsed());
    }
    let sequence_median = median(&mut sequence_times);
    let sequence_rgba_median = median(&mut sequence_rgba_times);
    println!(
        "sample={} frames={} iterations={} sequence_native_median_ms={} sequence_rgba_median_ms={}",
        sample.display(),
        animation.sequence.color_samples.len(),
        iterations,
        sequence_median.as_secs_f64() * 1000.0,
        sequence_rgba_median.as_secs_f64() * 1000.0
    );
}

fn decode_sequence(data: &[u8], convert_rgba: bool) -> usize {
    let mut decoder = avif_rust::AvifSequenceDecoder::new(data).unwrap();
    let mut frame_count = 0;
    while let Some(decoded) = decoder.next_frame().unwrap() {
        if convert_rgba {
            black_box(decoded.frame.to_rgba8().unwrap());
        } else {
            black_box(decoded.frame);
        }
        frame_count += 1;
    }
    frame_count
}

fn main() {
    let default_sample = || {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("samples")
            .join("WML2Viewer.avif")
    };
    let samples = std::env::var("AVIF_BENCH_SAMPLES")
        .ok()
        .map(|value| {
            value
                .split(';')
                .filter(|sample| !sample.trim().is_empty())
                .map(|sample| PathBuf::from(sample.trim()))
                .collect::<Vec<_>>()
        })
        .filter(|samples| !samples.is_empty())
        .unwrap_or_else(|| {
            vec![
                std::env::var_os("AVIF_BENCH_SAMPLE")
                    .map(PathBuf::from)
                    .unwrap_or_else(default_sample),
            ]
        });
    let iterations = std::env::var("AVIF_BENCH_ITERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value >= 3)
        .unwrap_or(10);
    for sample in samples {
        benchmark_sample(&sample, iterations);
    }
}
