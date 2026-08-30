use std::process::Command;

use avif_rust::{NativeDecodeLimits, StrictAvifSequenceDecoder};

fn limits() -> NativeDecodeLimits {
    NativeDecodeLimits::new(
        32 * 1024 * 1024,
        4096,
        4096,
        16 * 1024 * 1024,
        32 * 1024 * 1024,
        8 * 1024 * 1024,
        8 * 1024 * 1024,
        4096,
        4096,
        4096,
        64,
        64,
    )
}

fn generated_avis() -> Option<Vec<u8>> {
    let root = std::env::temp_dir().join(format!(
        ".test-avif-strict-sequence-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).expect("strict sequence temporary directory should exist");
    let output = root.join("sequence.avifs");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i", "color=c=red:size=64x64:rate=1"])
        .args(["-frames:v", "4", "-c:v", "libaom-av1"])
        .args([
            "-still-picture",
            "0",
            "-g",
            "1",
            "-lag-in-frames",
            "0",
            "-auto-alt-ref",
            "0",
            "-f",
            "avif",
        ])
        .arg(&output)
        .status();
    let Ok(status) = status else {
        let _ = std::fs::remove_dir_all(&root);
        return None;
    };
    if !status.success() {
        let _ = std::fs::remove_dir_all(&root);
        return None;
    }
    let data = std::fs::read(&output).expect("strict sequence output should be readable");
    let _ = std::fs::remove_dir_all(&root);
    Some(data)
}

#[test]
#[ignore = "requires the external ffmpeg/libaom AVIS fixture generator"]
fn strict_sequence_prepare_drop_retry_and_commit() {
    let data = generated_avis().expect("ffmpeg/libaom AVIS fixture generator is unavailable");
    let mut decoder = StrictAvifSequenceDecoder::new(&data, limits()).unwrap();
    assert_eq!(decoder.frame_count(), 4);
    assert!(decoder.timescale() > 0);
    assert!(decoder.duration_in_timescales() > 0);

    let dropped = decoder.prepare_next_frame().unwrap().unwrap();
    let first_timing = dropped.timing();
    assert_eq!(first_timing.pts_in_timescales, 0);
    assert!(dropped.additional_live_bytes() > 0);
    drop(dropped);

    let mut prepared = decoder.prepare_next_frame().unwrap().unwrap();
    let frame = prepared.take_frame().expect("prepared frame should be present");
    assert_eq!(frame.width, 64);
    assert_eq!(frame.height, 64);
    prepared.commit().unwrap();

    let second = decoder.prepare_next_frame().unwrap().unwrap();
    assert_eq!(second.timing().pts_in_timescales, first_timing.duration_in_timescales);
    second.commit().unwrap();
}
