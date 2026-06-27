#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    for split in 0..=data.len().min(32) {
        let (sequence_payload, frame_payload) = data.split_at(split);
        if let Ok(sequence) = avif_rust::av1::parse_sequence_header(sequence_payload) {
            let _ = avif_rust::av1::parse_frame_header(frame_payload, &sequence);
        }
    }
});
