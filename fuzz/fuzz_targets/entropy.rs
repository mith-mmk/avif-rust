#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(mut decoder) = avif_rust::av1::EntropyDecoder::new(data, false) {
        let _ = decoder.read_bool();
        let _ = decoder.read_literal(8);
        let mut cdf = [1 << 14, 1 << 15, 0];
        let _ = decoder.read_symbol(&mut cdf);
        let _ = decoder.exit();
    }
});
