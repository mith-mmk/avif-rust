#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = avif_rust::obu::parse_obu_stream(data);
});
