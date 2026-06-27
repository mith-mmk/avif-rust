#![allow(dead_code)]

use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub struct ExpectedPlane<'a> {
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    pub samples: &'a [u16],
}

pub fn assert_exact_decoded_planes(
    decoded: &avif_rust::DecodedFrame,
    expected: &[ExpectedPlane<'_>],
) {
    assert_eq!(decoded.buffers.planes.len(), expected.len());
    for (plane_index, (actual, expected)) in decoded
        .buffers
        .planes
        .iter()
        .zip(expected.iter())
        .enumerate()
    {
        assert_eq!(
            actual.layout.width, expected.width,
            "plane {plane_index} width"
        );
        assert_eq!(
            actual.layout.height, expected.height,
            "plane {plane_index} height"
        );
        assert_eq!(
            actual.layout.stride(),
            expected.stride,
            "plane {plane_index} stride"
        );
        assert_eq!(
            actual.samples, expected.samples,
            "plane {plane_index} samples"
        );
    }
}

pub fn assert_rgba8_max_error(actual: &[u8], expected: &[u8], max_allowed: u8, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} RGBA8 byte length");
    assert_eq!(actual.len() % 4, 0, "{label} RGBA8 alignment");
    let max = actual
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| actual.abs_diff(*expected))
        .max()
        .unwrap_or(0);
    assert!(max <= max_allowed, "{label} RGBA8 max error was {max}");
}

pub fn assert_rgba16_max_error(actual: &[u16], expected: &[u16], max_allowed: u16, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} RGBA16 sample length");
    assert_eq!(actual.len() % 4, 0, "{label} RGBA16 alignment");
    let max = actual
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| actual.abs_diff(*expected))
        .max()
        .unwrap_or(0);
    assert!(max <= max_allowed, "{label} RGBA16 max error was {max}");
}

pub fn read_u16le_samples(path: &Path, sample_count: usize) -> Vec<u16> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
    assert_eq!(
        bytes.len(),
        sample_count * 2,
        "unexpected u16 sample byte length for {}",
        path.display()
    );
    bytes
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
        .collect()
}
