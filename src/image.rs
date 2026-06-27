/// RGBA image returned by still-image decode helpers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageBuffer {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
}

/// 16-bit RGBA image returned by high-precision decode helpers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rgba16ImageBuffer {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u16>,
}
