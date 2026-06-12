/// RGBA image returned by still-image decode helpers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageBuffer {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
}
