//! Caller-owned bounds for the additive native decode API.

use crate::DecoderError;

/// Explicit resource bounds used before native AVIF parsing and allocation.
///
/// Fields stay private so the limit vocabulary can grow without making
/// struct literals a compatibility hazard. There is no unlimited default.
/// The current bounded decoder applies these limits to the input/container
/// metadata, ICC payloads, item/property counts, and the sequence/frame
/// geometry needed to validate a normal still `av01` item and its selected
/// auxiliary alpha before tile materialization. These limits are not yet a
/// total-live-allocation contract for deep tile/entropy, reference-frame, or
/// post-filter scratch; those C2/C3 guarantees remain future work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeDecodeLimits {
    max_input_bytes: usize,
    max_width: usize,
    max_height: usize,
    max_pixels: usize,
    max_plane_bytes: usize,
    max_metadata_bytes: usize,
    max_icc_bytes: usize,
    max_items: usize,
    max_properties: usize,
    max_grid_cells: usize,
    max_derived_depth: usize,
    max_frames: usize,
}

impl NativeDecodeLimits {
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        max_input_bytes: usize,
        max_width: usize,
        max_height: usize,
        max_pixels: usize,
        max_plane_bytes: usize,
        max_metadata_bytes: usize,
        max_icc_bytes: usize,
        max_items: usize,
        max_properties: usize,
        max_grid_cells: usize,
        max_derived_depth: usize,
        max_frames: usize,
    ) -> Self {
        Self {
            max_input_bytes,
            max_width,
            max_height,
            max_pixels,
            max_plane_bytes,
            max_metadata_bytes,
            max_icc_bytes,
            max_items,
            max_properties,
            max_grid_cells,
            max_derived_depth,
            max_frames,
        }
    }

    pub const fn max_input_bytes(self) -> usize {
        self.max_input_bytes
    }
    pub const fn max_width(self) -> usize {
        self.max_width
    }
    pub const fn max_height(self) -> usize {
        self.max_height
    }
    pub const fn max_pixels(self) -> usize {
        self.max_pixels
    }
    pub const fn max_plane_bytes(self) -> usize {
        self.max_plane_bytes
    }
    pub const fn max_metadata_bytes(self) -> usize {
        self.max_metadata_bytes
    }
    pub const fn max_icc_bytes(self) -> usize {
        self.max_icc_bytes
    }
    pub const fn max_items(self) -> usize {
        self.max_items
    }
    pub const fn max_properties(self) -> usize {
        self.max_properties
    }
    pub const fn max_grid_cells(self) -> usize {
        self.max_grid_cells
    }
    pub const fn max_derived_depth(self) -> usize {
        self.max_derived_depth
    }
    pub const fn max_frames(self) -> usize {
        self.max_frames
    }

    pub(crate) fn check_input_len(self, length: usize) -> Result<(), DecoderError> {
        if length > self.max_input_bytes {
            return Err(DecoderError::InvalidParam(format!(
                "native AVIF input size {length} exceeds limit {}",
                self.max_input_bytes
            )));
        }
        Ok(())
    }

    pub(crate) fn check_count(
        self,
        count: usize,
        limit: usize,
        label: &str,
    ) -> Result<(), DecoderError> {
        if count > limit {
            return Err(DecoderError::InvalidParam(format!(
                "native AVIF {label} count {count} exceeds limit {limit}"
            )));
        }
        Ok(())
    }

    pub(crate) fn check_dimensions(self, width: usize, height: usize) -> Result<(), DecoderError> {
        if width > self.max_width || height > self.max_height {
            return Err(DecoderError::InvalidParam(format!(
                "native AVIF dimensions {width}x{height} exceed limits {}x{}",
                self.max_width, self.max_height
            )));
        }
        let pixels = width.checked_mul(height).ok_or_else(|| {
            DecoderError::InvalidParam("native AVIF pixel count overflows".to_string())
        })?;
        if pixels > self.max_pixels {
            return Err(DecoderError::InvalidParam(format!(
                "native AVIF pixel count {pixels} exceeds limit {}",
                self.max_pixels
            )));
        }
        Ok(())
    }

    pub(crate) fn check_plane_bytes(self, bytes: usize) -> Result<(), DecoderError> {
        if bytes > self.max_plane_bytes {
            return Err(DecoderError::InvalidParam(format!(
                "native AVIF plane allocation {bytes} exceeds limit {}",
                self.max_plane_bytes
            )));
        }
        Ok(())
    }

    pub(crate) fn check_metadata(
        self,
        metadata_bytes: usize,
        icc_bytes: usize,
    ) -> Result<(), DecoderError> {
        if metadata_bytes > self.max_metadata_bytes {
            return Err(DecoderError::InvalidParam(format!(
                "native AVIF metadata size {metadata_bytes} exceeds limit {}",
                self.max_metadata_bytes
            )));
        }
        if icc_bytes > self.max_icc_bytes {
            return Err(DecoderError::InvalidParam(format!(
                "native AVIF ICC size {icc_bytes} exceeds limit {}",
                self.max_icc_bytes
            )));
        }
        Ok(())
    }
}
