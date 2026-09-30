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
    max_live_allocation_bytes: Option<usize>,
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
            max_live_allocation_bytes: None,
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

    /// Intersects two native limit sets without relaxing either operand.
    ///
    /// This is used by format bridges that have an outer resource budget. A
    /// missing live ceiling is treated as unbounded for the intersection;
    /// once either side supplies a ceiling it is retained, and two ceilings
    /// are reduced to their minimum.
    pub fn intersect(self, other: Self) -> Self {
        Self {
            max_input_bytes: self.max_input_bytes.min(other.max_input_bytes),
            max_width: self.max_width.min(other.max_width),
            max_height: self.max_height.min(other.max_height),
            max_pixels: self.max_pixels.min(other.max_pixels),
            max_plane_bytes: self.max_plane_bytes.min(other.max_plane_bytes),
            max_metadata_bytes: self.max_metadata_bytes.min(other.max_metadata_bytes),
            max_icc_bytes: self.max_icc_bytes.min(other.max_icc_bytes),
            max_items: self.max_items.min(other.max_items),
            max_properties: self.max_properties.min(other.max_properties),
            max_grid_cells: self.max_grid_cells.min(other.max_grid_cells),
            max_derived_depth: self.max_derived_depth.min(other.max_derived_depth),
            max_frames: self.max_frames.min(other.max_frames),
            max_live_allocation_bytes: match (
                self.max_live_allocation_bytes,
                other.max_live_allocation_bytes,
            ) {
                (Some(left), Some(right)) => Some(left.min(right)),
                (Some(value), None) | (None, Some(value)) => Some(value),
                (None, None) => None,
            },
        }
    }

    /// Adds an optional aggregate ceiling for the native parser's retained
    /// metadata and selected item payload owners.
    ///
    /// The existing twelve-argument constructor leaves this unset.  A zero
    /// ceiling is rejected because it cannot represent even a non-empty
    /// retained owner; use the existing class limits for an empty input.
    pub fn with_max_live_allocation_bytes(
        self,
        max_live_bytes: usize,
    ) -> Result<Self, DecoderError> {
        self.tighten_max_live_allocation_bytes(max_live_bytes)
    }

    /// Tightens the optional aggregate native allocation ceiling.
    ///
    /// This is intentionally monotonic: a caller-provided ceiling can only
    /// become smaller when a containing API applies its own budget.  Keeping
    /// the operation on the native limit value lets bridges derive one
    /// effective limit before constructing the decoder, rather than allowing
    /// native retained owners to be created under a looser temporary limit.
    pub fn tighten_max_live_allocation_bytes(
        mut self,
        max_live_bytes: usize,
    ) -> Result<Self, DecoderError> {
        if max_live_bytes == 0 {
            return Err(DecoderError::InvalidParam(
                "native AVIF live allocation limit must be positive".to_string(),
            ));
        }
        self.max_live_allocation_bytes = Some(
            self.max_live_allocation_bytes
                .map_or(max_live_bytes, |current| current.min(max_live_bytes)),
        );
        Ok(self)
    }

    pub(crate) const fn max_live_allocation_bytes(self) -> Option<usize> {
        self.max_live_allocation_bytes
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

#[cfg(test)]
mod tests {
    use super::NativeDecodeLimits;

    fn limits() -> NativeDecodeLimits {
        NativeDecodeLimits::new(4096, 16, 16, 256, 4096, 4096, 4096, 8, 8, 8, 8, 8)
    }

    #[test]
    fn live_allocation_limit_tightening_is_monotonic() {
        let narrowed = limits()
            .with_max_live_allocation_bytes(128)
            .unwrap()
            .tighten_max_live_allocation_bytes(64)
            .unwrap();
        assert_eq!(narrowed.max_live_allocation_bytes(), Some(64));

        let widened = limits()
            .tighten_max_live_allocation_bytes(64)
            .unwrap()
            .tighten_max_live_allocation_bytes(128)
            .unwrap();
        assert_eq!(widened.max_live_allocation_bytes(), Some(64));
        assert!(limits().tighten_max_live_allocation_bytes(0).is_err());
    }

    #[test]
    fn native_limit_intersection_never_relaxes_or_drops_live_ceiling() {
        let native = limits().tighten_max_live_allocation_bytes(64).unwrap();
        let outer = limits().tighten_max_live_allocation_bytes(1024).unwrap();
        assert_eq!(
            native.intersect(outer).max_live_allocation_bytes(),
            Some(64)
        );
        assert_eq!(
            outer.intersect(native).max_live_allocation_bytes(),
            Some(64)
        );

        let unbounded = limits();
        assert_eq!(
            unbounded.intersect(outer).max_live_allocation_bytes(),
            Some(1024)
        );
    }

    #[test]
    fn intersection_minimizes_each_native_scalar_independently() {
        let left = NativeDecodeLimits::new(
            101, 202, 303, 404, 505, 606, 707, 808, 909, 1_010, 1_111, 1_212,
        );
        let right = NativeDecodeLimits::new(
            201, 102, 403, 204, 605, 306, 807, 408, 1_009, 510, 1_311, 612,
        );
        let result = left.intersect(right);
        assert_eq!(result.max_input_bytes(), 101);
        assert_eq!(result.max_width(), 102);
        assert_eq!(result.max_height(), 303);
        assert_eq!(result.max_pixels(), 204);
        assert_eq!(result.max_plane_bytes(), 505);
        assert_eq!(result.max_metadata_bytes(), 306);
        assert_eq!(result.max_icc_bytes(), 707);
        assert_eq!(result.max_items(), 408);
        assert_eq!(result.max_properties(), 909);
        assert_eq!(result.max_grid_cells(), 510);
        assert_eq!(result.max_derived_depth(), 1_111);
        assert_eq!(result.max_frames(), 612);
        assert_eq!(left.intersect(right), right.intersect(left));
    }

    #[test]
    fn intersection_preserves_all_live_option_combinations_in_both_orders() {
        let base = limits();
        let finite_small = base.with_max_live_allocation_bytes(64).unwrap();
        let finite_large = base.with_max_live_allocation_bytes(128).unwrap();
        let cases = [
            (base, base, None),
            (finite_small, base, Some(64)),
            (base, finite_large, Some(128)),
            (finite_small, finite_large, Some(64)),
            (finite_large, finite_small, Some(64)),
        ];
        for (left, right, expected) in cases {
            assert_eq!(left.intersect(right).max_live_allocation_bytes(), expected);
            assert_eq!(right.intersect(left).max_live_allocation_bytes(), expected);
        }
    }
}
