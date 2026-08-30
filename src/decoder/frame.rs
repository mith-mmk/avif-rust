use super::sequence::{SequenceDecodeState, decode_hidden_key_frame_show_existing};
use super::*;
use crate::Rgba16ImageBuffer;
use crate::av1::{convert_linear_rgb_primaries, frame_buffers_to_rgba_16};
use crate::container::{
    AvifAnimation, AvifFrameTiming, AvifRepetitionCount, AvifSequence, parse_avif_animation,
    parse_avif_sequence, parse_gain_map_image,
};

#[cfg(test)]
#[path = "strict_sequence_tests.rs"]
mod strict_sequence_tests;

/// Decoded still-frame planes before colour conversion.
///
/// Samples are stored as native AV1 source planes in raster order. The first
/// three planes are Y/U/V (or the profile's native plane order); when an alpha
/// auxiliary or alpha grid is present, plane index three is the optional alpha
/// plane. The current decoder only supports a subset of still-image tools, but
/// this type is the conformance-test boundary for exact plane comparisons.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedFrame {
    pub width: usize,
    pub height: usize,
    pub render_width: usize,
    pub render_height: usize,
    pub bit_depth: u8,
    pub color_config: ColorConfig,
    pub color_information: Option<ColorInformation>,
    pub alpha_premultiplied: bool,
    pub buffers: FrameBuffers,
}

/// A decoded ISO 21496 gain-map image and its descriptor.
///
/// Gain-map pixels are returned as a normal native AV1 frame at the map's own
/// dimensions. Composition resamples them to the base frame dimensions when
/// needed; applications can still apply display-headroom policy themselves.
/// The default still-image API intentionally continues to return only the base
/// image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedGainMapFrame {
    pub metadata: crate::container::GainMapMetadata,
    pub frame: DecodedFrame,
}

/// One source-plane frame and the timing assigned to its color-track sample.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedSequenceFrame {
    pub frame: DecodedFrame,
    pub timing: AvifFrameTiming,
}

/// Incremental AVIS decoder.
///
/// The decoder exposes one color frame at a time and never retains decoded
/// RGBA output for later frames. Color and alpha tracks keep their AV1
/// reference, CDF, and motion state between calls, so a forward traversal
/// decodes every track sample exactly once.
pub struct AvifSequenceDecoder {
    animation: AvifAnimation,
    tracks: SequenceTracksDecoder,
    next_index: usize,
}

#[derive(Clone)]
struct SequenceTracksDecoder {
    info: AvifInfo,
    color_state: SequenceDecodeState,
    alpha_info: Option<AvifInfo>,
    alpha_state: Option<SequenceDecodeState>,
    static_alpha_frame: Option<DecodedFrame>,
    strict_alpha: bool,
}

/// The two heap classes owned by one strict sequence-track value.  A cloned
/// `AvifInfo` is metadata, while a cached static alpha frame is frame storage;
/// keeping separate tickets lets rollback release both owners in the same
/// order in which they were admitted.
struct SequenceTracksAllocation {
    metadata: AllocationTicket,
    frame: AllocationTicket,
    state: AllocationTicket,
}

/// Owns a candidate allocation while a prepared frame is being assembled.
///
/// All fallible work in `prepare_next_frame` runs while this guard is alive.
/// This makes metadata/track tickets transactional even when a capacity walk
/// or frame validation fails after the candidate has been decoded.
struct SequenceTracksAllocationGuard<'a> {
    budget: &'a mut crate::container::DecodeBudget,
    allocation: Option<SequenceTracksAllocation>,
}

impl<'a> SequenceTracksAllocationGuard<'a> {
    fn new(
        budget: &'a mut crate::container::DecodeBudget,
        allocation: SequenceTracksAllocation,
    ) -> Self {
        Self {
            budget,
            allocation: Some(allocation),
        }
    }

    fn reconcile_storage(
        &mut self,
        current: SequenceTracksStorage,
        target: SequenceTracksStorage,
    ) -> Result<(), DecoderError> {
        let allocation = self
            .allocation
            .as_mut()
            .expect("sequence allocation guard must own its allocation");
        SequenceTracksDecoder::reconcile_storage(allocation, current, target, self.budget)
    }

    fn reserve_state_delta(&mut self, bytes: usize) -> Result<(), DecoderError> {
        if bytes == 0 {
            return Ok(());
        }
        let allocation = self
            .allocation
            .as_mut()
            .expect("sequence allocation guard must own its allocation");
        let new_charge = allocation
            .state
            .charged_capacity_bytes
            .checked_add(bytes)
            .ok_or_else(|| {
                DecoderError::InvalidParam("AVIS sequence state ticket overflows".to_string())
            })?;
        let token = self.budget.reserve_existing_bytes(
            AllocationClass::Frame,
            bytes,
            "AVIS sequence state owners",
        )?;
        debug_assert_eq!(token.charged_capacity_bytes, bytes);
        allocation.state.charged_capacity_bytes = new_charge;
        Ok(())
    }

    fn reconcile_state_delta(&mut self, target: usize) -> Result<(), DecoderError> {
        let current = self
            .allocation
            .as_ref()
            .expect("sequence allocation guard must own its allocation")
            .state
            .charged_capacity_bytes;
        if current > target {
            self.budget
                .release_class_bytes(AllocationClass::Frame, current - target)?;
        } else if target > current {
            // A refresh plan is a pre-decode admission guarantee.  Never
            // repair an underestimated plan by reserving after decode: the
            // candidate allocations already exist and must be rolled back by
            // the guard instead.
            return Err(DecoderError::InvalidParam(
                "AVIS sequence refresh plan underestimated state owners".to_string(),
            ));
        }
        let allocation = self
            .allocation
            .as_mut()
            .expect("sequence allocation guard must own its allocation");
        allocation.state.charged_capacity_bytes = target;
        Ok(())
    }

    fn disarm(mut self) -> SequenceTracksAllocation {
        self.allocation
            .take()
            .expect("sequence allocation guard must own its allocation")
    }
}

impl Drop for SequenceTracksAllocationGuard<'_> {
    fn drop(&mut self) {
        if let Some(mut allocation) = self.allocation.take() {
            let _ = allocation.release(self.budget);
        }
    }
}

impl SequenceTracksAllocation {
    fn empty() -> Self {
        Self {
            metadata: AllocationTicket::new(AllocationClass::Metadata),
            frame: AllocationTicket::new(AllocationClass::Frame),
            state: AllocationTicket::new(AllocationClass::Frame),
        }
    }

    fn release(
        &mut self,
        budget: &mut crate::container::DecodeBudget,
    ) -> Result<(), DecoderError> {
        // Attempt both classes even if the first release reports an
        // accounting error.  This is used by the rollback guard, so one
        // broken ticket must not strand the other ticket in the budget.
        let state_result = budget.release_token(&mut self.state);
        let frame_result = budget.release_token(&mut self.frame);
        let metadata_result = budget.release_token(&mut self.metadata);
        match (state_result, frame_result, metadata_result) {
            (Err(error), _, _) | (_, Err(error), _) | (_, _, Err(error)) => Err(error),
            (Ok(()), Ok(()), Ok(())) => Ok(()),
        }
    }

    fn validate_release(
        &self,
        budget: &crate::container::DecodeBudget,
    ) -> Result<(), DecoderError> {
        for ticket in [&self.state, &self.frame, &self.metadata] {
            budget.validate_release(
                ticket.class,
                ticket.charged_capacity_bytes,
                "AVIS sequence allocation",
            )?;
        }
        Ok(())
    }

}

#[derive(Clone, Copy, Default)]
struct SequenceTracksStorage {
    metadata: usize,
    frame: usize,
    state: usize,
}

impl SequenceTracksStorage {
    fn total(self) -> Result<usize, DecoderError> {
        self.metadata
            .checked_add(self.frame)
            .and_then(|bytes| bytes.checked_add(self.state))
            .ok_or_else(|| DecoderError::InvalidParam("AVIS track clone size overflows".to_string()))
    }

}

fn checked_capacity_bytes<T>(capacity: usize, label: &str) -> Result<usize, DecoderError> {
    capacity
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| DecoderError::InvalidParam(format!("AVIS {label} size overflows")))
}

fn add_capacity<T>(
    total: &mut usize,
    capacity: usize,
    label: &str,
) -> Result<(), DecoderError> {
    *total = total
        .checked_add(checked_capacity_bytes::<T>(capacity, label)?)
        .ok_or_else(|| DecoderError::InvalidParam(format!("AVIS {label} size overflows")))?;
    Ok(())
}

fn color_information_storage_bytes(
    color: &Option<ColorInformation>,
) -> Result<usize, DecoderError> {
    color.as_ref().map_or(Ok(0), |color| {
        checked_capacity_bytes::<u8>(color.payload.capacity(), "colour metadata")
    })
}

fn pixel_information_storage_bytes(
    pixel: &Option<PixelInformation>,
) -> Result<usize, DecoderError> {
    let Some(pixel) = pixel else {
        return Ok(0);
    };
    let mut bytes = 0;
    add_capacity::<u8>(&mut bytes, pixel.bits_per_channel.capacity(), "pixi channels")?;
    if let Some(channels) = &pixel.extended_channels {
        add_capacity::<crate::container::PixelChannelInformation>(
            &mut bytes,
            channels.capacity(),
            "extended pixi channels",
        )?;
    }
    Ok(bytes)
}

fn grid_cell_storage_bytes(cell: &crate::container::GridCell) -> Result<usize, DecoderError> {
    let mut bytes = pixel_information_storage_bytes(&cell.pixel_information)?;
    bytes = bytes
        .checked_add(color_information_storage_bytes(&cell.color_information)?)
        .ok_or_else(|| DecoderError::InvalidParam("AVIS grid cell storage overflows".to_string()))?;
    if let Some(config) = &cell.av1_config {
        bytes = bytes
            .checked_add(checked_capacity_bytes::<u8>(config.capacity(), "grid AV1 config")?)
            .ok_or_else(|| DecoderError::InvalidParam("AVIS grid cell storage overflows".to_string()))?;
    }
    bytes = bytes
        .checked_add(checked_capacity_bytes::<u8>(cell.payload.capacity(), "grid cell payload")?)
        .ok_or_else(|| DecoderError::InvalidParam("AVIS grid cell storage overflows".to_string()))?;
    Ok(bytes)
}

fn grid_storage_bytes(grid: &Option<crate::container::GridImage>) -> Result<usize, DecoderError> {
    let Some(grid) = grid else {
        return Ok(0);
    };
    let mut bytes = checked_capacity_bytes::<u8>(grid.payload.capacity(), "grid payload")?;
    bytes = bytes
        .checked_add(checked_capacity_bytes::<crate::container::GridCell>(grid.cells.capacity(), "grid cells")?)
        .ok_or_else(|| DecoderError::InvalidParam("AVIS grid storage overflows".to_string()))?;
    for cell in &grid.cells {
        bytes = bytes
            .checked_add(grid_cell_storage_bytes(cell)?)
            .ok_or_else(|| DecoderError::InvalidParam("AVIS grid storage overflows".to_string()))?;
    }
    Ok(bytes)
}

/// Counts the allocations held by an `AvifInfo` after a strict sequence
/// clone.  This intentionally walks *actual capacities*, not logical lengths;
/// parser handoff vectors can retain spare capacity after their payloads are
/// cleared.
fn avif_info_storage_bytes(info: &AvifInfo) -> Result<usize, DecoderError> {
    let mut bytes = checked_capacity_bytes::<[u8; 4]>(
        info.compatible_brands.capacity(),
        "compatible brands",
    )?;
    bytes = bytes
        .checked_add(checked_capacity_bytes::<crate::container::AuxiliaryImage>(
            info.alpha_auxiliary_items.capacity(),
            "alpha items",
        )?)
        .ok_or_else(|| DecoderError::InvalidParam("AVIS info storage overflows".to_string()))?;
    for alpha in &info.alpha_auxiliary_items {
        bytes = bytes
            .checked_add(checked_capacity_bytes::<u8>(alpha.aux_type.capacity(), "alpha type")?)
            .ok_or_else(|| DecoderError::InvalidParam("AVIS info storage overflows".to_string()))?;
        bytes = bytes
            .checked_add(checked_capacity_bytes::<u8>(
                alpha.payload.capacity(),
                "alpha payload",
            )?)
            .ok_or_else(|| DecoderError::InvalidParam("AVIS info storage overflows".to_string()))?;
    }
    bytes = bytes
        .checked_add(pixel_information_storage_bytes(&info.pixel_information)?)
        .ok_or_else(|| DecoderError::InvalidParam("AVIS info storage overflows".to_string()))?;
    bytes = bytes
        .checked_add(color_information_storage_bytes(&info.color_information)?)
        .ok_or_else(|| DecoderError::InvalidParam("AVIS info storage overflows".to_string()))?;
    bytes = bytes
        .checked_add(grid_storage_bytes(&info.alpha_grid)?)
        .ok_or_else(|| DecoderError::InvalidParam("AVIS info storage overflows".to_string()))?;
    bytes = bytes
        .checked_add(grid_storage_bytes(&info.primary_grid)?)
        .ok_or_else(|| DecoderError::InvalidParam("AVIS info storage overflows".to_string()))?;
    if let Some(config) = &info.av1_config {
        bytes = bytes
            .checked_add(checked_capacity_bytes::<u8>(config.capacity(), "AV1 config")?)
            .ok_or_else(|| DecoderError::InvalidParam("AVIS info storage overflows".to_string()))?;
    }
    bytes = bytes
        .checked_add(checked_capacity_bytes::<u8>(
            info.primary_item_payload.capacity(),
            "primary payload",
        )?)
        .ok_or_else(|| DecoderError::InvalidParam("AVIS info storage overflows".to_string()))?;
    bytes = bytes
        .checked_add(checked_capacity_bytes::<Vec<u8>>(
            info.sequence_sample_payloads.capacity(),
            "sequence samples",
        )?)
        .ok_or_else(|| DecoderError::InvalidParam("AVIS info storage overflows".to_string()))?;
    for sample in &info.sequence_sample_payloads {
        bytes = bytes
            .checked_add(checked_capacity_bytes::<u8>(sample.capacity(), "sequence sample")?)
            .ok_or_else(|| DecoderError::InvalidParam("AVIS info storage overflows".to_string()))?;
    }
    Ok(bytes)
}

fn decoded_frame_storage_bytes(frame: &DecodedFrame) -> Result<usize, DecoderError> {
    frame_storage_bytes(frame)?.checked_add(color_information_storage_bytes(
        &frame.color_information,
    )?).ok_or_else(|| DecoderError::InvalidParam("AVIS frame metadata storage overflows".to_string()))
}

/// Returns the peak metadata-side storage needed while constructing strict
/// sequence tracks.  The alpha route temporarily has the source clone, the
/// second `AvifInfo` clone, and the first alpha payload clone alive at once;
/// account for all three before starting either clone.
fn sequence_tracks_clone_peak_bytes(
    info: &AvifInfo,
    alpha_samples: &[Vec<u8>],
) -> Result<usize, DecoderError> {
    let info_clone_bytes = avif_info_storage_bytes(info)?;
    let mut peak = info_clone_bytes;
    peak = peak
        .checked_add(
            SequenceDecodeState::sequence_prefix_upper_bound(info.primary_item_payload.len())?,
        )
        .ok_or_else(|| DecoderError::InvalidParam("AVIS track clone peak overflows".to_string()))?;
    let Some(alpha_sample) = alpha_samples.first() else {
        return Ok(peak);
    };
    let alpha_sample_bytes = checked_capacity_bytes::<u8>(
        alpha_sample.capacity(),
        "alpha sample temporary",
    )?;
    let alpha_prefix_bound = SequenceDecodeState::sequence_prefix_upper_bound(alpha_sample.len())?;
    peak
        .checked_add(info_clone_bytes)
        .and_then(|bytes| bytes.checked_add(alpha_prefix_bound))
        .and_then(|bytes| bytes.checked_add(alpha_sample_bytes))
        .ok_or_else(|| DecoderError::InvalidParam("AVIS track clone peak overflows".to_string()))
}

impl SequenceTracksDecoder {
    fn storage_bytes(&self) -> Result<SequenceTracksStorage, DecoderError> {
        let mut storage = SequenceTracksStorage {
            metadata: avif_info_storage_bytes(&self.info)?,
            frame: 0,
            state: 0,
        };
        if let Some(alpha_info) = &self.alpha_info {
            storage.metadata = storage
                .metadata
                .checked_add(avif_info_storage_bytes(alpha_info)?)
                .ok_or_else(|| DecoderError::InvalidParam("AVIS track metadata storage overflows".to_string()))?;
        }
        if let Some(frame) = &self.static_alpha_frame {
            storage.frame = decoded_frame_storage_bytes(frame)?;
        }
        storage.state = self.state_memory_bytes()?;
        Ok(storage)
    }

    fn storage_bytes_for_clone(&self) -> Result<SequenceTracksStorage, DecoderError> {
        // Use the currently retained capacities for the pre-copy check.  The
        // destination is measured again after construction because allocator
        // capacity is the value that the ticket must represent.
        let mut storage = self.storage_bytes()?;
        storage.state = 0;
        Ok(storage)
    }

    fn state_memory_bytes(&self) -> Result<usize, DecoderError> {
        SequenceDecodeState::memory_bytes_for_tracks(
            &self.color_state,
            self.alpha_state.as_ref(),
        )
    }

    fn additional_state_memory_bytes(
        &self,
        baseline: &Self,
    ) -> Result<usize, DecoderError> {
        SequenceDecodeState::additional_memory_bytes_for_tracks(
            &baseline.color_state,
            baseline.alpha_state.as_ref(),
            &self.color_state,
            self.alpha_state.as_ref(),
        )
    }

    fn state_refresh_plan(
        &self,
        color_sample: &[u8],
        alpha_sample: Option<&[u8]>,
        limits: &crate::limits::NativeDecodeLimits,
    ) -> Result<usize, DecoderError> {
        let mut bytes =
            SequenceDecodeState::refresh_plan_bytes(&self.info, color_sample, limits)?;
        if let Some(alpha_info) = &self.alpha_info {
            let alpha_sample = alpha_sample.ok_or_else(|| {
                DecoderError::Bitstream("AVIS alpha sample is missing for state plan".to_string())
            })?;
            bytes = bytes
                .checked_add(SequenceDecodeState::refresh_plan_bytes(
                    alpha_info,
                    alpha_sample,
                    limits,
                )?)
                .ok_or_else(|| {
                    DecoderError::InvalidParam("AVIS state refresh plan overflows".to_string())
                })?;
        }
        Ok(bytes)
    }

    fn strict_header_peak_plan(
        &self,
        color_sample: &[u8],
        alpha_sample: Option<&[u8]>,
        limits: &crate::limits::NativeDecodeLimits,
    ) -> Result<usize, DecoderError> {
        let mut bytes = SequenceDecodeState::strict_header_peak_bytes(
            &self.info,
            &self.color_state,
            color_sample,
            limits,
        )?;
        if let Some(alpha_info) = &self.alpha_info {
            let alpha_sample = alpha_sample.ok_or_else(|| {
                DecoderError::Bitstream("AVIS alpha sample is missing for header plan".to_string())
            })?;
            let alpha_state = self
                .alpha_state
                .as_ref()
                .expect("strict alpha info requires alpha state");
            bytes = bytes
                .checked_add(SequenceDecodeState::strict_header_peak_bytes(
                    alpha_info,
                    alpha_state,
                    alpha_sample,
                    limits,
                )?)
                .ok_or_else(|| {
                    DecoderError::InvalidParam("AVIS strict header peak overflows".to_string())
                })?;
        }
        Ok(bytes)
    }

    fn clone_with_budget(
        &self,
        budget: &mut crate::container::DecodeBudget,
    ) -> Result<(Self, SequenceTracksAllocation), DecoderError> {
        let estimate = self.storage_bytes_for_clone()?;
        budget.check_additional_frame(estimate.total()?)?;
        let candidate = self.clone();
        let storage = candidate.storage_bytes_for_clone()?;
        let allocation = Self::admit_storage(storage, budget)?;
        Ok((candidate, allocation))
    }

    fn admit_storage(
        storage: SequenceTracksStorage,
        budget: &mut crate::container::DecodeBudget,
    ) -> Result<SequenceTracksAllocation, DecoderError> {
        let mut allocation = SequenceTracksAllocation::empty();
        budget.check_additional_frame(storage.total()?)?;
        allocation.metadata = budget.reserve_existing_bytes(
            AllocationClass::Metadata,
            storage.metadata,
            "AVIS sequence metadata clone",
        )?;
        allocation.frame = match budget.reserve_existing_bytes(
            AllocationClass::Frame,
            storage.frame,
            "AVIS sequence static-alpha clone",
        ) {
            Ok(token) => token,
            Err(error) => {
                budget.release_token(&mut allocation.metadata)?;
                return Err(error);
            }
        };
        allocation.state = match budget.reserve_existing_bytes(
            AllocationClass::Frame,
            storage.state,
            "AVIS sequence state owners",
        ) {
            Ok(token) => token,
            Err(error) => {
                allocation.release(budget)?;
                return Err(error);
            }
        };
        Ok(allocation)
    }

    fn reconcile_storage(
        allocation: &mut SequenceTracksAllocation,
        current: SequenceTracksStorage,
        target: SequenceTracksStorage,
        budget: &mut crate::container::DecodeBudget,
    ) -> Result<(), DecoderError> {
        let metadata_current = allocation.metadata.charged_capacity_bytes;
        let frame_current = allocation.frame.charged_capacity_bytes;
        debug_assert_eq!(metadata_current, current.metadata);
        debug_assert_eq!(frame_current, current.frame);

        if target.metadata < metadata_current {
            budget.release_class_bytes(
                AllocationClass::Metadata,
                metadata_current - target.metadata,
            )?;
            allocation.metadata.charged_capacity_bytes = target.metadata;
        }
        if target.frame < frame_current {
            budget.release_class_bytes(AllocationClass::Frame, frame_current - target.frame)?;
            allocation.frame.charged_capacity_bytes = target.frame;
        }

        let metadata_delta = target.metadata.saturating_sub(
            allocation.metadata.charged_capacity_bytes,
        );
        let frame_delta = target
            .frame
            .saturating_sub(allocation.frame.charged_capacity_bytes);
        let total_delta = metadata_delta
            .checked_add(frame_delta)
            .ok_or_else(|| DecoderError::InvalidParam("AVIS track clone size overflows".to_string()))?;
        budget.check_additional_frame(total_delta)?;
        if metadata_delta != 0 {
            let new_charge = allocation
                .metadata
                .charged_capacity_bytes
                .checked_add(metadata_delta)
                .ok_or_else(|| DecoderError::InvalidParam("AVIS metadata ticket overflows".to_string()))?;
            let token = budget.reserve_existing_bytes(
                AllocationClass::Metadata,
                metadata_delta,
                "AVIS sequence metadata clone growth",
            )?;
            debug_assert_eq!(token.charged_capacity_bytes, metadata_delta);
            allocation.metadata.charged_capacity_bytes = new_charge;
        }
        if frame_delta != 0 {
            let new_charge = allocation
                .frame
                .charged_capacity_bytes
                .checked_add(frame_delta)
                .ok_or_else(|| DecoderError::InvalidParam("AVIS frame ticket overflows".to_string()))?;
            match budget.reserve_existing_bytes(
                AllocationClass::Frame,
                frame_delta,
                "AVIS sequence static-alpha clone growth",
            ) {
                Ok(token) => {
                    debug_assert_eq!(token.charged_capacity_bytes, frame_delta);
                    allocation.frame.charged_capacity_bytes = new_charge;
                }
                Err(error) => {
                    if metadata_delta != 0 {
                        budget.release_class_bytes(AllocationClass::Metadata, metadata_delta)?;
                        allocation.metadata.charged_capacity_bytes -= metadata_delta;
                    }
                    return Err(error);
                }
            }
        }
        Ok(())
    }
}

impl SequenceTracksDecoder {
    fn new(info: AvifInfo, sequence: &AvifSequence) -> Result<Self, DecoderError> {
        Self::new_with_policy(info, sequence, false)
    }

    fn new_strict(info: AvifInfo, sequence: &AvifSequence) -> Result<Self, DecoderError> {
        Self::new_with_policy(info, sequence, true)
    }

    fn new_with_policy(
        mut info: AvifInfo,
        sequence: &AvifSequence,
        strict_alpha: bool,
    ) -> Result<Self, DecoderError> {
        if sequence.color_samples.is_empty() {
            return Err(DecoderError::Bitstream(
                "AVIS color track has no samples".to_string(),
            ));
        }
        if !sequence.alpha_samples.is_empty()
            && sequence.alpha_samples.len() != sequence.color_samples.len()
        {
            return Err(DecoderError::Bitstream(format!(
                "AVIS alpha track has {} frames, expected {}",
                sequence.alpha_samples.len(),
                sequence.color_samples.len()
            )));
        }
        let color_state = SequenceDecodeState::new(&info)?;
        let (alpha_info, alpha_state) = if sequence.alpha_samples.is_empty() {
            (None, None)
        } else {
            let mut alpha_info = info.clone();
            alpha_info.primary_item_payload = sequence.alpha_samples[0].clone();
            alpha_info.sequence_sample_payloads.clear();
            alpha_info.alpha_auxiliary_items.clear();
            alpha_info.alpha_grid = None;
            alpha_info.av1_config = None;
            let alpha_state = SequenceDecodeState::new(&alpha_info)?;
            alpha_info.primary_item_payload.clear();
            (Some(alpha_info), Some(alpha_state))
        };
        // The caller's `AvifSequence` is the canonical owner of track samples.
        // Keep only container metadata in the private decode view rather than
        // a second retained copy of every color payload.
        info.primary_item_payload.clear();
        info.sequence_sample_payloads.clear();
        Ok(Self {
            info,
            color_state,
            alpha_info,
            alpha_state,
            static_alpha_frame: None,
            strict_alpha,
        })
    }

    #[cfg(test)]
    fn decoded_sample_counts(&self) -> (usize, Option<usize>) {
        (
            self.color_state.decoded_sample_count(),
            self.alpha_state
                .as_ref()
                .map(SequenceDecodeState::decoded_sample_count),
        )
    }

    fn next_frame(
        &mut self,
        sequence: &AvifSequence,
        next_index: usize,
    ) -> Result<Option<DecodedFrame>, DecoderError> {
        self.next_frame_with_options(sequence, next_index, None, None)
    }

    fn next_frame_with_budget(
        &mut self,
        sequence: &AvifSequence,
        next_index: usize,
        limits: &crate::limits::NativeDecodeLimits,
        budget: &mut crate::container::DecodeBudget,
    ) -> Result<Option<DecodedFrame>, DecoderError> {
        self.next_frame_with_options(sequence, next_index, Some(limits), Some(budget))
    }

    fn next_frame_with_options(
        &mut self,
        sequence: &AvifSequence,
        next_index: usize,
        strict_limits: Option<&crate::limits::NativeDecodeLimits>,
        mut strict_budget: Option<&mut crate::container::DecodeBudget>,
    ) -> Result<Option<DecodedFrame>, DecoderError> {
        let mut color_state = self.color_state.clone();
        let mut alpha_state = self.alpha_state.clone();
        let mut frame = if self.strict_alpha {
            let limits = strict_limits.expect("strict track requires native limits");
            let budget = strict_budget
                .as_deref_mut()
                .expect("strict track requires a decode budget");
            color_state.next_sample_strict(&self.info, &sequence.color_samples, limits, budget)?
        } else {
            color_state.next_sample(&self.info, &sequence.color_samples)?
        }
            .ok_or_else(|| {
                DecoderError::Bitstream(format!(
                    "AVIS color track ended before sample {}",
                    next_index
                ))
            })?;
        if let (Some(alpha_info), Some(alpha_state)) =
            (self.alpha_info.as_ref(), alpha_state.as_mut())
        {
            let alpha_frame = if self.strict_alpha {
                let limits = strict_limits.expect("strict track requires native limits");
                let budget = strict_budget
                    .as_deref_mut()
                    .expect("strict track requires a decode budget");
                alpha_state.next_sample_strict(
                    alpha_info,
                    &sequence.alpha_samples,
                    limits,
                    budget,
                )?
            } else {
                alpha_state.next_sample(alpha_info, &sequence.alpha_samples)?
            }
                .ok_or_else(|| {
                    DecoderError::Bitstream(format!(
                        "AVIS alpha track ended before sample {}",
                        next_index
                    ))
                })?;
            if self.strict_alpha {
                validate_strict_alpha(&frame, &alpha_frame)?;
            }
            append_alpha_plane(&mut frame, &alpha_frame)?;
        } else if !self.info.alpha_auxiliary_items.is_empty() {
            let new_static_alpha = if self.static_alpha_frame.is_none() {
                Some(decode_alpha_auxiliary_frame(&self.info)?)
            } else {
                None
            };
            let alpha_frame = self
                .static_alpha_frame
                .as_ref()
                .or(new_static_alpha.as_ref())
                .expect("static alpha frame should be cached or freshly decoded");
            if self.strict_alpha {
                validate_strict_alpha(&frame, alpha_frame)?;
            }
            append_alpha_plane(&mut frame, alpha_frame)?;
            if let Some(alpha_frame) = new_static_alpha {
                self.static_alpha_frame = Some(alpha_frame);
            }
        }
        self.color_state = color_state;
        self.alpha_state = alpha_state;
        Ok(Some(frame))
    }
}

impl AvifSequenceDecoder {
    pub fn new(data: &[u8]) -> Result<Self, DecoderError> {
        let info = parse_avif(data)?;
        validate_public_container_preflight(&info, false)?;
        let animation = parse_avif_animation(data)?;
        if animation.sequence.color_samples.len() != animation.color_timing.len() {
            return Err(DecoderError::Bitstream(
                "AVIS color timing does not match the sample count".to_string(),
            ));
        }
        let tracks = SequenceTracksDecoder::new(info, &animation.sequence)?;
        Ok(Self {
            animation,
            tracks,
            next_index: 0,
        })
    }

    pub fn animation(&self) -> &AvifAnimation {
        &self.animation
    }

    #[cfg(test)]
    pub(super) fn decoded_track_sample_counts(&self) -> (usize, Option<usize>) {
        self.tracks.decoded_sample_counts()
    }

    pub fn next_frame(&mut self) -> Result<Option<DecodedSequenceFrame>, DecoderError> {
        let Some(timing) = self.animation.color_timing.get(self.next_index).copied() else {
            return Ok(None);
        };
        let frame = self
            .tracks
            .next_frame(&self.animation.sequence, self.next_index)?
            .ok_or_else(|| {
                DecoderError::Bitstream(format!(
                    "AVIS sequence ended before sample {}",
                    self.next_index
                ))
            })?;
        self.next_index += 1;
        Ok(Some(DecodedSequenceFrame { frame, timing }))
    }
}

/// Bounded, transactional AVIS decoder for high-precision consumers.
///
/// Unlike [`AvifSequenceDecoder`], construction performs one native container
/// parse and applies [`NativeDecodeLimits`](crate::NativeDecodeLimits) before
/// retaining movie samples.  A frame is decoded against a cloned track state;
/// state advances only when its [`PreparedSequenceFrame`] is committed.
pub struct StrictAvifSequenceDecoder {
    information: crate::native::NativeAvifInformation,
    animation: AvifAnimation,
    tracks: SequenceTracksDecoder,
    tracks_allocation: SequenceTracksAllocation,
    limits: crate::limits::NativeDecodeLimits,
    budget: crate::container::DecodeBudget,
    next_index: usize,
}

impl StrictAvifSequenceDecoder {
    #[cfg(test)]
    pub(super) fn from_test_parts(
        info: AvifInfo,
        animation: AvifAnimation,
        limits: crate::limits::NativeDecodeLimits,
    ) -> Result<Self, DecoderError> {
        if animation.sequence.color_samples.is_empty()
            || animation.sequence.color_samples.len() != animation.color_timing.len()
        {
            return Err(DecoderError::Bitstream(
                "synthetic AVIS color samples and timing must match".to_string(),
            ));
        }
        if !animation.sequence.alpha_samples.is_empty()
            && (animation.sequence.alpha_samples.len() != animation.alpha_timing.len()
                || animation.sequence.alpha_samples.len() != animation.sequence.color_samples.len())
        {
            return Err(DecoderError::Bitstream(
                "synthetic AVIS alpha samples and timing must match".to_string(),
            ));
        }
        // Keep the synthetic constructor subject to the same frame-count
        // admission as the parsed constructor.  This is test-only, but it is
        // important that a synthetic multi-frame fixture cannot allocate
        // retained track state before a caller's frame limit is checked.
        limits.check_count(
            animation.sequence.color_samples.len(),
            limits.max_frames(),
            "frame",
        )?;
        let mut budget = crate::container::DecodeBudget::new(limits.max_live_allocation_bytes());
        let clone_peak_bytes =
            sequence_tracks_clone_peak_bytes(&info, &animation.sequence.alpha_samples)?;
        budget.check_additional_frame(clone_peak_bytes)?;
        let tracks = SequenceTracksDecoder::new_strict(info.clone(), &animation.sequence)?;
        let tracks_storage = tracks.storage_bytes()?;
        let tracks_allocation = SequenceTracksDecoder::admit_storage(tracks_storage, &mut budget)?;
        let information = crate::native::NativeAvifInformation::new(
            crate::container::RichAvifInfo {
                info,
                color_information: crate::container::ColorInformationSet::default(),
            },
            0,
            0,
            0,
            0,
            Some(*b"av01"),
            Vec::new(),
        );
        Ok(Self {
            information,
            animation,
            tracks,
            tracks_allocation,
            limits,
            budget,
            next_index: 0,
        })
    }

    pub fn new(
        data: &[u8],
        limits: crate::limits::NativeDecodeLimits,
    ) -> Result<Self, DecoderError> {
        let (information, mut budget, animation) =
            crate::native::parse_native_sequence_info_with_budget(data, &limits)?;
        if !information.info().is_avif_brand()
            || (!information.info().major_brand.eq(b"avis")
                && !information
                    .info()
                    .compatible_brands
                    .iter()
                    .any(|brand| brand == b"avis"))
        {
            return Err(DecoderError::Unsupported(
                "strict sequence decoder requires an AVIS container".to_string(),
            ));
        }
        if information.primary_item_type() != Some(*b"av01") {
            return Err(DecoderError::Unsupported(
                "strict sequence decoder requires a primary av01 item".to_string(),
            ));
        }
        if animation.sequence.color_samples.is_empty() {
            return Err(DecoderError::Bitstream(
                "AVIS color track has no samples".to_string(),
            ));
        }
        if animation.sequence.color_samples.len() != animation.color_timing.len() {
            return Err(DecoderError::Bitstream(
                "AVIS color timing does not match the sample count".to_string(),
            ));
        }
        if !animation.sequence.alpha_samples.is_empty() {
            if animation.sequence.alpha_samples.len() != animation.alpha_timing.len()
                || animation.sequence.alpha_samples.len() != animation.sequence.color_samples.len()
            {
                return Err(DecoderError::Bitstream(
                    "AVIS alpha timing does not match the color track".to_string(),
                ));
            }
            for (color, alpha) in animation
                .color_timing
                .iter()
                .zip(animation.alpha_timing.iter())
            {
                validate_exact_timing_sync(color, alpha)?;
            }
        }
        limits.check_count(
            animation.sequence.color_samples.len(),
            limits.max_frames(),
            "frame",
        )?;
        // The strict track view owns a deep metadata clone.  Preflight the
        // source capacities before cloning so a metadata limit failure cannot
        // occur after the clone has already been materialized.  The resulting
        // allocation is still measured again below because an allocator may
        // provide a larger capacity than requested.
        let clone_peak_bytes = sequence_tracks_clone_peak_bytes(
            information.info(),
            &animation.sequence.alpha_samples,
        )?;
        budget.check_additional_frame(clone_peak_bytes)?;
        let tracks =
            SequenceTracksDecoder::new_strict(information.info().clone(), &animation.sequence)?;
        let tracks_storage = tracks.storage_bytes()?;
        let tracks_allocation = SequenceTracksDecoder::admit_storage(tracks_storage, &mut budget)?;
        Ok(Self {
            information,
            animation,
            tracks,
            tracks_allocation,
            limits,
            budget,
            next_index: 0,
        })
    }

    pub fn information(&self) -> &crate::native::NativeAvifInformation {
        &self.information
    }

    pub fn frame_count(&self) -> usize {
        self.animation.sequence.color_samples.len()
    }

    pub fn timescale(&self) -> u64 {
        self.animation.color_timescale
    }

    pub fn duration_in_timescales(&self) -> u64 {
        self.animation.duration_in_timescales
    }

    pub fn repetition_count(&self) -> AvifRepetitionCount {
        self.animation.repetition_count
    }

    pub fn prepare_next_frame(
        &mut self,
    ) -> Result<Option<PreparedSequenceFrame<'_>>, DecoderError> {
        let Some(timing) = self.animation.color_timing.get(self.next_index).copied() else {
            return Ok(None);
        };
        let color_sample = self
            .animation
            .sequence
            .color_samples
            .get(self.next_index)
            .ok_or_else(|| {
                DecoderError::Bitstream("AVIS color sample is missing for state plan".to_string())
            })?;
        let alpha_sample = self
            .animation
            .sequence
            .alpha_samples
            .get(self.next_index)
            .map(Vec::as_slice);
        // Admit the clone and the entire strict refresh/split plan together.
        // This ordering is essential: no metadata clone, split vector, or AV1
        // decode may begin if their aggregate live peak is already over the
        // caller's limit.
        let aggregate_preflight = self.next_prepare_additional_bytes(color_sample, alpha_sample)?;
        let refresh_plan =
            self.tracks
                .state_refresh_plan(color_sample, alpha_sample, &self.limits)?;
        self.budget.check_additional_frame(aggregate_preflight)?;
        #[cfg(test)]
        let _clone_phase = crate::test_allocation_observer::begin_phase(1);
        let (mut candidate_tracks, mut candidate_allocation) =
            self.tracks.clone_with_budget(&mut self.budget)?;
        #[cfg(test)]
        drop(_clone_phase);
        let mut allocation_guard =
            SequenceTracksAllocationGuard::new(&mut self.budget, candidate_allocation);
        allocation_guard.reserve_state_delta(refresh_plan)?;
        let before_decode = candidate_tracks.storage_bytes_for_clone()?;
        let frame = match candidate_tracks.next_frame_with_budget(
            &self.animation.sequence,
            self.next_index,
            &self.limits,
            allocation_guard.budget,
        ) {
            Ok(Some(frame)) => frame,
            Ok(None) => {
                return Err(DecoderError::Bitstream(format!(
                    "AVIS sequence ended before sample {}",
                    self.next_index
                )));
            }
            Err(error) => {
                return Err(error);
            }
        };
        let after_decode = candidate_tracks.storage_bytes_for_clone()?;
        if let Err(error) = allocation_guard.reconcile_storage(before_decode, after_decode) {
            return Err(error);
        }
        // The candidate starts by sharing every state Arc with the committed
        // track.  Only owners introduced by decoding this sample are new
        // physical allocations; charge that union delta separately so the
        // clone itself remains a zero-delta operation.
        let state_delta = candidate_tracks.additional_state_memory_bytes(&self.tracks)?;
        allocation_guard.reconcile_state_delta(state_delta)?;
        if let Err(error) = validate_native_frame_limits(&frame, &self.limits) {
            return Err(error);
        }
        let frame_storage = match decoded_frame_storage_bytes(&frame) {
            Ok(bytes) => bytes,
            Err(error) => return Err(error),
        };
        let retained_live_bytes = allocation_guard.budget.aggregate_live_bytes();
        let additional_live_bytes = retained_live_bytes
            .checked_add(frame_storage)
            .ok_or_else(|| {
                DecoderError::InvalidParam("AVIS prepared live bytes overflow".to_string())
            })?;
        // Track clone storage is already charged by `clone_with_budget`; only
        // the newly decoded frame is additional to the current ledger state.
        allocation_guard
            .budget
            .check_additional_frame(frame_storage)?;
        candidate_allocation = allocation_guard.disarm();
        Ok(Some(PreparedSequenceFrame {
            decoder: self,
            candidate_tracks: Some(candidate_tracks),
            candidate_allocation: Some(candidate_allocation),
            frame: Some(frame),
            timing,
            additional_live_bytes,
            retained_live_bytes,
            committed: false,
        }))
    }

    fn next_prepare_additional_bytes(
        &self,
        color_sample: &[u8],
        alpha_sample: Option<&[u8]>,
    ) -> Result<usize, DecoderError> {
        let refresh_plan =
            self.tracks
                .state_refresh_plan(color_sample, alpha_sample, &self.limits)?;
        let header_peak = self
            .tracks
            .strict_header_peak_plan(color_sample, alpha_sample, &self.limits)?;
        let clone_bytes = self.tracks.storage_bytes_for_clone()?.total()?;
        clone_bytes.checked_add(refresh_plan).ok_or_else(|| {
            DecoderError::InvalidParam("AVIS strict preparation preflight overflows".to_string())
        })?.checked_add(header_peak).ok_or_else(|| {
            DecoderError::InvalidParam("AVIS strict preparation preflight overflows".to_string())
        })
    }
}

/// A decoded AVIS frame whose state transition is not yet committed.
pub struct PreparedSequenceFrame<'a> {
    decoder: &'a mut StrictAvifSequenceDecoder,
    candidate_tracks: Option<SequenceTracksDecoder>,
    candidate_allocation: Option<SequenceTracksAllocation>,
    frame: Option<DecodedFrame>,
    timing: AvifFrameTiming,
    additional_live_bytes: usize,
    retained_live_bytes: usize,
    committed: bool,
}

#[derive(Clone, Copy)]
struct SequenceCommitPlan {
    before: crate::container::ParseAccounting,
    after: crate::container::ParseAccounting,
    target_state_bytes: usize,
    next_index: usize,
}

impl SequenceCommitPlan {
    fn build(prepared: &PreparedSequenceFrame<'_>) -> Result<Self, DecoderError> {
        if prepared.frame.is_none() {
            return Err(DecoderError::InvalidParam(
                "AVIS prepared frame is already taken".to_string(),
            ));
        }
        let next_index = prepared.decoder.next_index.checked_add(1).ok_or_else(|| {
            DecoderError::InvalidParam("AVIS frame index overflows".to_string())
        })?;
        let candidate = prepared.candidate_tracks.as_ref().ok_or_else(|| {
            DecoderError::InvalidParam("AVIS prepared frame was already committed".to_string())
        })?;
        let candidate_allocation = prepared.candidate_allocation.as_ref().ok_or_else(|| {
            DecoderError::InvalidParam("AVIS prepared allocation was already committed".to_string())
        })?;
        for allocation in [&prepared.decoder.tracks_allocation, candidate_allocation] {
            if allocation.metadata.class != AllocationClass::Metadata
                || allocation.frame.class != AllocationClass::Frame
                || allocation.state.class != AllocationClass::Frame
            {
                return Err(DecoderError::InvalidParam(
                    "AVIS sequence allocation ticket class mismatch".to_string(),
                ));
            }
        }
        prepared
            .decoder
            .tracks_allocation
            .validate_release(&prepared.decoder.budget)?;
        candidate_allocation.validate_release(&prepared.decoder.budget)?;
        let target_state_bytes = candidate.state_memory_bytes()?;
        let current_state_bytes = prepared
            .decoder
            .tracks_allocation
            .state
            .charged_capacity_bytes
            .checked_add(candidate_allocation.state.charged_capacity_bytes)
            .ok_or_else(|| {
                DecoderError::InvalidParam("AVIS sequence state ticket overflows".to_string())
            })?;
        if target_state_bytes > current_state_bytes {
            return Err(DecoderError::InvalidParam(
                "AVIS sequence state ticket underestimates candidate owners".to_string(),
            ));
        }
        let before = prepared.decoder.budget.accounting_snapshot();
        debug_assert_eq!(
            prepared.decoder.budget.accounting_snapshot(),
            before,
            "sequence commit plan must snapshot accounting before mutation"
        );
        let old = &prepared.decoder.tracks_allocation;
        let remove_frame = old
            .frame
            .charged_capacity_bytes
            .checked_add(old.state.charged_capacity_bytes)
            .and_then(|bytes| {
                bytes.checked_add(candidate_allocation.frame.charged_capacity_bytes)
            })
            .and_then(|bytes| {
                bytes.checked_add(candidate_allocation.state.charged_capacity_bytes)
            })
            .ok_or_else(|| {
                DecoderError::InvalidParam("AVIS sequence frame accounting overflows".to_string())
            })?;
        let frame_without_owners = before.frame_live.checked_sub(remove_frame).ok_or_else(|| {
            DecoderError::InvalidParam("AVIS sequence frame accounting underflows".to_string())
        })?;
        let final_frame = frame_without_owners
            .checked_add(candidate_allocation.frame.charged_capacity_bytes)
            .and_then(|bytes| bytes.checked_add(target_state_bytes))
            .ok_or_else(|| {
                DecoderError::InvalidParam("AVIS sequence frame accounting overflows".to_string())
            })?;
        let final_metadata = before
            .metadata_live
            .checked_sub(old.metadata.charged_capacity_bytes)
            .ok_or_else(|| {
                DecoderError::InvalidParam("AVIS sequence metadata accounting underflows".to_string())
            })?;
        let aggregate = final_metadata
            .checked_add(before.payload_live)
            .and_then(|bytes| bytes.checked_add(final_frame))
            .ok_or_else(|| {
                DecoderError::InvalidParam("AVIS sequence aggregate accounting overflows".to_string())
            })?;
        let mut after = before;
        after.metadata_live = final_metadata;
        after.frame_live = final_frame;
        after.aggregate_live = aggregate;
        Ok(Self {
            before,
            after,
            target_state_bytes,
            next_index,
        })
    }
}

/// A validated, move-only sequence commit token.
///
/// The token owns the prepared frame until [`Self::try_map_and_commit`]
/// succeeds.  There is intentionally no public operation that consumes the
/// frame and then leaves the token commit-capable after a callback error.
pub struct PreparedSequenceCommit<'a> {
    prepared: Option<PreparedSequenceFrame<'a>>,
    plan: SequenceCommitPlan,
}

impl<'a> PreparedSequenceFrame<'a> {
    /// Borrows the decoded frame before commit. The frame remains owned by
    /// this preparation until [`Self::take_frame`] or drop.
    pub fn frame(&self) -> Option<&DecodedFrame> {
        self.frame.as_ref()
    }

    pub fn timing(&self) -> AvifFrameTiming {
        self.timing
    }

    pub const fn additional_live_bytes(&self) -> usize {
        self.additional_live_bytes
    }

    /// Returns the strict decoder storage retained while this preparation is
    /// alive, excluding the owned decoded frame and its native plane buffers.
    ///
    /// This value is suitable for a bridge that maps the returned frame into
    /// another checked ownership domain.  It includes the candidate track
    /// metadata/frame storage and sequence state, but not the frame that is
    /// transferred to the consumer by [`PreparedSequenceCommit::try_map_and_commit`].
    pub const fn retained_live_bytes(&self) -> usize {
        self.retained_live_bytes
    }

    /// Returns the zero-based index of this prepared frame.
    pub fn frame_index(&self) -> usize {
        self.decoder.next_index
    }

    /// Borrows the clone-free rich information associated with this frame.
    pub fn information(&self) -> &crate::native::NativeAvifInformation {
        self.decoder.information()
    }

    /// Takes the prepared frame exactly once. The sequence cursor is still
    /// unchanged until [`Self::commit`] succeeds; dropping after this call
    /// rolls back the candidate track state.
    pub fn take_frame(&mut self) -> Option<DecodedFrame> {
        self.frame.take()
    }

    /// Runs a consumer with the owned decoded frame and the decoder's rich
    /// container information without cloning either value.
    ///
    /// Converts this preparation into a validated commit token.
    ///
    /// The token is the only supported clone-free mapping boundary.  Its
    /// consuming mapping operation either drops the candidate on callback
    /// error or commits the prevalidated accounting snapshot on success.
    pub fn prepare_commit(self) -> Result<PreparedSequenceCommit<'a>, DecoderError> {
        let plan = SequenceCommitPlan::build(&self)?;
        Ok(PreparedSequenceCommit {
            prepared: Some(self),
            plan,
        })
    }

    /// Commits the cloned decoder state and advances the sequence cursor.
    /// Dropping a preparation without calling this method never advances it.
    pub fn commit(self) -> Result<(), DecoderError> {
        if self.frame.is_some() {
            let token = self.prepare_commit()?;
            token.commit();
            Ok(())
        } else {
            self.commit_legacy()
        }
    }

    fn commit_legacy(mut self) -> Result<(), DecoderError> {
        // Validate the cursor before taking or replacing either owner.  This
        // keeps an overflow failure fully transactional (P2 contract).
        let next_index = self.decoder.next_index.checked_add(1).ok_or_else(|| {
            DecoderError::InvalidParam("AVIS frame index overflows".to_string())
        })?;
        let candidate_ref = self.candidate_tracks.as_ref().ok_or_else(|| {
            DecoderError::InvalidParam("AVIS prepared frame was already committed".to_string())
        })?;
        let candidate_allocation_ref = self.candidate_allocation.as_ref().ok_or_else(|| {
            DecoderError::InvalidParam("AVIS prepared allocation was already committed".to_string())
        })?;
        let candidate_state_bytes = candidate_ref.state_memory_bytes()?;
        let old_state_bytes = self.decoder.tracks_allocation.state.charged_capacity_bytes;
        let candidate_state_ticket = candidate_allocation_ref.state.charged_capacity_bytes;
        let current_state_bytes = old_state_bytes
            .checked_add(candidate_state_ticket)
            .ok_or_else(|| {
                DecoderError::InvalidParam("AVIS sequence state ticket overflows".to_string())
            })?;
        if candidate_state_bytes > current_state_bytes {
            self.decoder
                .budget
                .check_additional_frame(candidate_state_bytes - current_state_bytes)?;
        } else if current_state_bytes > candidate_state_bytes {
            self.decoder.budget.validate_release(
                AllocationClass::Frame,
                current_state_bytes - candidate_state_bytes,
                "AVIS committed sequence state owners",
            )?;
        }
        self.decoder.tracks_allocation.validate_release(&self.decoder.budget)?;
        if candidate_state_bytes > current_state_bytes {
            let state_delta = candidate_state_bytes - current_state_bytes;
            let token = self.decoder.budget.reserve_existing_bytes(
                AllocationClass::Frame,
                state_delta,
                "AVIS committed sequence state owners",
            )?;
            debug_assert_eq!(token.charged_capacity_bytes, state_delta);
            self.candidate_allocation
                .as_mut()
                .expect("sequence allocation must remain present")
                .state
                .charged_capacity_bytes = candidate_state_bytes;
        } else if current_state_bytes > candidate_state_bytes {
            self.decoder.budget.release_class_bytes(
                AllocationClass::Frame,
                current_state_bytes - candidate_state_bytes,
            )?;
        }
        self.candidate_allocation
            .as_mut()
            .expect("sequence allocation must remain present")
            .state
            .charged_capacity_bytes = candidate_state_bytes;
        // Release only the old metadata/frame ticket while its state ticket
        // is retained by the shared candidate graph.  Validation above makes
        // this operation non-failing in normal execution; the old state value
        // is restored if an accounting authority reports an unexpected error.
        let old_state_ticket = self.decoder.tracks_allocation.state.charged_capacity_bytes;
        self.decoder.tracks_allocation.state.charged_capacity_bytes = 0;
        if let Err(error) = self.decoder.tracks_allocation.release(&mut self.decoder.budget) {
            self.decoder.tracks_allocation.state.charged_capacity_bytes = old_state_ticket;
            return Err(error);
        }
        let candidate = self.candidate_tracks.take().ok_or_else(|| {
            DecoderError::InvalidParam("AVIS prepared frame was already committed".to_string())
        })?;
        let candidate_allocation = self.candidate_allocation.take().ok_or_else(|| {
            DecoderError::InvalidParam("AVIS prepared allocation was already committed".to_string())
        })?;
        self.decoder.tracks_allocation = candidate_allocation;
        self.decoder.tracks = candidate;
        self.decoder.next_index = next_index;
        self.committed = true;
        Ok(())
    }
}

impl PreparedSequenceCommit<'_> {
    pub fn frame(&self) -> Option<&DecodedFrame> {
        self.prepared.as_ref().and_then(|prepared| prepared.frame())
    }

    pub fn information(&self) -> &crate::native::NativeAvifInformation {
        self.prepared
            .as_ref()
            .expect("sequence commit token must own preparation")
            .information()
    }

    pub const fn frame_index(&self) -> usize {
        self.plan.next_index - 1
    }

    pub fn timing(&self) -> AvifFrameTiming {
        self.prepared
            .as_ref()
            .expect("sequence commit token must own preparation")
            .timing
    }

    pub fn retained_live_bytes(&self) -> usize {
        self.prepared
            .as_ref()
            .expect("sequence commit token must own preparation")
            .retained_live_bytes
    }

    /// Maps the owned native frame and commits the sequence transition only
    /// after the callback succeeds.  A callback error consumes the token and
    /// rolls back through the prepared owner's Drop implementation.
    pub fn try_map_and_commit<T, E, F>(mut self, consumer: F) -> Result<T, E>
    where
        F: FnOnce(
            DecodedFrame,
            &crate::container::RichAvifInfo,
            AvifFrameTiming,
            usize,
        ) -> Result<T, E>,
    {
        let prepared = self
            .prepared
            .as_mut()
            .expect("sequence commit token must own preparation");
        let frame = prepared
            .frame
            .take()
            .expect("sequence commit token frame must be present");
        let value = consumer(
            frame,
            prepared.decoder.information.rich(),
            prepared.timing,
            prepared.decoder.next_index,
        )?;
        self.commit();
        Ok(value)
    }

    /// Applies the prevalidated accounting snapshot and swaps the candidate
    /// track owners.  No fallible budget operation remains after the mapping
    /// callback has returned successfully.
    pub fn commit(mut self) {
        let prepared_ref = self
            .prepared
            .as_ref()
            .expect("sequence commit token must own preparation");
        debug_assert_eq!(
            prepared_ref.decoder.budget.accounting_snapshot(),
            self.plan.before,
            "sequence commit accounting changed after preflight"
        );
        let mut prepared = self
            .prepared
            .take()
            .expect("sequence commit token must own preparation");
        let mut candidate_allocation = prepared
            .candidate_allocation
            .take()
            .expect("sequence commit token allocation must remain present");
        candidate_allocation.state.charged_capacity_bytes = self.plan.target_state_bytes;
        let candidate = prepared
            .candidate_tracks
            .take()
            .expect("sequence commit token tracks must remain present");
        prepared.decoder.tracks_allocation.metadata.charged_capacity_bytes = 0;
        prepared.decoder.tracks_allocation.frame.charged_capacity_bytes = 0;
        prepared.decoder.tracks_allocation.state.charged_capacity_bytes = 0;
        prepared.decoder.budget.apply_prevalidated_sequence_commit(
            self.plan.before,
            self.plan.after,
        );
        prepared.decoder.tracks_allocation = candidate_allocation;
        prepared.decoder.tracks = candidate;
        prepared.decoder.next_index = self.plan.next_index;
        prepared.committed = true;
    }
}

impl Drop for PreparedSequenceFrame<'_> {
    fn drop(&mut self) {
        // Candidate track state and frame are dropped without touching the
        // decoder. This is the rollback path for both explicit and panic-free
        // early exits before `commit`.
        if !self.committed {
            self.candidate_tracks.take();
            self.frame.take();
            if let Some(mut allocation) = self.candidate_allocation.take() {
                let _ = allocation.release(&mut self.decoder.budget);
            }
        }
    }
}

fn validate_exact_timing_sync(
    color: &AvifFrameTiming,
    alpha: &AvifFrameTiming,
) -> Result<(), DecoderError> {
    if color.timescale == 0 || alpha.timescale == 0 {
        return Err(DecoderError::Bitstream(
            "AVIS track timescale must be non-zero".to_string(),
        ));
    }
    let pts_left = u128::from(color.pts_in_timescales)
        .checked_mul(u128::from(alpha.timescale))
        .ok_or_else(|| DecoderError::Bitstream("AVIS PTS comparison overflows".to_string()))?;
    let pts_right = u128::from(alpha.pts_in_timescales)
        .checked_mul(u128::from(color.timescale))
        .ok_or_else(|| DecoderError::Bitstream("AVIS PTS comparison overflows".to_string()))?;
    let duration_left = u128::from(color.duration_in_timescales)
        .checked_mul(u128::from(alpha.timescale))
        .ok_or_else(|| DecoderError::Bitstream("AVIS duration comparison overflows".to_string()))?;
    let duration_right = u128::from(alpha.duration_in_timescales)
        .checked_mul(u128::from(color.timescale))
        .ok_or_else(|| DecoderError::Bitstream("AVIS duration comparison overflows".to_string()))?;
    if pts_left != pts_right || duration_left != duration_right {
        return Err(DecoderError::Bitstream(
            "AVIS alpha PTS or duration is not synchronized with color".to_string(),
        ));
    }
    Ok(())
}

fn frame_storage_bytes(frame: &DecodedFrame) -> Result<usize, DecoderError> {
    let mut bytes = frame
        .buffers
        .planes
        .capacity()
        .checked_mul(std::mem::size_of::<crate::av1::PlaneBuffer>())
        .ok_or_else(|| DecoderError::InvalidParam("AVIS frame storage overflows".to_string()))?;
    for plane in &frame.buffers.planes {
        bytes = bytes
            .checked_add(
                plane
                    .samples
                    .capacity()
                    .checked_mul(std::mem::size_of::<u16>())
                    .ok_or_else(|| {
                        DecoderError::InvalidParam("AVIS plane storage overflows".to_string())
                    })?,
            )
            .ok_or_else(|| DecoderError::InvalidParam("AVIS frame storage overflows".to_string()))?;
    }
    Ok(bytes)
}

impl DecodedFrame {
    pub fn to_rgba8(&self) -> Result<ImageBuffer, DecoderError> {
        if self
            .color_information
            .as_ref()
            .and_then(ColorInformation::icc_profile)
            .is_none()
        {
            let mut image = crate::av1::frame_buffers_to_rgba_8(&self.buffers, &self.color_config)?;
            if self.alpha_premultiplied {
                unpremultiply_rgba8(&mut image.rgba);
            }
            return Ok(image);
        }
        let rgba16 = self.to_rgba16()?;
        let rgba = rgba16
            .rgba
            .iter()
            .map(|sample| ((u32::from(*sample) * 255 + 32767) / 65535) as u8)
            .collect();
        Ok(ImageBuffer {
            width: rgba16.width,
            height: rgba16.height,
            rgba,
        })
    }

    pub fn to_rgba16(&self) -> Result<Rgba16ImageBuffer, DecoderError> {
        let mut image = frame_buffers_to_rgba_16(&self.buffers, &self.color_config)?;
        if let Some(profile) = self
            .color_information
            .as_ref()
            .and_then(ColorInformation::icc_profile)
        {
            crate::icc::apply_to_rgba16(&mut image.rgba, profile)?;
        }
        if self.alpha_premultiplied {
            unpremultiply_rgba16(&mut image.rgba);
        }
        Ok(image)
    }

    /// Applies an explicitly decoded ISO 21496 gain map to this frame.
    ///
    /// Gain-map frames may use a different native size and are resampled to
    /// the base dimensions during composition. Base-colour maps and alternate
    /// maps in supported CICP RGB primary sets are
    /// supported; matrix-shaper and linear-affine ICC LUT/mAB alternate
    /// conversions are supported while non-linear or reverse-direction
    /// profiles fail closed because applying their tone curves to scalar gain
    /// samples would change the gain semantics.
    /// `hdr_headroom` is expressed in log2 headroom units; a value at the base
    /// headroom returns the base RGBA16 image unchanged. The default AVIF
    /// decode path never applies this method implicitly.
    pub fn to_rgba16_with_gain_map(
        &self,
        gain_map: &DecodedGainMapFrame,
        hdr_headroom: f32,
    ) -> Result<Rgba16ImageBuffer, DecoderError> {
        if !hdr_headroom.is_finite() || hdr_headroom < 0.0 {
            return Err(DecoderError::InvalidParam(
                "gain-map HDR headroom must be finite and non-negative".to_string(),
            ));
        }
        let weight = gain_map_weight(hdr_headroom, &gain_map.metadata)?;
        let mut base = self.to_rgba16()?;
        if weight == 0.0 {
            return Ok(base);
        }
        let declared_base_primaries = self
            .color_config
            .color_description
            .map(|description| description.color_primaries)
            .unwrap_or(2);
        let declared_alternate_primaries = gain_map
            .frame
            .color_config
            .color_description
            .map(|description| description.color_primaries)
            .unwrap_or(declared_base_primaries);
        // A gain-map item carries scalar/log gain samples, and libavif files
        // commonly leave its CICP primaries unspecified (2). Treat an
        // unspecified side as inheriting the other side rather than trying
        // to construct a chromaticity matrix for CICP 2.
        let (base_primaries, alternate_primaries) =
            if declared_base_primaries == 2 && declared_alternate_primaries != 2 {
                (declared_alternate_primaries, declared_alternate_primaries)
            } else if declared_alternate_primaries == 2 {
                (declared_base_primaries, declared_base_primaries)
            } else {
                (declared_base_primaries, declared_alternate_primaries)
            };
        let convert_alternate =
            !gain_map.metadata.use_base_colour_space && base_primaries != alternate_primaries;
        let alternate_icc = gain_map
            .frame
            .color_information
            .as_ref()
            .and_then(ColorInformation::icc_profile);
        let decoded_map = gain_map.frame.to_rgba16()?;
        let map = if decoded_map.width == base.width && decoded_map.height == base.height {
            decoded_map
        } else {
            resample_gain_map(&decoded_map, base.width, base.height)?
        };
        if map.rgba.len() != base.rgba.len() {
            return Err(DecoderError::Bitstream(
                "gain-map and base RGBA buffers do not match".to_string(),
            ));
        }
        let channels = gain_map.metadata.channels.as_slice();
        if !matches!(channels.len(), 1 | 3) {
            return Err(DecoderError::Unsupported(format!(
                "gain-map channel count {} is not supported",
                channels.len()
            )));
        }
        let mut gamma = [0.0; 3];
        let mut minimum = [0.0; 3];
        let mut maximum = [0.0; 3];
        let mut base_offset = [0.0; 3];
        let mut alternate_offset = [0.0; 3];
        for channel in 0..3 {
            let metadata = channels[if channels.len() == 1 { 0 } else { channel }];
            gamma[channel] = rational_to_f64(metadata.gamma, "gain-map gamma")?;
            if gamma[channel] <= 0.0 {
                return Err(DecoderError::Bitstream(
                    "gain-map gamma must be positive".to_string(),
                ));
            }
            minimum[channel] = rational_to_f64(metadata.gain_map_min, "gain-map minimum")?;
            maximum[channel] = rational_to_f64(metadata.gain_map_max, "gain-map maximum")?;
            base_offset[channel] = rational_to_f64(metadata.base_offset, "base offset")?;
            alternate_offset[channel] =
                rational_to_f64(metadata.alternate_offset, "alternate offset")?;
        }
        for (base_pixel, map_pixel) in base.rgba.chunks_exact_mut(4).zip(map.rgba.chunks_exact(4)) {
            let mut base_linear = [
                srgb_to_linear(f64::from(base_pixel[0]) / f64::from(u16::MAX)),
                srgb_to_linear(f64::from(base_pixel[1]) / f64::from(u16::MAX)),
                srgb_to_linear(f64::from(base_pixel[2]) / f64::from(u16::MAX)),
            ];
            if convert_alternate {
                if let Some(profile) = alternate_icc {
                    crate::icc::convert_linear_srgb_with_profile(&mut base_linear, profile, true)?;
                } else {
                    convert_linear_rgb_primaries(
                        &mut base_linear,
                        base_primaries,
                        alternate_primaries,
                    )?;
                }
            }
            let mut tone_mapped = [0.0; 3];
            for channel in 0..3 {
                let map_value = f64::from(map_pixel[channel]) / f64::from(u16::MAX);
                let gain_map_log2 = minimum[channel]
                    + (maximum[channel] - minimum[channel]) * map_value.powf(1.0 / gamma[channel]);
                tone_mapped[channel] = (base_linear[channel] + base_offset[channel])
                    * (gain_map_log2 * f64::from(weight)).exp2()
                    - alternate_offset[channel];
            }
            if convert_alternate {
                if let Some(profile) = alternate_icc {
                    crate::icc::convert_linear_srgb_with_profile(&mut tone_mapped, profile, false)?;
                } else {
                    convert_linear_rgb_primaries(
                        &mut tone_mapped,
                        alternate_primaries,
                        base_primaries,
                    )?;
                }
            }
            for channel in 0..3 {
                base_pixel[channel] = (linear_to_srgb(tone_mapped[channel].max(0.0))
                    * f64::from(u16::MAX))
                .round()
                .clamp(0.0, f64::from(u16::MAX)) as u16;
            }
        }
        Ok(base)
    }
}

fn gain_map_weight(
    hdr_headroom: f32,
    metadata: &crate::container::GainMapMetadata,
) -> Result<f32, DecoderError> {
    let base = rational_to_f64(metadata.base_hdr_headroom, "base HDR headroom")?;
    let alternate = rational_to_f64(metadata.alternate_hdr_headroom, "alternate HDR headroom")?;
    if (alternate - base).abs() < f64::EPSILON {
        return Ok(0.0);
    }
    let normalized = ((f64::from(hdr_headroom) - base) / (alternate - base)).clamp(0.0, 1.0);
    Ok(if metadata.backward_direction {
        -(normalized as f32)
    } else {
        normalized as f32
    })
}

fn rational_to_f64(
    rational: crate::container::GainMapRational,
    name: &str,
) -> Result<f64, DecoderError> {
    if rational.denominator == 0 {
        return Err(DecoderError::Bitstream(format!(
            "{name} denominator is zero"
        )));
    }
    Ok(rational.numerator as f64 / f64::from(rational.denominator))
}

fn srgb_to_linear(encoded: f64) -> f64 {
    if encoded <= 0.04045 {
        encoded / 12.92
    } else {
        ((encoded + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(linear: f64) -> f64 {
    if linear <= 0.0031308 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

pub(super) fn resample_gain_map(
    input: &Rgba16ImageBuffer,
    width: usize,
    height: usize,
) -> Result<Rgba16ImageBuffer, DecoderError> {
    if input.width == 0 || input.height == 0 || width == 0 || height == 0 {
        return Err(DecoderError::Bitstream(
            "gain-map resampling dimensions must be non-zero".to_string(),
        ));
    }
    let input_pixels = input
        .width
        .checked_mul(input.height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| DecoderError::InvalidParam("gain-map buffer size overflows".to_string()))?;
    if input.rgba.len() != input_pixels {
        return Err(DecoderError::Bitstream(
            "gain-map RGBA buffer length does not match dimensions".to_string(),
        ));
    }
    let output_pixels = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| DecoderError::InvalidParam("gain-map output size overflows".to_string()))?;
    let mut rgba = vec![0_u16; output_pixels];
    for y in 0..height {
        let source_y = (((y as f64 + 0.5) * input.height as f64 / height as f64) - 0.5)
            .clamp(0.0, (input.height - 1) as f64);
        let y0 = source_y.floor() as usize;
        let y1 = (y0 + 1).min(input.height - 1);
        let fy = source_y - y0 as f64;
        for x in 0..width {
            let source_x = (((x as f64 + 0.5) * input.width as f64 / width as f64) - 0.5)
                .clamp(0.0, (input.width - 1) as f64);
            let x0 = source_x.floor() as usize;
            let x1 = (x0 + 1).min(input.width - 1);
            let fx = source_x - x0 as f64;
            let top = (y0 * input.width + x0) * 4;

            let top_right = (y0 * input.width + x1) * 4;
            let bottom = (y1 * input.width + x0) * 4;
            let bottom_right = (y1 * input.width + x1) * 4;
            let destination = (y * width + x) * 4;
            for channel in 0..4 {
                let top_value = f64::from(input.rgba[top + channel])
                    + (f64::from(input.rgba[top_right + channel])
                        - f64::from(input.rgba[top + channel]))
                        * fx;
                let bottom_value = f64::from(input.rgba[bottom + channel])
                    + (f64::from(input.rgba[bottom_right + channel])
                        - f64::from(input.rgba[bottom + channel]))
                        * fx;
                rgba[destination + channel] = (top_value + (bottom_value - top_value) * fy)
                    .round()
                    .clamp(0.0, f64::from(u16::MAX))
                    as u16;
            }
        }
    }
    Ok(Rgba16ImageBuffer {
        width,
        height,
        rgba,
    })
}

pub(super) fn unpremultiply_rgba8(rgba: &mut [u8]) {
    for pixel in rgba.chunks_exact_mut(4) {
        let alpha = u32::from(pixel[3]);
        if alpha == 0 {
            pixel[..3].fill(0);
            continue;
        }
        for channel in &mut pixel[..3] {
            *channel = ((u32::from(*channel) * 255 + alpha / 2) / alpha).min(255) as u8;
        }
    }
}

pub(super) fn unpremultiply_rgba16(rgba: &mut [u16]) {
    for pixel in rgba.chunks_exact_mut(4) {
        let alpha = u64::from(pixel[3]);
        if alpha == 0 {
            pixel[..3].fill(0);
            continue;
        }
        for channel in &mut pixel[..3] {
            *channel = ((u64::from(*channel) * u64::from(u16::MAX) + alpha / 2) / alpha)
                .min(u64::from(u16::MAX)) as u16;
        }
    }
}

/// Decodes a still AVIF image from memory into high-precision source planes.
pub fn decode_frame_bytes(data: &[u8]) -> Result<DecodedFrame, DecoderError> {
    let info = parse_avif(data)?;
    validate_public_container_preflight(&info, false)?;
    if let Some(frame) = decode_sample_transform_frame(data, &info)? {
        return Ok(frame);
    }
    if info.primary_grid.is_some() {
        return decode_grid_frame(&info);
    }
    let mut frame = if let Some(frame) = decode_hidden_key_frame_show_existing(&info)? {
        frame
    } else {
        let headers = parse_av1_headers(&info)?;
        decode_still_frame(&headers, Some(&info))?
    };
    if !info.alpha_auxiliary_items.is_empty() {
        let alpha_frame = decode_alpha_auxiliary_frame(&info)?;
        append_alpha_plane(&mut frame, &alpha_frame)?;
    }
    Ok(frame)
}

/// Strict native still-image decode used by high-precision consumers.
///
/// The historical [`decode_frame_bytes`] path keeps its established alpha
/// normalization behaviour for compatibility.  This additive entry point
/// validates AVIF's master/auxiliary constraints before attaching alpha, so a
/// depth mismatch can never be rounded into a successful native frame.
pub fn decode_frame_bytes_strict(data: &[u8]) -> Result<DecodedFrame, DecoderError> {
    let info = parse_avif(data)?;
    validate_public_container_preflight(&info, false)?;
    decode_frame_bytes_strict_from_info(data, &info, None)
}

/// Bounded native still decode. The rich container parse is performed once;
/// the legacy unbounded entry point is not used as a fallback.
///
/// At this stage the limits cover the container/metadata projection and the
/// pre-tile geometry/header checks for one ordinary still `av01` item and its
/// selected auxiliary alpha. AVIS sequences and derived images are rejected
/// by this entry point. Deep tile/entropy traversal, reference-frame state,
/// post-filter scratch, and a total-live-allocation guarantee are not covered
/// yet; those are the planned C2/C3 bounded-decoding stages.
pub fn decode_frame_bytes_strict_with_limits(
    data: &[u8],
    limits: &crate::limits::NativeDecodeLimits,
) -> Result<crate::native::NativeDecodedFrame, DecoderError> {
    let (information, mut decode_budget) =
        crate::native::parse_native_info_with_budget(data, limits)?;
    let info = information.info();
    validate_public_container_preflight(info, false)?;
    if !info.sequence_sample_payloads.is_empty() {
        return Err(DecoderError::Unsupported(
            "bounded native decode accepts one still av01 item, not an AVIS sequence".to_string(),
        ));
    }
    if information.primary_item_type() != Some(*b"av01") {
        return Err(DecoderError::Unsupported(
            "bounded native decode accepts only a primary av01 item".to_string(),
        ));
    }
    let derived = information.primary_item_type() == Some(*b"sato") || info.primary_grid.is_some();
    if derived {
        limits.check_count(1, limits.max_derived_depth(), "derived image depth")?;
        return Err(DecoderError::Unsupported(
            "bounded native decode does not yet support derived images".to_string(),
        ));
    }
    let frame =
        decode_frame_bytes_strict_with_budget_from_info(data, info, limits, &mut decode_budget)?;
    validate_native_frame_limits(&frame, limits)?;
    Ok(crate::native::NativeDecodedFrame::new(frame, information))
}

/// Strict bounded decode after the container parser has transferred its
/// retained-owner budget.  Only the ordinary still plus selected alpha route
/// enters this function; AVIS and derived formats were rejected by the caller.
fn decode_frame_bytes_strict_with_budget_from_info(
    _data: &[u8],
    info: &AvifInfo,
    limits: &crate::limits::NativeDecodeLimits,
    budget: &mut crate::container::DecodeBudget,
) -> Result<DecodedFrame, DecoderError> {
    let sequence = validate_native_sequence_limits(info, limits)?;
    let frame_prefix = validate_native_frame_prefix_limits(info, &sequence, limits)?;
    let alpha_header = if info.alpha_auxiliary_items.is_empty() {
        None
    } else {
        let header = super::still::validate_alpha_auxiliary_header_limits(
            info,
            limits,
            usize::try_from(frame_prefix.geometry().0).map_err(|_| {
                DecoderError::InvalidParam("AV1 frame width is too large".to_string())
            })?,
            usize::try_from(frame_prefix.geometry().1).map_err(|_| {
                DecoderError::InvalidParam("AV1 frame height is too large".to_string())
            })?,
            sequence.color_config.bit_depth,
        )?;
        validate_bounded_item_payload(
            &info.alpha_auxiliary_items[0].payload,
            &header.sequence,
            "alpha",
        )?;
        Some(header)
    };
    validate_bounded_item_payload(&info.primary_item_payload, &sequence, "primary")?;
    // Header/tile/plan vectors are strict-native temporary owners.  Keep them
    // charged while the master consumes them, then release their tickets and
    // drop the headers before the selected alpha starts materializing its own
    // header.  This is deliberately unlike the legacy wrapper, whose public
    // allocation timing remains unchanged.
    let (mut frame, mut frame_allocation) = {
        let headers =
            super::parse_av1_headers_with_frame_prefix_and_budget(info, frame_prefix, budget)?;
        let result = (|| {
            validate_native_decode_plan_limits(&headers.decode_plan, limits)?;
            super::decode_still_frame_with_native_budget(&headers, Some(info), budget, limits)
        })();
        super::drop_native_headers_and_release(headers, budget)?;
        result?
    };
    if let Some(alpha_header) = alpha_header {
        let alpha = super::still::decode_alpha_auxiliary_frame_with_prefix_and_budget(
            info,
            alpha_header.parts,
            alpha_header.prefix,
            budget,
            limits,
        );
        let (alpha_frame, mut alpha_allocation) = match alpha {
            Ok(alpha) => alpha,
            Err(error) => {
                frame_allocation.release(budget)?;
                return Err(error);
            }
        };
        if let Err(error) = append_native_alpha_plane_with_budget(
            &mut frame,
            &mut frame_allocation,
            alpha_frame,
            &mut alpha_allocation,
            budget,
        ) {
            alpha_allocation.release(budget)?;
            frame_allocation.release(budget)?;
            return Err(error);
        }
    }
    // The tickets deliberately remain charged through the result construction:
    // the budget is dropped with this private decode operation after all strict
    // allocations have been admitted. They are never exposed through the
    // public `DecodedFrame` API.
    Ok(frame)
}

fn decode_frame_bytes_strict_from_info(
    data: &[u8],
    info: &AvifInfo,
    limits: Option<&crate::limits::NativeDecodeLimits>,
) -> Result<DecodedFrame, DecoderError> {
    if limits.is_none() {
        if let Some(frame) = decode_sample_transform_frame(data, info)? {
            reject_strict_derived_alpha(
                "sato",
                info.alpha_grid.is_some(),
                !info.alpha_auxiliary_items.is_empty(),
            )?;
            return Ok(frame);
        }
        if info.primary_grid.is_some() {
            reject_strict_derived_alpha(
                "grid",
                info.alpha_grid.is_some(),
                !info.alpha_auxiliary_items.is_empty(),
            )?;
            return decode_grid_frame_raw(info);
        }
    }
    let mut bounded_frame_prefix = None;
    let mut bounded_alpha_prefix = None;
    let bounded_sequence = if let Some(limits) = limits {
        let sequence = validate_native_sequence_limits(info, limits)?;
        let frame_prefix = validate_native_frame_prefix_limits(info, &sequence, limits)?;
        if !info.alpha_auxiliary_items.is_empty() {
            let alpha_header = super::still::validate_alpha_auxiliary_header_limits(
                info,
                limits,
                usize::try_from(frame_prefix.geometry().0).map_err(|_| {
                    DecoderError::InvalidParam("AV1 frame width is too large".to_string())
                })?,
                usize::try_from(frame_prefix.geometry().1).map_err(|_| {
                    DecoderError::InvalidParam("AV1 frame height is too large".to_string())
                })?,
                sequence.color_config.bit_depth,
            )?;
            validate_bounded_item_payload(
                &info.alpha_auxiliary_items[0].payload,
                &alpha_header.sequence,
                "alpha",
            )?;
            bounded_alpha_prefix = Some(alpha_header);
        }
        validate_bounded_item_payload(&info.primary_item_payload, &sequence, "primary")?;
        bounded_frame_prefix = Some(frame_prefix);
        Some(sequence)
    } else {
        None
    };
    let mut frame = if bounded_sequence.is_some() {
        let headers = parse_av1_headers_with_frame_prefix(
            info,
            bounded_frame_prefix
                .take()
                .expect("bounded native path has a validated frame prefix"),
        )?;
        validate_native_decode_plan_limits(&headers.decode_plan, limits.expect("bounded path"))?;
        decode_still_frame(&headers, Some(info))?
    } else if let Some(frame) = decode_hidden_key_frame_show_existing(info)? {
        frame
    } else {
        let headers = parse_av1_headers(info)?;
        decode_still_frame(&headers, Some(info))?
    };
    if !info.alpha_auxiliary_items.is_empty() {
        let alpha_frame = if limits.is_some() {
            let prefix = bounded_alpha_prefix
                .take()
                .expect("bounded native path has a validated alpha frame prefix");
            super::still::decode_alpha_auxiliary_frame_with_prefix(
                info,
                prefix.parts,
                prefix.prefix,
            )?
        } else {
            decode_alpha_auxiliary_frame(info)?
        };
        if limits.is_some() {
            append_native_alpha_plane(&mut frame, alpha_frame)?;
        } else {
            validate_strict_alpha(&frame, &alpha_frame)?;
            append_alpha_plane_buffer(
                &mut frame,
                alpha_frame.buffers.planes[0].clone(),
                alpha_frame.bit_depth,
            )?;
        }
    }
    Ok(frame)
}

/// Attaches the already-decoded alpha owner on the strict native path.
///
/// The ownership move is intentionally kept in the production helper used by
/// the bounded decoder.  The legacy path above continues to clone its plane,
/// preserving the historical callback/API behavior.
pub(super) fn append_native_alpha_plane(
    frame: &mut DecodedFrame,
    alpha_frame: DecodedFrame,
) -> Result<(), DecoderError> {
    validate_strict_alpha(frame, &alpha_frame)?;
    let alpha_bit_depth = alpha_frame.bit_depth;
    let alpha_plane = alpha_frame
        .buffers
        .planes
        .into_iter()
        .next()
        .ok_or_else(|| {
            DecoderError::Bitstream("AVIF alpha auxiliary plane is missing".to_string())
        })?;
    append_alpha_plane_buffer(frame, alpha_plane, alpha_bit_depth)
}

fn append_native_alpha_plane_with_budget(
    frame: &mut DecodedFrame,
    frame_allocation: &mut crate::av1::NativeFrameAllocation,
    alpha_frame: DecodedFrame,
    alpha_allocation: &mut crate::av1::NativeFrameAllocation,
    budget: &mut crate::container::DecodeBudget,
) -> Result<(), DecoderError> {
    validate_strict_alpha(frame, &alpha_frame)?;
    if alpha_frame.buffers.planes.len() != 1 {
        return Err(DecoderError::Bitstream(
            "AVIF alpha auxiliary plane count is invalid".to_string(),
        ));
    }
    frame_allocation.grow_outer_for_alpha(&mut frame.buffers.planes, budget)?;
    let alpha_bit_depth = alpha_frame.bit_depth;
    let mut alpha_planes = alpha_frame.buffers.planes;
    let mut alpha_plane = alpha_planes.pop().ok_or_else(|| {
        DecoderError::Bitstream("AVIF alpha auxiliary plane is missing".to_string())
    })?;
    drop(alpha_planes);
    alpha_allocation.release_outer(budget)?;
    let ticket = alpha_allocation.take_plane_ticket(0)?;
    frame_allocation.adopt_alpha(ticket)?;
    alpha_plane.layout.plane = 3;
    frame.buffers.planes.push(alpha_plane);
    debug_assert_eq!(alpha_bit_depth, frame.bit_depth);
    Ok(())
}

pub(super) fn validate_native_sequence_limits(
    info: &AvifInfo,
    limits: &crate::limits::NativeDecodeLimits,
) -> Result<crate::av1::SequenceHeader, DecoderError> {
    let sequence_payload = super::select_strict_native_sequence_header(&info.primary_item_payload)?;
    let sequence = crate::av1::parse_sequence_header(sequence_payload)?;
    limits.check_dimensions(
        usize::try_from(sequence.max_frame_width).map_err(|_| {
            DecoderError::InvalidParam("AV1 maximum width is too large".to_string())
        })?,
        usize::try_from(sequence.max_frame_height).map_err(|_| {
            DecoderError::InvalidParam("AV1 maximum height is too large".to_string())
        })?,
    )?;
    if sequence.enable_superres {
        return Err(DecoderError::Unsupported(
            "bounded native decode does not yet budget AV1 super-resolution".to_string(),
        ));
    }
    Ok(sequence)
}

fn validate_bounded_item_payload(
    payload: &[u8],
    sequence: &crate::av1::SequenceHeader,
    item_label: &str,
) -> Result<(), DecoderError> {
    let mut displayed_frames = 0usize;
    for obu in crate::obu::ObuIter::new(payload) {
        let obu = obu?;
        if matches!(
            obu.obu_type,
            crate::obu::ObuType::Frame | crate::obu::ObuType::FrameHeader
        ) {
            displayed_frames = displayed_frames.checked_add(1).ok_or_else(|| {
                DecoderError::InvalidParam("AV1 frame count overflows".to_string())
            })?;
            if !sequence.reduced_still_picture_header
                && crate::av1::parse_show_existing_frame_index(obu.payload)?.is_some()
            {
                return Err(DecoderError::Unsupported(
                    "bounded native decode does not yet support show_existing_frame".to_string(),
                ));
            }
        }
    }
    if displayed_frames > 1 {
        return Err(DecoderError::Unsupported(format!(
            "bounded native decode accepts one displayed {item_label} AV1 frame"
        )));
    }
    Ok(())
}

/// Parse only the borrowed AV1 frame header and account for every visible
/// geometry before the materializer is allowed to copy a tile payload.
///
/// This is deliberately separate from `parse_av1_headers`: that routine owns
/// the frame/tile data used by the legacy decoder.  Native callers need the
/// header decision first so a plane limit cannot be discovered after the
/// frame OBU has already been copied.
pub(super) fn validate_native_frame_prefix_limits<'a>(
    info: &'a AvifInfo,
    sequence: &crate::av1::SequenceHeader,
    limits: &crate::limits::NativeDecodeLimits,
) -> Result<crate::av1::FramePrefix<'a, 'static>, DecoderError> {
    let payload = match super::select_strict_native_frame_route(&info.primary_item_payload)? {
        super::StrictNativeFrameRoute::Normal { frame_payload } => frame_payload,
        super::StrictNativeFrameRoute::Split {
            frame_header_payload,
        } => frame_header_payload,
    };
    let sequence_payload = super::select_strict_native_sequence_header(&info.primary_item_payload)?;
    let (_, sequence_metadata) = crate::av1::parse_sequence_header_with_metadata(sequence_payload)?;
    let prefix = crate::av1::parse_frame_prefix(
        payload,
        sequence,
        &sequence_metadata,
        &crate::av1::NO_REFERENCES,
    )?;
    if !prefix.show_frame() {
        return Err(DecoderError::Unsupported(
            "bounded native decode requires a displayed AV1 frame".to_string(),
        ));
    }
    validate_native_frame_prefix_geometry_limits(sequence, &prefix, limits)?;
    Ok(prefix)
}

pub(super) fn validate_native_frame_prefix_geometry_limits(
    sequence: &crate::av1::SequenceHeader,
    prefix: &crate::av1::FramePrefix<'_, '_>,
    limits: &crate::limits::NativeDecodeLimits,
) -> Result<(), DecoderError> {
    let (width, height, upscaled_width, render_width, render_height) = prefix.geometry();
    validate_native_frame_geometry_limits(
        sequence,
        width,
        height,
        upscaled_width,
        render_width,
        render_height,
        limits,
    )
}

fn validate_native_frame_geometry_limits(
    sequence: &crate::av1::SequenceHeader,
    frame_width: u32,
    frame_height: u32,
    upscaled_width: u32,
    render_width: u32,
    render_height: u32,
    limits: &crate::limits::NativeDecodeLimits,
) -> Result<(), DecoderError> {
    let width = usize::try_from(frame_width)
        .map_err(|_| DecoderError::InvalidParam("AV1 frame width is too large".to_string()))?;
    let height = usize::try_from(frame_height)
        .map_err(|_| DecoderError::InvalidParam("AV1 frame height is too large".to_string()))?;
    let upscaled_width = usize::try_from(upscaled_width)
        .map_err(|_| DecoderError::InvalidParam("AV1 upscaled width is too large".to_string()))?;
    let render_width = usize::try_from(render_width)
        .map_err(|_| DecoderError::InvalidParam("AV1 render width is too large".to_string()))?;
    let render_height = usize::try_from(render_height)
        .map_err(|_| DecoderError::InvalidParam("AV1 render height is too large".to_string()))?;
    validate_native_geometry(sequence, width, height, limits, "visible")?;
    validate_native_geometry(sequence, upscaled_width, height, limits, "upscaled")?;
    validate_native_geometry(sequence, render_width, render_height, limits, "render")?;

    let coded_width = width
        .div_ceil(8)
        .checked_mul(8)
        .ok_or_else(|| DecoderError::InvalidParam("AV1 coded width overflows".to_string()))?;
    let coded_height = height
        .div_ceil(8)
        .checked_mul(8)
        .ok_or_else(|| DecoderError::InvalidParam("AV1 coded height overflows".to_string()))?;
    validate_native_geometry(sequence, coded_width, coded_height, limits, "coded")
}

fn validate_native_geometry(
    sequence: &crate::av1::SequenceHeader,
    width: usize,
    height: usize,
    limits: &crate::limits::NativeDecodeLimits,
    label: &str,
) -> Result<(), DecoderError> {
    limits.check_dimensions(width, height)?;
    for plane in 0..3 {
        if let Some(layout) = crate::av1::plane_layout_for_geometry(sequence, width, height, plane)?
        {
            let bytes = layout
                .sample_count
                .checked_mul(std::mem::size_of::<u16>())
                .ok_or_else(|| {
                    DecoderError::InvalidParam(format!("AV1 {label} plane size overflows"))
                })?;
            limits.check_plane_bytes(bytes)?;
        }
    }
    Ok(())
}

pub(super) fn validate_native_decode_plan_limits(
    plan: &FrameDecodePlan,
    limits: &crate::limits::NativeDecodeLimits,
) -> Result<(), DecoderError> {
    limits.check_dimensions(plan.width, plan.height)?;
    limits.check_dimensions(plan.upscaled_width, plan.render_height)?;
    limits.check_dimensions(plan.render_width, plan.render_height)?;
    for layout in &plan.planes {
        let bytes = layout
            .sample_count
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| DecoderError::InvalidParam("native plane size overflows".to_string()))?;
        limits.check_plane_bytes(bytes)?;
        let coded_width = plan.width.div_ceil(8).checked_mul(8).ok_or_else(|| {
            DecoderError::InvalidParam("native coded width overflows".to_string())
        })?;
        let coded_height = plan.height.div_ceil(8).checked_mul(8).ok_or_else(|| {
            DecoderError::InvalidParam("native coded height overflows".to_string())
        })?;
        let coded_width = coded_width.div_ceil(1usize << layout.subsampling_x);
        let coded_height = coded_height.div_ceil(1usize << layout.subsampling_y);
        let coded_bytes = coded_width
            .checked_mul(coded_height)
            .and_then(|samples| samples.checked_mul(std::mem::size_of::<u16>()))
            .ok_or_else(|| {
                DecoderError::InvalidParam("native coded plane size overflows".to_string())
            })?;
        limits.check_plane_bytes(coded_bytes)?;
    }
    Ok(())
}

fn validate_native_frame_limits(
    frame: &DecodedFrame,
    limits: &crate::limits::NativeDecodeLimits,
) -> Result<(), DecoderError> {
    limits.check_dimensions(frame.width, frame.height)?;
    for plane in &frame.buffers.planes {
        let bytes = plane
            .samples
            .len()
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| DecoderError::InvalidParam("native plane size overflows".to_string()))?;
        limits.check_plane_bytes(bytes)?;
    }
    Ok(())
}

pub(super) fn reject_strict_derived_alpha(
    derived_kind: &str,
    has_alpha_grid: bool,
    has_alpha_auxiliary: bool,
) -> Result<(), DecoderError> {
    if has_alpha_grid || has_alpha_auxiliary {
        return Err(DecoderError::Unsupported(format!(
            "strict native decode does not yet support alpha on a derived {derived_kind} image"
        )));
    }
    Ok(())
}

pub(super) fn validate_strict_alpha(
    master: &DecodedFrame,
    alpha: &DecodedFrame,
) -> Result<(), DecoderError> {
    if alpha.width != master.width || alpha.height != master.height {
        return Err(DecoderError::Bitstream(format!(
            "AVIF alpha dimensions {}x{} do not match master dimensions {}x{}",
            alpha.width, alpha.height, master.width, master.height
        )));
    }
    let alpha_plane = alpha.buffers.planes.first().ok_or_else(|| {
        DecoderError::Bitstream("AVIF alpha auxiliary plane is missing".to_string())
    })?;
    if alpha_plane.layout.width != alpha.width || alpha_plane.layout.height != alpha.height {
        return Err(DecoderError::Bitstream(
            "AVIF alpha plane geometry does not match its decoded header".to_string(),
        ));
    }
    if alpha.buffers.planes.len() != 1 || !alpha.color_config.monochrome {
        return Err(DecoderError::Unsupported(
            "AVIF alpha auxiliary image must be monochrome".to_string(),
        ));
    }
    if alpha.bit_depth != master.bit_depth {
        return Err(DecoderError::Unsupported(format!(
            "AVIF alpha bit depth {} does not match master bit depth {}",
            alpha.bit_depth, master.bit_depth
        )));
    }
    if alpha.color_config.color_range != ColorRange::Full {
        return Err(DecoderError::Unsupported(
            "AVIF alpha auxiliary image must use full range".to_string(),
        ));
    }
    Ok(())
}

/// Decodes the AV1 gain-map item referenced by a `tmap` derived image.
///
/// `Ok(None)` means that the input has no `tmap` item. Unsupported gain-map
/// item layouts fail closed while the ordinary [`decode_frame_bytes`] API
/// remains available for the base image.
pub fn decode_gain_map_frame_bytes(
    data: &[u8],
) -> Result<Option<DecodedGainMapFrame>, DecoderError> {
    let Some(gain_map) = parse_gain_map_image(data)? else {
        return Ok(None);
    };
    let info = AvifInfo {
        major_brand: *b"avif",
        compatible_brands: vec![*b"avif"],
        primary_item_id: None,
        width: Some(gain_map.width),
        height: Some(gain_map.height),
        pixel_information: gain_map.pixel_information,
        color_information: gain_map.color_information,
        alpha_premultiplied: false,
        alpha_auxiliary_items: Vec::new(),
        alpha_grid: None,
        primary_grid: gain_map.grid,

        clean_aperture: None,
        rotation: None,
        mirror: None,
        av1_config: gain_map.av1_config,
        primary_item_payload: gain_map.payload,
        sequence_sample_payloads: Vec::new(),
    };
    validate_public_container_preflight(&info, false)?;
    let frame = if info.primary_grid.is_some() {
        decode_grid_frame(&info)?
    } else {
        let headers = parse_av1_headers(&info)?;
        decode_still_frame(&headers, Some(&info))?
    };
    Ok(Some(DecodedGainMapFrame {
        metadata: gain_map.metadata,
        frame,
    }))
}

/// Decodes one sample from an AVIS sequence into source planes.
///
/// Key and intra-only samples are decoded independently while sharing the
/// sequence header from the primary item. A `show_existing_frame` sample can
/// reuse a previously decoded reference slot, and inter/switch samples use the
/// same reference-slot state for reconstruction.
pub fn decode_sequence_frame_bytes(
    data: &[u8],
    frame_index: usize,
) -> Result<DecodedFrame, DecoderError> {
    let info = parse_avif(data)?;
    validate_public_container_preflight(&info, false)?;
    let sequence = parse_avif_sequence(data)?;
    let sample_count = sequence.color_samples.len();
    if frame_index >= sample_count {
        return Err(DecoderError::InvalidParam(format!(
            "AVIS frame index {frame_index} is outside the {sample_count}-sample sequence"
        )));
    }
    let mut tracks = SequenceTracksDecoder::new(info, &sequence)?;
    for index in 0..=frame_index {
        let frame = tracks.next_frame(&sequence, index)?.ok_or_else(|| {
            DecoderError::Bitstream(format!("AVIS sequence ended before sample {frame_index}"))
        })?;
        if index == frame_index {
            return Ok(frame);
        }
    }
    unreachable!("validated AVIS frame index should be returned from the decode loop")
}

/// Decodes every independently addressable AVIS sample into source planes.
///
/// This animation-oriented API accepts Key/IntraOnly, inter/switch, and
/// show-existing samples.
pub fn decode_sequence_frames_bytes(data: &[u8]) -> Result<Vec<DecodedFrame>, DecoderError> {
    let info = parse_avif(data)?;
    validate_public_container_preflight(&info, false)?;
    let sequence = parse_avif_sequence(data)?;
    let mut tracks = SequenceTracksDecoder::new(info, &sequence)?;
    let mut frames = Vec::with_capacity(sequence.color_samples.len());
    for index in 0..sequence.color_samples.len() {
        let frame = tracks.next_frame(&sequence, index)?.ok_or_else(|| {
            DecoderError::Bitstream(format!("AVIS sequence ended before sample {index}"))
        })?;
        frames.push(frame);
    }
    Ok(frames)
}
