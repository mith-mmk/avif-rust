//! Checked heap ownership accounting for the private AVIS sequence state.
//!
//! The sequence state is deliberately clone-cheap: a speculative decode
//! clones `Arc` handles and only the allocations which become reachable from
//! the decoded state are new.  This module walks the graph without allocating
//! an image-sized temporary set and deduplicates every Arc by its live address.

use super::{FrameReferenceSlots, ReferenceFrame, SequenceDecodeState};
use crate::DecoderError;
use crate::av1::{CdfContext, FrameBuffers, MotionField, PlaneBuffer};
use crate::container::{AvifInfo, ColorInformation};
use crate::limits::NativeDecodeLimits;
use std::sync::Arc;

const ARC_HEADER_BYTES: usize = 2 * std::mem::size_of::<usize>();
// One transition keeps baseline/candidate color+alpha graphs alive at once.
// The fixed inventory intentionally has room for all four graphs plus the
// maximum eight reference slots in each graph.
const MAX_ARCS: usize = 4 * (1 + 1 + 1 + (8 * 4));

struct SeenArcs {
    pointers: [usize; MAX_ARCS],
    len: usize,
}

impl Default for SeenArcs {
    fn default() -> Self {
        Self {
            pointers: [0; MAX_ARCS],
            len: 0,
        }
    }
}

impl SeenArcs {
    fn insert<T: ?Sized>(&mut self, value: &Arc<T>) -> Result<bool, DecoderError> {
        let pointer = Arc::as_ptr(value).cast::<()>() as usize;
        if self.pointers[..self.len].contains(&pointer) {
            return Ok(false);
        }
        if self.len == self.pointers.len() {
            return Err(DecoderError::InvalidParam(
                "AVIS sequence state Arc inventory overflows".to_string(),
            ));
        }
        self.pointers[self.len] = pointer;
        self.len += 1;
        Ok(true)
    }
}

fn add(total: &mut usize, bytes: usize, label: &str) -> Result<(), DecoderError> {
    *total = total
        .checked_add(bytes)
        .ok_or_else(|| DecoderError::InvalidParam(format!("AVIS {label} size overflows")))?;
    Ok(())
}

fn capacity_bytes<T>(capacity: usize, label: &str) -> Result<usize, DecoderError> {
    capacity
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| DecoderError::InvalidParam(format!("AVIS {label} size overflows")))
}

fn arc_bytes<T>(payload: usize) -> Result<usize, DecoderError> {
    ARC_HEADER_BYTES
        .checked_add(std::mem::size_of::<T>())
        .and_then(|bytes| bytes.checked_add(payload))
        .ok_or_else(|| DecoderError::InvalidParam("AVIS Arc storage overflows".to_string()))
}

fn arc_slice_bytes(len: usize) -> Result<usize, DecoderError> {
    if len == 0 {
        return Ok(0);
    }
    ARC_HEADER_BYTES
        .checked_add(len.checked_mul(std::mem::size_of::<u8>()).ok_or_else(|| {
            DecoderError::InvalidParam("AVIS Arc slice storage overflows".to_string())
        })?)
        .ok_or_else(|| DecoderError::InvalidParam("AVIS Arc slice storage overflows".to_string()))
}

/// Upper bound used before `SequenceDecodeState::new` materializes its
/// sequence-header prefix.  The encoded header is always a subslice-derived
/// OBU no larger than the source payload, so this bound is conservative while
/// remaining independent of any allocation.
pub(super) fn prefix_upper_bound(payload_len: usize) -> Result<usize, DecoderError> {
    if payload_len == 0 {
        return Ok(0);
    }
    ARC_HEADER_BYTES.checked_add(payload_len).ok_or_else(|| {
        DecoderError::InvalidParam("AVIS sequence prefix bound overflows".to_string())
    })
}

fn color_information_bytes(
    color_information: &Option<ColorInformation>,
) -> Result<usize, DecoderError> {
    color_information.as_ref().map_or(Ok(0), |color| {
        capacity_bytes::<u8>(color.payload.capacity(), "reference colour metadata")
    })
}

fn frame_buffers_bytes(buffers: &FrameBuffers) -> Result<usize, DecoderError> {
    let mut bytes = capacity_bytes::<PlaneBuffer>(buffers.planes.capacity(), "reference planes")?;
    for plane in &buffers.planes {
        add(
            &mut bytes,
            capacity_bytes::<u16>(plane.samples.capacity(), "reference plane samples")?,
            "reference frame buffers",
        )?;
    }
    Ok(bytes)
}

fn motion_field_bytes(motion: &MotionField) -> Result<usize, DecoderError> {
    let mut bytes = capacity_bytes::<Option<u8>>(
        motion.reference_frames.capacity(),
        "reference motion frames",
    )?;
    add(
        &mut bytes,
        capacity_bytes::<Option<(i32, i32)>>(motion.motion_vectors.capacity(), "motion vectors")?,
        "reference motion field",
    )?;
    add(
        &mut bytes,
        capacity_bytes::<Option<i32>>(motion.reference_offsets.capacity(), "motion offsets")?,
        "reference motion field",
    )?;
    Ok(bytes)
}

fn cdf_bytes(cdf: &Vec<CdfContext>) -> Result<usize, DecoderError> {
    capacity_bytes::<CdfContext>(cdf.capacity(), "CDF states")
}

fn reference_metadata_bytes(reference: &ReferenceFrame) -> Result<usize, DecoderError> {
    color_information_bytes(&reference.metadata.color_information)
}

fn insert_cdf(
    total: &mut usize,
    seen: &mut SeenArcs,
    cdf: &Arc<Vec<CdfContext>>,
) -> Result<(), DecoderError> {
    if seen.insert(cdf)? {
        add(
            total,
            arc_bytes::<Vec<CdfContext>>(cdf_bytes(cdf)?)?,
            "CDF Arc",
        )?;
    }
    Ok(())
}

fn insert_motion(
    total: &mut usize,
    seen: &mut SeenArcs,
    motion: &Arc<MotionField>,
) -> Result<(), DecoderError> {
    if seen.insert(motion)? {
        add(
            total,
            arc_bytes::<MotionField>(motion_field_bytes(motion)?)?,
            "motion Arc",
        )?;
    }
    Ok(())
}

fn insert_reference(
    total: &mut usize,
    seen: &mut SeenArcs,
    reference: &ReferenceFrame,
) -> Result<(), DecoderError> {
    if seen.insert(&reference.metadata)? {
        add(
            total,
            arc_bytes::<super::ReferenceFrameMetadata>(reference_metadata_bytes(reference)?)?,
            "reference metadata Arc",
        )?;
    }
    if seen.insert(&reference.buffers)? {
        add(
            total,
            arc_bytes::<FrameBuffers>(frame_buffers_bytes(&reference.buffers)?)?,
            "reference buffers Arc",
        )?;
    }
    insert_cdf(total, seen, &reference.cdf_states)?;
    insert_motion(total, seen, &reference.motion_field)
}

fn references_bytes(
    total: &mut usize,
    seen: &mut SeenArcs,
    references: &FrameReferenceSlots,
) -> Result<(), DecoderError> {
    for reference in references.slots.iter().flatten() {
        insert_reference(total, seen, reference)?;
    }
    if let Some(motion) = &references.previous_motion_field
        && seen.insert(motion)?
    {
        add(
            total,
            arc_bytes::<MotionField>(motion_field_bytes(motion)?)?,
            "previous motion Arc",
        )?;
    }
    Ok(())
}

/// Counts the state-owned heap graph using checked actual vector capacities.
/// Arc payloads are charged once even when all eight reference slots share
/// their decoded planes, metadata, or CDF state.
#[cfg(test)]
pub(super) fn state_memory_bytes(state: &SequenceDecodeState) -> Result<usize, DecoderError> {
    let mut total = 0;
    let mut seen = SeenArcs::default();
    account_state(state, &mut seen, &mut total)?;
    Ok(total)
}

fn account_state(
    state: &SequenceDecodeState,
    seen: &mut SeenArcs,
    total: &mut usize,
) -> Result<(), DecoderError> {
    if seen.insert(&state.sequence_prefix)? {
        add(
            total,
            arc_slice_bytes(state.sequence_prefix.len())?,
            "sequence prefix Arc",
        )?;
    }
    if let Some(cdf) = &state.cdf_states {
        insert_cdf(total, seen, cdf)?;
    }
    references_bytes(total, seen, &state.references)
}

fn mark_state(state: &SequenceDecodeState, seen: &mut SeenArcs) -> Result<(), DecoderError> {
    let mut ignored_bytes = 0;
    account_state(state, seen, &mut ignored_bytes)
}

/// Returns only allocations reachable from `candidate` that were not already
/// reachable from `baseline`.  A speculative state clone therefore reports
/// zero, while newly decoded CDF/reference/motion owners report their exact
/// capacity delta even when the old state remains alive for rollback.
#[cfg(test)]
pub(super) fn additional_state_memory_bytes(
    baseline: &SequenceDecodeState,
    candidate: &SequenceDecodeState,
) -> Result<usize, DecoderError> {
    let mut seen = SeenArcs::default();
    mark_state(baseline, &mut seen)?;
    let mut total = 0;
    account_state(candidate, &mut seen, &mut total)?;
    Ok(total)
}

pub(super) fn state_memory_bytes_for_tracks(
    color: &SequenceDecodeState,
    alpha: Option<&SequenceDecodeState>,
) -> Result<usize, DecoderError> {
    let mut seen = SeenArcs::default();
    let mut total = 0;
    account_state(color, &mut seen, &mut total)?;
    if let Some(alpha) = alpha {
        account_state(alpha, &mut seen, &mut total)?;
    }
    Ok(total)
}

pub(super) fn additional_state_memory_bytes_for_tracks(
    baseline_color: &SequenceDecodeState,
    baseline_alpha: Option<&SequenceDecodeState>,
    candidate_color: &SequenceDecodeState,
    candidate_alpha: Option<&SequenceDecodeState>,
) -> Result<usize, DecoderError> {
    let mut seen = SeenArcs::default();
    mark_state(baseline_color, &mut seen)?;
    if let Some(alpha) = baseline_alpha {
        mark_state(alpha, &mut seen)?;
    }
    let mut total = 0;
    account_state(candidate_color, &mut seen, &mut total)?;
    if let Some(alpha) = candidate_alpha {
        account_state(alpha, &mut seen, &mut total)?;
    }
    Ok(total)
}

/// A conservative, allocation-free upper bound for the state owners a frame
/// refresh can introduce.  It is deliberately independent of the current
/// state, because a first key frame can create CDF, reference, and motion
/// owners from an empty state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct StateRefreshPlan {
    pub(super) additional_bytes: usize,
}

impl StateRefreshPlan {
    pub(super) const fn additional_bytes(self) -> usize {
        self.additional_bytes
    }
}

pub(super) fn state_refresh_plan(
    info: &AvifInfo,
    sample: &[u8],
    limits: &NativeDecodeLimits,
) -> Result<StateRefreshPlan, DecoderError> {
    let width = info.width.ok_or_else(|| {
        DecoderError::InvalidParam("AVIS state plan requires frame width".to_string())
    })? as usize;
    let height = info.height.ok_or_else(|| {
        DecoderError::InvalidParam("AVIS state plan requires frame height".to_string())
    })? as usize;
    // Reference buffers are allocated in coded/upscaled coordinates.  The
    // native sequence header can request super-resolution, whose dimensions
    // are at most twice the displayed dimensions.  Use the current retained
    // track geometry rather than the caller's maximum dimensions: a large
    // limit must not turn every small sample into a large speculative
    // allocation.  The native parser has already checked this geometry, but
    // keep the check here so direct strict-state planning remains bounded.
    limits.check_dimensions(width, height)?;
    let width = width
        .checked_mul(2)
        .ok_or_else(|| DecoderError::InvalidParam("AVIS state plan width overflows".to_string()))?;
    let height = height.checked_mul(2).ok_or_else(|| {
        DecoderError::InvalidParam("AVIS state plan height overflows".to_string())
    })?;
    let mi_cols = width
        .checked_add(3)
        .ok_or_else(|| DecoderError::InvalidParam("AVIS state plan width overflows".to_string()))?
        / 4;
    let mi_rows = height.checked_add(3).ok_or_else(|| {
        DecoderError::InvalidParam("AVIS state plan height overflows".to_string())
    })? / 4;
    let mi_count = mi_cols.checked_mul(mi_rows).ok_or_else(|| {
        DecoderError::InvalidParam("AVIS state plan motion count overflows".to_string())
    })?;
    let tile_cols = width
        .checked_add(63)
        .ok_or_else(|| DecoderError::InvalidParam("AVIS state plan width overflows".to_string()))?
        / 64;
    let tile_rows = height.checked_add(63).ok_or_else(|| {
        DecoderError::InvalidParam("AVIS state plan height overflows".to_string())
    })? / 64;
    let tile_count = tile_cols.checked_mul(tile_rows).ok_or_else(|| {
        DecoderError::InvalidParam("AVIS state plan tile count overflows".to_string())
    })?;

    let cdf_bytes = capacity_bytes::<CdfContext>(tile_count.max(1), "planned CDF states")?;
    // One root CDF plus one CDF owner per refreshable reference slot.  Keep
    // one spare owner for a transient CDF vector while a multi-unit sample
    // is being assembled.
    let cdf_arc = arc_bytes::<Vec<CdfContext>>(cdf_bytes)?;
    let mut bytes = cdf_arc.checked_mul(10).ok_or_else(|| {
        DecoderError::InvalidParam("AVIS state plan CDF owners overflow".to_string())
    })?;
    add(&mut bytes, cdf_bytes, "planned reference CDF states")?;
    let frame_samples = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .and_then(|samples| samples.checked_mul(std::mem::size_of::<u16>()))
        .ok_or_else(|| DecoderError::InvalidParam("AVIS state plan frame overflows".to_string()))?;
    let frame_buffers = arc_bytes::<FrameBuffers>(
        capacity_bytes::<PlaneBuffer>(4, "planned reference planes")?
            .checked_add(frame_samples)
            .ok_or_else(|| {
                DecoderError::InvalidParam("AVIS state plan frame overflows".to_string())
            })?,
    )?;
    add(
        &mut bytes,
        frame_buffers.checked_mul(9).ok_or_else(|| {
            DecoderError::InvalidParam("AVIS state plan reference buffers overflow".to_string())
        })?,
        "planned reference buffers",
    )?;
    let metadata = arc_bytes::<super::ReferenceFrameMetadata>(color_information_bytes(
        &info.color_information,
    )?)?;
    add(
        &mut bytes,
        metadata.checked_mul(9).ok_or_else(|| {
            DecoderError::InvalidParam("AVIS state plan reference metadata overflow".to_string())
        })?,
        "planned reference metadata",
    )?;
    let motion_frames = capacity_bytes::<Option<u8>>(mi_count, "planned motion frames")?;
    let motion_vectors = capacity_bytes::<Option<(i32, i32)>>(mi_count, "planned motion vectors")?;
    let motion_offsets = capacity_bytes::<Option<i32>>(mi_count, "planned motion offsets")?;
    let motion_vectors = motion_frames
        .checked_add(motion_vectors)
        .and_then(|value| value.checked_add(motion_offsets))
        .ok_or_else(|| {
            DecoderError::InvalidParam("AVIS state plan motion overflows".to_string())
        })?;
    let motion_arc = arc_bytes::<MotionField>(motion_vectors)?;
    // One previous-motion owner and one owner per refreshable reference slot,
    // plus one spare owner for a second in-flight unit's temporal field.
    add(
        &mut bytes,
        motion_arc.checked_mul(10).ok_or_else(|| {
            DecoderError::InvalidParam("AVIS state plan motion overflows".to_string())
        })?,
        "planned motion owners",
    )?;
    add(
        &mut bytes,
        super::strict_sequence_split_plan(sample)?.scratch_bytes()?,
        "planned sequence unit scratch",
    )?;
    Ok(StateRefreshPlan {
        additional_bytes: bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av1::{ColorConfig, ColorRange, FrameType, GlobalMotionParams, PlaneLayout};

    fn minimal_info(width: u32, height: u32) -> AvifInfo {
        AvifInfo {
            major_brand: *b"avis",
            compatible_brands: Vec::new(),
            primary_item_id: Some(1),
            width: Some(width),
            height: Some(height),
            pixel_information: None,
            color_information: None,
            alpha_premultiplied: false,
            alpha_auxiliary_items: Vec::new(),
            alpha_grid: None,
            primary_grid: None,
            clean_aperture: None,
            rotation: None,
            mirror: None,
            av1_config: None,
            primary_item_payload: Vec::new(),
            sequence_sample_payloads: Vec::new(),
        }
    }

    fn plan_limits(max_width: usize, max_height: usize) -> NativeDecodeLimits {
        NativeDecodeLimits::new(
            usize::MAX,
            max_width,
            max_height,
            usize::MAX,
            usize::MAX,
            usize::MAX,
            usize::MAX,
            usize::MAX,
            usize::MAX,
            usize::MAX,
            usize::MAX,
            usize::MAX,
        )
    }

    fn sample_with_sequence_prefix(payload_len: usize, frame_count: usize) -> Vec<u8> {
        let mut sample = vec![0x0a, payload_len as u8];
        sample.extend(std::iter::repeat(0x55).take(payload_len));
        for _ in 0..frame_count {
            sample.extend([0x32, 1, 0]);
        }
        sample
    }

    fn state_with_shared_reference_slots(slot_count: usize) -> SequenceDecodeState {
        let metadata = Arc::new(super::super::ReferenceFrameMetadata {
            width: 1,
            height: 1,
            render_width: 1,
            render_height: 1,
            bit_depth: 8,
            color_config: ColorConfig {
                high_bitdepth: false,
                twelve_bit: false,
                bit_depth: 8,
                monochrome: true,
                color_description: None,
                color_range: ColorRange::Full,
                subsampling_x: false,
                subsampling_y: false,
                chroma_sample_position: None,
                separate_uv_delta_q: false,
            },
            color_information: None,
            alpha_premultiplied: false,
        });
        let buffers = Arc::new(FrameBuffers {
            width: 1,
            height: 1,
            planes: vec![PlaneBuffer {
                layout: PlaneLayout {
                    plane: 0,
                    width: 1,
                    height: 1,
                    subsampling_x: 0,
                    subsampling_y: 0,
                    sample_count: 1,
                },
                samples: vec![0],
            }],
        });
        let cdf_states = Arc::new(vec![CdfContext::new(0)]);
        let motion_field = Arc::new(MotionField::empty(1, 1));
        let reference = || super::super::ReferenceFrame {
            metadata: Arc::clone(&metadata),
            buffers: Arc::clone(&buffers),
            frame_width: 1,
            frame_height: 1,
            upscaled_width: 1,
            render_width: 1,
            render_height: 1,
            order_hint: 0,
            frame_type: FrameType::Key,
            film_grain: None,
            frame_id: None,
            global_motion: GlobalMotionParams::default(),
            cdf_states: Arc::clone(&cdf_states),
            motion_field: Arc::clone(&motion_field),
        };
        let mut state = SequenceDecodeState {
            sequence_prefix: Arc::from(vec![1, 2].into_boxed_slice()),
            references: FrameReferenceSlots::default(),
            cdf_states: Some(Arc::clone(&cdf_states)),
            next_sample_index: 0,
            #[cfg(test)]
            decoded_sample_count: 0,
        };
        state.references.previous_motion_field = Some(Arc::clone(&motion_field));
        for slot in state.references.slots.iter_mut().take(slot_count) {
            *slot = Some(reference());
        }
        state
    }

    #[test]
    fn arc_pointer_inventory_deduplicates_and_checks_overflow() {
        let bytes = Arc::new(vec![CdfContext::new(0); 2]);
        let mut seen = SeenArcs::default();
        assert!(seen.insert(&bytes).unwrap());
        assert!(!seen.insert(&bytes).unwrap());
        assert_eq!(seen.len, 1);
        assert_eq!(
            cdf_bytes(&bytes).unwrap(),
            2 * std::mem::size_of::<CdfContext>()
        );
        assert!(capacity_bytes::<u16>(usize::MAX, "test").is_err());
    }

    #[test]
    fn state_inventory_deduplicates_shared_reference_arcs() {
        let one = state_memory_bytes(&state_with_shared_reference_slots(1)).unwrap();
        let eight = state_memory_bytes(&state_with_shared_reference_slots(8)).unwrap();
        assert_eq!(one, eight, "shared Arc payloads must be charged once");
    }

    #[test]
    fn state_clone_has_zero_additional_owned_bytes() {
        let state = state_with_shared_reference_slots(8);
        let clone = state.clone();
        assert_eq!(additional_state_memory_bytes(&state, &clone).unwrap(), 0);
    }

    #[test]
    fn color_and_alpha_graphs_share_one_arc_inventory() {
        let baseline = state_with_shared_reference_slots(8);
        let candidate = baseline.clone();
        let one = state_memory_bytes_for_tracks(&baseline, Some(&baseline)).unwrap();
        let separate = state_memory_bytes_for_tracks(&baseline, None).unwrap();
        assert_eq!(
            one, separate,
            "color and alpha shared Arcs must be charged once"
        );
        assert_eq!(
            additional_state_memory_bytes_for_tracks(
                &baseline,
                Some(&baseline),
                &candidate,
                Some(&candidate),
            )
            .unwrap(),
            0,
            "a cross-track clone must not duplicate shared owners"
        );
    }

    #[test]
    fn four_state_transition_inventory_has_no_valid_graph_overflow() {
        let baseline_color = state_with_shared_reference_slots(8);
        let baseline_alpha = state_with_shared_reference_slots(8);
        let candidate_color = state_with_shared_reference_slots(8);
        let candidate_alpha = state_with_shared_reference_slots(8);
        assert!(
            additional_state_memory_bytes_for_tracks(
                &baseline_color,
                Some(&baseline_alpha),
                &candidate_color,
                Some(&candidate_alpha),
            )
            .is_ok()
        );
    }

    #[test]
    fn state_inventory_reports_new_cdf_owner_after_decode() {
        let baseline = state_with_shared_reference_slots(1);
        let mut candidate = baseline.clone();
        candidate.cdf_states = Some(Arc::new(vec![CdfContext::new(1); 3]));
        let delta = additional_state_memory_bytes(&baseline, &candidate).unwrap();
        assert_eq!(
            delta,
            arc_bytes::<Vec<CdfContext>>(3 * std::mem::size_of::<CdfContext>()).unwrap()
        );
    }

    #[test]
    fn strict_split_rejects_extension_header_without_touching_legacy_splitter() {
        // OBU sequence header, extension flag, one-byte extension value,
        // one-byte payload.  The strict path must not rebuild this OBU while
        // the legacy splitter remains the compatibility path.
        let sample = [0x0e, 0x07, 0x01, 0x00];
        let error = super::super::strict_sequence_split_plan(&sample)
            .expect_err("strict split must reject extension OBUs explicitly");
        assert!(matches!(
            error,
            DecoderError::Unsupported(message)
                if message.contains("extension headers")
        ));
        assert!(super::super::split_av1_sequence_sample(&sample).is_ok());
    }

    #[test]
    fn strict_split_overcapacity_drops_candidate_and_retries_exact_plan() {
        let sample = sample_with_sequence_prefix(1, 2);
        let plan = super::super::strict_sequence_split_plan(&sample).unwrap();
        let planned_bytes = plan.scratch_bytes().unwrap();
        let mut budget = crate::container::DecodeBudget::new(Some(planned_bytes));
        let token = budget
            .reserve_existing_bytes(
                crate::allocation::AllocationClass::Frame,
                planned_bytes,
                "strict split test plan",
            )
            .unwrap();
        let before = budget.accounting();
        let error = match super::super::strict_split_with_capacity_extra_for_test(&sample, 1) {
            Ok(_) => panic!("a second-pass capacity overage must be rejected"),
            Err(error) => error,
        };
        assert!(
            matches!(error, DecoderError::InvalidParam(message) if message.contains("exceeded"))
        );
        assert_eq!(budget.accounting(), before);

        let (retry_plan, retry) =
            super::super::strict_split_with_capacity_extra_for_test(&sample, 0)
                .expect("exact planned capacities must retry");
        assert_eq!(retry_plan, plan);
        assert_eq!(retry.actual_bytes().unwrap(), planned_bytes);
        drop(retry);
        let mut token = token;
        budget.release_token(&mut token).unwrap();
        assert_eq!(budget.accounting().aggregate_live, 0);
    }

    #[test]
    fn state_inventory_addition_is_checked() {
        let mut total = usize::MAX;
        assert!(add(&mut total, 1, "test").is_err());
    }

    #[test]
    fn refresh_plan_is_conservative_and_checked_before_decode() {
        let plan = state_refresh_plan(&minimal_info(64, 32), &[0], &plan_limits(64, 32)).unwrap();
        let metadata_only =
            state_refresh_plan(&minimal_info(1, 1), &[0], &plan_limits(1, 1)).unwrap();
        assert!(plan.additional_bytes() > 0);
        assert!(
            plan.additional_bytes() > metadata_only.additional_bytes(),
            "strict geometry limits must affect the preflight plan"
        );
        let under = crate::container::DecodeBudget::new(Some(plan.additional_bytes() - 1));
        assert!(
            under
                .check_additional_frame(plan.additional_bytes())
                .is_err()
        );
        assert_eq!(under.accounting().aggregate_live, 0);
        assert!(
            state_refresh_plan(
                &minimal_info(u32::MAX, u32::MAX),
                &[0],
                &plan_limits(64, 32),
            )
            .is_err()
        );
    }

    #[test]
    fn sequence_unit_plan_uses_each_sample_prefix_without_allocation() {
        let limits = plan_limits(1, 1);
        let short = sample_with_sequence_prefix(1, 1);
        let large = sample_with_sequence_prefix(64, 2);
        let split = super::super::strict_sequence_split_plan(&large).unwrap();
        assert_eq!(split.coded_unit_count, 2);
        assert_eq!(split.outer_capacity, split.coded_unit_count);
        assert!(split.max_current_encoded_len > split.sequence_prefix_encoded_len);
        let short_plan = state_refresh_plan(&minimal_info(1, 1), &short, &limits).unwrap();
        let large_plan = state_refresh_plan(&minimal_info(1, 1), &large, &limits).unwrap();
        assert!(large_plan.additional_bytes() > short_plan.additional_bytes());
        let (_, requests) = crate::test_allocation_observer::count_allocation_requests(|| {
            state_refresh_plan(&minimal_info(1, 1), &large, &limits)
        });
        assert_eq!(requests, 0, "refresh planning must remain allocation-free");
    }

    #[test]
    fn strict_split_materialization_uses_the_two_pass_plan() {
        let sample = sample_with_sequence_prefix(12, 3);
        let plan = super::super::strict_sequence_split_plan(&sample).unwrap();
        let (planned, requests) =
            crate::test_allocation_observer::count_allocation_requests(|| {
                super::super::strict_sequence_split_plan(&sample)
            });
        assert_eq!(requests, 0, "strict split planning must stay borrowed");
        assert_eq!(planned.unwrap(), plan);

        let (plan, split) = super::super::split_av1_sequence_sample_strict(&sample).unwrap();
        assert_eq!(split.units.len(), plan.coded_unit_count);
        assert_eq!(split.actual_bytes, plan.scratch_bytes().unwrap());
        assert_eq!(split.units.capacity(), plan.outer_capacity);
        assert!(
            split
                .units
                .iter()
                .all(|unit| unit.payload.capacity() >= unit.payload.len())
        );
    }
}
