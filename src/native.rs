//! Additive native decode result wrappers.

use crate::container::{
    AvifAnimation, AvifSequence, ColorInformationSet, DecodeBudget, NativePropertyRecord,
    RichAvifInfo, parse_rich_info_with_limits, parse_rich_info_with_limits_and_budget,
    parse_rich_info_with_limits_and_budget_sequence,
};
use crate::decoder::DecodedFrame;
use crate::limits::NativeDecodeLimits;

/// Rich metadata and accounting from one bounded native parse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeAvifInformation {
    rich: RichAvifInfo,
    input_bytes: usize,
    metadata_bytes: usize,
    item_count: usize,
    property_count: usize,
    primary_item_type: Option<[u8; 4]>,
    ordered_primary_properties: Vec<NativePropertyRecord>,
}

impl NativeAvifInformation {
    pub(crate) fn new(
        rich: RichAvifInfo,
        input_bytes: usize,
        metadata_bytes: usize,
        item_count: usize,
        property_count: usize,
        primary_item_type: Option<[u8; 4]>,
        ordered_primary_properties: Vec<NativePropertyRecord>,
    ) -> Self {
        Self {
            rich,
            input_bytes,
            metadata_bytes,
            item_count,
            property_count,
            primary_item_type,
            ordered_primary_properties,
        }
    }

    pub fn rich(&self) -> &RichAvifInfo {
        &self.rich
    }
    pub fn info(&self) -> &crate::container::AvifInfo {
        &self.rich.info
    }
    pub fn color_information(&self) -> &ColorInformationSet {
        &self.rich.color_information
    }
    pub const fn input_bytes(&self) -> usize {
        self.input_bytes
    }
    pub const fn metadata_bytes(&self) -> usize {
        self.metadata_bytes
    }
    pub const fn item_count(&self) -> usize {
        self.item_count
    }
    pub const fn property_count(&self) -> usize {
        self.property_count
    }
    pub const fn primary_item_type(&self) -> Option<[u8; 4]> {
        self.primary_item_type
    }
    pub fn ordered_primary_properties(&self) -> &[NativePropertyRecord] {
        &self.ordered_primary_properties
    }

    /// Returns the primary item's ordered `pasp` value, when present.
    pub fn pixel_aspect_ratio(&self) -> Option<(u32, u32)> {
        self.ordered_primary_properties
            .iter()
            .find_map(|property| match property {
                NativePropertyRecord::PixelAspectRatio(ratio) => {
                    Some((ratio.h_spacing, ratio.v_spacing))
                }
                _ => None,
            })
    }
}

/// Performs one bounded rich parse and retains its accounting metadata.
pub fn parse_native_info(
    data: &[u8],
    limits: &NativeDecodeLimits,
) -> Result<NativeAvifInformation, crate::DecoderError> {
    let (rich, stats) = parse_rich_info_with_limits(data, limits)?;
    Ok(NativeAvifInformation::new(
        rich,
        data.len(),
        stats.metadata_bytes,
        stats.item_count,
        stats.property_count,
        stats.primary_item_type,
        stats.ordered_primary_properties,
    ))
}

/// Internal native parse handoff. The public parse API deliberately drops the
/// parser budget at return; strict decode moves it into this private path so
/// retained metadata and selected payload owners share one accounting state.
pub(crate) fn parse_native_info_with_budget(
    data: &[u8],
    limits: &NativeDecodeLimits,
) -> Result<(NativeAvifInformation, DecodeBudget), crate::DecoderError> {
    let (rich, stats, budget) = parse_rich_info_with_limits_and_budget(data, limits)?;
    Ok((
        NativeAvifInformation::new(
            rich,
            data.len(),
            stats.metadata_bytes,
            stats.item_count,
            stats.property_count,
            stats.primary_item_type,
            stats.ordered_primary_properties,
        ),
        budget,
    ))
}

/// Performs one bounded parse for an AVIS sequence and returns the timing
/// handoff together with the retained metadata budget. Compressed samples are
/// moved out of the rich info projection into the animation owner, avoiding a
/// second sample materialization.
pub(crate) fn parse_native_sequence_info_with_budget(
    data: &[u8],
    limits: &NativeDecodeLimits,
) -> Result<(NativeAvifInformation, DecodeBudget, AvifAnimation), crate::DecoderError> {
    let (mut rich, mut stats, budget) =
        parse_rich_info_with_limits_and_budget_sequence(data, limits)?;
    let timing = stats.animation_timing.take().ok_or_else(|| {
        crate::DecoderError::Unsupported("AVIS animation timing is missing".to_string())
    })?;
    let color_timing = timing.color_timing;
    let alpha_timing = timing.alpha_timing;
    let color_durations_ms = timing.color_durations_ms;
    let alpha_durations_ms = timing.alpha_durations_ms;
    let alpha_samples = timing.alpha_samples;
    let sequence = AvifSequence {
        color_samples: std::mem::take(&mut rich.info.sequence_sample_payloads),
        color_durations_ms,
        alpha_samples,
        alpha_durations_ms,
    };
    let animation = AvifAnimation {
        sequence,
        color_timing,
        alpha_timing,
        color_timescale: timing.color_timescale,
        duration_in_timescales: timing.duration_in_timescales,
        repetition_count: timing.repetition_count,
    };
    let information = NativeAvifInformation::new(
        rich,
        data.len(),
        stats.metadata_bytes,
        stats.item_count,
        stats.property_count,
        stats.primary_item_type,
        stats.ordered_primary_properties,
    );
    Ok((information, budget, animation))
}

/// Native decoded frame paired with bounded container metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeDecodedFrame {
    frame: DecodedFrame,
    information: NativeAvifInformation,
}

impl NativeDecodedFrame {
    pub(crate) fn new(frame: DecodedFrame, information: NativeAvifInformation) -> Self {
        Self { frame, information }
    }

    pub fn frame(&self) -> &DecodedFrame {
        &self.frame
    }
    pub fn into_frame(self) -> DecodedFrame {
        self.frame
    }

    /// Transfers the decoded frame and its rich container metadata to a
    /// consumer which needs to build a typed native representation.  The
    /// bounded decoder has already performed the parse and validation; this
    /// handoff therefore does not trigger a second decode or metadata parse.
    pub fn into_frame_and_rich(self) -> (DecodedFrame, RichAvifInfo) {
        (self.frame, self.information.rich)
    }
    pub fn information(&self) -> &NativeAvifInformation {
        &self.information
    }
}
