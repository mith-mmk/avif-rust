//! Additive native decode result wrappers.

use crate::container::{
    ColorInformationSet, NativePropertyRecord, RichAvifInfo, parse_rich_info_with_limits,
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
    pub fn information(&self) -> &NativeAvifInformation {
        &self.information
    }
}
