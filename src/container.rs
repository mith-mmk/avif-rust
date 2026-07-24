use std::borrow::Cow;

use crate::DecoderError;
use crate::obu::{ObuType, parse_obu_stream};

const BRAND_AVIF: &[u8; 4] = b"avif";
const BRAND_AVIS: &[u8; 4] = b"avis";
const BRAND_AVIO: &[u8; 4] = b"avio";
const ALPHA_AUX_TYPE: &str = "urn:mpeg:mpegB:cicp:systems:auxiliary:alpha";

/// The coded-frame kind found in one AVIS track sample.
///
/// This is intentionally a header-level classification. It does not claim
/// that the decoder can reconstruct inter frames yet, but lets callers audit
/// a sequence without treating all non-Key samples as opaque bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AvifSequenceSampleKind {
    Key,
    Inter,
    IntraOnly,
    Switch,
    ShowExisting { frame_to_show_map_idx: u8 },
}

/// Header-only metadata used by the decoder when it needs to inspect a
/// sequence sample before reconstructing it. Keeping the OBU scan result
/// together avoids reparsing the same sample to detect both its frame kind and
/// an optional per-sample sequence header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AvifSequenceSampleInfo {
    pub kind: Option<AvifSequenceSampleKind>,
    pub has_sequence_header: bool,
}

/// Classifies the first coded-frame OBU in one AVIS track sample.
pub fn classify_av1_sequence_sample(
    payload: &[u8],
) -> Result<Option<AvifSequenceSampleKind>, DecoderError> {
    Ok(inspect_av1_sequence_sample(payload)?.kind)
}

/// Inspects one AVIS sample once, retaining the small amount of metadata that
/// the sequence decoder needs for classification and header composition.
pub(crate) fn inspect_av1_sequence_sample(
    payload: &[u8],
) -> Result<AvifSequenceSampleInfo, DecoderError> {
    let obus = parse_obu_stream(payload)?;
    let has_sequence_header = obus
        .iter()
        .any(|obu| obu.obu_type == ObuType::SequenceHeader);
    let Some(frame) = obus
        .iter()
        .find(|obu| matches!(obu.obu_type, ObuType::Frame | ObuType::FrameHeader))
    else {
        return Ok(AvifSequenceSampleInfo {
            kind: None,
            has_sequence_header,
        });
    };
    let first = *frame
        .payload
        .first()
        .ok_or_else(|| DecoderError::NotEnoughData("AV1 frame header is empty".to_string()))?;
    if first & 0x80 != 0 {
        return Ok(AvifSequenceSampleInfo {
            kind: Some(AvifSequenceSampleKind::ShowExisting {
                frame_to_show_map_idx: (first >> 4) & 0x07,
            }),
            has_sequence_header,
        });
    }
    Ok(AvifSequenceSampleInfo {
        kind: Some(match (first >> 5) & 0x03 {
            0 => AvifSequenceSampleKind::Key,
            1 => AvifSequenceSampleKind::Inter,
            2 => AvifSequenceSampleKind::IntraOnly,
            3 => AvifSequenceSampleKind::Switch,
            _ => unreachable!("two-bit AV1 frame type is always in range"),
        }),
        has_sequence_header,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BoxHeader {
    offset: usize,
    size: usize,
    header_size: usize,
    box_type: [u8; 4],
}

/// Parsed AVIF information needed before AV1 sample decode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AvifInfo {
    pub major_brand: [u8; 4],
    pub compatible_brands: Vec<[u8; 4]>,
    pub primary_item_id: Option<u32>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub pixel_information: Option<PixelInformation>,
    pub color_information: Option<ColorInformation>,
    pub alpha_premultiplied: bool,
    pub alpha_auxiliary_items: Vec<AuxiliaryImage>,
    pub alpha_grid: Option<GridImage>,
    pub primary_grid: Option<GridImage>,
    pub clean_aperture: Option<CleanAperture>,
    pub rotation: Option<ImageRotation>,
    pub mirror: Option<ImageMirror>,
    pub av1_config: Option<Vec<u8>>,
    pub primary_item_payload: Vec<u8>,
    /// AV1 samples following the primary item in an AVIS image sequence.
    ///
    /// The primary item remains in `primary_item_payload`; these samples are
    /// exposed separately so still-image callers do not accidentally decode a
    /// sequence as one concatenated OBU stream.
    pub sequence_sample_payloads: Vec<Vec<u8>>,
}

impl AvifInfo {
    pub fn is_avif_brand(&self) -> bool {
        brand_is_avif(&self.major_brand) || self.compatible_brands.iter().any(brand_is_avif)
    }
}

/// `ispe` image spatial extents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSpatialExtents {
    pub width: u32,
    pub height: u32,
}

/// `pixi` pixel information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PixelInformation {
    pub bits_per_channel: Vec<u8>,
    /// Optional channel descriptors from the extended `pixi` form.
    ///
    /// The ordinary AVIF form has no channel descriptors. When the `pixi`
    /// full-box flags signal the extended form, each channel carries its
    /// component format and (when present) chroma subsampling location.
    pub extended_channels: Option<Vec<PixelChannelInformation>>,
}

/// One channel descriptor carried by an extended `pixi` property.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PixelChannelInformation {
    /// ISO/IEC 23008-12 channel identifier (0 is colour or grayscale).
    pub channel_idc: u8,
    /// ISO/IEC 23001-17 component format (0 is unsigned integer).
    pub component_format: u8,
    pub subsampling: Option<PixelSubsampling>,
}

/// Chroma subsampling type and sample location from an extended `pixi` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PixelSubsampling {
    pub subsampling_type: u8,
    pub subsampling_location: u8,
}

/// `colr` color information box payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColorInformation {
    pub color_type: [u8; 4],
    pub payload: Vec<u8>,
}

impl ColorInformation {
    pub fn nclx(&self) -> Option<NclxColorInformation> {
        if &self.color_type != b"nclx" || self.payload.len() < 7 {
            return None;
        }
        Some(NclxColorInformation {
            color_primaries: u16::from_be_bytes([self.payload[0], self.payload[1]]),
            transfer_characteristics: u16::from_be_bytes([self.payload[2], self.payload[3]]),
            matrix_coefficients: u16::from_be_bytes([self.payload[4], self.payload[5]]),
            full_range_flag: self.payload[6] & 0x80 != 0,
        })
    }

    pub fn icc_profile(&self) -> Option<&[u8]> {
        if &self.color_type == b"prof" || &self.color_type == b"rICC" {
            Some(&self.payload)
        } else {
            None
        }
    }
}

/// Structured `nclx` CICP colour information from an AVIF `colr` property.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NclxColorInformation {
    pub color_primaries: u16,
    pub transfer_characteristics: u16,
    pub matrix_coefficients: u16,
    pub full_range_flag: bool,
}

/// Auxiliary image item payload, such as an AVIF alpha plane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuxiliaryImage {
    pub item_id: u32,
    pub aux_type: String,
    pub payload: Vec<u8>,
}

/// A signed or unsigned rational value from an ISO 21496-1 gain-map
/// descriptor. Unsigned wire numerators are widened to `i64` for one stable
/// public representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GainMapRational {
    pub numerator: i64,
    pub denominator: u32,
}

/// One channel's gain-map conversion parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GainMapChannel {
    pub gain_map_min: GainMapRational,
    pub gain_map_max: GainMapRational,
    pub gamma: GainMapRational,
    pub base_offset: GainMapRational,
    pub alternate_offset: GainMapRational,
}

/// Parsed ISO 21496-1 Annex C.2 metadata carried by a `tmap` item.
///
/// This is metadata-only support: public RGBA decode continues to return the
/// base image until a display-headroom-aware composition API is added.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GainMapMetadata {
    pub minimum_version: u16,
    pub writer_version: u16,
    pub is_multichannel: bool,
    pub use_base_colour_space: bool,
    pub backward_direction: bool,
    pub base_hdr_headroom: GainMapRational,
    pub alternate_hdr_headroom: GainMapRational,
    pub channels: Vec<GainMapChannel>,
}

impl GainMapMetadata {
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }
}

/// Decodable AV1 gain-map item paired with a `tmap` descriptor.
///
/// The item is exposed separately from the base image so callers can choose
/// whether and how much HDR headroom to apply. The default RGBA API continues
/// to return the base image unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GainMapImage {
    pub metadata: GainMapMetadata,
    pub width: u32,
    pub height: u32,
    pub pixel_information: PixelInformation,
    pub color_information: Option<ColorInformation>,
    pub av1_config: Vec<u8>,
    pub payload: Vec<u8>,
}

/// Parsed AVIF image grid derived item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GridImage {
    pub item_id: u32,
    pub rows: u8,
    pub columns: u8,
    pub output_width: u32,
    pub output_height: u32,
    pub payload: Vec<u8>,
    pub cells: Vec<GridCell>,
}

/// One AV1 image item referenced by a `grid` derived image item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GridCell {
    pub item_id: u32,
    pub width: u32,
    pub height: u32,
    pub pixel_information: Option<PixelInformation>,
    pub color_information: Option<ColorInformation>,
    pub av1_config: Option<Vec<u8>>,
    pub payload: Vec<u8>,
}

/// AVIF 1.2 Sample Transform (`sato`) expression associated with a derived
/// image item. This stays crate-private because it is an implementation detail
/// of the still-image decoder rather than a new public container API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SampleTransform {
    pub output_width: u32,
    pub output_height: u32,
    pub output_bit_depth: u8,
    pub intermediate_bit_depth: u8,
    pub tokens: Vec<SampleTransformToken>,
    pub inputs: Vec<SampleTransformInput>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SampleTransformToken {
    Constant(i64),
    Input(usize),
    Unary(u8),
    Binary(u8),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SampleTransformInput {
    pub item_id: u32,
    pub width: u32,
    pub height: u32,
    pub pixel_information: PixelInformation,
    pub color_information: Option<ColorInformation>,
    pub av1_config: Vec<u8>,
    pub payload: Vec<u8>,
    /// A `grid` input is composed before the sample-transform expression is
    /// evaluated. Plain `av01` inputs keep this as `None`.
    pub grid: Option<GridImage>,
}

/// `clap` clean aperture item property.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CleanAperture {
    pub width_n: u32,
    pub width_d: u32,
    pub height_n: u32,
    pub height_d: u32,
    pub horizontal_offset_n: u32,
    pub horizontal_offset_d: u32,
    pub vertical_offset_n: u32,
    pub vertical_offset_d: u32,
}

/// `irot` image rotation item property.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageRotation {
    pub angle: u8,
}

/// `imir` image mirror item property.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageMirror {
    pub axis: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ItemLocation {
    item_id: u32,
    base_offset: u64,
    extents: Vec<ItemExtent>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ItemExtent {
    offset: u64,
    length: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ItemReference {
    reference_type: [u8; 4],
    from_item_id: u32,
    to_item_ids: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ItemPropertyAssociation {
    item_id: u32,
    associations: Vec<PropertyAssociation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PropertyAssociation {
    index: u16,
    essential: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ItemInfo {
    item_id: u32,
    item_type: [u8; 4],
    item_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ItemProperty {
    AuxiliaryType(String),
    CleanAperture(CleanAperture),
    Rotation(ImageRotation),
    Mirror(ImageMirror),
    SpatialExtents(ImageSpatialExtents),
    PixelInformation(PixelInformation),
    Av1Config(Vec<u8>),
    ColorInformation(ColorInformation),
    Premultiplied,
    OperatingPointSelector(u8),
    LayerSelector(u16),
    LayerIndexing([u64; 3]),
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PropertyKind {
    AuxiliaryType,
    CleanAperture,
    Rotation,
    Mirror,
    SpatialExtents,
    PixelInformation,
    Av1Config,
    ColorInformation,
    Premultiplied,
    OperatingPointSelector,
    LayerSelector,
    LayerIndexing,
    Other,
}

impl PropertyKind {
    fn is_singleton(self) -> bool {
        !matches!(self, Self::Other)
    }
}

fn property_kind(property: &ItemProperty) -> PropertyKind {
    match property {
        ItemProperty::AuxiliaryType(_) => PropertyKind::AuxiliaryType,
        ItemProperty::CleanAperture(_) => PropertyKind::CleanAperture,
        ItemProperty::Rotation(_) => PropertyKind::Rotation,
        ItemProperty::Mirror(_) => PropertyKind::Mirror,
        ItemProperty::SpatialExtents(_) => PropertyKind::SpatialExtents,
        ItemProperty::PixelInformation(_) => PropertyKind::PixelInformation,
        ItemProperty::Av1Config(_) => PropertyKind::Av1Config,
        ItemProperty::ColorInformation(_) => PropertyKind::ColorInformation,
        ItemProperty::Premultiplied => PropertyKind::Premultiplied,
        ItemProperty::OperatingPointSelector(_) => PropertyKind::OperatingPointSelector,
        ItemProperty::LayerSelector(_) => PropertyKind::LayerSelector,
        ItemProperty::LayerIndexing(_) => PropertyKind::LayerIndexing,
        ItemProperty::Other => PropertyKind::Other,
    }
}

#[derive(Debug, Default)]
struct MetaState {
    primary_item_id: Option<u32>,
    item_locations: Vec<ItemLocation>,
    item_construction_methods: Vec<(u32, u16)>,
    item_extent_indexes: Vec<(u32, Vec<u64>)>,
    idat_payload: Option<Vec<u8>>,
    item_infos: Vec<ItemInfo>,
    item_references: Vec<ItemReference>,
    item_property_associations: Vec<ItemPropertyAssociation>,
    item_properties: Vec<ItemProperty>,
}

pub fn is_avif_file(data: &[u8]) -> bool {
    parse_ftyp(data).is_ok_and(|(major_brand, compatible_brands)| {
        brand_is_avif(&major_brand) || compatible_brands.iter().any(brand_is_avif)
    })
}

pub fn parse_avif(data: &[u8]) -> Result<AvifInfo, DecoderError> {
    let (major_brand, compatible_brands) = parse_ftyp(data)?;
    if !brand_is_avif(&major_brand) && !compatible_brands.iter().any(brand_is_avif) {
        return Err(DecoderError::Bitstream(
            "file type box does not advertise AVIF".to_string(),
        ));
    }

    let mut meta = MetaState::default();
    for_each_top_level_box(data, |header| {
        match &header.box_type {
            b"meta" => parse_meta(data, header, &mut meta)?,
            // The primary still image item remains independently decodable in
            // an AVIS file. The movie box describes later frames; the public
            // still-image API intentionally selects the primary item.
            b"moov" => {}
            _ => {}
        }
        Ok(())
    })?;

    let decode_primary_item_id = effective_primary_item_id(&meta)?;
    let primary_item_payload = item_payload(data, &meta, decode_primary_item_id)?;
    let sequence_sample_payloads =
        sequence_sample_payloads(data, &major_brand, &compatible_brands)?;
    validate_primary_item_metadata(&meta)?;
    let alpha_auxiliary_items =
        alpha_auxiliary_items_for(data, &meta, Some(decode_primary_item_id))?;
    let primary_grid = primary_grid(data, &primary_item_payload, &meta)?;
    let alpha_grid = alpha_grid(data, &alpha_auxiliary_items, &meta)?;
    let primary_metadata = item_metadata(&meta, decode_primary_item_id)?;
    Ok(AvifInfo {
        major_brand,
        compatible_brands,
        primary_item_id: meta.primary_item_id,
        width: primary_metadata.width,
        height: primary_metadata.height,
        pixel_information: primary_metadata.pixel_information,
        color_information: primary_metadata.color_information,
        alpha_premultiplied: primary_metadata.alpha_premultiplied,
        alpha_auxiliary_items,
        alpha_grid,
        primary_grid,
        clean_aperture: primary_metadata.clean_aperture,
        rotation: primary_metadata.rotation,
        mirror: primary_metadata.mirror,
        av1_config: primary_metadata.av1_config,
        primary_item_payload,
        sequence_sample_payloads,
    })
}

/// Parses the first `tmap` item descriptor in an AVIF file.
///
/// `None` means that no `tmap` item is present. An unknown descriptor version
/// is reported as [`DecoderError::Unsupported`] so callers can deliberately
/// fall back to the base image, while malformed supported-version payloads
/// are rejected as bitstream errors.
pub fn parse_gain_map_metadata(data: &[u8]) -> Result<Option<GainMapMetadata>, DecoderError> {
    let Some((meta, item_id)) = gain_map_meta_state(data)? else {
        return Ok(None);
    };
    let payload = item_payload(data, &meta, item_id)?;
    Ok(Some(parse_gain_map_metadata_payload(&payload)?))
}

fn gain_map_meta_state(data: &[u8]) -> Result<Option<(MetaState, u32)>, DecoderError> {
    if !data.windows(4).any(|window| window == b"tmap") {
        return Ok(None);
    }
    let mut meta = MetaState::default();
    for_each_top_level_box(data, |header| {
        if &header.box_type == b"meta" {
            parse_meta(data, header, &mut meta)?;
        }
        Ok(())
    })?;
    let Some(item_id) = meta
        .item_infos
        .iter()
        .find(|item| item.item_type == *b"tmap")
        .map(|item| item.item_id)
    else {
        return Ok(None);
    };
    Ok(Some((meta, item_id)))
}

/// Locates and parses the AV1 image referenced as the second `tmap` input.
pub(crate) fn parse_gain_map_image(data: &[u8]) -> Result<Option<GainMapImage>, DecoderError> {
    let Some((meta, tmap_id)) = gain_map_meta_state(data)? else {
        return Ok(None);
    };
    let reference = meta
        .item_references
        .iter()
        .find(|reference| reference.reference_type == *b"dimg" && reference.from_item_id == tmap_id)
        .ok_or_else(|| {
            DecoderError::Bitstream("tmap item is missing dimg input references".to_string())
        })?;
    if reference.to_item_ids.len() != 2 {
        return Err(DecoderError::Bitstream(
            "tmap item must reference one base image and one gain map".to_string(),
        ));
    }
    let base_id = reference.to_item_ids[0];
    let gain_map_id = reference.to_item_ids[1];
    let gain_map_item = meta
        .item_infos
        .iter()
        .find(|item| item.item_id == gain_map_id)
        .ok_or_else(|| {
            DecoderError::Bitstream(format!(
                "tmap gain map item {gain_map_id} is missing item info"
            ))
        })?;
    if gain_map_item.item_type != *b"av01" {
        return Err(DecoderError::Unsupported(format!(
            "tmap gain map item {gain_map_id} has unsupported type {:?}",
            gain_map_item.item_type
        )));
    }
    item_metadata(&meta, base_id)?;
    let gain_metadata = item_metadata(&meta, gain_map_id)?;
    let width = gain_metadata.width.ok_or_else(|| {
        DecoderError::Bitstream("tmap gain map item is missing ispe dimensions".to_string())
    })?;
    let height = gain_metadata.height.ok_or_else(|| {
        DecoderError::Bitstream("tmap gain map item is missing ispe dimensions".to_string())
    })?;
    if width == 0 || height == 0 {
        return Err(DecoderError::Bitstream(
            "tmap gain map dimensions must be non-zero".to_string(),
        ));
    }
    let pixel_information = gain_metadata
        .pixel_information
        .ok_or_else(|| DecoderError::Bitstream("tmap gain map item is missing pixi".to_string()))?;
    let av1_config = gain_metadata
        .av1_config
        .ok_or_else(|| DecoderError::Bitstream("tmap gain map item is missing av1C".to_string()))?;
    let tmap_payload = item_payload(data, &meta, tmap_id)?;
    let metadata = parse_gain_map_metadata_payload(&tmap_payload)?;
    Ok(Some(GainMapImage {
        metadata,
        width,
        height,
        pixel_information,
        color_information: gain_metadata.color_information,
        av1_config,
        payload: item_payload(data, &meta, gain_map_id)?,
    }))
}

fn parse_gain_map_metadata_payload(payload: &[u8]) -> Result<GainMapMetadata, DecoderError> {
    let mut reader = GainMapReader::new(payload);
    let minimum_version = reader.read_u16("minimum_version")?;
    if minimum_version != 0 {
        return Err(DecoderError::Unsupported(format!(
            "gain map metadata minimum version {minimum_version} is not supported"
        )));
    }
    let writer_version = reader.read_u16("writer_version")?;
    if writer_version < minimum_version {
        return Err(DecoderError::Bitstream(
            "gain map writer version is below the minimum version".to_string(),
        ));
    }
    let flags = reader.read_u8("flags")?;
    let is_multichannel = flags & 0x80 != 0;
    let use_base_colour_space = flags & 0x40 != 0;
    let backward_direction = flags & 0x04 != 0;
    let use_common_denominator = flags & 0x08 != 0;
    let channel_count = if is_multichannel { 3 } else { 1 };

    let (base_hdr_headroom, alternate_hdr_headroom, channels) = if use_common_denominator {
        let denominator = reader.read_nonzero_u32("common denominator")?;
        let base_hdr_headroom = GainMapRational {
            numerator: i64::from(reader.read_u32("base HDR headroom numerator")?),
            denominator,
        };
        let alternate_hdr_headroom = GainMapRational {
            numerator: i64::from(reader.read_u32("alternate HDR headroom numerator")?),
            denominator,
        };
        let channels = (0..channel_count)
            .map(|_| parse_gain_map_channel_common(&mut reader, denominator))
            .collect::<Result<Vec<_>, _>>()?;
        (base_hdr_headroom, alternate_hdr_headroom, channels)
    } else {
        let base_hdr_headroom = reader.read_unsigned_rational("base HDR headroom")?;
        let alternate_hdr_headroom = reader.read_unsigned_rational("alternate HDR headroom")?;
        let channels = (0..channel_count)
            .map(|_| parse_gain_map_channel(&mut reader))
            .collect::<Result<Vec<_>, _>>()?;
        (base_hdr_headroom, alternate_hdr_headroom, channels)
    };

    if base_hdr_headroom == alternate_hdr_headroom {
        return Err(DecoderError::Bitstream(
            "gain map base and alternate HDR headroom must differ".to_string(),
        ));
    }
    for channel in &channels {
        let max = i128::from(channel.gain_map_max.numerator)
            * i128::from(channel.gain_map_min.denominator);
        let min = i128::from(channel.gain_map_min.numerator)
            * i128::from(channel.gain_map_max.denominator);
        if max < min {
            return Err(DecoderError::Bitstream(
                "gain map maximum is below gain map minimum".to_string(),
            ));
        }
        if channel.gamma.numerator <= 0 {
            return Err(DecoderError::Bitstream(
                "gain map gamma numerator must be non-zero".to_string(),
            ));
        }
    }

    Ok(GainMapMetadata {
        minimum_version,
        writer_version,
        is_multichannel,
        use_base_colour_space,
        backward_direction,
        base_hdr_headroom,
        alternate_hdr_headroom,
        channels,
    })
}

fn parse_gain_map_channel_common(
    reader: &mut GainMapReader<'_>,
    denominator: u32,
) -> Result<GainMapChannel, DecoderError> {
    Ok(GainMapChannel {
        gain_map_min: GainMapRational {
            numerator: i64::from(reader.read_i32("gain map minimum numerator")?),
            denominator,
        },
        gain_map_max: GainMapRational {
            numerator: i64::from(reader.read_i32("gain map maximum numerator")?),
            denominator,
        },
        gamma: GainMapRational {
            numerator: i64::from(reader.read_u32("gain map gamma numerator")?),
            denominator,
        },
        base_offset: GainMapRational {
            numerator: i64::from(reader.read_i32("base offset numerator")?),
            denominator,
        },
        alternate_offset: GainMapRational {
            numerator: i64::from(reader.read_i32("alternate offset numerator")?),
            denominator,
        },
    })
}

fn parse_gain_map_channel(reader: &mut GainMapReader<'_>) -> Result<GainMapChannel, DecoderError> {
    Ok(GainMapChannel {
        gain_map_min: reader.read_signed_rational("gain map minimum")?,
        gain_map_max: reader.read_signed_rational("gain map maximum")?,
        gamma: reader.read_unsigned_rational("gain map gamma")?,
        base_offset: reader.read_signed_rational("base offset")?,
        alternate_offset: reader.read_signed_rational("alternate offset")?,
    })
}

struct GainMapReader<'a> {
    payload: &'a [u8],
    offset: usize,
}

impl<'a> GainMapReader<'a> {
    fn new(payload: &'a [u8]) -> Self {
        Self { payload, offset: 0 }
    }

    fn read_bytes(&mut self, count: usize, name: &str) -> Result<&'a [u8], DecoderError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| DecoderError::Bitstream(format!("gain map {name} offset overflows")))?;
        let bytes = self
            .payload
            .get(self.offset..end)
            .ok_or_else(|| DecoderError::NotEnoughData(format!("gain map {name} is truncated")))?;
        self.offset = end;
        Ok(bytes)
    }

    fn read_u8(&mut self, name: &str) -> Result<u8, DecoderError> {
        Ok(self.read_bytes(1, name)?[0])
    }

    fn read_u16(&mut self, name: &str) -> Result<u16, DecoderError> {
        Ok(u16::from_be_bytes(
            self.read_bytes(2, name)?.try_into().unwrap(),
        ))
    }

    fn read_u32(&mut self, name: &str) -> Result<u32, DecoderError> {
        Ok(u32::from_be_bytes(
            self.read_bytes(4, name)?.try_into().unwrap(),
        ))
    }

    fn read_i32(&mut self, name: &str) -> Result<i32, DecoderError> {
        Ok(i32::from_be_bytes(
            self.read_bytes(4, name)?.try_into().unwrap(),
        ))
    }

    fn read_nonzero_u32(&mut self, name: &str) -> Result<u32, DecoderError> {
        let value = self.read_u32(name)?;
        if value == 0 {
            return Err(DecoderError::Bitstream(format!(
                "gain map {name} must be non-zero"
            )));
        }
        Ok(value)
    }

    fn read_unsigned_rational(&mut self, name: &str) -> Result<GainMapRational, DecoderError> {
        let numerator = self.read_u32(&format!("{name} numerator"))?;
        let denominator = self.read_nonzero_u32(&format!("{name} denominator"))?;
        Ok(GainMapRational {
            numerator: i64::from(numerator),
            denominator,
        })
    }

    fn read_signed_rational(&mut self, name: &str) -> Result<GainMapRational, DecoderError> {
        let numerator = self.read_i32(&format!("{name} numerator"))?;
        let denominator = self.read_nonzero_u32(&format!("{name} denominator"))?;
        Ok(GainMapRational {
            numerator: i64::from(numerator),
            denominator,
        })
    }
}

/// Parses the first `sato` derived item that references the primary image.
///
/// AVIF permits multiple alternative derived items. The still decoder only
/// selects a transform whose `dimg` inputs include the primary item. This
/// keeps the still-image API deterministic while allowing the input order
/// prescribed by the file to place hidden/residual inputs before the primary.
pub(crate) fn parse_sample_transform(data: &[u8]) -> Result<Option<SampleTransform>, DecoderError> {
    // Avoid a second full metadata walk for the overwhelmingly common case.
    // `sato` is an item type, so this cheap byte preflight is only a hint; the
    // structured parser below still validates every box and reference.
    if !data.windows(4).any(|window| window == b"sato") {
        return Ok(None);
    }
    let mut meta = MetaState::default();
    for_each_top_level_box(data, |header| {
        if &header.box_type == b"meta" {
            parse_meta(data, header, &mut meta)?;
        }
        Ok(())
    })?;
    let Some(primary_item_id) = meta.primary_item_id else {
        return Ok(None);
    };
    let primary_item_is_sato = meta
        .item_infos
        .iter()
        .any(|item| item.item_id == primary_item_id && item.item_type == *b"sato");
    let Some(sato) = meta.item_infos.iter().find(|item| {
        item.item_type == *b"sato"
            && (item.item_id == primary_item_id
                || meta.item_references.iter().any(|reference| {
                    reference.reference_type == *b"dimg"
                        && reference.from_item_id == item.item_id
                        && reference.to_item_ids.contains(&primary_item_id)
                }))
    }) else {
        return Ok(None);
    };
    let inputs = meta
        .item_references
        .iter()
        .find(|reference| {
            reference.reference_type == *b"dimg" && reference.from_item_id == sato.item_id
        })
        .map(|reference| reference.to_item_ids.clone())
        .ok_or_else(|| {
            DecoderError::Bitstream("sato item is missing dimg input references".to_string())
        })?;
    if inputs.is_empty() {
        return Err(DecoderError::Bitstream(
            "sato input reference list is empty".to_string(),
        ));
    }
    if !primary_item_is_sato && !inputs.contains(&primary_item_id) {
        return Err(DecoderError::Unsupported(
            "sato inputs do not reference the primary item".to_string(),
        ));
    }

    let output_metadata = item_metadata(&meta, sato.item_id)?;
    let output_width = output_metadata.width.ok_or_else(|| {
        DecoderError::Bitstream("sato item is missing ispe dimensions".to_string())
    })?;
    let output_height = output_metadata.height.ok_or_else(|| {
        DecoderError::Bitstream("sato item is missing ispe dimensions".to_string())
    })?;
    let output_bit_depth = uniform_bit_depth(output_metadata.pixel_information.as_ref())?;
    if !(8..=16).contains(&output_bit_depth) {
        return Err(DecoderError::Unsupported(format!(
            "sato output bit depth {output_bit_depth} is not supported"
        )));
    }
    let mut transform_inputs = Vec::with_capacity(inputs.len());
    for item_id in inputs {
        let item = meta
            .item_infos
            .iter()
            .find(|item| item.item_id == item_id)
            .ok_or_else(|| {
                DecoderError::Bitstream(format!("sato input item {item_id} is missing item info"))
            })?;
        if item.item_type != *b"av01" && item.item_type != *b"grid" {
            return Err(DecoderError::Unsupported(format!(
                "sato input item {item_id} has unsupported type {:?}",
                item.item_type
            )));
        }
        let metadata = item_metadata(&meta, item_id)?;
        let width = metadata.width.ok_or_else(|| {
            DecoderError::Bitstream(format!("sato input item {item_id} is missing ispe"))
        })?;
        let height = metadata.height.ok_or_else(|| {
            DecoderError::Bitstream(format!("sato input item {item_id} is missing ispe"))
        })?;
        let metadata_pixel_information = metadata.pixel_information;
        let metadata_av1_config = metadata.av1_config;
        let color_information = metadata.color_information;
        let grid = if item.item_type == *b"grid" {
            let payload = item_payload(data, &meta, item_id)?;
            Some(parse_grid_item(data, &payload, &meta, item_id)?)
        } else {
            None
        };
        let av1_config = match metadata_av1_config {
            Some(config) => config,
            None => grid
                .as_ref()
                .and_then(|grid| grid.cells.first())
                .and_then(|cell| cell.av1_config.clone())
                .ok_or_else(|| {
                    DecoderError::Bitstream(format!("sato input item {item_id} is missing av1C"))
                })?,
        };
        let pixel_information = match metadata_pixel_information {
            Some(pixel_information) => pixel_information,
            None => grid
                .as_ref()
                .and_then(|grid| grid.cells.first())
                .and_then(|cell| cell.pixel_information.clone())
                .ok_or_else(|| {
                    DecoderError::Bitstream(format!("sato input item {item_id} is missing pixi"))
                })?,
        };
        transform_inputs.push(SampleTransformInput {
            item_id,
            width,
            height,
            pixel_information,
            color_information,
            av1_config,
            payload: item_payload(data, &meta, item_id)?,
            grid,
        });
    }
    let payload = item_payload(data, &meta, sato.item_id)?;
    let (intermediate_bit_depth, tokens) = parse_sample_transform_payload(&payload)?;
    Ok(Some(SampleTransform {
        output_width,
        output_height,
        output_bit_depth,
        intermediate_bit_depth,
        tokens,
        inputs: transform_inputs,
    }))
}

fn uniform_bit_depth(pixi: Option<&PixelInformation>) -> Result<u8, DecoderError> {
    let pixi =
        pixi.ok_or_else(|| DecoderError::Bitstream("pixi property is missing".to_string()))?;
    let Some(&depth) = pixi.bits_per_channel.first() else {
        return Err(DecoderError::Bitstream("pixi has no channels".to_string()));
    };
    if pixi.bits_per_channel.iter().any(|value| *value != depth) {
        return Err(DecoderError::Unsupported(
            "sato requires a uniform bit depth across planes".to_string(),
        ));
    }
    Ok(depth)
}

fn parse_sample_transform_payload(
    payload: &[u8],
) -> Result<(u8, Vec<SampleTransformToken>), DecoderError> {
    if payload.len() < 2 {
        return Err(DecoderError::NotEnoughData(
            "sato payload is missing its header".to_string(),
        ));
    }
    let header = payload[0];
    let version = header >> 6;
    if version != 0 || header & 0x3c != 0 {
        return Err(DecoderError::Unsupported(
            "unsupported sato version or reserved header bits".to_string(),
        ));
    }
    let intermediate_bit_depth = match header & 0x03 {
        0 => 8,
        1 => 16,
        2 => 32,
        _ => 64,
    };
    let token_count = usize::from(payload[1]);
    if token_count == 0 || payload.len() < token_count + 2 {
        return Err(DecoderError::Bitstream(
            "sato token count exceeds payload".to_string(),
        ));
    }
    let mut cursor = 2usize;
    let mut stack_depth = 0usize;
    let mut tokens = Vec::with_capacity(token_count);
    for _ in 0..token_count {
        let token = payload[cursor];
        cursor += 1;
        match token {
            0 => {
                let bytes = usize::from(intermediate_bit_depth / 8);
                let end = cursor.checked_add(bytes).ok_or_else(|| {
                    DecoderError::Bitstream("sato constant length overflow".to_string())
                })?;
                if end > payload.len() {
                    return Err(DecoderError::NotEnoughData(
                        "sato constant is truncated".to_string(),
                    ));
                }
                let value = match bytes {
                    1 => i64::from(i8::from_be_bytes([payload[cursor]])),
                    2 => i64::from(i16::from_be_bytes([payload[cursor], payload[cursor + 1]])),
                    4 => i64::from(i32::from_be_bytes([
                        payload[cursor],
                        payload[cursor + 1],
                        payload[cursor + 2],
                        payload[cursor + 3],
                    ])),
                    8 => i64::from_be_bytes([
                        payload[cursor],
                        payload[cursor + 1],
                        payload[cursor + 2],
                        payload[cursor + 3],
                        payload[cursor + 4],
                        payload[cursor + 5],
                        payload[cursor + 6],
                        payload[cursor + 7],
                    ]),
                    _ => unreachable!("sato intermediate depth is byte-aligned"),
                };
                cursor = end;
                tokens.push(SampleTransformToken::Constant(value));
                stack_depth += 1;
            }
            1..=32 => {
                tokens.push(SampleTransformToken::Input(usize::from(token - 1)));
                stack_depth += 1;
            }
            64..=67 => {
                if stack_depth < 1 {
                    return Err(DecoderError::Bitstream(
                        "sato unary stack underflow".to_string(),
                    ));
                }
                tokens.push(SampleTransformToken::Unary(token - 64));
            }
            128..=137 => {
                if stack_depth < 2 {
                    return Err(DecoderError::Bitstream(
                        "sato binary stack underflow".to_string(),
                    ));
                }
                stack_depth -= 1;
                tokens.push(SampleTransformToken::Binary(token - 128));
            }
            _ => {
                return Err(DecoderError::Unsupported(format!(
                    "reserved sato token {token}"
                )));
            }
        }
    }
    if stack_depth != 1 || cursor != payload.len() {
        return Err(DecoderError::Bitstream(
            "sato expression does not leave exactly one result".to_string(),
        ));
    }
    Ok((intermediate_bit_depth, tokens))
}

fn brand_is_avif(brand: &[u8; 4]) -> bool {
    brand == BRAND_AVIF || brand == BRAND_AVIS || brand == BRAND_AVIO
}

fn parse_ftyp(data: &[u8]) -> Result<([u8; 4], Vec<[u8; 4]>), DecoderError> {
    let header = read_box_header(data, 0, data.len())?;
    if &header.box_type != b"ftyp" {
        return Err(DecoderError::Bitstream("first box is not ftyp".to_string()));
    }
    let payload = box_payload(data, header)?;
    if payload.len() < 8 {
        return Err(DecoderError::NotEnoughData(
            "ftyp payload is too short".to_string(),
        ));
    }
    let major_brand = read_fourcc(payload, 0)?;
    let mut compatible_brands = Vec::new();
    let mut offset = 8;
    while offset + 4 <= payload.len() {
        compatible_brands.push(read_fourcc(payload, offset)?);
        offset += 4;
    }
    Ok((major_brand, compatible_brands))
}

fn for_each_top_level_box<F>(data: &[u8], mut callback: F) -> Result<(), DecoderError>
where
    F: FnMut(BoxHeader) -> Result<(), DecoderError>,
{
    let mut offset = 0;
    while offset < data.len() {
        let header = read_box_header(data, offset, data.len())?;
        if header.size == 0 {
            return Err(DecoderError::Bitstream(
                "zero-sized top-level box".to_string(),
            ));
        }
        offset = checked_add(header.offset, header.size, "top-level box end")?;
        callback(header)?;
    }
    Ok(())
}

fn parse_meta(data: &[u8], header: BoxHeader, state: &mut MetaState) -> Result<(), DecoderError> {
    let payload = box_payload(data, header)?;
    if payload.len() < 4 {
        return Err(DecoderError::NotEnoughData(
            "meta full-box header is missing".to_string(),
        ));
    }
    parse_meta_children(data, payload, 4, state)
}

fn parse_meta_children(
    source: &[u8],
    payload: &[u8],
    mut offset: usize,
    state: &mut MetaState,
) -> Result<(), DecoderError> {
    while offset < payload.len() {
        let header = read_box_header(payload, offset, payload.len())?;
        let child_payload = box_payload(payload, header)?;
        match &header.box_type {
            b"pitm" => state.primary_item_id = parse_pitm(child_payload)?,
            b"iinf" => state.item_infos = parse_iinf(child_payload)?,
            b"iloc" => {
                let (locations, construction_methods, extent_indexes) =
                    parse_iloc_with_indexes(child_payload)?;
                state.item_locations = locations;
                state.item_construction_methods = construction_methods;
                state.item_extent_indexes = extent_indexes;
            }
            b"idat" => state.idat_payload = Some(child_payload.to_vec()),
            b"iref" => state.item_references = parse_iref(child_payload)?,
            b"iprp" => parse_iprp(source, child_payload, state)?,
            _ => {}
        }
        offset = checked_add(header.offset, header.size, "meta child box end")?;
    }
    Ok(())
}

fn parse_iprp(source: &[u8], payload: &[u8], state: &mut MetaState) -> Result<(), DecoderError> {
    let mut offset = 0;
    while offset < payload.len() {
        let header = read_box_header(payload, offset, payload.len())?;
        let child_payload = box_payload(payload, header)?;
        match &header.box_type {
            b"ipco" => parse_ipco(source, child_payload, state)?,
            b"ipma" => merge_ipma(
                &mut state.item_property_associations,
                parse_ipma(child_payload)?,
            )?,
            _ => {}
        }
        offset = checked_add(header.offset, header.size, "iprp child box end")?;
    }
    Ok(())
}

fn parse_ipco(_source: &[u8], payload: &[u8], state: &mut MetaState) -> Result<(), DecoderError> {
    let mut offset = 0;
    while offset < payload.len() {
        let header = read_box_header(payload, offset, payload.len())?;
        let child_payload = box_payload(payload, header)?;
        let property = match &header.box_type {
            b"auxC" => ItemProperty::AuxiliaryType(parse_auxc(child_payload)?),
            b"clap" => ItemProperty::CleanAperture(parse_clap(child_payload)?),
            b"irot" => ItemProperty::Rotation(parse_irot(child_payload)?),
            b"imir" => ItemProperty::Mirror(parse_imir(child_payload)?),
            b"ispe" => ItemProperty::SpatialExtents(parse_ispe(child_payload)?),
            b"pixi" => ItemProperty::PixelInformation(parse_pixi(child_payload)?),
            b"av1C" => ItemProperty::Av1Config(child_payload.to_vec()),
            b"colr" => ItemProperty::ColorInformation(parse_colr(child_payload)?),
            b"prem" => ItemProperty::Premultiplied,
            b"a1op" => ItemProperty::OperatingPointSelector(parse_a1op(child_payload)?),
            b"lsel" => ItemProperty::LayerSelector(parse_lsel(child_payload)?),
            b"a1lx" => ItemProperty::LayerIndexing(parse_a1lx(child_payload)?),
            _ => ItemProperty::Other,
        };
        state.item_properties.push(property);
        offset = checked_add(header.offset, header.size, "ipco child box end")?;
    }
    Ok(())
}

fn parse_pitm(payload: &[u8]) -> Result<Option<u32>, DecoderError> {
    if payload.len() < 6 {
        return Err(DecoderError::NotEnoughData(
            "pitm payload is too short".to_string(),
        ));
    }
    let version = payload[0];
    let item_id = match version {
        0 => read_u16(payload, 4)? as u32,
        1 => read_u32(payload, 4)?,
        _ => {
            return Err(DecoderError::Unsupported(format!(
                "pitm version {version} is not supported"
            )));
        }
    };
    Ok(Some(item_id))
}

fn parse_iinf(payload: &[u8]) -> Result<Vec<ItemInfo>, DecoderError> {
    if payload.len() < 6 {
        return Err(DecoderError::NotEnoughData(
            "iinf payload is too short".to_string(),
        ));
    }
    let version = payload[0];
    if version > 1 {
        return Err(DecoderError::Unsupported(format!(
            "iinf version {version} is not supported"
        )));
    }
    let (entry_count, mut offset) = if version == 0 {
        (read_u16(payload, 4)? as usize, 6usize)
    } else {
        (read_u32(payload, 4)? as usize, 8usize)
    };
    validate_collection_count(
        entry_count,
        payload.len().saturating_sub(offset),
        8,
        "iinf entry",
    )?;
    let mut infos = Vec::with_capacity(entry_count);
    for _ in 0..entry_count {
        let header = read_box_header(payload, offset, payload.len())?;
        let child_payload = box_payload(payload, header)?;
        if &header.box_type == b"infe" {
            infos.push(parse_infe(child_payload)?);
        }
        offset = checked_add(header.offset, header.size, "iinf child box end")?;
    }
    Ok(infos)
}

fn parse_infe(payload: &[u8]) -> Result<ItemInfo, DecoderError> {
    if payload.len() < 12 {
        return Err(DecoderError::NotEnoughData(
            "infe payload is too short".to_string(),
        ));
    }
    let version = payload[0];
    let mut cursor = 4usize;
    let item_id = match version {
        2 => {
            let value = read_u16(payload, cursor)? as u32;
            cursor += 2;
            value
        }
        3 => {
            let value = read_u32(payload, cursor)?;
            cursor += 4;
            value
        }
        _ => {
            return Err(DecoderError::Unsupported(format!(
                "infe version {version} is not supported"
            )));
        }
    };
    let _item_protection_index = read_u16(payload, cursor)?;
    cursor += 2;
    let item_type = read_fourcc(payload, cursor)?;
    cursor += 4;
    let item_name = read_c_string(payload, cursor)?;
    Ok(ItemInfo {
        item_id,
        item_type,
        item_name,
    })
}

#[cfg(test)]
fn parse_iloc(payload: &[u8]) -> Result<Vec<ItemLocation>, DecoderError> {
    parse_iloc_with_methods(payload).map(|(locations, _)| locations)
}

#[cfg(test)]
fn parse_iloc_with_methods(
    payload: &[u8],
) -> Result<(Vec<ItemLocation>, Vec<(u32, u16)>), DecoderError> {
    parse_iloc_with_indexes(payload).map(|(locations, methods, _)| (locations, methods))
}

fn parse_iloc_with_indexes(
    payload: &[u8],
) -> Result<(Vec<ItemLocation>, Vec<(u32, u16)>, Vec<(u32, Vec<u64>)>), DecoderError> {
    if payload.len() < 8 {
        return Err(DecoderError::NotEnoughData(
            "iloc payload is too short".to_string(),
        ));
    }
    let version = payload[0];
    if version > 2 {
        return Err(DecoderError::Unsupported(format!(
            "iloc version {version} is not supported"
        )));
    }

    let size_byte = payload[4];
    let offset_size = (size_byte >> 4) & 0x0f;
    let length_size = size_byte & 0x0f;
    let base_byte = payload[5];
    let base_offset_size = (base_byte >> 4) & 0x0f;
    let index_size = if version == 1 || version == 2 {
        base_byte & 0x0f
    } else {
        0
    };
    validate_field_size(offset_size, "iloc offset_size")?;
    validate_field_size(length_size, "iloc length_size")?;
    validate_field_size(base_offset_size, "iloc base_offset_size")?;
    validate_field_size(index_size, "iloc index_size")?;

    let (item_count, mut cursor) = if version < 2 {
        (read_u16(payload, 6)? as usize, 8usize)
    } else {
        (read_u32(payload, 6)? as usize, 10usize)
    };

    let item_id_size = if version < 2 { 2usize } else { 4usize };
    let construction_method_size = usize::from(version == 1 || version == 2) * 2;
    let minimum_item_size = item_id_size
        .checked_add(construction_method_size)
        .and_then(|size| size.checked_add(2))
        .and_then(|size| size.checked_add(usize::from(base_offset_size)))
        .and_then(|size| size.checked_add(2))
        .ok_or_else(|| DecoderError::Bitstream("iloc item size overflow".to_string()))?;
    validate_collection_count(
        item_count,
        payload.len().saturating_sub(cursor),
        minimum_item_size,
        "iloc item",
    )?;

    let mut locations = Vec::with_capacity(item_count);
    let mut construction_methods = Vec::with_capacity(item_count);
    let mut all_extent_indexes = Vec::with_capacity(item_count);
    for _ in 0..item_count {
        let item_id = if version < 2 {
            let value = read_u16(payload, cursor)? as u32;
            cursor += 2;
            value
        } else {
            let value = read_u32(payload, cursor)?;
            cursor += 4;
            value
        };

        let construction_method = if version == 1 || version == 2 {
            let construction_method = read_u16(payload, cursor)? & 0x000f;
            cursor += 2;
            if construction_method > 2 {
                return Err(DecoderError::Unsupported(format!(
                    "iloc construction_method {construction_method} is not supported"
                )));
            }
            construction_method
        } else {
            0
        };
        construction_methods.push((item_id, construction_method));

        let data_reference_index = read_u16(payload, cursor)?;
        cursor += 2;
        if data_reference_index != 0 {
            return Err(DecoderError::Unsupported(
                "external AVIF item data references are not supported".to_string(),
            ));
        }

        let base_offset = read_sized_int(payload, &mut cursor, base_offset_size)?;
        let extent_count = read_u16(payload, cursor)? as usize;
        cursor += 2;
        let minimum_extent_size = usize::from(index_size)
            .checked_add(usize::from(offset_size))
            .and_then(|size| size.checked_add(usize::from(length_size)))
            .ok_or_else(|| DecoderError::Bitstream("iloc extent size overflow".to_string()))?;
        if minimum_extent_size > 0 {
            validate_collection_count(
                extent_count,
                payload.len().saturating_sub(cursor),
                minimum_extent_size,
                "iloc extent",
            )?;
        }
        let mut extents = Vec::with_capacity(extent_count);
        let mut extent_indexes = Vec::with_capacity(extent_count);
        for _ in 0..extent_count {
            let extent_index = if version == 1 || version == 2 {
                let value = read_sized_int(payload, &mut cursor, index_size)?;
                if construction_method == 2 && index_size != 0 && value == 0 {
                    return Err(DecoderError::Bitstream(
                        "iloc item_offset extent index is zero".to_string(),
                    ));
                }
                if index_size == 0 { 1 } else { value }
            } else {
                1
            };
            let offset = read_sized_int(payload, &mut cursor, offset_size)?;
            let length = read_sized_int(payload, &mut cursor, length_size)?;
            extent_indexes.push(extent_index);
            extents.push(ItemExtent { offset, length });
        }
        locations.push(ItemLocation {
            item_id,
            base_offset,
            extents,
        });
        all_extent_indexes.push((item_id, extent_indexes));
    }

    Ok((locations, construction_methods, all_extent_indexes))
}

fn parse_iref(payload: &[u8]) -> Result<Vec<ItemReference>, DecoderError> {
    if payload.len() < 4 {
        return Err(DecoderError::NotEnoughData(
            "iref full-box header is missing".to_string(),
        ));
    }
    let version = payload[0];
    if version > 1 {
        return Err(DecoderError::Unsupported(format!(
            "iref version {version} is not supported"
        )));
    }
    let large_ids = version == 1;
    let mut references = Vec::new();
    let mut offset = 4;
    while offset < payload.len() {
        let header = read_box_header(payload, offset, payload.len())?;
        let child_payload = box_payload(payload, header)?;
        let mut cursor = 0usize;
        let from_item_id = read_item_id(child_payload, &mut cursor, large_ids)?;
        let reference_count = read_u16(child_payload, cursor)? as usize;
        cursor += 2;
        let mut to_item_ids = Vec::with_capacity(reference_count);
        for _ in 0..reference_count {
            to_item_ids.push(read_item_id(child_payload, &mut cursor, large_ids)?);
        }
        references.push(ItemReference {
            reference_type: header.box_type,
            from_item_id,
            to_item_ids,
        });
        offset = checked_add(header.offset, header.size, "iref child box end")?;
    }
    Ok(references)
}

fn parse_ipma(payload: &[u8]) -> Result<Vec<ItemPropertyAssociation>, DecoderError> {
    if payload.len() < 8 {
        return Err(DecoderError::NotEnoughData(
            "ipma payload is too short".to_string(),
        ));
    }
    let version = payload[0];
    if version > 1 {
        return Err(DecoderError::Unsupported(format!(
            "ipma version {version} is not supported"
        )));
    }
    let flags = read_u24(payload, 1)?;
    let large_property_index = flags & 1 != 0;
    let entry_count = read_u32(payload, 4)? as usize;
    let mut cursor = 8usize;
    let item_id_size = if version == 1 { 4usize } else { 2usize };
    validate_collection_count(
        entry_count,
        payload.len().saturating_sub(cursor),
        item_id_size + 1,
        "ipma entry",
    )?;
    let mut item_associations = Vec::with_capacity(entry_count);
    for _ in 0..entry_count {
        let item_id = read_item_id(payload, &mut cursor, version == 1)?;
        let association_count = read_u8(payload, cursor)? as usize;
        cursor += 1;
        validate_collection_count(
            association_count,
            payload.len().saturating_sub(cursor),
            if large_property_index { 2 } else { 1 },
            "ipma association",
        )?;
        let mut property_associations = Vec::with_capacity(association_count);
        for _ in 0..association_count {
            let (index, essential) = if large_property_index {
                let value = read_u16(payload, cursor)?;
                cursor += 2;
                (value & 0x7fff, value & 0x8000 != 0)
            } else {
                let value = read_u8(payload, cursor)?;
                cursor += 1;
                (u16::from(value & 0x7f), value & 0x80 != 0)
            };
            property_associations.push(PropertyAssociation { index, essential });
        }
        item_associations.push(ItemPropertyAssociation {
            item_id,
            associations: property_associations,
        });
    }
    Ok(item_associations)
}

fn merge_ipma(
    target: &mut Vec<ItemPropertyAssociation>,
    incoming: Vec<ItemPropertyAssociation>,
) -> Result<(), DecoderError> {
    for incoming_item in incoming {
        let Some(existing) = target
            .iter_mut()
            .find(|association| association.item_id == incoming_item.item_id)
        else {
            target.push(incoming_item);
            continue;
        };
        for incoming_property in incoming_item.associations {
            if existing
                .associations
                .iter()
                .any(|property| property.index == incoming_property.index)
            {
                return Err(DecoderError::Bitstream(format!(
                    "item {} has duplicate property association index {}",
                    existing.item_id, incoming_property.index
                )));
            }
            existing.associations.push(incoming_property);
        }
    }
    Ok(())
}

fn validate_primary_item_metadata(state: &MetaState) -> Result<(), DecoderError> {
    let primary_item_id = effective_primary_item_id(state)?;
    if !state
        .item_infos
        .iter()
        .any(|item| item.item_id == primary_item_id)
    {
        return Err(DecoderError::Bitstream(format!(
            "primary item {primary_item_id} has no item information"
        )));
    }
    if !state
        .item_locations
        .iter()
        .any(|location| location.item_id == primary_item_id)
    {
        return Err(DecoderError::Bitstream(format!(
            "primary item {primary_item_id} has no item location"
        )));
    }
    for association in &state.item_property_associations {
        if !state
            .item_infos
            .iter()
            .any(|item| item.item_id == association.item_id)
        {
            return Err(DecoderError::Bitstream(format!(
                "property association refers to unknown item {}",
                association.item_id
            )));
        }
        let mut seen_kinds = Vec::new();
        for property in &association.associations {
            let index = property.index;
            if index == 0 || usize::from(index) > state.item_properties.len() {
                return Err(DecoderError::Bitstream(format!(
                    "item {} property association index {} is out of range",
                    association.item_id, index
                )));
            }
            let kind = property_kind(&state.item_properties[usize::from(index) - 1]);
            if property.essential && kind == PropertyKind::Other {
                return Err(DecoderError::Unsupported(format!(
                    "item {} has an essential unsupported property at index {}",
                    association.item_id, index
                )));
            }
            match &state.item_properties[usize::from(index) - 1] {
                // The decoder's default AV1 operating point is index 0 and
                // its still-image path currently reconstructs the first
                // spatial layer. Accept only selectors that are equivalent
                // to that existing policy; other selectors must not be
                // silently ignored.
                ItemProperty::OperatingPointSelector(op_index) if *op_index != 0 => {
                    return Err(DecoderError::Unsupported(format!(
                        "item {} a1op operating point {} is not supported",
                        association.item_id, op_index
                    )));
                }
                ItemProperty::LayerSelector(layer_id) if *layer_id != 0 => {
                    return Err(DecoderError::Unsupported(format!(
                        "item {} lsel layer {} is not supported",
                        association.item_id, layer_id
                    )));
                }
                _ => {}
            }
            if kind.is_singleton() && seen_kinds.contains(&kind) {
                return Err(DecoderError::Bitstream(format!(
                    "item {} has duplicate {kind:?} property association",
                    association.item_id
                )));
            }
            seen_kinds.push(kind);
        }
    }
    let Some(primary_association) = state
        .item_property_associations
        .iter()
        .find(|association| association.item_id == primary_item_id)
    else {
        return Err(DecoderError::Bitstream(format!(
            "primary item {primary_item_id} has no property associations"
        )));
    };
    let kinds = primary_association
        .associations
        .iter()
        .map(|association| {
            property_kind(&state.item_properties[usize::from(association.index) - 1])
        })
        .collect::<Vec<_>>();
    let is_derived = state
        .item_infos
        .iter()
        .find(|info| info.item_id == state.primary_item_id.unwrap_or(primary_item_id))
        .is_some_and(|info| matches!(&info.item_type, b"grid" | b"sato"));
    let required: &[PropertyKind] = if is_derived {
        &[PropertyKind::SpatialExtents, PropertyKind::PixelInformation]
    } else {
        &[
            PropertyKind::SpatialExtents,
            PropertyKind::PixelInformation,
            PropertyKind::Av1Config,
        ]
    };
    for required in required {
        if !kinds.contains(required) {
            return Err(DecoderError::Bitstream(format!(
                "primary item {primary_item_id} is missing required {required:?} property"
            )));
        }
    }
    Ok(())
}

#[derive(Debug, Default)]
struct PrimaryItemMetadata {
    width: Option<u32>,
    height: Option<u32>,
    pixel_information: Option<PixelInformation>,
    color_information: Option<ColorInformation>,
    alpha_premultiplied: bool,
    av1_config: Option<Vec<u8>>,
    clean_aperture: Option<CleanAperture>,
    rotation: Option<ImageRotation>,
    mirror: Option<ImageMirror>,
}

#[cfg(test)]
fn primary_item_metadata(state: &MetaState) -> Result<PrimaryItemMetadata, DecoderError> {
    let Some(primary_item_id) = state.primary_item_id else {
        return Err(DecoderError::Bitstream(
            "primary item is missing".to_string(),
        ));
    };
    item_metadata(state, primary_item_id)
}

fn item_metadata(state: &MetaState, item_id: u32) -> Result<PrimaryItemMetadata, DecoderError> {
    let association = state
        .item_property_associations
        .iter()
        .find(|association| association.item_id == item_id)
        .ok_or_else(|| {
            DecoderError::Bitstream(format!("item {item_id} has no property associations"))
        })?;
    let mut metadata = PrimaryItemMetadata::default();
    for association in &association.associations {
        let property = &state.item_properties[usize::from(association.index) - 1];
        match property {
            ItemProperty::SpatialExtents(extents) => {
                metadata.width = Some(extents.width);
                metadata.height = Some(extents.height);
            }
            ItemProperty::PixelInformation(pixi) => {
                metadata.pixel_information = Some(pixi.clone());
            }
            ItemProperty::Av1Config(config) => metadata.av1_config = Some(config.clone()),
            ItemProperty::ColorInformation(color) => {
                metadata.color_information = Some(color.clone());
            }
            ItemProperty::Premultiplied => metadata.alpha_premultiplied = true,
            ItemProperty::CleanAperture(clap) => metadata.clean_aperture = Some(*clap),
            ItemProperty::Rotation(rotation) => metadata.rotation = Some(*rotation),
            ItemProperty::Mirror(mirror) => metadata.mirror = Some(*mirror),
            ItemProperty::AuxiliaryType(_)
            | ItemProperty::OperatingPointSelector(_)
            | ItemProperty::LayerSelector(_)
            | ItemProperty::LayerIndexing(_)
            | ItemProperty::Other => {}
        }
    }
    Ok(metadata)
}

fn parse_auxc(payload: &[u8]) -> Result<String, DecoderError> {
    if payload.len() < 4 {
        return Err(DecoderError::NotEnoughData(
            "auxC full-box header is missing".to_string(),
        ));
    }
    let aux_type = &payload[4..];
    let end = aux_type
        .iter()
        .position(|value| *value == 0)
        .unwrap_or(aux_type.len());
    std::str::from_utf8(&aux_type[..end])
        .map(|value| value.to_string())
        .map_err(|_| DecoderError::Bitstream("auxC auxiliary type is not UTF-8".to_string()))
}

fn parse_a1op(payload: &[u8]) -> Result<u8, DecoderError> {
    if payload.len() != 1 {
        return Err(DecoderError::Bitstream(
            "a1op payload must contain one byte".to_string(),
        ));
    }
    Ok(payload[0])
}

fn parse_lsel(payload: &[u8]) -> Result<u16, DecoderError> {
    if payload.len() != 2 {
        return Err(DecoderError::Bitstream(
            "lsel payload must contain one 16-bit layer id".to_string(),
        ));
    }
    Ok(u16::from_be_bytes([payload[0], payload[1]]))
}

fn parse_a1lx(payload: &[u8]) -> Result<[u64; 3], DecoderError> {
    let large_size = payload
        .first()
        .ok_or_else(|| DecoderError::NotEnoughData("a1lx payload is empty".to_string()))?;
    if large_size & 0xfe != 0 {
        return Err(DecoderError::Bitstream(
            "a1lx reserved bits are not zero".to_string(),
        ));
    }
    let field_bytes = if large_size & 1 == 0 { 2 } else { 4 };
    let expected = 1 + field_bytes * 3;
    if payload.len() != expected {
        return Err(DecoderError::Bitstream(format!(
            "a1lx payload length {} does not match {}-byte layer sizes",
            payload.len(),
            field_bytes
        )));
    }
    let mut sizes = [0; 3];
    for (index, size) in sizes.iter_mut().enumerate() {
        let offset = 1 + index * field_bytes;
        *size = if field_bytes == 2 {
            u64::from(u16::from_be_bytes([payload[offset], payload[offset + 1]]))
        } else {
            u64::from(u32::from_be_bytes([
                payload[offset],
                payload[offset + 1],
                payload[offset + 2],
                payload[offset + 3],
            ]))
        };
    }
    Ok(sizes)
}

fn parse_clap(payload: &[u8]) -> Result<CleanAperture, DecoderError> {
    if payload.len() < 32 {
        return Err(DecoderError::NotEnoughData(
            "clap payload is too short".to_string(),
        ));
    }
    Ok(CleanAperture {
        width_n: read_u32(payload, 0)?,
        width_d: read_u32(payload, 4)?,
        height_n: read_u32(payload, 8)?,
        height_d: read_u32(payload, 12)?,
        horizontal_offset_n: read_u32(payload, 16)?,
        horizontal_offset_d: read_u32(payload, 20)?,
        vertical_offset_n: read_u32(payload, 24)?,
        vertical_offset_d: read_u32(payload, 28)?,
    })
}

fn parse_irot(payload: &[u8]) -> Result<ImageRotation, DecoderError> {
    let value = read_u8(payload, 0)?;
    Ok(ImageRotation {
        angle: value & 0x03,
    })
}

fn parse_imir(payload: &[u8]) -> Result<ImageMirror, DecoderError> {
    let value = read_u8(payload, 0)?;
    Ok(ImageMirror { axis: value & 0x01 })
}

fn read_item_id(payload: &[u8], cursor: &mut usize, large_ids: bool) -> Result<u32, DecoderError> {
    if large_ids {
        let value = read_u32(payload, *cursor)?;
        *cursor += 4;
        Ok(value)
    } else {
        let value = read_u16(payload, *cursor)? as u32;
        *cursor += 2;
        Ok(value)
    }
}

fn parse_ispe(payload: &[u8]) -> Result<ImageSpatialExtents, DecoderError> {
    if payload.len() < 12 {
        return Err(DecoderError::NotEnoughData(
            "ispe payload is too short".to_string(),
        ));
    }
    Ok(ImageSpatialExtents {
        width: read_u32(payload, 4)?,
        height: read_u32(payload, 8)?,
    })
}

fn parse_pixi(payload: &[u8]) -> Result<PixelInformation, DecoderError> {
    if payload.len() < 5 {
        return Err(DecoderError::NotEnoughData(
            "pixi payload is too short".to_string(),
        ));
    }
    if payload[0] != 0 {
        return Err(DecoderError::Unsupported(format!(
            "pixi version {} is not supported",
            payload[0]
        )));
    }
    let channel_count = payload[4] as usize;
    if payload.len() < 5 + channel_count {
        return Err(DecoderError::NotEnoughData(
            "pixi channel depth list is too short".to_string(),
        ));
    }
    let flags = u32::from_be_bytes([0, payload[1], payload[2], payload[3]]);
    if flags & !1 != 0 {
        return Err(DecoderError::Unsupported(format!(
            "pixi flags 0x{flags:06x} are not supported"
        )));
    }
    let extended_channels = if flags & 1 == 0 {
        None
    } else {
        let mut reader = PixiBitReader::new(&payload[5 + channel_count..]);
        let mut channels = Vec::with_capacity(channel_count);
        for channel in 0..channel_count {
            let channel_idc = reader.read_bits(3, "pixi channel_idc")? as u8;
            let reserved = reader.read_bits(1, "pixi reserved")?;
            let component_format = reader.read_bits(2, "pixi component_format")? as u8;
            let subsampling_flag = reader.read_bits(1, "pixi subsampling_flag")? != 0;
            let channel_label_flag = reader.read_bits(1, "pixi channel_label_flag")? != 0;
            if reserved != 0 {
                return Err(DecoderError::Bitstream(format!(
                    "pixi channel {channel} has non-zero reserved bits"
                )));
            }
            if channel_idc != 0 {
                return Err(DecoderError::Unsupported(format!(
                    "pixi channel {channel} idc {channel_idc} is not supported"
                )));
            }
            if component_format != 0 {
                return Err(DecoderError::Unsupported(format!(
                    "pixi channel {channel} component format {component_format} is not supported"
                )));
            }
            let subsampling = if subsampling_flag {
                let subsampling_type = reader.read_bits(4, "pixi subsampling_type")? as u8;
                let subsampling_location = reader.read_bits(4, "pixi subsampling_location")? as u8;
                if subsampling_type >= 5 {
                    return Err(DecoderError::Bitstream(format!(
                        "pixi channel {channel} subsampling type {subsampling_type} is reserved"
                    )));
                }
                if subsampling_location > 4 {
                    return Err(DecoderError::Bitstream(format!(
                        "pixi channel {channel} subsampling location {subsampling_location} is reserved"
                    )));
                }
                Some(PixelSubsampling {
                    subsampling_type,
                    subsampling_location,
                })
            } else {
                None
            };
            if channel_label_flag {
                reader.skip_utf8_string(channel)?;
            }
            channels.push(PixelChannelInformation {
                channel_idc,
                component_format,
                subsampling,
            });
        }
        Some(channels)
    };
    Ok(PixelInformation {
        bits_per_channel: payload[5..5 + channel_count].to_vec(),
        extended_channels,
    })
}

struct PixiBitReader<'a> {
    data: &'a [u8],
    bit_position: usize,
}

impl<'a> PixiBitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            bit_position: 0,
        }
    }

    fn read_bits(&mut self, count: usize, field: &str) -> Result<u32, DecoderError> {
        if count > 32 {
            return Err(DecoderError::Bitstream(format!(
                "{field} requests too many bits"
            )));
        }
        let end = self
            .bit_position
            .checked_add(count)
            .ok_or_else(|| DecoderError::Bitstream(format!("{field} bit position overflows")))?;
        if end > self.data.len().saturating_mul(8) {
            return Err(DecoderError::NotEnoughData(format!("{field} is truncated")));
        }
        let mut value = 0u32;
        for _ in 0..count {
            let byte = self.data[self.bit_position / 8];
            let shift = 7 - (self.bit_position % 8);
            value = (value << 1) | u32::from((byte >> shift) & 1);
            self.bit_position += 1;
        }
        Ok(value)
    }

    fn skip_utf8_string(&mut self, channel: usize) -> Result<(), DecoderError> {
        if self.bit_position % 8 != 0 {
            return Err(DecoderError::Bitstream(format!(
                "pixi channel {channel} label is not byte aligned"
            )));
        }
        let mut cursor = self.bit_position / 8;
        let Some(relative_end) = self.data[cursor..].iter().position(|byte| *byte == 0) else {
            return Err(DecoderError::NotEnoughData(format!(
                "pixi channel {channel} label is unterminated"
            )));
        };
        cursor += relative_end + 1;
        self.bit_position = cursor * 8;
        Ok(())
    }
}

fn parse_colr(payload: &[u8]) -> Result<ColorInformation, DecoderError> {
    if payload.len() < 4 {
        return Err(DecoderError::NotEnoughData(
            "colr payload is too short".to_string(),
        ));
    }
    Ok(ColorInformation {
        color_type: read_fourcc(payload, 0)?,
        payload: payload[4..].to_vec(),
    })
}

#[cfg(test)]
fn primary_item_payload(data: &[u8], state: &MetaState) -> Result<Vec<u8>, DecoderError> {
    let primary_item_id = effective_primary_item_id(state)?;
    item_payload(data, state, primary_item_id)
}

fn effective_primary_item_id(state: &MetaState) -> Result<u32, DecoderError> {
    let primary_item_id = state
        .primary_item_id
        .ok_or_else(|| DecoderError::Bitstream("primary item is missing".to_string()))?;
    let Some(primary_info) = state
        .item_infos
        .iter()
        .find(|item| item.item_id == primary_item_id)
    else {
        // Keep the payload helper's low-level error behavior for callers that
        // construct a partial MetaState in tests; full parse validation still
        // requires item information before this helper is used for decoding.
        return Ok(primary_item_id);
    };
    if primary_info.item_type != *b"tmap" {
        return Ok(primary_item_id);
    }
    let base_item_id = state
        .item_references
        .iter()
        .filter(|reference| {
            reference.reference_type == *b"dimg" && reference.from_item_id == primary_item_id
        })
        .flat_map(|reference| reference.to_item_ids.iter().copied())
        .find(|item_id| {
            state
                .item_infos
                .iter()
                .any(|item| item.item_id == *item_id && item.item_type == *b"av01")
        })
        .ok_or_else(|| {
            DecoderError::Unsupported(
                "tmap primary item has no referenced base av01 image".to_string(),
            )
        })?;
    Ok(base_item_id)
}

fn sequence_sample_payloads(
    data: &[u8],
    major_brand: &[u8; 4],
    compatible_brands: &[[u8; 4]],
) -> Result<Vec<Vec<u8>>, DecoderError> {
    if major_brand != BRAND_AVIS && !compatible_brands.iter().any(|brand| brand == BRAND_AVIS) {
        return Ok(Vec::new());
    }
    let mut samples = None;
    for_each_top_level_box(data, |header| {
        if header.box_type == *b"moov" && samples.is_none() {
            let payload = box_payload(data, header)?;
            if let Some(track_samples) = parse_sequence_track(data, payload)? {
                samples = Some(track_samples);
            }
        }
        Ok(())
    })?;
    Ok(samples.unwrap_or_default())
}

fn parse_sequence_track(
    data: &[u8],
    moov_payload: &[u8],
) -> Result<Option<Vec<Vec<u8>>>, DecoderError> {
    for trak in child_boxes(moov_payload)? {
        let trak_payload = box_payload(moov_payload, trak)?;
        let Some(mdia) = child_box(trak_payload, b"mdia")? else {
            continue;
        };
        let mdia_payload = box_payload(trak_payload, mdia)?;
        let Some(hdlr) = child_box(mdia_payload, b"hdlr")? else {
            continue;
        };
        let hdlr_payload = box_payload(mdia_payload, hdlr)?;
        if hdlr_payload.len() < 12
            || (&hdlr_payload[8..12] != b"vide" && &hdlr_payload[8..12] != b"pict")
        {
            continue;
        }
        let Some(minf) = child_box(mdia_payload, b"minf")? else {
            continue;
        };
        let minf_payload = box_payload(mdia_payload, minf)?;
        let Some(stbl) = child_box(minf_payload, b"stbl")? else {
            continue;
        };
        let stbl_payload = box_payload(minf_payload, stbl)?;
        let Some(stsd) = child_box(stbl_payload, b"stsd")? else {
            continue;
        };
        let stsd_payload = box_payload(stbl_payload, stsd)?;
        let sample_descriptions = stsd_av01_descriptions(stsd_payload)?;
        if sample_descriptions.iter().all(Option::is_none) {
            continue;
        }
        let Some(stsc) = child_box(stbl_payload, b"stsc")? else {
            continue;
        };
        let Some(stsz) = child_box(stbl_payload, b"stsz")? else {
            continue;
        };
        let (chunk_offset_payload, chunk_offset_width) =
            if let Some(header) = child_box(stbl_payload, b"stco")? {
                (Some(box_payload(stbl_payload, header)?), 4)
            } else if let Some(header) = child_box(stbl_payload, b"co64")? {
                (Some(box_payload(stbl_payload, header)?), 8)
            } else {
                (None, 0)
            };
        let Some(chunk_offset_payload) = chunk_offset_payload else {
            continue;
        };
        let chunk_offsets = parse_chunk_offsets(chunk_offset_payload, chunk_offset_width)?;
        let sizes = parse_sample_sizes(box_payload(stbl_payload, stsz)?)?;
        let samples_per_chunk = parse_samples_per_chunk(box_payload(stbl_payload, stsc)?)?;
        let (sample_offsets, sample_description_indices) = build_sample_offsets_with_descriptions(
            &chunk_offsets,
            &samples_per_chunk,
            &sizes,
            &sample_descriptions,
        )?;
        let mut samples = Vec::with_capacity(sizes.len());
        for ((offset, size), description_index) in sample_offsets
            .into_iter()
            .zip(sizes)
            .zip(sample_description_indices)
        {
            let start = usize::try_from(offset).map_err(|_| {
                DecoderError::Bitstream("AVIS sample offset is too large".to_string())
            })?;
            let end = start.checked_add(size).ok_or_else(|| {
                DecoderError::Bitstream("AVIS sample end overflows usize".to_string())
            })?;
            if end > data.len() {
                return Err(DecoderError::NotEnoughData(
                    "AVIS sample extends beyond the file".to_string(),
                ));
            }
            samples.push(data[start..end].to_vec());
            let description = sample_descriptions
                .get(description_index)
                .and_then(Option::as_ref)
                .expect("sample description was validated while building offsets");
            let first_description = sample_descriptions
                .iter()
                .find_map(Option::as_ref)
                .expect("stsd AV1 presence checked by caller");
            validate_sequence_sample_description(
                description,
                first_description,
                samples.last().expect("sample was pushed"),
            )?;
        }
        return Ok(Some(samples));
    }
    Ok(None)
}

fn sample_contains_sequence_header(sample: &[u8]) -> Result<bool, DecoderError> {
    Ok(parse_obu_stream(sample)?
        .iter()
        .any(|obu| obu.obu_type == ObuType::SequenceHeader))
}

fn validate_sequence_sample_description(
    description: &[u8],
    first_description: &[u8],
    sample: &[u8],
) -> Result<(), DecoderError> {
    if description != first_description && !sample_contains_sequence_header(sample)? {
        return Err(DecoderError::Unsupported(
            "AVIS differing sample descriptions require a sequence header in each changed sample"
                .to_string(),
        ));
    }
    Ok(())
}

fn child_boxes(payload: &[u8]) -> Result<Vec<BoxHeader>, DecoderError> {
    let mut headers = Vec::new();
    let mut offset = 0;
    while offset < payload.len() {
        let header = read_box_header(payload, offset, payload.len())?;
        offset = checked_add(header.offset, header.size, "child box end")?;
        headers.push(header);
    }
    Ok(headers)
}

fn child_box(payload: &[u8], box_type: &[u8; 4]) -> Result<Option<BoxHeader>, DecoderError> {
    Ok(child_boxes(payload)?
        .into_iter()
        .find(|header| header.box_type == *box_type))
}

fn stsd_av01_descriptions(payload: &[u8]) -> Result<Vec<Option<Vec<u8>>>, DecoderError> {
    if payload.len() < 8 {
        return Err(DecoderError::NotEnoughData(
            "stsd payload is too short".to_string(),
        ));
    }
    let entry_count = read_u32(payload, 4)? as usize;
    let mut offset = 8;
    let mut descriptions = Vec::with_capacity(entry_count);
    for _ in 0..entry_count {
        let header = read_box_header(payload, offset, payload.len())?;
        descriptions.push(if header.box_type == *b"av01" {
            let end = checked_add(header.offset, header.size, "stsd av01 entry end")?;
            Some(payload[header.offset..end].to_vec())
        } else {
            None
        });
        offset = checked_add(header.offset, header.size, "stsd entry end")?;
    }
    Ok(descriptions)
}

fn parse_sample_sizes(payload: &[u8]) -> Result<Vec<usize>, DecoderError> {
    if payload.len() < 12 {
        return Err(DecoderError::NotEnoughData(
            "stsz payload is too short".to_string(),
        ));
    }
    let sample_size = read_u32(payload, 4)? as usize;
    let sample_count = read_u32(payload, 8)? as usize;
    if sample_size != 0 {
        return Ok(vec![sample_size; sample_count]);
    }
    let end = 12usize
        .checked_add(sample_count.checked_mul(4).ok_or_else(|| {
            DecoderError::Bitstream("stsz sample count overflows usize".to_string())
        })?)
        .ok_or_else(|| DecoderError::Bitstream("stsz payload size overflows usize".to_string()))?;
    if end > payload.len() {
        return Err(DecoderError::NotEnoughData(
            "stsz entries are truncated".to_string(),
        ));
    }
    (0..sample_count)
        .map(|index| {
            usize::try_from(read_u32(payload, 12 + index * 4)?)
                .map_err(|_| DecoderError::Bitstream("stsz sample size is too large".to_string()))
        })
        .collect()
}

fn parse_chunk_offsets(payload: &[u8], width: usize) -> Result<Vec<u64>, DecoderError> {
    if width != 4 && width != 8 {
        return Err(DecoderError::Bitstream(
            "invalid chunk offset width".to_string(),
        ));
    }
    if payload.len() < 8 {
        return Err(DecoderError::NotEnoughData(
            "chunk offset table is too short".to_string(),
        ));
    }
    let count = read_u32(payload, 4)? as usize;
    let end = 8usize
        .checked_add(
            count.checked_mul(width).ok_or_else(|| {
                DecoderError::Bitstream("chunk count overflows usize".to_string())
            })?,
        )
        .ok_or_else(|| DecoderError::Bitstream("chunk table size overflows usize".to_string()))?;
    if end > payload.len() {
        return Err(DecoderError::NotEnoughData(
            "chunk offsets are truncated".to_string(),
        ));
    }
    (0..count)
        .map(|index| {
            if width == 8 {
                read_u64(payload, 8 + index * width)
            } else {
                read_u32(payload, 8 + index * width).map(u64::from)
            }
        })
        .collect()
}

fn parse_samples_per_chunk(payload: &[u8]) -> Result<Vec<(u32, u32, u32)>, DecoderError> {
    if payload.len() < 8 {
        return Err(DecoderError::NotEnoughData(
            "stsc payload is too short".to_string(),
        ));
    }
    let count = read_u32(payload, 4)? as usize;
    let end = 8usize
        .checked_add(
            count
                .checked_mul(12)
                .ok_or_else(|| DecoderError::Bitstream("stsc count overflows usize".to_string()))?,
        )
        .ok_or_else(|| DecoderError::Bitstream("stsc table size overflows usize".to_string()))?;
    if end > payload.len() {
        return Err(DecoderError::NotEnoughData(
            "stsc entries are truncated".to_string(),
        ));
    }
    (0..count)
        .map(|index| {
            let offset = 8 + index * 12;
            Ok((
                read_u32(payload, offset)?,
                read_u32(payload, offset + 4)?,
                read_u32(payload, offset + 8)?,
            ))
        })
        .collect()
}

fn build_sample_offsets_with_descriptions(
    chunk_offsets: &[u64],
    samples_per_chunk: &[(u32, u32, u32)],
    sizes: &[usize],
    sample_descriptions: &[Option<Vec<u8>>],
) -> Result<(Vec<u64>, Vec<usize>), DecoderError> {
    if samples_per_chunk.is_empty() {
        return Err(DecoderError::Bitstream("stsc table is empty".to_string()));
    }
    let mut offsets = Vec::with_capacity(sizes.len());
    let mut description_indices = Vec::with_capacity(sizes.len());
    let mut sample_index = 0usize;
    for (chunk_index, &chunk_offset) in chunk_offsets.iter().enumerate() {
        let chunk_number = u32::try_from(chunk_index + 1)
            .map_err(|_| DecoderError::Bitstream("chunk number is too large".to_string()))?;
        let record = samples_per_chunk
            .iter()
            .rev()
            .find(|(first_chunk, _, _)| *first_chunk <= chunk_number)
            .ok_or_else(|| DecoderError::Bitstream("stsc first_chunk is invalid".to_string()))?;
        let description_index = usize::try_from(record.2).map_err(|_| {
            DecoderError::Bitstream("AVIS sample description index is too large".to_string())
        })?;
        if description_index == 0 {
            return Err(DecoderError::Bitstream(
                "AVIS sample description index is zero".to_string(),
            ));
        }
        sample_descriptions
            .get(description_index - 1)
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                DecoderError::Bitstream(format!(
                    "AVIS sample description index {} is missing an av01 entry",
                    record.2
                ))
            })?;
        let mut offset = chunk_offset;
        for _ in 0..record.1 {
            let size = *sizes.get(sample_index).ok_or_else(|| {
                DecoderError::Bitstream("stsc references more samples than stsz".to_string())
            })?;
            offsets.push(offset);
            description_indices.push(description_index - 1);
            offset = offset.checked_add(size as u64).ok_or_else(|| {
                DecoderError::Bitstream("AVIS sample offset overflows u64".to_string())
            })?;
            sample_index += 1;
        }
    }
    if sample_index != sizes.len() {
        return Err(DecoderError::Bitstream(
            "stsz contains samples not covered by stsc".to_string(),
        ));
    }
    Ok((offsets, description_indices))
}

#[cfg(test)]
fn alpha_auxiliary_items(
    data: &[u8],
    state: &MetaState,
) -> Result<Vec<AuxiliaryImage>, DecoderError> {
    alpha_auxiliary_items_for(data, state, state.primary_item_id)
}

fn alpha_auxiliary_items_for(
    data: &[u8],
    state: &MetaState,
    owner_item_id: Option<u32>,
) -> Result<Vec<AuxiliaryImage>, DecoderError> {
    let mut item_ids: Vec<(u32, String)> = state
        .item_property_associations
        .iter()
        .filter_map(|association| {
            association
                .associations
                .iter()
                .filter_map(|index| {
                    state
                        .item_properties
                        .get(usize::from(index.index).saturating_sub(1))
                })
                .find_map(|property| match property {
                    ItemProperty::AuxiliaryType(aux_type) if aux_type == ALPHA_AUX_TYPE => {
                        Some((association.item_id, aux_type.clone()))
                    }
                    _ => None,
                })
        })
        .collect();

    if item_ids.is_empty()
        && let Some(primary_item_id) = owner_item_id
    {
        for reference in &state.item_references {
            if &reference.reference_type == b"auxl" && reference.from_item_id == primary_item_id {
                item_ids.extend(
                    reference
                        .to_item_ids
                        .iter()
                        .map(|item_id| (*item_id, ALPHA_AUX_TYPE.to_string())),
                );
            }
        }
    }

    item_ids.sort_by_key(|(item_id, _)| *item_id);
    item_ids.dedup_by_key(|(item_id, _)| *item_id);
    item_ids
        .into_iter()
        .map(|(item_id, aux_type)| {
            Ok(AuxiliaryImage {
                item_id,
                aux_type,
                payload: item_payload(data, state, item_id)?,
            })
        })
        .collect()
}

fn primary_grid(
    data: &[u8],
    payload: &[u8],
    state: &MetaState,
) -> Result<Option<GridImage>, DecoderError> {
    let Some(primary_item_id) = state.primary_item_id else {
        return Ok(None);
    };
    let Some(item_info) = state
        .item_infos
        .iter()
        .find(|info| info.item_id == primary_item_id)
    else {
        return Ok(None);
    };
    if &item_info.item_type != b"grid" {
        return Ok(None);
    }
    parse_grid_item(data, payload, state, primary_item_id).map(Some)
}

fn alpha_grid(
    data: &[u8],
    auxiliary_items: &[AuxiliaryImage],
    state: &MetaState,
) -> Result<Option<GridImage>, DecoderError> {
    let Some(auxiliary) = auxiliary_items.iter().find(|item| {
        state
            .item_infos
            .iter()
            .any(|info| info.item_id == item.item_id && info.item_type == *b"grid")
    }) else {
        return Ok(None);
    };
    parse_grid_item(data, &auxiliary.payload, state, auxiliary.item_id).map(Some)
}

fn parse_grid_item(
    data: &[u8],
    payload: &[u8],
    state: &MetaState,
    grid_item_id: u32,
) -> Result<GridImage, DecoderError> {
    let parsed = parse_grid_payload(payload)?;
    let references = state
        .item_references
        .iter()
        .filter(|reference| {
            reference.reference_type == *b"dimg" && reference.from_item_id == grid_item_id
        })
        .collect::<Vec<_>>();
    if references.len() != 1 {
        return Err(DecoderError::Bitstream(format!(
            "grid item {grid_item_id} must have exactly one dimg reference"
        )));
    }
    let cell_count = usize::from(parsed.rows)
        .checked_mul(usize::from(parsed.columns))
        .ok_or_else(|| DecoderError::Bitstream("grid cell count overflow".to_string()))?;
    let cell_ids = &references[0].to_item_ids;
    if cell_ids.len() != cell_count {
        return Err(DecoderError::Bitstream(format!(
            "grid item {grid_item_id} references {} cells, expected {cell_count}",
            cell_ids.len()
        )));
    }
    let mut cells = Vec::with_capacity(cell_count);
    for &item_id in cell_ids {
        let item_info = state
            .item_infos
            .iter()
            .find(|info| info.item_id == item_id)
            .ok_or_else(|| {
                DecoderError::Bitstream(format!("grid cell item {item_id} info is missing"))
            })?;
        if &item_info.item_type != b"av01" {
            return Err(DecoderError::Unsupported(format!(
                "grid cell item {item_id} type {:?} is not av01",
                item_info.item_type
            )));
        }
        let metadata = item_metadata(state, item_id)?;
        let width = metadata.width.ok_or_else(|| {
            DecoderError::Bitstream(format!("grid cell item {item_id} width is missing"))
        })?;
        let height = metadata.height.ok_or_else(|| {
            DecoderError::Bitstream(format!("grid cell item {item_id} height is missing"))
        })?;
        let av1_config = metadata.av1_config.ok_or_else(|| {
            DecoderError::Bitstream(format!("grid cell item {item_id} av1C is missing"))
        })?;
        cells.push(GridCell {
            item_id,
            width,
            height,
            pixel_information: metadata.pixel_information,
            color_information: metadata.color_information,
            av1_config: Some(av1_config),
            payload: item_payload(data, state, item_id)?,
        });
    }
    Ok(GridImage {
        item_id: grid_item_id,
        rows: parsed.rows,
        columns: parsed.columns,
        output_width: parsed.output_width,
        output_height: parsed.output_height,
        payload: payload.to_vec(),
        cells,
    })
}

#[cfg(test)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct PrimaryItemTransforms {
    clean_aperture: Option<CleanAperture>,
    rotation: Option<ImageRotation>,
    mirror: Option<ImageMirror>,
}

#[cfg(test)]
fn primary_item_transforms(state: &MetaState) -> PrimaryItemTransforms {
    let Some(primary_item_id) = state.primary_item_id else {
        return PrimaryItemTransforms::default();
    };
    let Some(association) = state
        .item_property_associations
        .iter()
        .find(|association| association.item_id == primary_item_id)
    else {
        return PrimaryItemTransforms::default();
    };
    let mut transforms = PrimaryItemTransforms::default();
    for property_association in &association.associations {
        let Some(property) = state
            .item_properties
            .get(usize::from(property_association.index).saturating_sub(1))
        else {
            continue;
        };
        match property {
            ItemProperty::CleanAperture(clap) => transforms.clean_aperture = Some(*clap),
            ItemProperty::Rotation(rotation) => transforms.rotation = Some(*rotation),
            ItemProperty::Mirror(mirror) => transforms.mirror = Some(*mirror),
            _ => {}
        }
    }
    transforms
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ParsedGridPayload {
    rows: u8,
    columns: u8,
    output_width: u32,
    output_height: u32,
}

fn parse_grid_payload(payload: &[u8]) -> Result<ParsedGridPayload, DecoderError> {
    if payload.len() < 8 {
        return Err(DecoderError::NotEnoughData(
            "grid payload is too short".to_string(),
        ));
    }
    let version = read_u8(payload, 0)?;
    if version != 0 {
        return Err(DecoderError::Unsupported(format!(
            "grid version {version} is not supported"
        )));
    }
    let flags = read_u8(payload, 1)?;
    let rows = read_u8(payload, 2)?
        .checked_add(1)
        .ok_or_else(|| DecoderError::Bitstream("grid row count overflow".to_string()))?;
    let columns = read_u8(payload, 3)?
        .checked_add(1)
        .ok_or_else(|| DecoderError::Bitstream("grid column count overflow".to_string()))?;
    if flags & 1 == 0 {
        Ok(ParsedGridPayload {
            rows,
            columns,
            output_width: read_u16(payload, 4)? as u32,
            output_height: read_u16(payload, 6)? as u32,
        })
    } else {
        if payload.len() < 12 {
            return Err(DecoderError::NotEnoughData(
                "large grid payload is too short".to_string(),
            ));
        }
        Ok(ParsedGridPayload {
            rows,
            columns,
            output_width: read_u32(payload, 4)?,
            output_height: read_u32(payload, 8)?,
        })
    }
}

fn item_payload(data: &[u8], state: &MetaState, item_id: u32) -> Result<Vec<u8>, DecoderError> {
    item_payload_with_stack(data, state, item_id, &mut Vec::new())
}

fn item_payload_with_stack(
    data: &[u8],
    state: &MetaState,
    item_id: u32,
    stack: &mut Vec<u32>,
) -> Result<Vec<u8>, DecoderError> {
    if stack.contains(&item_id) {
        return Err(DecoderError::Bitstream(format!(
            "iloc item reference cycle includes item {item_id}"
        )));
    }
    stack.push(item_id);
    let location = state
        .item_locations
        .iter()
        .find(|location| location.item_id == item_id)
        .ok_or_else(|| DecoderError::Bitstream(format!("item {item_id} location is missing")))?;
    let construction_method = state
        .item_construction_methods
        .iter()
        .find_map(|(id, method)| (*id == item_id).then_some(*method))
        .unwrap_or(0);
    let payload_len = location.extents.iter().try_fold(0usize, |sum, extent| {
        let length = usize::try_from(extent.length)
            .map_err(|_| DecoderError::Bitstream("item extent length is too large".to_string()))?;
        sum.checked_add(length).ok_or_else(|| {
            DecoderError::Bitstream("item extent payload length overflow".to_string())
        })
    })?;
    let mut payload = Vec::with_capacity(payload_len);
    match construction_method {
        0 | 1 => {
            let source: Cow<'_, [u8]> = if construction_method == 0 {
                Cow::Borrowed(data)
            } else {
                Cow::Borrowed(state.idat_payload.as_deref().ok_or_else(|| {
                    DecoderError::Bitstream(format!("item {item_id} references missing idat box"))
                })?)
            };
            if payload_len > source.len() {
                return Err(DecoderError::Bitstream(
                    "item extent payload length exceeds file size".to_string(),
                ));
            }
            for extent in &location.extents {
                append_item_extent(&mut payload, location, extent, &source)?;
            }
        }
        2 => {
            for (extent_position, extent) in location.extents.iter().enumerate() {
                let extent_index = state
                    .item_extent_indexes
                    .iter()
                    .find(|(id, _)| *id == item_id)
                    .and_then(|(_, indexes)| indexes.get(extent_position).copied())
                    .unwrap_or(1);
                let target = state
                    .item_references
                    .iter()
                    .find(|reference| {
                        reference.reference_type == *b"iloc" && reference.from_item_id == item_id
                    })
                    .and_then(|reference| {
                        usize::try_from(extent_index.saturating_sub(1))
                            .ok()
                            .and_then(|index| reference.to_item_ids.get(index).copied())
                    })
                    .ok_or_else(|| {
                        DecoderError::Bitstream(format!(
                            "item {item_id} item_offset reference index {extent_index} is missing"
                        ))
                    })?;
                let source = item_payload_with_stack(data, state, target, stack)?;
                append_item_extent(&mut payload, location, extent, &source)?;
            }
        }
        method => {
            return Err(DecoderError::Unsupported(format!(
                "iloc construction_method {method} is not supported"
            )));
        }
    }
    stack.pop();
    Ok(payload)
}

fn append_item_extent(
    payload: &mut Vec<u8>,
    location: &ItemLocation,
    extent: &ItemExtent,
    source: &[u8],
) -> Result<(), DecoderError> {
    let start = location
        .base_offset
        .checked_add(extent.offset)
        .ok_or_else(|| DecoderError::Bitstream("item extent offset overflow".to_string()))?;
    let end = start
        .checked_add(extent.length)
        .ok_or_else(|| DecoderError::Bitstream("item extent length overflow".to_string()))?;
    let start = usize::try_from(start)
        .map_err(|_| DecoderError::Bitstream("item extent start is too large".to_string()))?;
    let end = usize::try_from(end)
        .map_err(|_| DecoderError::Bitstream("item extent end is too large".to_string()))?;
    if end > source.len() || start > end {
        return Err(DecoderError::NotEnoughData(
            "item extent points outside the file".to_string(),
        ));
    }
    payload.extend_from_slice(&source[start..end]);
    Ok(())
}

fn read_box_header(data: &[u8], offset: usize, limit: usize) -> Result<BoxHeader, DecoderError> {
    if offset + 8 > limit || limit > data.len() {
        return Err(DecoderError::NotEnoughData(
            "box header is truncated".to_string(),
        ));
    }

    let small_size = read_u32(data, offset)? as usize;
    let box_type = read_fourcc(data, offset + 4)?;
    let (size, header_size) = if small_size == 1 {
        if offset + 16 > limit {
            return Err(DecoderError::NotEnoughData(
                "large box header is truncated".to_string(),
            ));
        }
        let large_size = read_u64(data, offset + 8)?;
        let size = usize::try_from(large_size)
            .map_err(|_| DecoderError::Bitstream("box size is too large".to_string()))?;
        (size, 16)
    } else if small_size == 0 {
        (limit - offset, 8)
    } else {
        (small_size, 8)
    };

    if size < header_size {
        return Err(DecoderError::Bitstream(
            "box size is smaller than its header".to_string(),
        ));
    }
    let end = checked_add(offset, size, "box end")?;
    if end > limit {
        return Err(DecoderError::NotEnoughData(
            "box extends beyond parent".to_string(),
        ));
    }

    Ok(BoxHeader {
        offset,
        size,
        header_size,
        box_type,
    })
}

fn box_payload(data: &[u8], header: BoxHeader) -> Result<&[u8], DecoderError> {
    let start = checked_add(header.offset, header.header_size, "box payload start")?;
    let end = checked_add(header.offset, header.size, "box payload end")?;
    Ok(&data[start..end])
}

fn validate_field_size(size: u8, name: &str) -> Result<(), DecoderError> {
    match size {
        0 | 4 | 8 => Ok(()),
        _ => Err(DecoderError::Unsupported(format!("{name}={size}"))),
    }
}

fn read_sized_int(data: &[u8], cursor: &mut usize, size: u8) -> Result<u64, DecoderError> {
    let value = match size {
        0 => 0,
        4 => {
            let value = read_u32(data, *cursor)? as u64;
            *cursor += 4;
            value
        }
        8 => {
            let value = read_u64(data, *cursor)?;
            *cursor += 8;
            value
        }
        _ => return Err(DecoderError::Unsupported(format!("integer size {size}"))),
    };
    Ok(value)
}

fn read_fourcc(data: &[u8], offset: usize) -> Result<[u8; 4], DecoderError> {
    Ok(data
        .get(offset..offset + 4)
        .ok_or_else(|| DecoderError::NotEnoughData("fourcc is truncated".to_string()))?
        .try_into()
        .expect("slice length checked"))
}

fn read_c_string(data: &[u8], offset: usize) -> Result<String, DecoderError> {
    let bytes = data
        .get(offset..)
        .ok_or_else(|| DecoderError::NotEnoughData("string offset exceeds input".to_string()))?;
    let end = bytes
        .iter()
        .position(|value| *value == 0)
        .unwrap_or(bytes.len());
    std::str::from_utf8(&bytes[..end])
        .map(|value| value.to_string())
        .map_err(|_| DecoderError::Bitstream("box string is not UTF-8".to_string()))
}

fn read_u8(data: &[u8], offset: usize) -> Result<u8, DecoderError> {
    data.get(offset)
        .copied()
        .ok_or_else(|| DecoderError::NotEnoughData("u8 is truncated".to_string()))
}

fn read_u16(data: &[u8], offset: usize) -> Result<u16, DecoderError> {
    Ok(u16::from_be_bytes(
        data.get(offset..offset + 2)
            .ok_or_else(|| DecoderError::NotEnoughData("u16 is truncated".to_string()))?
            .try_into()
            .expect("slice length checked"),
    ))
}

fn read_u24(data: &[u8], offset: usize) -> Result<u32, DecoderError> {
    let bytes = data
        .get(offset..offset + 3)
        .ok_or_else(|| DecoderError::NotEnoughData("u24 is truncated".to_string()))?;
    Ok((u32::from(bytes[0]) << 16) | (u32::from(bytes[1]) << 8) | u32::from(bytes[2]))
}

fn read_u32(data: &[u8], offset: usize) -> Result<u32, DecoderError> {
    Ok(u32::from_be_bytes(
        data.get(offset..offset + 4)
            .ok_or_else(|| DecoderError::NotEnoughData("u32 is truncated".to_string()))?
            .try_into()
            .expect("slice length checked"),
    ))
}

fn read_u64(data: &[u8], offset: usize) -> Result<u64, DecoderError> {
    Ok(u64::from_be_bytes(
        data.get(offset..offset + 8)
            .ok_or_else(|| DecoderError::NotEnoughData("u64 is truncated".to_string()))?
            .try_into()
            .expect("slice length checked"),
    ))
}

fn checked_add(left: usize, right: usize, label: &str) -> Result<usize, DecoderError> {
    left.checked_add(right)
        .ok_or_else(|| DecoderError::Bitstream(format!("{label} overflow")))
}

fn validate_collection_count(
    count: usize,
    remaining_payload: usize,
    minimum_entry_size: usize,
    label: &str,
) -> Result<(), DecoderError> {
    let minimum_size = count
        .checked_mul(minimum_entry_size)
        .ok_or_else(|| DecoderError::Bitstream(format!("{label} count overflow")))?;
    if minimum_size > remaining_payload {
        return Err(DecoderError::NotEnoughData(format!(
            "{label} count exceeds the remaining payload"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_u16(payload: &mut Vec<u8>, value: u16) {
        payload.extend_from_slice(&value.to_be_bytes());
    }

    fn push_u32(payload: &mut Vec<u8>, value: u32) {
        payload.extend_from_slice(&value.to_be_bytes());
    }

    fn push_i32(payload: &mut Vec<u8>, value: i32) {
        payload.extend_from_slice(&value.to_be_bytes());
    }

    fn boxed(box_type: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let size = u32::try_from(payload.len() + 8).unwrap();
        let mut output = Vec::with_capacity(payload.len() + 8);
        output.extend_from_slice(&size.to_be_bytes());
        output.extend_from_slice(box_type);
        output.extend_from_slice(payload);
        output
    }

    fn common_gain_map_payload() -> Vec<u8> {
        let mut payload = Vec::new();
        push_u16(&mut payload, 0);
        push_u16(&mut payload, 0);
        payload.push(0xc8 | 0x04); // RGB channels, base colour space, backward direction, common denominator.
        push_u32(&mut payload, 100);
        push_u32(&mut payload, 100);
        push_u32(&mut payload, 200);
        for channel in 0..3 {
            push_i32(&mut payload, -10 - channel);
            push_i32(&mut payload, 40 + channel);
            push_u32(&mut payload, 100);
            push_i32(&mut payload, 0);
            push_i32(&mut payload, 5);
        }
        payload.extend_from_slice(&[0, 1]);
        payload
    }

    #[test]
    fn parses_gain_map_metadata_with_common_denominator() {
        let metadata = parse_gain_map_metadata_payload(&common_gain_map_payload()).unwrap();

        assert_eq!(metadata.channel_count(), 3);
        assert!(metadata.is_multichannel);
        assert!(metadata.use_base_colour_space);
        assert!(metadata.backward_direction);
        assert_eq!(metadata.base_hdr_headroom.numerator, 100);
        assert_eq!(metadata.alternate_hdr_headroom.numerator, 200);
        assert_eq!(metadata.channels[0].gain_map_min.numerator, -10);
        assert_eq!(metadata.channels[2].gain_map_max.numerator, 42);
        assert_eq!(metadata.channels[1].gamma.denominator, 100);
    }

    #[test]
    fn parses_gain_map_metadata_with_per_field_denominators() {
        let mut payload = vec![0, 0, 0, 0, 0];
        push_u32(&mut payload, 1); // base headroom 1/1
        push_u32(&mut payload, 1);
        push_u32(&mut payload, 2); // alternate headroom 2/1
        push_u32(&mut payload, 1);
        push_i32(&mut payload, -1);
        push_u32(&mut payload, 1);
        push_i32(&mut payload, 3);
        push_u32(&mut payload, 1);
        push_u32(&mut payload, 1); // gamma numerator
        push_u32(&mut payload, 1);
        push_i32(&mut payload, 0);
        push_u32(&mut payload, 1);
        push_i32(&mut payload, 1);
        push_u32(&mut payload, 1);

        let metadata = parse_gain_map_metadata_payload(&payload).unwrap();
        assert_eq!(metadata.channel_count(), 1);
        assert_eq!(metadata.channels[0].gain_map_max.numerator, 3);
        assert_eq!(metadata.channels[0].alternate_offset.numerator, 1);
    }

    #[test]
    fn rejects_invalid_gain_map_metadata_without_partial_result() {
        let mut unsupported = common_gain_map_payload();
        unsupported[0] = 1;
        assert!(matches!(
            parse_gain_map_metadata_payload(&unsupported),
            Err(DecoderError::Unsupported(message)) if message.contains("minimum version")
        ));

        let mut zero_denominator = common_gain_map_payload();
        zero_denominator[5..9].fill(0);
        assert!(matches!(
            parse_gain_map_metadata_payload(&zero_denominator),
            Err(DecoderError::Bitstream(message)) if message.contains("denominator")
        ));

        let mut invalid_range = common_gain_map_payload();
        invalid_range[21..25].copy_from_slice(&(-20_i32).to_be_bytes());
        assert!(matches!(
            parse_gain_map_metadata_payload(&invalid_range),
            Err(DecoderError::Bitstream(message)) if message.contains("maximum")
        ));
    }

    #[test]
    fn gain_map_metadata_is_absent_without_tmap_item() {
        assert_eq!(parse_gain_map_metadata(b"not an avif").unwrap(), None);
    }

    #[test]
    fn parses_gain_map_metadata_from_tmap_item_payload() {
        let gain_map = common_gain_map_payload();
        let mut infe = vec![2, 0, 0, 0, 0, 1, 0, 0];
        infe.extend_from_slice(b"tmapgain\0");
        let iinf_entry = boxed(b"infe", &infe);
        let mut iinf = vec![0, 0, 0, 0, 0, 1];
        iinf.extend_from_slice(&iinf_entry);

        let mut iloc = vec![
            1, 0, 0, 0, 0x04, 0, 0, 1, // version, sizes, one item
            0, 1, // item id
            0, 1, // construction method 1
            0, 0, // data reference index
            0, 1, // one extent
        ];
        iloc.extend_from_slice(&(u32::try_from(gain_map.len()).unwrap()).to_be_bytes());

        let mut meta_payload = vec![0, 0, 0, 0];
        meta_payload.extend_from_slice(&boxed(b"pitm", &[0, 0, 0, 0, 0, 1]));
        meta_payload.extend_from_slice(&boxed(b"iinf", &iinf));
        meta_payload.extend_from_slice(&boxed(b"iloc", &iloc));
        meta_payload.extend_from_slice(&boxed(b"idat", &gain_map));

        let metadata = parse_gain_map_metadata(&boxed(b"meta", &meta_payload))
            .unwrap()
            .expect("tmap metadata should be discovered from the item payload");
        assert_eq!(metadata.channel_count(), 3);
        assert_eq!(metadata.channels[2].gain_map_max.numerator, 42);
    }

    fn one_frame_obu(obu_type: u8, header_byte: u8) -> Vec<u8> {
        vec![(obu_type << 3) | 0x02, 1, header_byte]
    }

    #[test]
    fn classifies_sequence_sample_frame_header_kinds() {
        assert_eq!(
            classify_av1_sequence_sample(&one_frame_obu(6, 0x00)).unwrap(),
            Some(AvifSequenceSampleKind::Key)
        );
        assert_eq!(
            classify_av1_sequence_sample(&one_frame_obu(6, 0x20)).unwrap(),
            Some(AvifSequenceSampleKind::Inter)
        );
        assert_eq!(
            classify_av1_sequence_sample(&one_frame_obu(3, 0xa0)).unwrap(),
            Some(AvifSequenceSampleKind::ShowExisting {
                frame_to_show_map_idx: 2
            })
        );
    }

    #[test]
    fn inspects_sequence_header_presence_alongside_frame_kind() {
        let mut payload = vec![(1 << 3) | 0x02, 1, 0x00];
        payload.extend_from_slice(&one_frame_obu(6, 0x40));
        let info = inspect_av1_sequence_sample(&payload).unwrap();
        assert_eq!(info.kind, Some(AvifSequenceSampleKind::IntraOnly));
        assert!(info.has_sequence_header);
        assert_eq!(classify_av1_sequence_sample(&payload).unwrap(), info.kind);
    }

    #[test]
    fn parses_sato_postfix_expression_with_16_bit_constant() {
        // input0 * 256 + input1, the canonical 8-bit-to-16-bit suffix.
        let payload = [1, 5, 1, 0, 1, 0, 130, 2, 128];
        let (depth, tokens) = parse_sample_transform_payload(&payload).unwrap();
        assert_eq!(depth, 16);
        assert_eq!(
            tokens,
            vec![
                SampleTransformToken::Input(0),
                SampleTransformToken::Constant(256),
                SampleTransformToken::Binary(2),
                SampleTransformToken::Input(1),
                SampleTransformToken::Binary(0),
            ]
        );
    }

    #[test]
    fn rejects_sato_reserved_token_and_stack_leaks() {
        let reserved = [0, 1, 33];
        assert!(matches!(
            parse_sample_transform_payload(&reserved),
            Err(DecoderError::Unsupported(message)) if message.contains("reserved sato token")
        ));
        let leaked = [0, 2, 1, 2];
        assert!(matches!(
            parse_sample_transform_payload(&leaked),
            Err(DecoderError::Bitstream(message)) if message.contains("exactly one result")
        ));
    }

    #[test]
    fn parses_sato_32_bit_signed_constants_and_unary_operators() {
        // -2, absolute value, then bitwise-not: ~abs(-2) == -3.
        let payload = [2, 3, 0, 0xff, 0xff, 0xff, 0xfe, 65, 66];
        let (depth, tokens) = parse_sample_transform_payload(&payload).unwrap();
        assert_eq!(depth, 32);
        assert_eq!(
            tokens,
            vec![
                SampleTransformToken::Constant(-2),
                SampleTransformToken::Unary(1),
                SampleTransformToken::Unary(2),
            ]
        );
    }

    #[test]
    fn parses_sato_64_bit_signed_constants() {
        // The 64-bit form is a signed big-endian integer, not a reserved
        // expression width. Keep the value at the signed boundary so both
        // byte width and sign extension are covered.
        let payload = [3, 1, 0, 0x80, 0, 0, 0, 0, 0, 0, 0];
        let (depth, tokens) = parse_sample_transform_payload(&payload).unwrap();
        assert_eq!(depth, 64);
        assert_eq!(tokens, vec![SampleTransformToken::Constant(i64::MIN)]);
    }

    #[test]
    fn rejects_empty_coded_frame_header_during_sequence_classification() {
        let error = classify_av1_sequence_sample(&[0x1a, 0]).unwrap_err();
        assert!(matches!(
            error,
            DecoderError::NotEnoughData(message) if message.contains("frame header is empty")
        ));
    }

    #[test]
    fn rejects_truncated_top_level_box_header() {
        let err = parse_avif(&[0, 0, 0]).unwrap_err();

        assert!(
            matches!(err, DecoderError::NotEnoughData(message) if message.contains("box header"))
        );
    }

    #[test]
    fn rejects_box_size_smaller_than_header() {
        let err = read_box_header(&[0, 0, 0, 4, b'f', b't', b'y', b'p'], 0, 8).unwrap_err();

        assert!(
            matches!(err, DecoderError::Bitstream(message) if message.contains("smaller than its header"))
        );
    }

    #[test]
    fn rejects_box_extending_beyond_parent() {
        let err = read_box_header(&[0, 0, 0, 12, b'f', b't', b'y', b'p'], 0, 8).unwrap_err();

        assert!(
            matches!(err, DecoderError::NotEnoughData(message) if message.contains("beyond parent"))
        );
    }

    #[test]
    fn rejects_item_extent_payload_length_overflow() {
        let state = MetaState {
            primary_item_id: Some(1),
            item_locations: vec![ItemLocation {
                item_id: 1,
                base_offset: 0,
                extents: vec![
                    ItemExtent {
                        offset: 0,
                        length: u64::MAX,
                    },
                    ItemExtent {
                        offset: 0,
                        length: 1,
                    },
                ],
            }],
            ..MetaState::default()
        };

        let err = primary_item_payload(&[], &state).unwrap_err();

        assert!(
            matches!(err, DecoderError::Bitstream(message) if message.contains("payload length overflow") || message.contains("too large"))
        );
    }

    #[test]
    fn rejects_item_extent_offset_overflow() {
        let state = MetaState {
            primary_item_id: Some(1),
            item_locations: vec![ItemLocation {
                item_id: 1,
                base_offset: u64::MAX,
                extents: vec![ItemExtent {
                    offset: 1,
                    length: 0,
                }],
            }],
            ..MetaState::default()
        };

        let err = primary_item_payload(&[], &state).unwrap_err();

        assert!(
            matches!(err, DecoderError::Bitstream(message) if message.contains("offset overflow"))
        );
    }

    #[test]
    fn rejects_item_extent_outside_file() {
        let state = MetaState {
            primary_item_id: Some(1),
            item_locations: vec![ItemLocation {
                item_id: 1,
                base_offset: 0,
                extents: vec![ItemExtent {
                    offset: 1,
                    length: 2,
                }],
            }],
            ..MetaState::default()
        };

        let err = primary_item_payload(&[0, 1], &state).unwrap_err();

        assert!(
            matches!(err, DecoderError::NotEnoughData(message) if message.contains("outside the file"))
        );
    }

    #[test]
    fn rejects_iinf_entry_count_that_cannot_fit_in_payload() {
        let err = parse_iinf(&[
            1, 0, 0, 0, // version and flags
            0xff, 0xff, 0xff, 0xff, // entry count
        ])
        .unwrap_err();

        assert!(
            matches!(err, DecoderError::NotEnoughData(message) if message.contains("iinf entry count"))
        );
    }

    #[test]
    fn rejects_iloc_item_count_that_cannot_fit_in_payload() {
        let err = parse_iloc(&[
            2, 0, 0, 0, // version and flags
            0, 0, // field sizes
            0xff, 0xff, 0xff, 0xff, // item count
        ])
        .unwrap_err();

        assert!(
            matches!(err, DecoderError::NotEnoughData(message) if message.contains("iloc item count"))
        );
    }

    #[test]
    fn rejects_iloc_extent_count_that_cannot_fit_in_payload() {
        let err = parse_iloc(&[
            0, 0, 0, 0, // version and flags
            0x44, 0, // four-byte offsets and lengths, no base offset
            0, 1, // item count
            0, 1, // item id
            0, 0, // data reference index
            0xff, 0xff, // extent count
        ])
        .unwrap_err();

        assert!(
            matches!(err, DecoderError::NotEnoughData(message) if message.contains("iloc extent count"))
        );
    }

    #[test]
    fn parses_idat_construction_method() {
        let (locations, methods) = parse_iloc_with_methods(&[
            1, 0, 0, 0, // version and flags
            0x44, 0, // four-byte offsets and lengths, no base/index
            0, 1, // item count
            0, 1, // item id
            0, 1, // idat construction method
            0, 0, // data reference index
            0, 1, // extent count
            0, 0, 0, 1, // extent offset
            0, 0, 0, 3, // extent length
        ])
        .unwrap();

        assert_eq!(methods, vec![(1, 1)]);
        assert_eq!(locations[0].item_id, 1);
        assert_eq!(locations[0].extents[0].offset, 1);
        assert_eq!(locations[0].extents[0].length, 3);
    }

    #[test]
    fn parses_item_offset_construction_method() {
        let (_, methods) = parse_iloc_with_methods(&[
            1, 0, 0, 0, // version and flags
            0x44, 0, // four-byte offsets and lengths, no base/index
            0, 1, // item count
            0, 1, // item id
            0, 2, // item-offset construction method
            0, 0, // data reference index
            0, 1, // extent count
            0, 0, 0, 1, // item offset
            0, 0, 0, 2, // extent length
        ])
        .unwrap();

        assert_eq!(methods, vec![(1, 2)]);
    }

    #[test]
    fn parses_explicit_item_offset_extent_indexes() {
        let (_, methods, indexes) = parse_iloc_with_indexes(&[
            1, 0, 0, 0, // version and flags
            0x44, 0x04, // four-byte offsets, lengths and extent indexes
            0, 1, // item count
            0, 1, // item id
            0, 2, // item-offset construction method
            0, 0, // data reference index
            0, 2, // extent count
            0, 0, 0, 2, // extent index 2
            0, 0, 0, 1, // item offset
            0, 0, 0, 2, // extent length
            0, 0, 0, 1, // extent index 1
            0, 0, 0, 0, // item offset
            0, 0, 0, 2, // extent length
        ])
        .unwrap();

        assert_eq!(methods, vec![(1, 2)]);
        assert_eq!(indexes, vec![(1, vec![2, 1])]);
    }

    #[test]
    fn rejects_zero_item_offset_extent_index() {
        let error = parse_iloc_with_indexes(&[
            1, 0, 0, 0, // version and flags
            0x44, 0x04, // four-byte offsets, lengths and extent indexes
            0, 1, // item count
            0, 1, // item id
            0, 2, // item-offset construction method
            0, 0, // data reference index
            0, 1, // extent count
            0, 0, 0, 0, // reserved zero extent index
            0, 0, 0, 0, // item offset
            0, 0, 0, 1, // extent length
        ])
        .unwrap_err();

        assert!(
            matches!(error, DecoderError::Bitstream(message) if message.contains("extent index is zero"))
        );
    }

    #[test]
    fn resolves_item_payload_from_idat() {
        let state = MetaState {
            primary_item_id: Some(1),
            item_locations: vec![ItemLocation {
                item_id: 1,
                base_offset: 0,
                extents: vec![ItemExtent {
                    offset: 1,
                    length: 3,
                }],
            }],
            item_construction_methods: vec![(1, 1)],
            idat_payload: Some(b"xyz123".to_vec()),
            ..MetaState::default()
        };

        assert_eq!(primary_item_payload(b"unused", &state).unwrap(), b"yz1");
    }

    #[test]
    fn resolves_item_payload_from_item_offset_reference() {
        let state = MetaState {
            primary_item_id: Some(1),
            item_locations: vec![
                ItemLocation {
                    item_id: 1,
                    base_offset: 0,
                    extents: vec![ItemExtent {
                        offset: 1,
                        length: 2,
                    }],
                },
                ItemLocation {
                    item_id: 2,
                    base_offset: 0,
                    extents: vec![ItemExtent {
                        offset: 0,
                        length: 5,
                    }],
                },
            ],
            item_construction_methods: vec![(1, 2), (2, 0)],
            item_references: vec![ItemReference {
                reference_type: *b"iloc",
                from_item_id: 1,
                to_item_ids: vec![2],
            }],
            ..MetaState::default()
        };

        assert_eq!(primary_item_payload(b"abcde", &state).unwrap(), b"bc");
    }

    #[test]
    fn resolves_item_payload_from_explicit_item_offset_indexes() {
        let state = MetaState {
            primary_item_id: Some(1),
            item_locations: vec![
                ItemLocation {
                    item_id: 1,
                    base_offset: 0,
                    extents: vec![
                        ItemExtent {
                            offset: 1,
                            length: 2,
                        },
                        ItemExtent {
                            offset: 2,
                            length: 2,
                        },
                    ],
                },
                ItemLocation {
                    item_id: 2,
                    base_offset: 0,
                    extents: vec![ItemExtent {
                        offset: 0,
                        length: 5,
                    }],
                },
                ItemLocation {
                    item_id: 3,
                    base_offset: 5,
                    extents: vec![ItemExtent {
                        offset: 0,
                        length: 5,
                    }],
                },
            ],
            item_construction_methods: vec![(1, 2), (2, 0), (3, 0)],
            item_extent_indexes: vec![(1, vec![2, 1])],
            item_references: vec![ItemReference {
                reference_type: *b"iloc",
                from_item_id: 1,
                to_item_ids: vec![2, 3],
            }],
            ..MetaState::default()
        };

        assert_eq!(
            primary_item_payload(b"abcdefghij", &state).unwrap(),
            b"ghcd"
        );
    }

    #[test]
    fn rejects_item_offset_reference_cycles() {
        let state = MetaState {
            primary_item_id: Some(1),
            item_locations: vec![
                ItemLocation {
                    item_id: 1,
                    base_offset: 0,
                    extents: vec![ItemExtent {
                        offset: 0,
                        length: 1,
                    }],
                },
                ItemLocation {
                    item_id: 2,
                    base_offset: 0,
                    extents: vec![ItemExtent {
                        offset: 0,
                        length: 1,
                    }],
                },
            ],
            item_construction_methods: vec![(1, 2), (2, 2)],
            item_references: vec![
                ItemReference {
                    reference_type: *b"iloc",
                    from_item_id: 1,
                    to_item_ids: vec![2],
                },
                ItemReference {
                    reference_type: *b"iloc",
                    from_item_id: 2,
                    to_item_ids: vec![1],
                },
            ],
            ..MetaState::default()
        };

        let error = primary_item_payload(b"unused", &state).unwrap_err();
        assert!(
            matches!(error, DecoderError::Bitstream(message) if message.contains("reference cycle"))
        );
    }

    #[test]
    fn rejects_idat_item_without_idat_box() {
        let state = MetaState {
            primary_item_id: Some(1),
            item_locations: vec![ItemLocation {
                item_id: 1,
                base_offset: 0,
                extents: vec![ItemExtent {
                    offset: 0,
                    length: 1,
                }],
            }],
            item_construction_methods: vec![(1, 1)],
            ..MetaState::default()
        };

        let err = primary_item_payload(b"unused", &state).unwrap_err();
        assert!(
            matches!(err, DecoderError::Bitstream(message) if message.contains("missing idat box"))
        );
    }

    #[test]
    fn rejects_ipma_entry_count_that_cannot_fit_in_payload() {
        let err = parse_ipma(&[
            0, 0, 0, 0, // version and flags
            0xff, 0xff, 0xff, 0xff, // entry count
        ])
        .unwrap_err();

        assert!(
            matches!(err, DecoderError::NotEnoughData(message) if message.contains("ipma entry count"))
        );
    }

    #[test]
    fn rejects_item_extent_payload_larger_than_source_file() {
        let state = MetaState {
            primary_item_id: Some(1),
            item_locations: vec![ItemLocation {
                item_id: 1,
                base_offset: 0,
                extents: vec![
                    ItemExtent {
                        offset: 0,
                        length: 2,
                    },
                    ItemExtent {
                        offset: 0,
                        length: 2,
                    },
                ],
            }],
            ..MetaState::default()
        };

        let err = primary_item_payload(&[0, 1], &state).unwrap_err();

        assert!(
            matches!(err, DecoderError::Bitstream(message) if message.contains("exceeds file size"))
        );
    }

    #[test]
    fn rejects_truncated_large_grid_payload() {
        let err = parse_grid_payload(&[
            0, 1, // version, large-field flag
            0, 0, // rows minus 1, columns minus 1
            0, 0, 0, 1, // width
        ])
        .unwrap_err();

        assert!(
            matches!(err, DecoderError::NotEnoughData(message) if message.contains("large grid"))
        );
    }

    #[test]
    fn color_information_exposes_nclx_and_icc_payloads() {
        let nclx = parse_colr(&[
            b'n', b'c', b'l', b'x', //
            0, 1, 0, 13, 0, 0, 0x80,
        ])
        .unwrap();
        assert_eq!(
            nclx.nclx(),
            Some(NclxColorInformation {
                color_primaries: 1,
                transfer_characteristics: 13,
                matrix_coefficients: 0,
                full_range_flag: true,
            })
        );
        assert_eq!(nclx.icc_profile(), None);

        let icc = parse_colr(&[b'p', b'r', b'o', b'f', 1, 2, 3, 4]).unwrap();
        assert_eq!(icc.nclx(), None);
        assert_eq!(icc.icc_profile(), Some(&[1, 2, 3, 4][..]));
    }

    #[test]
    fn parses_extended_pixi_channel_subsampling() {
        let pixi = parse_pixi(&[
            0, 0, 0, 1, // version and extended-pixi flag
            3, 8, 8, 8, // three 8-bit channels
            0x02, 0x00, // Y: subsampling type 0, location 0
            0x02, 0x20, // U: subsampling type 2, location 0
            0x02, 0x20, // V: subsampling type 2, location 0
        ])
        .unwrap();
        assert_eq!(pixi.bits_per_channel, vec![8, 8, 8]);
        assert_eq!(
            pixi.extended_channels,
            Some(vec![
                PixelChannelInformation {
                    channel_idc: 0,
                    component_format: 0,
                    subsampling: Some(PixelSubsampling {
                        subsampling_type: 0,
                        subsampling_location: 0,
                    }),
                },
                PixelChannelInformation {
                    channel_idc: 0,
                    component_format: 0,
                    subsampling: Some(PixelSubsampling {
                        subsampling_type: 2,
                        subsampling_location: 0,
                    }),
                },
                PixelChannelInformation {
                    channel_idc: 0,
                    component_format: 0,
                    subsampling: Some(PixelSubsampling {
                        subsampling_type: 2,
                        subsampling_location: 0,
                    }),
                },
            ])
        );
    }

    #[test]
    fn rejects_unsupported_extended_pixi_channel_fields() {
        let base = vec![0, 0, 0, 1, 1, 8, 0x02, 0x00];
        let mut version = base.clone();
        version[0] = 1;
        assert!(matches!(
            parse_pixi(&version),
            Err(DecoderError::Unsupported(message)) if message.contains("version")
        ));

        let mut flags = base.clone();
        flags[3] = 2;
        assert!(matches!(
            parse_pixi(&flags),
            Err(DecoderError::Unsupported(message)) if message.contains("flags")
        ));

        let mut channel_idc = base.clone();
        channel_idc[6] = 0x22;
        assert!(matches!(
            parse_pixi(&channel_idc),
            Err(DecoderError::Unsupported(message)) if message.contains("idc")
        ));

        let mut reserved = base.clone();
        reserved[6] = 0x12;
        assert!(matches!(
            parse_pixi(&reserved),
            Err(DecoderError::Bitstream(message)) if message.contains("reserved")
        ));

        let mut component_format = base.clone();
        component_format[6] = 0x06;
        assert!(matches!(
            parse_pixi(&component_format),
            Err(DecoderError::Unsupported(message)) if message.contains("component format")
        ));

        let mut subsampling_type = base;
        subsampling_type[7] = 0x50;
        assert!(matches!(
            parse_pixi(&subsampling_type),
            Err(DecoderError::Bitstream(message)) if message.contains("subsampling type")
        ));
    }

    #[test]
    fn parses_alpha_auxiliary_item_metadata_and_payload() {
        let state = MetaState {
            primary_item_id: Some(1),
            item_locations: vec![
                ItemLocation {
                    item_id: 1,
                    base_offset: 0,
                    extents: vec![ItemExtent {
                        offset: 0,
                        length: 4,
                    }],
                },
                ItemLocation {
                    item_id: 2,
                    base_offset: 0,
                    extents: vec![ItemExtent {
                        offset: 4,
                        length: 4,
                    }],
                },
            ],
            item_property_associations: vec![ItemPropertyAssociation {
                item_id: 2,
                associations: vec![PropertyAssociation {
                    index: 1,
                    essential: false,
                }],
            }],
            item_properties: vec![ItemProperty::AuxiliaryType(ALPHA_AUX_TYPE.to_string())],
            ..MetaState::default()
        };

        let alpha_items = alpha_auxiliary_items(b"mainalph", &state).unwrap();

        assert_eq!(
            alpha_items,
            vec![AuxiliaryImage {
                item_id: 2,
                aux_type: ALPHA_AUX_TYPE.to_string(),
                payload: b"alph".to_vec(),
            }]
        );
    }

    #[test]
    fn parses_auxiliary_item_references_as_alpha_fallback() {
        let payload = [
            0, 0, 0, 0, // iref version and flags
            0, 0, 0, 14, b'a', b'u', b'x', b'l', // auxl box header
            0, 1, // from item id
            0, 1, // reference count
            0, 2, // to item id
        ];

        let references = parse_iref(&payload).unwrap();

        assert_eq!(
            references,
            vec![ItemReference {
                reference_type: *b"auxl",
                from_item_id: 1,
                to_item_ids: vec![2],
            }]
        );
    }

    #[test]
    fn parses_auxiliary_type_and_property_association() {
        let auxc = parse_auxc(
            &[
                0, 0, 0, 0, // auxC full-box header
            ]
            .into_iter()
            .chain(ALPHA_AUX_TYPE.bytes())
            .chain([0])
            .collect::<Vec<_>>(),
        )
        .unwrap();
        assert_eq!(auxc, ALPHA_AUX_TYPE);

        let associations = parse_ipma(&[
            0, 0, 0, 0, // version and flags
            0, 0, 0, 1, // entry count
            0, 2,    // item id
            1,    // association count
            0x81, // essential property index 1
        ])
        .unwrap();

        assert_eq!(
            associations,
            vec![ItemPropertyAssociation {
                item_id: 2,
                associations: vec![PropertyAssociation {
                    index: 1,
                    essential: true,
                }],
            }]
        );
    }

    #[test]
    fn parses_layered_image_selector_properties() {
        assert_eq!(parse_a1op(&[0]).unwrap(), 0);
        assert_eq!(parse_lsel(&[0, 0]).unwrap(), 0);
        assert_eq!(parse_lsel(&[0xff, 0xff]).unwrap(), u16::MAX);
        assert_eq!(parse_a1lx(&[0, 0, 4, 0, 8, 0, 0]).unwrap(), [4, 8, 0]);
        assert_eq!(
            parse_a1lx(&[1, 0, 0, 0, 3, 0, 0, 0, 7, 0, 0, 0, 0]).unwrap(),
            [3, 7, 0]
        );
    }

    #[test]
    fn layered_image_selectors_fail_closed_outside_default_layer_policy() {
        let state = MetaState {
            primary_item_id: Some(1),
            item_infos: vec![ItemInfo {
                item_id: 1,
                item_type: *b"av01",
                item_name: "primary".to_string(),
            }],
            item_locations: vec![ItemLocation {
                item_id: 1,
                base_offset: 0,
                extents: vec![ItemExtent {
                    offset: 0,
                    length: 1,
                }],
            }],
            item_property_associations: vec![ItemPropertyAssociation {
                item_id: 1,
                associations: vec![
                    PropertyAssociation {
                        index: 1,
                        essential: true,
                    },
                    PropertyAssociation {
                        index: 2,
                        essential: true,
                    },
                    PropertyAssociation {
                        index: 3,
                        essential: true,
                    },
                    PropertyAssociation {
                        index: 4,
                        essential: true,
                    },
                ],
            }],
            item_properties: vec![
                ItemProperty::SpatialExtents(ImageSpatialExtents {
                    width: 1,
                    height: 1,
                }),
                ItemProperty::PixelInformation(PixelInformation {
                    bits_per_channel: vec![8, 8, 8],
                    extended_channels: None,
                }),
                ItemProperty::Av1Config(vec![0x81, 0, 0, 0]),
                ItemProperty::LayerSelector(1),
            ],
            ..MetaState::default()
        };

        let error = validate_primary_item_metadata(&state).unwrap_err();
        assert!(matches!(
            error,
            DecoderError::Unsupported(message) if message.contains("lsel layer 1")
        ));
    }

    #[test]
    fn tmap_primary_selects_the_referenced_base_av1_item() {
        let state = MetaState {
            primary_item_id: Some(10),
            item_infos: vec![
                ItemInfo {
                    item_id: 10,
                    item_type: *b"tmap",
                    item_name: "tone map".to_string(),
                },
                ItemInfo {
                    item_id: 11,
                    item_type: *b"av01",
                    item_name: "base".to_string(),
                },
                ItemInfo {
                    item_id: 12,
                    item_type: *b"av01",
                    item_name: "gain map".to_string(),
                },
            ],
            item_references: vec![ItemReference {
                reference_type: *b"dimg",
                from_item_id: 10,
                to_item_ids: vec![11, 12],
            }],
            ..MetaState::default()
        };

        assert_eq!(effective_primary_item_id(&state).unwrap(), 11);

        let mut without_base = state;
        without_base.item_references[0].to_item_ids = Vec::new();
        let error = effective_primary_item_id(&without_base).unwrap_err();
        assert!(matches!(
            error,
            DecoderError::Unsupported(message) if message.contains("base av01")
        ));
    }

    #[test]
    fn rejects_essential_unknown_item_property() {
        let state = MetaState {
            primary_item_id: Some(1),
            item_infos: vec![ItemInfo {
                item_id: 1,
                item_type: *b"av01",
                item_name: "primary".to_string(),
            }],
            item_locations: vec![ItemLocation {
                item_id: 1,
                base_offset: 0,
                extents: vec![ItemExtent {
                    offset: 0,
                    length: 1,
                }],
            }],
            item_property_associations: vec![ItemPropertyAssociation {
                item_id: 1,
                associations: vec![
                    PropertyAssociation {
                        index: 1,
                        essential: true,
                    },
                    PropertyAssociation {
                        index: 2,
                        essential: true,
                    },
                    PropertyAssociation {
                        index: 3,
                        essential: true,
                    },
                    PropertyAssociation {
                        index: 4,
                        essential: true,
                    },
                ],
            }],
            item_properties: vec![
                ItemProperty::SpatialExtents(ImageSpatialExtents {
                    width: 1,
                    height: 1,
                }),
                ItemProperty::PixelInformation(PixelInformation {
                    bits_per_channel: vec![8, 8, 8],
                    extended_channels: None,
                }),
                ItemProperty::Av1Config(vec![0x81, 0, 0, 0]),
                ItemProperty::Other,
            ],
            ..MetaState::default()
        };

        let error = validate_primary_item_metadata(&state).unwrap_err();
        assert!(matches!(
            error,
            DecoderError::Unsupported(message) if message.contains("essential unsupported property")
        ));
    }

    #[test]
    fn rejects_primary_metadata_with_out_of_range_property_association() {
        let state = MetaState {
            primary_item_id: Some(1),
            item_infos: vec![ItemInfo {
                item_id: 1,
                item_type: *b"av01",
                item_name: "primary".to_string(),
            }],
            item_locations: vec![ItemLocation {
                item_id: 1,
                base_offset: 0,
                extents: vec![ItemExtent {
                    offset: 0,
                    length: 1,
                }],
            }],
            item_property_associations: vec![ItemPropertyAssociation {
                item_id: 1,
                associations: vec![PropertyAssociation {
                    index: 2,
                    essential: false,
                }],
            }],
            item_properties: vec![ItemProperty::Other],
            ..MetaState::default()
        };

        let error = validate_primary_item_metadata(&state).unwrap_err();
        assert!(
            matches!(error, DecoderError::Bitstream(message) if message.contains("out of range"))
        );
    }

    #[test]
    fn parses_item_info_for_grid_items() {
        let payload = [
            0, 0, 0, 0, // iinf version and flags
            0, 1, // entry count
            0, 0, 0, 24, b'i', b'n', b'f', b'e', // infe box header
            2, 0, 0, 0, // infe version and flags
            0, 7, // item id
            0, 0, // protection index
            b'g', b'r', b'i', b'd', // item type
            b'g', b'r', b'i', b'd', 0, // item name
        ];

        let infos = parse_iinf(&payload).unwrap();

        assert_eq!(
            infos,
            vec![ItemInfo {
                item_id: 7,
                item_type: *b"grid",
                item_name: "grid".to_string(),
            }]
        );
    }

    #[test]
    fn parses_grid_payload_small_and_large_dimensions() {
        assert_eq!(
            parse_grid_payload(&[
                0, 0, // version, flags
                1, 2, // rows_minus_one, columns_minus_one
                0x03, 0x20, // output width 800
                0x02, 0x58, // output height 600
            ])
            .unwrap(),
            ParsedGridPayload {
                rows: 2,
                columns: 3,
                output_width: 800,
                output_height: 600,
            }
        );

        assert_eq!(
            parse_grid_payload(&[
                0, 1, // version, large-fields flag
                0, 0, // rows_minus_one, columns_minus_one
                0, 0, 0x10, 0, // output width 4096
                0, 0, 0x08, 0, // output height 2048
            ])
            .unwrap(),
            ParsedGridPayload {
                rows: 1,
                columns: 1,
                output_width: 4096,
                output_height: 2048,
            }
        );
    }

    #[test]
    fn primary_grid_requires_ordered_cell_references() {
        let state = MetaState {
            primary_item_id: Some(7),
            item_infos: vec![ItemInfo {
                item_id: 7,
                item_type: *b"grid",
                item_name: "grid".to_string(),
            }],
            ..MetaState::default()
        };
        let payload = [
            0, 0, // version, flags
            1, 1, // rows_minus_one, columns_minus_one
            0, 10, // output width
            0, 20, // output height
        ];

        let error = primary_grid(&[0; 4], &payload, &state).unwrap_err();
        assert!(matches!(error, DecoderError::Bitstream(message) if message.contains("dimg")));
    }

    #[test]
    fn parses_clap_irot_and_imir_properties() {
        assert_eq!(
            parse_clap(&[
                0, 0, 3, 32, // width_n 800
                0, 0, 0, 1, // width_d
                0, 0, 2, 88, // height_n 600
                0, 0, 0, 1, // height_d
                0, 0, 0, 0, // horiz offset n
                0, 0, 0, 1, // horiz offset d
                0, 0, 0, 0, // vert offset n
                0, 0, 0, 1, // vert offset d
            ])
            .unwrap(),
            CleanAperture {
                width_n: 800,
                width_d: 1,
                height_n: 600,
                height_d: 1,
                horizontal_offset_n: 0,
                horizontal_offset_d: 1,
                vertical_offset_n: 0,
                vertical_offset_d: 1,
            }
        );
        assert_eq!(parse_irot(&[5]).unwrap(), ImageRotation { angle: 1 });
        assert_eq!(parse_imir(&[3]).unwrap(), ImageMirror { axis: 1 });
    }

    #[test]
    fn primary_item_transforms_are_exposed_from_property_associations() {
        let clap = CleanAperture {
            width_n: 800,
            width_d: 1,
            height_n: 600,
            height_d: 1,
            horizontal_offset_n: 0,
            horizontal_offset_d: 1,
            vertical_offset_n: 0,
            vertical_offset_d: 1,
        };
        let state = MetaState {
            primary_item_id: Some(7),
            item_property_associations: vec![ItemPropertyAssociation {
                item_id: 7,
                associations: vec![
                    PropertyAssociation {
                        index: 1,
                        essential: false,
                    },
                    PropertyAssociation {
                        index: 2,
                        essential: false,
                    },
                    PropertyAssociation {
                        index: 3,
                        essential: false,
                    },
                ],
            }],
            item_properties: vec![
                ItemProperty::CleanAperture(clap),
                ItemProperty::Rotation(ImageRotation { angle: 2 }),
                ItemProperty::Mirror(ImageMirror { axis: 1 }),
            ],
            ..MetaState::default()
        };

        assert_eq!(
            primary_item_transforms(&state),
            PrimaryItemTransforms {
                clean_aperture: Some(clap),
                rotation: Some(ImageRotation { angle: 2 }),
                mirror: Some(ImageMirror { axis: 1 }),
            }
        );
    }

    #[test]
    fn primary_metadata_preserves_premultiplied_alpha_property() {
        let state = MetaState {
            item_property_associations: vec![ItemPropertyAssociation {
                item_id: 7,
                associations: vec![PropertyAssociation {
                    index: 1,
                    essential: false,
                }],
            }],
            item_properties: vec![ItemProperty::Premultiplied],
            ..MetaState::default()
        };

        assert!(item_metadata(&state, 7).unwrap().alpha_premultiplied);
    }

    #[test]
    fn primary_metadata_resolves_only_properties_associated_with_primary_item() {
        let primary_color = ColorInformation {
            color_type: *b"nclx",
            payload: vec![0, 1, 0, 13, 0, 0, 0x80],
        };
        let auxiliary_color = ColorInformation {
            color_type: *b"prof",
            payload: vec![9, 8, 7],
        };
        let state = MetaState {
            primary_item_id: Some(1),
            item_infos: vec![
                ItemInfo {
                    item_id: 1,
                    item_type: *b"av01",
                    item_name: "primary".to_string(),
                },
                ItemInfo {
                    item_id: 2,
                    item_type: *b"av01",
                    item_name: "auxiliary".to_string(),
                },
            ],
            item_locations: vec![
                ItemLocation {
                    item_id: 1,
                    base_offset: 0,
                    extents: vec![ItemExtent {
                        offset: 0,
                        length: 1,
                    }],
                },
                ItemLocation {
                    item_id: 2,
                    base_offset: 0,
                    extents: vec![ItemExtent {
                        offset: 0,
                        length: 1,
                    }],
                },
            ],
            item_properties: vec![
                ItemProperty::SpatialExtents(ImageSpatialExtents {
                    width: 1204,
                    height: 800,
                }),
                ItemProperty::PixelInformation(PixelInformation {
                    bits_per_channel: vec![8, 8, 8],
                    extended_channels: None,
                }),
                ItemProperty::Av1Config(vec![1, 2, 3]),
                ItemProperty::ColorInformation(primary_color.clone()),
                ItemProperty::SpatialExtents(ImageSpatialExtents {
                    width: 10,
                    height: 10,
                }),
                ItemProperty::PixelInformation(PixelInformation {
                    bits_per_channel: vec![10],
                    extended_channels: None,
                }),
                ItemProperty::Av1Config(vec![9]),
                ItemProperty::ColorInformation(auxiliary_color),
            ],
            item_property_associations: vec![
                ItemPropertyAssociation {
                    item_id: 1,
                    associations: vec![
                        PropertyAssociation {
                            index: 1,
                            essential: true,
                        },
                        PropertyAssociation {
                            index: 2,
                            essential: false,
                        },
                        PropertyAssociation {
                            index: 3,
                            essential: true,
                        },
                        PropertyAssociation {
                            index: 4,
                            essential: false,
                        },
                    ],
                },
                ItemPropertyAssociation {
                    item_id: 2,
                    associations: vec![
                        PropertyAssociation {
                            index: 5,
                            essential: true,
                        },
                        PropertyAssociation {
                            index: 6,
                            essential: false,
                        },
                        PropertyAssociation {
                            index: 7,
                            essential: true,
                        },
                        PropertyAssociation {
                            index: 8,
                            essential: false,
                        },
                    ],
                },
            ],
            ..MetaState::default()
        };

        validate_primary_item_metadata(&state).unwrap();
        let metadata = primary_item_metadata(&state).unwrap();
        assert_eq!(metadata.width, Some(1204));
        assert_eq!(metadata.height, Some(800));
        assert_eq!(
            metadata.pixel_information.unwrap().bits_per_channel,
            vec![8, 8, 8]
        );
        assert_eq!(metadata.av1_config, Some(vec![1, 2, 3]));
        assert_eq!(metadata.color_information, Some(primary_color));
    }

    #[test]
    fn sato_primary_item_requires_derived_image_properties_not_av1c() {
        let state = MetaState {
            primary_item_id: Some(7),
            item_infos: vec![ItemInfo {
                item_id: 7,
                item_type: *b"sato",
                item_name: "sample transform".to_string(),
            }],
            item_locations: vec![ItemLocation {
                item_id: 7,
                base_offset: 0,
                extents: vec![ItemExtent {
                    offset: 0,
                    length: 2,
                }],
            }],
            item_properties: vec![
                ItemProperty::SpatialExtents(ImageSpatialExtents {
                    width: 1,
                    height: 1,
                }),
                ItemProperty::PixelInformation(PixelInformation {
                    bits_per_channel: vec![16, 16, 16],
                    extended_channels: None,
                }),
            ],
            item_property_associations: vec![ItemPropertyAssociation {
                item_id: 7,
                associations: vec![
                    PropertyAssociation {
                        index: 1,
                        essential: true,
                    },
                    PropertyAssociation {
                        index: 2,
                        essential: true,
                    },
                ],
            }],
            ..MetaState::default()
        };
        validate_primary_item_metadata(&state).unwrap();
    }

    #[test]
    fn official_sato_item_can_be_selected_as_primary_when_fixture_is_present() {
        let Some(path) = std::env::var_os("AVIF_SATO_SAMPLE") else {
            return;
        };
        let mut data = std::fs::read(path).expect("sato fixture should be readable");
        let mut state = MetaState::default();
        let mut pitm_payload_offset = None;
        for_each_top_level_box(&data, |top| {
            if &top.box_type != b"meta" {
                return Ok(());
            }
            let payload = box_payload(&data, top)?;
            parse_meta(&data, top, &mut state)?;
            let mut offset = 4usize;
            while offset < payload.len() {
                let child = read_box_header(payload, offset, payload.len())?;
                if &child.box_type == b"pitm" {
                    pitm_payload_offset =
                        Some(top.offset + top.header_size + child.offset + child.header_size);
                    break;
                }
                offset = checked_add(child.offset, child.size, "test meta child end")?;
            }
            Ok(())
        })
        .unwrap();
        let sato_id = state
            .item_infos
            .iter()
            .find(|item| item.item_type == *b"sato")
            .map(|item| item.item_id)
            .expect("official sato fixture should contain a sato item");
        let pitm = pitm_payload_offset.expect("sato fixture should contain pitm");
        match data[pitm] {
            0 => data[pitm + 4..pitm + 6].copy_from_slice(&(sato_id as u16).to_be_bytes()),
            1 => data[pitm + 4..pitm + 8].copy_from_slice(&sato_id.to_be_bytes()),
            version => panic!("unexpected pitm version {version}"),
        }
        let info = parse_avif(&data).expect("primary sato metadata should parse");
        assert_eq!(info.primary_item_id, Some(sato_id));
        let transform = parse_sample_transform(&data)
            .unwrap()
            .expect("primary sato transform should be selected");
        assert_eq!(transform.output_bit_depth, 16);
        let frame = crate::decode_frame_bytes(&data).expect("primary sato frame should decode");
        assert_eq!((frame.width, frame.height), (1024, 684));
        assert_eq!(frame.bit_depth, 16);
        let image = crate::image_from_bytes(&data).expect("primary sato RGBA should decode");
        assert_eq!((image.width, image.height), (1024, 684));
        assert_eq!(image.rgba.len(), 1024 * 684 * 4);
    }

    #[test]
    fn merges_ipma_boxes_preserving_order_and_essential_flags() {
        let mut target = vec![ItemPropertyAssociation {
            item_id: 1,
            associations: vec![PropertyAssociation {
                index: 2,
                essential: true,
            }],
        }];
        merge_ipma(
            &mut target,
            vec![ItemPropertyAssociation {
                item_id: 1,
                associations: vec![PropertyAssociation {
                    index: 1,
                    essential: false,
                }],
            }],
        )
        .unwrap();
        assert_eq!(
            target[0].associations,
            vec![
                PropertyAssociation {
                    index: 2,
                    essential: true,
                },
                PropertyAssociation {
                    index: 1,
                    essential: false,
                },
            ]
        );
    }

    #[test]
    fn rejects_duplicate_ipma_and_singleton_property_associations() {
        let mut target = vec![ItemPropertyAssociation {
            item_id: 1,
            associations: vec![PropertyAssociation {
                index: 1,
                essential: false,
            }],
        }];
        let duplicate = merge_ipma(
            &mut target,
            vec![ItemPropertyAssociation {
                item_id: 1,
                associations: vec![PropertyAssociation {
                    index: 1,
                    essential: true,
                }],
            }],
        )
        .unwrap_err();
        assert!(
            matches!(duplicate, DecoderError::Bitstream(message) if message.contains("duplicate property association"))
        );

        let state = MetaState {
            primary_item_id: Some(1),
            item_infos: vec![ItemInfo {
                item_id: 1,
                item_type: *b"av01",
                item_name: "primary".to_string(),
            }],
            item_locations: vec![ItemLocation {
                item_id: 1,
                base_offset: 0,
                extents: vec![ItemExtent {
                    offset: 0,
                    length: 1,
                }],
            }],
            item_properties: vec![
                ItemProperty::SpatialExtents(ImageSpatialExtents {
                    width: 1,
                    height: 1,
                }),
                ItemProperty::SpatialExtents(ImageSpatialExtents {
                    width: 2,
                    height: 2,
                }),
                ItemProperty::PixelInformation(PixelInformation {
                    bits_per_channel: vec![8, 8, 8],
                    extended_channels: None,
                }),
                ItemProperty::Av1Config(vec![1]),
            ],
            item_property_associations: vec![ItemPropertyAssociation {
                item_id: 1,
                associations: vec![
                    PropertyAssociation {
                        index: 1,
                        essential: false,
                    },
                    PropertyAssociation {
                        index: 2,
                        essential: false,
                    },
                    PropertyAssociation {
                        index: 3,
                        essential: false,
                    },
                    PropertyAssociation {
                        index: 4,
                        essential: false,
                    },
                ],
            }],
            ..MetaState::default()
        };
        let duplicate_singleton = validate_primary_item_metadata(&state).unwrap_err();
        assert!(
            matches!(duplicate_singleton, DecoderError::Bitstream(message) if message.contains("duplicate SpatialExtents"))
        );
    }

    #[test]
    fn rejects_zero_ipma_property_index() {
        let associations = parse_ipma(&[
            0, 0, 0, 0, // version and flags
            0, 0, 0, 1, // entry count
            0, 1, // item id
            1, // association count
            0, // reserved/out-of-range property index
        ])
        .unwrap();
        let state = MetaState {
            primary_item_id: Some(1),
            item_infos: vec![ItemInfo {
                item_id: 1,
                item_type: *b"av01",
                item_name: "primary".to_string(),
            }],
            item_locations: vec![ItemLocation {
                item_id: 1,
                base_offset: 0,
                extents: vec![ItemExtent {
                    offset: 0,
                    length: 1,
                }],
            }],
            item_property_associations: associations,
            ..MetaState::default()
        };
        let error = validate_primary_item_metadata(&state).unwrap_err();
        assert!(
            matches!(error, DecoderError::Bitstream(message) if message.contains("out of range"))
        );
    }

    #[test]
    fn accepts_repeated_identical_avis_sample_descriptions() {
        let descriptions = vec![
            Some(vec![b'a', b'v', b'0', b'1']),
            Some(vec![b'a', b'v', b'0', b'1']),
        ];
        let (offsets, indices) =
            build_sample_offsets_with_descriptions(&[100], &[(1, 2, 2)], &[3, 4], &descriptions)
                .unwrap();
        assert_eq!(offsets, vec![100, 103]);
        assert_eq!(indices, vec![1, 1]);
    }

    #[test]
    fn accepts_differing_avis_sample_descriptions_for_later_validation() {
        let descriptions = vec![
            Some(vec![b'a', b'v', b'0', b'1']),
            Some(vec![b'a', b'v', b'0', b'2']),
        ];
        let (offsets, indices) =
            build_sample_offsets_with_descriptions(&[100], &[(1, 1, 2)], &[3], &descriptions)
                .unwrap();
        assert_eq!(offsets, vec![100]);
        assert_eq!(indices, vec![1]);
    }

    #[test]
    fn differing_avis_description_requires_a_per_sample_sequence_header() {
        let first = *b"av01-description-one";
        let changed = *b"av01-description-two";
        let sequence_header = [0x0a, 0x00];
        validate_sequence_sample_description(&changed, &first, &sequence_header).unwrap();

        let frame_only = [0x32, 0x01, 0x00];
        let error =
            validate_sequence_sample_description(&changed, &first, &frame_only).unwrap_err();
        assert!(matches!(
            error,
            DecoderError::Unsupported(message)
                if message.contains("require a sequence header")
        ));
    }
}
