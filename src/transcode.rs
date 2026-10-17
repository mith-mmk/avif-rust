//! Native AVIF preservation planning, separate from shared pixel storage.
//! The writer rebuilds a primary/alpha container. Byte/property-order/graph
//! preservation is a separate unsupported request, never implied by lossless.
use crate::buffer::*;
use crate::encoder::{self, ColorPolicyEx, EncodeOptionsEx, EncodeReportEx, NativeEncodeOptions};
use crate::{NativeDecodeLimits, NativeImageEx, NativePropertyRecord};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreservationFeature {
    CleanAperture,
    Rotation,
    Mirror,
    PixelAspectRatio,
    RenderDimensions,
    UnknownProperty([u8; 4]),
    UnknownColr([u8; 4]),
    ExtendedPixelInformation,
    ExtraItems,
    /// AV1 Metadata OBUs are not re-emitted by the current writer.
    Av1Metadata,
    /// Keep the decoder-selected first alpha and omit the additional items.
    AdditionalAlphaItems,
    DerivedStructure,
    SequenceOrGrid,
    OtherTopLevel,
    OtherMetadata,
    OtherReferences,
    EntityGroups,
    ItemNames,
    AuxiliaryMetadata,
    UnassociatedProperties,
    MissingSourceInventory,
    IccProfile,
    IccPropertyType,
    ContainerStructure,
    NativeSampleHandoff,
    Depth,
    Subsampling,
    Alpha,
    ColorSignalling,
}
/// Resolve source color authorities explicitly; none of these applies a color transform.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum ColorAuthority {
    #[default]
    RequireAgreement,
    Av1,
    Container,
}
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PreservationPolicy {
    /// Individual fields/features which the caller permits this writer to omit.
    pub allow_drop: Vec<PreservationFeature>,
    pub color_authority: ColorAuthority,
    /// The current writer cannot preserve original graph/property order/bytes.
    pub require_container_structure: bool,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignallingChange {
    ContainerNclx,
    Av1Description,
    Av1RangeInterpretation,
}
/// Preflight previews have encoded=false; success reports verify output signalling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeTranscodeReport {
    pub encoded: bool,
    pub encode: EncodeReportEx,
    pub preserved: Vec<PreservationFeature>,
    pub dropped: Vec<PreservationFeature>,
    /// Source item selected by the decoder; output item IDs may be rebuilt.
    pub retained_alpha_item_id: Option<u32>,
    /// Source alpha IDs omitted with explicit AdditionalAlphaItems permission.
    pub dropped_alpha_item_ids: Vec<u32>,
    /// Count omitted with explicit Av1Metadata permission (item data + configOBUs).
    pub dropped_av1_metadata_obus: usize,
    pub signalling_changes: Vec<SignallingChange>,
    pub color_authority: ColorAuthority,
    pub container_signalling_verified: bool,
    /// No automatic output pixel decode is performed by this API.
    pub pixel_roundtrip_verified: bool,
    pub container_structure_preserved: bool,
}
#[derive(Debug)]
pub struct NativeTranscodeResult {
    pub bytes: Vec<u8>,
    pub report: NativeTranscodeReport,
}
#[derive(Debug)]
pub enum PreservationError {
    PreservationUnsupported(PreservationFeature),
    UnsupportedStructure(PreservationFeature),
    ColorAuthorityConflict(&'static str),
    ConversionRequired(&'static str),
    Buffer(BufferError),
    Encoder(encoder::EncodeErrorEx),
    Decoder(crate::DecoderError),
    OutputVerification(&'static str),
}
impl fmt::Display for PreservationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PreservationUnsupported(v) => write!(
                f,
                "preservation unsupported: {v:?}; select an individual drop or supported explicit conversion"
            ),
            Self::UnsupportedStructure(v) => write!(f, "source structure unsupported: {v:?}"),
            Self::ColorAuthorityConflict(v) => write!(f, "color authority conflict: {v}"),
            Self::ConversionRequired(v) => write!(f, "explicit conversion required: {v}"),
            Self::Buffer(e) => write!(f, "{e}"),
            Self::Encoder(e) => write!(f, "{e}"),
            Self::Decoder(e) => write!(f, "{e}"),
            Self::OutputVerification(v) => write!(f, "encoded signalling verification failed: {v}"),
        }
    }
}
impl std::error::Error for PreservationError {}
impl From<BufferError> for PreservationError {
    fn from(e: BufferError) -> Self {
        Self::Buffer(e)
    }
}
impl From<encoder::EncodeErrorEx> for PreservationError {
    fn from(e: encoder::EncodeErrorEx) -> Self {
        Self::Encoder(e)
    }
}
impl From<crate::DecoderError> for PreservationError {
    fn from(e: crate::DecoderError) -> Self {
        Self::Decoder(e)
    }
}

fn push<T: PartialEq>(values: &mut Vec<T>, value: T) -> Result<(), BufferError> {
    if !values.contains(&value) {
        values.try_reserve(1).map_err(|_| BufferError::Allocation)?;
        values.push(value);
    }
    Ok(())
}
fn drop_or_reject(
    feature: PreservationFeature,
    policy: &PreservationPolicy,
    drops: &mut Vec<PreservationFeature>,
) -> Result<(), PreservationError> {
    if !policy.allow_drop.contains(&feature) {
        return Err(match feature {
            PreservationFeature::ExtraItems
            | PreservationFeature::DerivedStructure
            | PreservationFeature::SequenceOrGrid
            | PreservationFeature::ContainerStructure => {
                PreservationError::UnsupportedStructure(feature)
            }
            _ => PreservationError::PreservationUnsupported(feature),
        });
    }
    push(drops, feature)?;
    Ok(())
}
fn nclx_tuple(c: crate::NclxColorInformation) -> Cicp {
    Cicp {
        primaries: c.color_primaries,
        transfer: c.transfer_characteristics,
        matrix: c.matrix_coefficients,
    }
}
fn output_tuple(c: encoder::NclxColorInformation) -> Cicp {
    Cicp {
        primaries: c.color_primaries,
        transfer: c.transfer_characteristics,
        matrix: c.matrix_coefficients,
    }
}

// The AVIF planner owns a short-lived encode request. Copy its metadata
// fallibly rather than using Clone on caller-controlled ICC/advanced values.
fn copied<T: Copy>(values: &[T]) -> Result<Vec<T>, BufferError> {
    let mut out = Vec::new();
    out.try_reserve_exact(values.len())
        .map_err(|_| BufferError::Allocation)?;
    out.extend_from_slice(values);
    Ok(out)
}
fn copied_string(value: &str) -> Result<String, BufferError> {
    let mut out = String::new();
    out.try_reserve_exact(value.len())
        .map_err(|_| BufferError::Allocation)?;
    out.push_str(value);
    Ok(out)
}
fn copied_options(native: &NativeEncodeOptions) -> Result<NativeEncodeOptions, BufferError> {
    let e = &native.encoder;
    let mut advanced = encoder::Metadata::new();
    advanced
        .try_reserve(e.advanced.len())
        .map_err(|_| BufferError::Allocation)?;
    for (key, value) in &e.advanced {
        let value = match value {
            encoder::DataMap::UInt(v) => encoder::DataMap::UInt(*v),
            encoder::DataMap::UIntAllay(v) => encoder::DataMap::UIntAllay(copied(v)?),
            encoder::DataMap::Raw(v) => encoder::DataMap::Raw(copied(v)?),
            encoder::DataMap::Ascii(v) => encoder::DataMap::Ascii(copied_string(v)?),
            encoder::DataMap::None => encoder::DataMap::None,
        };
        advanced.insert(copied_string(key)?, value);
    }
    let color_information = e
        .color_information
        .as_ref()
        .map(|v| {
            Ok::<_, BufferError>(encoder::ColorInformation {
                color_type: v.color_type,
                payload: copied(&v.payload)?,
            })
        })
        .transpose()?;
    let pixel_information = e
        .pixel_information
        .as_ref()
        .map(|v| {
            Ok::<_, BufferError>(encoder::PixelInformation {
                bits_per_channel: copied(&v.bits_per_channel)?,
                extended_channels: v.extended_channels.as_deref().map(copied).transpose()?,
            })
        })
        .transpose()?;
    Ok(NativeEncodeOptions {
        bit_depth: native.bit_depth,
        subsampling: native.subsampling,
        encoder: encoder::EncoderOptions {
            quality: e.quality,
            quality_alpha: e.quality_alpha,
            speed: e.speed,
            lossless: e.lossless,
            jobs: e.jobs,
            tile_rows_log2: e.tile_rows_log2,
            tile_cols_log2: e.tile_cols_log2,
            advanced,
            color_information,
            pixel_information,
        },
        color_information: encoder::NativeColorInformation {
            nclx: native.color_information.nclx,
            icc_profile: native
                .color_information
                .icc_profile
                .as_deref()
                .map(copied)
                .transpose()?,
            icc_color_type: native.color_information.icc_color_type,
        },
    })
}

fn prepare<'a>(
    source: &'a NativeImageEx,
    native: &NativeEncodeOptions,
    policy: &PreservationPolicy,
) -> Result<(ImageViewEx<'a>, EncodeOptionsEx, NativeTranscodeReport), PreservationError> {
    let context = source.context();
    let information = context.information();
    let info = information.info();
    let colors = information.color_information();
    let mut drops = Vec::new();
    let mut changes = Vec::new();
    if policy.require_container_structure {
        return Err(PreservationError::UnsupportedStructure(
            PreservationFeature::ContainerStructure,
        ));
    }
    let inventory = information.source_inventory();
    for (present, feature) in [
        (
            !inventory.scan_complete,
            PreservationFeature::MissingSourceInventory,
        ),
        (
            inventory.other_top_level,
            PreservationFeature::OtherTopLevel,
        ),
        (inventory.other_metadata, PreservationFeature::OtherMetadata),
        (
            inventory.other_references,
            PreservationFeature::OtherReferences,
        ),
        (inventory.entity_groups, PreservationFeature::EntityGroups),
        (
            inventory.av1_metadata_obus != 0,
            PreservationFeature::Av1Metadata,
        ),
        (
            info.alpha_auxiliary_items.len() > 1,
            PreservationFeature::AdditionalAlphaItems,
        ),
        (inventory.custom_item_names, PreservationFeature::ItemNames),
        (
            inventory.unmodelled_auxiliary,
            PreservationFeature::AuxiliaryMetadata,
        ),
        (
            inventory.unassociated_properties,
            PreservationFeature::UnassociatedProperties,
        ),
        (
            information.primary_item_type() != Some(*b"av01"),
            PreservationFeature::DerivedStructure,
        ),
        (
            info.primary_grid.is_some()
                || info.alpha_grid.is_some()
                || !info.sequence_sample_payloads.is_empty()
                || info.major_brand == *b"avis",
            PreservationFeature::SequenceOrGrid,
        ),
        (
            information.item_count() > 1 + info.alpha_auxiliary_items.len(),
            PreservationFeature::ExtraItems,
        ),
        (
            info.clean_aperture.is_some(),
            PreservationFeature::CleanAperture,
        ),
        (info.rotation.is_some(), PreservationFeature::Rotation),
        (info.mirror.is_some(), PreservationFeature::Mirror),
        (
            information.pixel_aspect_ratio().is_some(),
            PreservationFeature::PixelAspectRatio,
        ),
        (
            context.coded_dimensions() != context.render_dimensions(),
            PreservationFeature::RenderDimensions,
        ),
        (
            info.pixel_information
                .as_ref()
                .is_some_and(|p| p.extended_channels.is_some()),
            PreservationFeature::ExtendedPixelInformation,
        ),
    ] {
        if present {
            drop_or_reject(feature, policy, &mut drops)?;
        }
    }
    for record in information.ordered_primary_properties() {
        if let NativePropertyRecord::Other(kind) = record {
            drop_or_reject(
                PreservationFeature::UnknownProperty(*kind),
                policy,
                &mut drops,
            )?;
        }
    }
    for colr in &colors.unknown_colr {
        drop_or_reject(
            PreservationFeature::UnknownColr(colr.color_type),
            policy,
            &mut drops,
        )?;
    }
    if colors.icc_profile.as_deref() != native.color_information.icc_profile.as_deref() {
        if colors.icc_profile.is_some() && native.color_information.icc_profile.is_none() {
            drop_or_reject(PreservationFeature::IccProfile, policy, &mut drops)?;
        } else {
            return Err(PreservationError::ConversionRequired(
                "ICC replacement/application is a separate explicit pixel conversion",
            ));
        }
    }
    if native.color_information.icc_profile.is_some()
        && colors.icc_color_type != native.color_information.icc_color_type
    {
        drop_or_reject(PreservationFeature::IccPropertyType, policy, &mut drops)?;
    }
    let cfg = context.color_config();
    let av1 = cfg.color_description.map(|c| Cicp {
        primaries: u16::from(c.color_primaries),
        transfer: u16::from(c.transfer_characteristics),
        matrix: u16::from(c.matrix_coefficients),
    });
    let av1_full = cfg.color_range == crate::av1::ColorRange::Full;
    let container = colors.nclx.map(nclx_tuple);
    let (chosen, full) = match policy.color_authority {
        ColorAuthority::RequireAgreement => {
            let (Some(a), Some(c)) = (av1, colors.nclx) else {
                return Err(PreservationError::ColorAuthorityConflict(
                    "strict signalling requires original nclx and explicit AV1 description",
                ));
            };
            if a != nclx_tuple(c) || av1_full != c.full_range_flag {
                return Err(PreservationError::ColorAuthorityConflict(
                    "nclx/AV1 description or range disagree",
                ));
            }
            (a, av1_full)
        }
        ColorAuthority::Av1 => {
            let Some(a) = av1 else {
                return Err(PreservationError::ColorAuthorityConflict(
                    "no AV1 description to select",
                ));
            };
            (a, av1_full)
        }
        ColorAuthority::Container => {
            let Some(c) = colors.nclx else {
                return Err(PreservationError::ColorAuthorityConflict(
                    "no container nclx to select",
                ));
            };
            (nclx_tuple(c), c.full_range_flag)
        }
    };
    let out = native
        .color_information
        .nclx
        .ok_or(PreservationError::ColorAuthorityConflict(
            "explicit output nclx missing",
        ))?;
    if output_tuple(out) != chosen || out.full_range_flag != full {
        return Err(PreservationError::ColorAuthorityConflict(
            "output options do not match the selected authority",
        ));
    }
    if container != Some(chosen) || colors.nclx.is_some_and(|c| c.full_range_flag != full) {
        push(&mut changes, SignallingChange::ContainerNclx)?;
    }
    if av1 != Some(chosen) {
        push(&mut changes, SignallingChange::Av1Description)?;
    }
    if av1_full != full {
        push(&mut changes, SignallingChange::Av1RangeInterpretation)?;
    }
    let mut view = source.view()?;
    if let ImageViewEx::NativePlanar(ref mut planar) = view {
        // Explicit authority resolution changes interpretation only, never samples.
        if policy.color_authority == ColorAuthority::Container {
            planar.descriptor.range = Some(if full {
                ColorRange::Full
            } else {
                ColorRange::Limited
            });
        }
    } else {
        return Err(PreservationError::ConversionRequired(
            "native preservation cannot fall back through packed RGBA",
        ));
    }
    if context.bit_depth() != native.bit_depth {
        return Err(PreservationError::ConversionRequired(
            "native encoded depth differs; graph flattening permission does not permit precision reduction",
        ));
    }
    let options = EncodeOptionsEx {
        native: copied_options(native)?,
        precision: PrecisionPolicy::Exact,
        color: ColorPolicyEx::Explicit {
            allow_cicp_change: policy.color_authority != ColorAuthority::RequireAgreement,
            allow_icc_change: drops.contains(&PreservationFeature::IccProfile),
        },
    };
    let encode = encoder::preflight_ex(view, &options)?;
    let mut preserved = Vec::new();
    for field in [
        PreservationFeature::NativeSampleHandoff,
        PreservationFeature::Depth,
        PreservationFeature::Subsampling,
        PreservationFeature::Alpha,
    ] {
        push(&mut preserved, field)?;
    }
    if changes.is_empty() {
        push(&mut preserved, PreservationFeature::ColorSignalling)?;
    }
    if colors.icc_profile.is_some() && !drops.contains(&PreservationFeature::IccProfile) {
        push(&mut preserved, PreservationFeature::IccProfile)?;
    }
    if colors.icc_color_type.is_some()
        && !drops.contains(&PreservationFeature::IccPropertyType)
        && !drops.contains(&PreservationFeature::IccProfile)
    {
        push(&mut preserved, PreservationFeature::IccPropertyType)?;
    }
    let retained_alpha_item_id = info.alpha_auxiliary_items.first().map(|a| a.item_id);
    let mut dropped_alpha_item_ids = Vec::new();
    dropped_alpha_item_ids
        .try_reserve_exact(info.alpha_auxiliary_items.len().saturating_sub(1))
        .map_err(|_| BufferError::Allocation)?;
    dropped_alpha_item_ids.extend(info.alpha_auxiliary_items.iter().skip(1).map(|a| a.item_id));
    let report = NativeTranscodeReport {
        encoded: false,
        encode,
        preserved,
        dropped: drops,
        retained_alpha_item_id,
        dropped_alpha_item_ids,
        dropped_av1_metadata_obus: inventory.av1_metadata_obus,
        signalling_changes: changes,
        color_authority: policy.color_authority,
        container_signalling_verified: false,
        pixel_roundtrip_verified: false,
        container_structure_preserved: false,
    };
    Ok((view, options, report))
}

/// Plan preservation on an already bounded native decode. No encode or fallback.
pub fn preflight_native_transcode(
    source: &NativeImageEx,
    options: &NativeEncodeOptions,
    policy: &PreservationPolicy,
) -> Result<NativeTranscodeReport, PreservationError> {
    prepare(source, options, policy).map(|(_, _, report)| report)
}

/// Re-encode native samples with separate source context and fail-closed preflight.
/// Input parse/decode is not repeated; output metadata is independently parsed
/// under the supplied limits before bytes become a successful result.
pub fn transcode_native(
    source: &NativeImageEx,
    options: &NativeEncodeOptions,
    policy: &PreservationPolicy,
    output_limits: &NativeDecodeLimits,
) -> Result<NativeTranscodeResult, PreservationError> {
    let (view, options, mut report) = prepare(source, options, policy)?;
    let encoded = encoder::encode_ex(view, &options)?;
    let information = crate::parse_native_info(&encoded.bytes, output_limits)?;
    let info = information.info();
    let out = options
        .native
        .color_information
        .nclx
        .ok_or(PreservationError::OutputVerification("nclx missing"))?;
    let actual = information.color_information();
    let nclx = actual
        .nclx
        .ok_or(PreservationError::OutputVerification("writer omitted nclx"))?;
    if nclx_tuple(nclx) != output_tuple(out)
        || nclx.full_range_flag != out.full_range_flag
        || actual.icc_profile.as_deref() != options.native.color_information.icc_profile.as_deref()
        || actual.icc_color_type != options.native.color_information.icc_color_type
    {
        return Err(PreservationError::OutputVerification(
            "color/ICC bytes or property type changed",
        ));
    }
    if info.width.map(|v| v as usize) != Some(view.width())
        || info.height.map(|v| v as usize) != Some(view.height())
        || info.pixel_information.as_ref().is_none_or(|p| {
            p.bits_per_channel
                .iter()
                .any(|b| *b != options.native.bit_depth)
        })
    {
        return Err(PreservationError::OutputVerification(
            "dimensions/pixi depth",
        ));
    }
    let config = crate::av1::parse_av1_config(
        info.av1_config
            .as_deref()
            .ok_or(PreservationError::OutputVerification("av1C missing"))?,
    )?;
    let actual_depth = if config.high_bitdepth {
        if config.twelve_bit { 12 } else { 10 }
    } else {
        8
    };
    let expected = match options.native.subsampling {
        encoder::NativeSubsampling::Cs400 | encoder::NativeSubsampling::Cs420 => (true, true),
        encoder::NativeSubsampling::Cs422 => (true, false),
        encoder::NativeSubsampling::Cs444 => (false, false),
    };
    if actual_depth != options.native.bit_depth
        || (config.chroma_subsampling_x, config.chroma_subsampling_y) != expected
        || config.monochrome != (encoded.report.model == NativeModel::Monochrome)
        || info.alpha_auxiliary_items.len()
            != usize::from(encoded.report.alpha != AlphaMode::Absent)
        || information.item_count() != 1 + info.alpha_auxiliary_items.len()
    {
        return Err(PreservationError::OutputVerification(
            "AV1 depth/sampling/alpha",
        ));
    }
    report.encoded = true;
    report.encode = encoded.report;
    report.container_signalling_verified = true;
    Ok(NativeTranscodeResult {
        bytes: encoded.bytes,
        report,
    })
}
