//! Borrowed native selected-item planning and retained-result accounting.
//!
//! The compatibility parser intentionally remains in `container.rs`; this
//! module contains only the native still preflight and result ownership walk.

use super::container_budget::RetainedOwnerKind;
use super::*;

/// Visits the unique alpha items selected by the native still policy. An
/// `auxl` reference with no targets does not suppress the association-based
/// fallback used by the materializer.
pub(super) fn for_each_native_selected_alpha_item(
    state: &MetaState,
    owner_item_id: u32,
    mut visit: impl FnMut(u32) -> Result<(), DecoderError>,
) -> Result<(), DecoderError> {
    let mut has_target = false;
    for (reference_index, reference) in state.item_references.iter().enumerate() {
        if reference.reference_type != *b"auxl" || reference.from_item_id != owner_item_id {
            continue;
        }
        for (target_index, item_id) in reference.to_item_ids.iter().copied().enumerate() {
            has_target = true;
            let duplicate = state
                .item_references
                .iter()
                .enumerate()
                .take(reference_index + 1)
                .filter(|(_, previous)| {
                    previous.reference_type == *b"auxl" && previous.from_item_id == owner_item_id
                })
                .any(|(previous_index, previous)| {
                    let count = if previous_index == reference_index {
                        target_index
                    } else {
                        previous.to_item_ids.len()
                    };
                    previous
                        .to_item_ids
                        .iter()
                        .take(count)
                        .any(|previous_id| *previous_id == item_id)
                });
            if !duplicate {
                visit(item_id)?;
            }
        }
    }
    if has_target {
        return Ok(());
    }
    for (association_index, association) in state.item_property_associations.iter().enumerate() {
        let is_alpha = association.associations.iter().any(|association| {
            state
                .item_properties
                .get(usize::from(association.index).saturating_sub(1))
                .is_some_and(|property| {
                    matches!(property, ItemProperty::AuxiliaryType(value) if value == ALPHA_AUX_TYPE)
                })
        });
        if !is_alpha {
            continue;
        }
        let duplicate = state
            .item_property_associations
            .iter()
            .take(association_index)
            .filter(|previous| {
                previous.associations.iter().any(|association| {
                    state
                        .item_properties
                        .get(usize::from(association.index).saturating_sub(1))
                        .is_some_and(|property| {
                            matches!(property, ItemProperty::AuxiliaryType(value) if value == ALPHA_AUX_TYPE)
                        })
                })
            })
            .any(|previous| previous.item_id == association.item_id);
        if !duplicate {
            visit(association.item_id)?;
        }
    }
    Ok(())
}

/// Validates the complete native still selection before materializing any
/// selected item payload. This deliberately walks borrowed metadata only;
/// legacy parsing keeps its historical primary/sequence/metadata/alpha order.
pub(super) fn validate_native_selected_plan(
    data: &[u8],
    state: &MetaState,
    primary_item_id: u32,
    context: &ParseContext<'_>,
) -> Result<(), DecoderError> {
    validate_primary_item_metadata(state)?;
    let mut payload_total = validate_native_selected_item(data, state, primary_item_id, "primary")?;

    for_each_native_selected_alpha_item(state, primary_item_id, |item_id| {
        payload_total = payload_total
            .checked_add(validate_native_alpha_item(data, state, item_id)?)
            .ok_or_else(|| {
                DecoderError::InvalidParam("selected payload size overflows".to_string())
            })?;
        Ok(())
    })?;
    if let Some(limits) = context.limits() {
        limits.check_count(
            payload_total,
            limits.max_input_bytes(),
            "selected payload byte",
        )?;
    }
    Ok(())
}

/// Materializes a directly located native item without creating the
/// compatibility resolver's recursion stack. Native still parsing has
/// already rejected item-offset construction, but keeping that check here as
/// well makes this allocation boundary explicit and atomic.
#[cfg(test)]
pub(super) fn native_direct_item_payload(
    data: &[u8],
    state: &MetaState,
    item_id: u32,
    context: &mut ParseContext<'_>,
) -> Result<Vec<u8>, DecoderError> {
    native_direct_item_payload_with_owner(
        data,
        state,
        item_id,
        context,
        RetainedOwnerKind::PrimaryPayload,
    )
}

pub(super) fn native_direct_item_payload_with_owner(
    data: &[u8],
    state: &MetaState,
    item_id: u32,
    context: &mut ParseContext<'_>,
    owner_kind: RetainedOwnerKind,
) -> Result<Vec<u8>, DecoderError> {
    let construction_method = state
        .item_construction_methods
        .iter()
        .find_map(|(id, method)| (*id == item_id).then_some(*method))
        .unwrap_or(0);
    if construction_method == 2 {
        return Err(DecoderError::Unsupported(
            "bounded native decode does not support item_offset construction".to_string(),
        ));
    }
    if construction_method != 0 && construction_method != 1 {
        return Err(DecoderError::Unsupported(format!(
            "bounded native decode does not support construction_method {construction_method}"
        )));
    }

    let location = state
        .item_locations
        .iter()
        .find(|location| location.item_id == item_id)
        .ok_or_else(|| DecoderError::Bitstream(format!("item {item_id} location is missing")))?;
    let source = if construction_method == 0 {
        data
    } else if let Some((start, end)) = state.idat_payload_range {
        data.get(start..end).ok_or_else(|| {
            DecoderError::NotEnoughData("idat payload is outside the input".to_string())
        })?
    } else {
        state.idat_payload.as_deref().ok_or_else(|| {
            DecoderError::Bitstream(format!("item {item_id} references missing idat box"))
        })?
    };

    // Plan and validate every extent before creating the returned payload.
    // The second walk consumes the same checked bounds without a temporary
    // extent Vec, so malformed ranges cannot leave a charged allocation.
    let payload_len = location.extents.iter().try_fold(0usize, |total, extent| {
        let (start, end) = super::item_extent_bounds(location, extent, source.len())?;
        total.checked_add(end - start).ok_or_else(|| {
            DecoderError::Bitstream("item extent payload length overflow".to_string())
        })
    })?;
    let mut payload = Vec::new();
    let mut payload_token = AllocationToken::new(AllocationClass::Payload);
    context.try_reserve_class_with_token(
        &mut payload,
        &mut payload_token,
        payload_len,
        AllocationClass::Payload,
        "item payload",
    )?;
    for extent in &location.extents {
        let (start, end) = super::item_extent_bounds(location, extent, source.len())?;
        payload.extend_from_slice(&source[start..end]);
    }
    context.retain_native_token(owner_kind, payload_token)?;
    Ok(payload)
}

fn validate_native_alpha_item(
    data: &[u8],
    state: &MetaState,
    item_id: u32,
) -> Result<usize, DecoderError> {
    let item_type = state
        .item_infos
        .iter()
        .find(|item| item.item_id == item_id)
        .map(|item| item.item_type);
    if item_type != Some(*b"av01") {
        return Err(DecoderError::Unsupported(
            "bounded native decode accepts only an av01 alpha item".to_string(),
        ));
    }
    validate_native_selected_item(data, state, item_id, "alpha")
}

fn validate_native_selected_item(
    data: &[u8],
    state: &MetaState,
    item_id: u32,
    role: &str,
) -> Result<usize, DecoderError> {
    let construction_method = state
        .item_construction_methods
        .iter()
        .find_map(|(id, method)| (*id == item_id).then_some(*method))
        .unwrap_or(0);
    if construction_method == 2 {
        return Err(DecoderError::Unsupported(format!(
            "bounded native decode does not support {role} item_offset construction"
        )));
    }
    if construction_method != 0 && construction_method != 1 {
        return Err(DecoderError::Unsupported(format!(
            "bounded native decode does not support {role} construction_method {construction_method}"
        )));
    }
    let location = state
        .item_locations
        .iter()
        .find(|location| location.item_id == item_id)
        .ok_or_else(|| DecoderError::Bitstream(format!("item {item_id} location is missing")))?;
    let source_len = if construction_method == 0 {
        data.len()
    } else {
        state
            .idat_payload_range
            .map(|(start, end)| {
                end.checked_sub(start).ok_or_else(|| {
                    DecoderError::Bitstream("idat payload range is inverted".to_string())
                })
            })
            .transpose()?
            .or_else(|| state.idat_payload.as_ref().map(Vec::len))
            .ok_or_else(|| {
                DecoderError::Bitstream(format!("item {item_id} references missing idat box"))
            })?
    };
    let mut total = 0usize;
    for extent in &location.extents {
        let length = usize::try_from(extent.length)
            .map_err(|_| DecoderError::Bitstream("item extent length is too large".to_string()))?;
        total = total.checked_add(length).ok_or_else(|| {
            DecoderError::Bitstream("item extent payload length overflow".to_string())
        })?;
        validate_item_extent(location, extent, source_len)?;
    }
    Ok(total)
}

pub(super) fn retained_metadata_capacity(
    info: &AvifInfo,
    colors: &ColorInformationSet,
    ordered: &Vec<NativePropertyRecord>,
) -> Result<usize, DecoderError> {
    let mut total = 0usize;
    let mut add = |bytes: usize| {
        total = total.checked_add(bytes).ok_or_else(|| {
            DecoderError::InvalidParam("retained metadata capacity overflows".to_string())
        })?;
        Ok::<(), DecoderError>(())
    };
    let bytes_for = |count: usize, size: usize| {
        count.checked_mul(size).ok_or_else(|| {
            DecoderError::InvalidParam("retained metadata capacity overflows".to_string())
        })
    };
    add(info
        .compatible_brands
        .capacity()
        .checked_mul(std::mem::size_of::<[u8; 4]>())
        .ok_or_else(|| {
            DecoderError::InvalidParam("retained metadata capacity overflows".to_string())
        })?)?;
    if let Some(pixi) = info.pixel_information.as_ref() {
        add(pixi.bits_per_channel.capacity())?;
        if let Some(channels) = pixi.extended_channels.as_ref() {
            add(bytes_for(
                channels.capacity(),
                std::mem::size_of::<PixelChannelInformation>(),
            )?)?;
        }
    }
    if let Some(color) = info.color_information.as_ref() {
        add(color.payload.capacity())?;
    }
    if let Some(config) = info.av1_config.as_ref() {
        add(config.capacity())?;
    }
    add(bytes_for(
        info.alpha_auxiliary_items.capacity(),
        std::mem::size_of::<AuxiliaryImage>(),
    )?)?;
    for auxiliary in &info.alpha_auxiliary_items {
        add(auxiliary.aux_type.capacity())?;
    }
    add(bytes_for(
        info.sequence_sample_payloads.capacity(),
        std::mem::size_of::<Vec<u8>>(),
    )?)?;
    if let Some(icc) = colors.icc_profile.as_ref() {
        add(icc.capacity())?;
    }
    add(bytes_for(
        colors.unknown_colr.capacity(),
        std::mem::size_of::<ColorInformation>(),
    )?)?;
    for color in &colors.unknown_colr {
        add(color.payload.capacity())?;
    }
    add(retained_ordered_metadata_capacity(ordered)?)?;
    Ok(total)
}

pub(super) fn retained_ordered_metadata_capacity(
    ordered: &Vec<NativePropertyRecord>,
) -> Result<usize, DecoderError> {
    let mut total = ordered
        .capacity()
        .checked_mul(std::mem::size_of::<NativePropertyRecord>())
        .ok_or_else(|| {
            DecoderError::InvalidParam("retained metadata capacity overflows".to_string())
        })?;
    for property in ordered {
        let mut add = |bytes: usize| {
            total = total.checked_add(bytes).ok_or_else(|| {
                DecoderError::InvalidParam("retained metadata capacity overflows".to_string())
            })?;
            Ok::<(), DecoderError>(())
        };
        match property {
            NativePropertyRecord::AuxiliaryType(value) => add(value.capacity())?,
            NativePropertyRecord::PixelInformation(pixi) => {
                add(pixi.bits_per_channel.capacity())?;
                if let Some(channels) = pixi.extended_channels.as_ref() {
                    add(channels
                        .capacity()
                        .checked_mul(std::mem::size_of::<PixelChannelInformation>())
                        .ok_or_else(|| {
                            DecoderError::InvalidParam(
                                "retained metadata capacity overflows".to_string(),
                            )
                        })?)?;
                }
            }
            NativePropertyRecord::Av1Config(value) => add(value.capacity())?,
            NativePropertyRecord::ColorInformation(color) => add(color.payload.capacity())?,
            NativePropertyRecord::CleanAperture(_)
            | NativePropertyRecord::Rotation(_)
            | NativePropertyRecord::Mirror(_)
            | NativePropertyRecord::SpatialExtents(_)
            | NativePropertyRecord::Premultiplied
            | NativePropertyRecord::PixelAspectRatio(_)
            | NativePropertyRecord::Other(_) => {}
        }
    }
    Ok(total)
}

/// Counts ICC payloads in the owners that survive the parser handoff.
///
/// The parser may temporarily own the source `colr` payload in addition to
/// the legacy projection, rich projection, and ordered property record.  Its
/// ICC counter therefore cannot be reused for the retained budget: only the
/// copies reachable from the returned values belong in the handoff budget.
pub(super) fn retained_icc_capacity(
    info: &AvifInfo,
    colors: &ColorInformationSet,
    ordered: &Vec<NativePropertyRecord>,
) -> Result<usize, DecoderError> {
    let mut total = 0usize;
    let mut add = |bytes: usize| {
        total = total.checked_add(bytes).ok_or_else(|| {
            DecoderError::InvalidParam("retained ICC capacity overflows".to_string())
        })?;
        Ok::<(), DecoderError>(())
    };
    if let Some(color) = info
        .color_information
        .as_ref()
        .filter(|color| color.icc_profile().is_some())
    {
        add(color.payload.capacity())?;
    }
    if let Some(icc) = colors.icc_profile.as_ref() {
        add(icc.capacity())?;
    }
    for property in ordered {
        if let Some(color) = match property {
            NativePropertyRecord::ColorInformation(color) if color.icc_profile().is_some() => {
                Some(color)
            }
            _ => None,
        } {
            add(color.payload.capacity())?;
        }
    }
    Ok(total)
}

/// Counts only payload vectors that survive the parser-to-native handoff.
/// Metadata capacities intentionally stay in the separate metadata total.
pub(super) fn retained_payload_capacity(info: &AvifInfo) -> Result<usize, DecoderError> {
    let mut total = 0usize;
    let mut add = |bytes: usize| {
        total = total.checked_add(bytes).ok_or_else(|| {
            DecoderError::InvalidParam("retained payload capacity overflows".to_string())
        })?;
        Ok::<(), DecoderError>(())
    };
    add(info.primary_item_payload.capacity())?;
    for auxiliary in &info.alpha_auxiliary_items {
        add(auxiliary.payload.capacity())?;
    }
    for payload in &info.sequence_sample_payloads {
        add(payload.capacity())?;
    }
    Ok(total)
}
