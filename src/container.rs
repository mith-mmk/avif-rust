use std::borrow::Cow;

use crate::DecoderError;

const BRAND_AVIF: &[u8; 4] = b"avif";
const BRAND_AVIS: &[u8; 4] = b"avis";
const ALPHA_AUX_TYPE: &str = "urn:mpeg:mpegB:cicp:systems:auxiliary:alpha";

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
        ItemProperty::Other => PropertyKind::Other,
    }
}

#[derive(Debug, Default)]
struct MetaState {
    primary_item_id: Option<u32>,
    item_locations: Vec<ItemLocation>,
    item_construction_methods: Vec<(u32, u16)>,
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

    let primary_item_payload = primary_item_payload(data, &meta)?;
    validate_primary_item_metadata(&meta)?;
    let alpha_auxiliary_items = alpha_auxiliary_items(data, &meta)?;
    let primary_grid = primary_grid(data, &primary_item_payload, &meta)?;
    let alpha_grid = alpha_grid(data, &alpha_auxiliary_items, &meta)?;
    let primary_metadata = primary_item_metadata(&meta)?;
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
    })
}

fn brand_is_avif(brand: &[u8; 4]) -> bool {
    brand == BRAND_AVIF || brand == BRAND_AVIS
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
                let (locations, construction_methods) = parse_iloc_with_methods(child_payload)?;
                state.item_locations = locations;
                state.item_construction_methods = construction_methods;
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

fn parse_iloc_with_methods(
    payload: &[u8],
) -> Result<(Vec<ItemLocation>, Vec<(u32, u16)>), DecoderError> {
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
            if construction_method == 2 && index_size != 0 {
                return Err(DecoderError::Unsupported(
                    "iloc item_offset with explicit extent indexes is not supported".to_string(),
                ));
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
        for _ in 0..extent_count {
            if version == 1 || version == 2 {
                let _extent_index = read_sized_int(payload, &mut cursor, index_size)?;
            }
            let offset = read_sized_int(payload, &mut cursor, offset_size)?;
            let length = read_sized_int(payload, &mut cursor, length_size)?;
            extents.push(ItemExtent { offset, length });
        }
        locations.push(ItemLocation {
            item_id,
            base_offset,
            extents,
        });
    }

    Ok((locations, construction_methods))
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
    let Some(primary_item_id) = state.primary_item_id else {
        return Err(DecoderError::Bitstream(
            "primary item is missing".to_string(),
        ));
    };
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
    let is_grid = state
        .item_infos
        .iter()
        .find(|info| info.item_id == primary_item_id)
        .is_some_and(|info| &info.item_type == b"grid");
    let required: &[PropertyKind] = if is_grid {
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
            ItemProperty::AuxiliaryType(_) | ItemProperty::Other => {}
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
    let channel_count = payload[4] as usize;
    if payload.len() < 5 + channel_count {
        return Err(DecoderError::NotEnoughData(
            "pixi channel depth list is too short".to_string(),
        ));
    }
    Ok(PixelInformation {
        bits_per_channel: payload[5..5 + channel_count].to_vec(),
    })
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

fn primary_item_payload(data: &[u8], state: &MetaState) -> Result<Vec<u8>, DecoderError> {
    let primary_item_id = state
        .primary_item_id
        .ok_or_else(|| DecoderError::Bitstream("primary item is missing".to_string()))?;
    item_payload(data, state, primary_item_id)
}

fn alpha_auxiliary_items(
    data: &[u8],
    state: &MetaState,
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
        && let Some(primary_item_id) = state.primary_item_id
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
    let source: Cow<'_, [u8]> = match construction_method {
        0 => Cow::Borrowed(data),
        1 => Cow::Borrowed(state.idat_payload.as_deref().ok_or_else(|| {
            DecoderError::Bitstream(format!("item {item_id} references missing idat box"))
        })?),
        2 => {
            let target = state
                .item_references
                .iter()
                .find(|reference| {
                    reference.reference_type == *b"iloc" && reference.from_item_id == item_id
                })
                .and_then(|reference| reference.to_item_ids.first().copied())
                .ok_or_else(|| {
                    DecoderError::Bitstream(format!(
                        "item {item_id} item_offset reference is missing"
                    ))
                })?;
            Cow::Owned(item_payload_with_stack(data, state, target, stack)?)
        }
        method => {
            return Err(DecoderError::Unsupported(format!(
                "iloc construction_method {method} is not supported"
            )));
        }
    };
    if construction_method != 2 && payload_len > source.len() {
        return Err(DecoderError::Bitstream(
            "item extent payload length exceeds file size".to_string(),
        ));
    }
    let mut payload = Vec::with_capacity(payload_len);
    for extent in &location.extents {
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
    }
    stack.pop();
    Ok(payload)
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
                }),
                ItemProperty::Av1Config(vec![1, 2, 3]),
                ItemProperty::ColorInformation(primary_color.clone()),
                ItemProperty::SpatialExtents(ImageSpatialExtents {
                    width: 10,
                    height: 10,
                }),
                ItemProperty::PixelInformation(PixelInformation {
                    bits_per_channel: vec![10],
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
}
