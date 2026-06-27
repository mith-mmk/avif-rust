use crate::DecoderError;

const BRAND_AVIF: &[u8; 4] = b"avif";
const BRAND_AVIS: &[u8; 4] = b"avis";

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

#[derive(Debug, Default)]
struct MetaState {
    primary_item_id: Option<u32>,
    item_locations: Vec<ItemLocation>,
    width: Option<u32>,
    height: Option<u32>,
    pixel_information: Option<PixelInformation>,
    color_information: Option<ColorInformation>,
    alpha_premultiplied: bool,
    av1_config: Option<Vec<u8>>,
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
            b"moov" => {
                return Err(DecoderError::Unsupported(
                    "AVIF sequences are not supported".to_string(),
                ));
            }
            _ => {}
        }
        Ok(())
    })?;

    let primary_item_payload = primary_item_payload(data, &meta)?;
    Ok(AvifInfo {
        major_brand,
        compatible_brands,
        primary_item_id: meta.primary_item_id,
        width: meta.width,
        height: meta.height,
        pixel_information: meta.pixel_information,
        color_information: meta.color_information,
        alpha_premultiplied: meta.alpha_premultiplied,
        av1_config: meta.av1_config,
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
            b"iloc" => state.item_locations = parse_iloc(child_payload)?,
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
        if &header.box_type == b"ipco" {
            parse_ipco(source, child_payload, state)?;
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
        match &header.box_type {
            b"ispe" => {
                let extents = parse_ispe(child_payload)?;
                state.width = Some(extents.width);
                state.height = Some(extents.height);
            }
            b"pixi" => state.pixel_information = Some(parse_pixi(child_payload)?),
            b"av1C" => state.av1_config = Some(child_payload.to_vec()),
            b"colr" => state.color_information = Some(parse_colr(child_payload)?),
            b"prem" => state.alpha_premultiplied = true,
            _ => {}
        }
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

fn parse_iloc(payload: &[u8]) -> Result<Vec<ItemLocation>, DecoderError> {
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

    let mut locations = Vec::with_capacity(item_count);
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

        if version == 1 || version == 2 {
            let construction_method = read_u16(payload, cursor)? & 0x000f;
            cursor += 2;
            if construction_method != 0 {
                return Err(DecoderError::Unsupported(format!(
                    "iloc construction_method {construction_method} is not supported"
                )));
            }
        }

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

    Ok(locations)
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
    let location = state
        .item_locations
        .iter()
        .find(|location| location.item_id == primary_item_id)
        .ok_or_else(|| DecoderError::Bitstream("primary item location is missing".to_string()))?;

    let payload_len = location.extents.iter().try_fold(0usize, |sum, extent| {
        let length = usize::try_from(extent.length)
            .map_err(|_| DecoderError::Bitstream("item extent length is too large".to_string()))?;
        sum.checked_add(length).ok_or_else(|| {
            DecoderError::Bitstream("item extent payload length overflow".to_string())
        })
    })?;
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
        if end > data.len() || start > end {
            return Err(DecoderError::NotEnoughData(
                "item extent points outside the file".to_string(),
            ));
        }
        payload.extend_from_slice(&data[start..end]);
    }
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

fn read_u16(data: &[u8], offset: usize) -> Result<u16, DecoderError> {
    Ok(u16::from_be_bytes(
        data.get(offset..offset + 2)
            .ok_or_else(|| DecoderError::NotEnoughData("u16 is truncated".to_string()))?
            .try_into()
            .expect("slice length checked"),
    ))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_avif() -> Vec<u8> {
        std::fs::read(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("samples")
                .join("WML2Viewer.avif"),
        )
        .expect("sample AVIF should exist")
    }

    #[test]
    fn ftyp_recognizes_avif_brand() {
        let data = sample_avif();
        assert!(is_avif_file(&data));
    }

    #[test]
    fn parse_sample_container_metadata() {
        let data = sample_avif();
        let info = parse_avif(&data).unwrap();

        assert!(info.is_avif_brand());
        assert_eq!(info.width, Some(900));
        assert_eq!(info.height, Some(900));
        assert_eq!(
            info.pixel_information
                .as_ref()
                .map(|pixi| pixi.bits_per_channel.as_slice()),
            Some(&[8, 8, 8][..])
        );
        assert!(!info.primary_item_payload.is_empty());
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
}
