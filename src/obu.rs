use crate::DecoderError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObuType {
    Reserved0,
    SequenceHeader,
    TemporalDelimiter,
    FrameHeader,
    TileGroup,
    Metadata,
    Frame,
    RedundantFrameHeader,
    TileList,
    Reserved(u8),
    Padding,
}

impl ObuType {
    fn from_bits(value: u8) -> Self {
        match value {
            0 => Self::Reserved0,
            1 => Self::SequenceHeader,
            2 => Self::TemporalDelimiter,
            3 => Self::FrameHeader,
            4 => Self::TileGroup,
            5 => Self::Metadata,
            6 => Self::Frame,
            7 => Self::RedundantFrameHeader,
            8 => Self::TileList,
            15 => Self::Padding,
            value => Self::Reserved(value),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Obu<'a> {
    pub obu_type: ObuType,
    pub extension_header: Option<u8>,
    pub payload: &'a [u8],
}

pub fn parse_obu_stream(data: &[u8]) -> Result<Vec<Obu<'_>>, DecoderError> {
    let mut offset = 0usize;
    let mut obus = Vec::new();
    while let Some(obu) = read_next_obu(data, &mut offset)? {
        obus.push(obu);
    }
    Ok(obus)
}

pub fn find_obu_payload(data: &[u8], target: ObuType) -> Result<Option<&[u8]>, DecoderError> {
    let mut offset = 0usize;
    while let Some(obu) = read_next_obu(data, &mut offset)? {
        if obu.obu_type == target {
            return Ok(Some(obu.payload));
        }
    }
    Ok(None)
}

pub fn count_obus(data: &[u8], target: ObuType) -> Result<usize, DecoderError> {
    let mut offset = 0usize;
    let mut count = 0usize;
    while let Some(obu) = read_next_obu(data, &mut offset)? {
        if obu.obu_type == target {
            count = count
                .checked_add(1)
                .ok_or_else(|| DecoderError::Bitstream("OBU count overflow".to_string()))?;
        }
    }
    Ok(count)
}

pub fn find_obu_payloads<const N: usize>(
    data: &[u8],
    targets: [ObuType; N],
) -> Result<[Option<&[u8]>; N], DecoderError> {
    let mut offset = 0usize;
    let mut payloads = [None; N];
    while let Some(obu) = read_next_obu(data, &mut offset)? {
        for (index, target) in targets.iter().enumerate() {
            if payloads[index].is_none() && obu.obu_type == *target {
                payloads[index] = Some(obu.payload);
            }
        }
        if payloads.iter().all(Option::is_some) {
            break;
        }
    }
    Ok(payloads)
}

fn read_next_obu<'a>(data: &'a [u8], offset: &mut usize) -> Result<Option<Obu<'a>>, DecoderError> {
    if *offset >= data.len() {
        return Ok(None);
    }
    let start = *offset;
    let header = *data
        .get(*offset)
        .ok_or_else(|| DecoderError::NotEnoughData("OBU header is missing".to_string()))?;
    *offset += 1;

    if header & 0x80 != 0 {
        return Err(DecoderError::Bitstream(
            "OBU forbidden bit is set".to_string(),
        ));
    }
    if header & 0x01 != 0 {
        return Err(DecoderError::Bitstream(
            "OBU reserved bit is set".to_string(),
        ));
    }

    let obu_type = ObuType::from_bits((header >> 3) & 0x0f);
    let has_extension = header & 0x04 != 0;
    let has_size_field = header & 0x02 != 0;
    let extension_header = if has_extension {
        let extension = *data.get(*offset).ok_or_else(|| {
            DecoderError::NotEnoughData("OBU extension header is missing".to_string())
        })?;
        *offset += 1;
        Some(extension)
    } else {
        None
    };

    let payload_len = if has_size_field {
        read_leb128(data, offset)?
    } else {
        data.len() - *offset
    };
    let payload_end = (*offset)
        .checked_add(payload_len)
        .ok_or_else(|| DecoderError::Bitstream("OBU payload length overflow".to_string()))?;
    if payload_end > data.len() {
        return Err(DecoderError::NotEnoughData(format!(
            "OBU payload extends beyond item data at byte {start}"
        )));
    }
    let payload_start = *offset;
    *offset = payload_end;
    Ok(Some(Obu {
        obu_type,
        extension_header,
        payload: &data[payload_start..payload_end],
    }))
}

fn read_leb128(data: &[u8], offset: &mut usize) -> Result<usize, DecoderError> {
    let mut value = 0usize;
    for byte_index in 0..8 {
        let byte = *data.get(*offset).ok_or_else(|| {
            DecoderError::NotEnoughData("OBU leb128 size is truncated".to_string())
        })?;
        *offset += 1;

        let shifted = (byte & 0x7f) as usize;
        let shift = byte_index * 7;
        value =
            value
                .checked_add(shifted.checked_shl(shift as u32).ok_or_else(|| {
                    DecoderError::Bitstream("OBU leb128 size overflow".to_string())
                })?)
                .ok_or_else(|| DecoderError::Bitstream("OBU leb128 size overflow".to_string()))?;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err(DecoderError::Bitstream(
        "OBU leb128 size uses more than 8 bytes".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_truncated_obu_extension_header() {
        let err = parse_obu_stream(&[0x0e]).unwrap_err();

        assert!(
            matches!(err, DecoderError::NotEnoughData(message) if message.contains("extension header"))
        );
    }

    #[test]
    fn rejects_truncated_obu_leb128_size_field() {
        let err = parse_obu_stream(&[0x0a, 0x80]).unwrap_err();

        assert!(
            matches!(err, DecoderError::NotEnoughData(message) if message.contains("leb128 size"))
        );
    }

    #[test]
    fn rejects_obu_payload_extending_beyond_item_data() {
        let err = parse_obu_stream(&[0x0a, 0x02, 0xff]).unwrap_err();

        assert!(
            matches!(err, DecoderError::NotEnoughData(message) if message.contains("payload extends"))
        );
    }

    #[test]
    fn rejects_forbidden_obu_bit() {
        let err = parse_obu_stream(&[0x80]).unwrap_err();
        assert!(err.to_string().contains("forbidden bit"));
    }
}
