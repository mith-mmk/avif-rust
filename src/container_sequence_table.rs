//! Borrowed sample-table views used by the bounded AVIS parser.
//!
//! The compatibility parser intentionally materializes sample tables.  The
//! native sequence parser cannot do that: a hostile `stsz`/`stco`/`stsc` table
//! must be validated without first allocating a vector proportional to the
//! number of samples.  These views keep only references and small scalar
//! counts, and `walk_samples` derives one sample mapping at a time.

use super::*;
use crate::obu::ObuIter;

pub(super) struct StsdView<'a> {
    payload: &'a [u8],
    entries_offset: usize,
    entry_count: usize,
}

impl<'a> StsdView<'a> {
    pub(super) fn parse(payload: &'a [u8]) -> Result<Self, DecoderError> {
        if payload.len() < 8 {
            return Err(DecoderError::NotEnoughData(
                "stsd payload is too short".to_string(),
            ));
        }
        let entry_count = usize::try_from(read_u32(payload, 4)?)
            .map_err(|_| DecoderError::Bitstream("stsd entry count is too large".to_string()))?;
        let view = Self {
            payload,
            entries_offset: 8,
            entry_count,
        };
        view.validate_entries()?;
        Ok(view)
    }

    fn entry(&self, wanted: usize) -> Result<Option<&'a [u8]>, DecoderError> {
        if wanted >= self.entry_count {
            return Ok(None);
        }
        let mut offset = self.entries_offset;
        for index in 0..self.entry_count {
            let header = read_box_header(self.payload, offset, self.payload.len())?;
            let end = checked_add(header.offset, header.size, "stsd entry end")?;
            if index == wanted {
                return Ok(Some(&self.payload[offset..end]));
            }
            offset = end;
        }
        Ok(None)
    }

    fn validate_entries(&self) -> Result<(), DecoderError> {
        let mut offset = self.entries_offset;
        for _ in 0..self.entry_count {
            let header = read_box_header(self.payload, offset, self.payload.len())?;
            offset = checked_add(header.offset, header.size, "stsd entry end")?;
        }
        Ok(())
    }

    pub(super) fn first_av01(&self) -> Result<Option<&'a [u8]>, DecoderError> {
        let mut offset = self.entries_offset;
        for _ in 0..self.entry_count {
            let header = read_box_header(self.payload, offset, self.payload.len())?;
            let end = checked_add(header.offset, header.size, "stsd av01 entry end")?;
            if header.box_type == *b"av01" {
                return Ok(Some(&self.payload[offset..end]));
            }
            offset = end;
        }
        Ok(None)
    }

    pub(super) fn av01(&self, index: usize) -> Result<Option<&'a [u8]>, DecoderError> {
        let Some(entry) = self.entry(index)? else {
            return Ok(None);
        };
        if entry.len() < 8 || &entry[4..8] != b"av01" {
            return Ok(None);
        }
        Ok(Some(entry))
    }
}

pub(super) struct StszView<'a> {
    payload: &'a [u8],
    sample_size: usize,
    sample_count: usize,
}

impl<'a> StszView<'a> {
    pub(super) fn parse(payload: &'a [u8]) -> Result<Self, DecoderError> {
        if payload.len() < 12 {
            return Err(DecoderError::NotEnoughData(
                "stsz payload is too short".to_string(),
            ));
        }
        let sample_size = usize::try_from(read_u32(payload, 4)?)
            .map_err(|_| DecoderError::Bitstream("stsz sample size is too large".to_string()))?;
        let sample_count = usize::try_from(read_u32(payload, 8)?)
            .map_err(|_| DecoderError::Bitstream("stsz sample count is too large".to_string()))?;
        if sample_size == 0 {
            let bytes = sample_count.checked_mul(4).ok_or_else(|| {
                DecoderError::Bitstream("stsz sample count overflows usize".to_string())
            })?;
            let end = checked_add(12, bytes, "stsz payload size")?;
            if end > payload.len() {
                return Err(DecoderError::NotEnoughData(
                    "stsz entries are truncated".to_string(),
                ));
            }
        }
        Ok(Self {
            payload,
            sample_size,
            sample_count,
        })
    }

    pub(super) fn len(&self) -> usize {
        self.sample_count
    }

    pub(super) fn size(&self, index: usize) -> Result<usize, DecoderError> {
        if index >= self.sample_count {
            return Err(DecoderError::Bitstream(
                "stsc references more samples than stsz".to_string(),
            ));
        }
        if self.sample_size != 0 {
            return Ok(self.sample_size);
        }
        usize::try_from(read_u32(
            self.payload,
            checked_add(
                12,
                index.checked_mul(4).ok_or_else(|| {
                    DecoderError::Bitstream("stsz sample index overflows usize".to_string())
                })?,
                "stsz sample offset",
            )?,
        )?)
        .map_err(|_| DecoderError::Bitstream("stsz sample size is too large".to_string()))
    }
}

pub(super) struct ChunkOffsetView<'a> {
    payload: &'a [u8],
    width: usize,
    count: usize,
}

impl<'a> ChunkOffsetView<'a> {
    pub(super) fn parse(payload: &'a [u8], width: usize) -> Result<Self, DecoderError> {
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
        let count = usize::try_from(read_u32(payload, 4)?)
            .map_err(|_| DecoderError::Bitstream("chunk count is too large".to_string()))?;
        let bytes = count
            .checked_mul(width)
            .ok_or_else(|| DecoderError::Bitstream("chunk count overflows usize".to_string()))?;
        let end = checked_add(8, bytes, "chunk table size")?;
        if end > payload.len() {
            return Err(DecoderError::NotEnoughData(
                "chunk offsets are truncated".to_string(),
            ));
        }
        Ok(Self {
            payload,
            width,
            count,
        })
    }

    pub(super) fn len(&self) -> usize {
        self.count
    }

    pub(super) fn offset(&self, index: usize) -> Result<u64, DecoderError> {
        if index >= self.count {
            return Err(DecoderError::Bitstream(
                "chunk offset index is out of range".to_string(),
            ));
        }
        let at = checked_add(
            8,
            index.checked_mul(self.width).ok_or_else(|| {
                DecoderError::Bitstream("chunk offset overflows usize".to_string())
            })?,
            "chunk offset",
        )?;
        if self.width == 8 {
            read_u64(self.payload, at)
        } else {
            read_u32(self.payload, at).map(u64::from)
        }
    }
}

pub(super) struct StscView<'a> {
    payload: &'a [u8],
    count: usize,
}

impl<'a> StscView<'a> {
    pub(super) fn parse(payload: &'a [u8]) -> Result<Self, DecoderError> {
        if payload.len() < 8 {
            return Err(DecoderError::NotEnoughData(
                "stsc payload is too short".to_string(),
            ));
        }
        let count = usize::try_from(read_u32(payload, 4)?)
            .map_err(|_| DecoderError::Bitstream("stsc count is too large".to_string()))?;
        let bytes = count
            .checked_mul(12)
            .ok_or_else(|| DecoderError::Bitstream("stsc count overflows usize".to_string()))?;
        let end = checked_add(8, bytes, "stsc table size")?;
        if end > payload.len() {
            return Err(DecoderError::NotEnoughData(
                "stsc entries are truncated".to_string(),
            ));
        }
        if count == 0 {
            return Err(DecoderError::Bitstream("stsc table is empty".to_string()));
        }
        let view = Self { payload, count };
        view.validate_records()?;
        Ok(view)
    }

    fn record(&self, index: usize) -> Result<(u32, u32, u32), DecoderError> {
        if index >= self.count {
            return Err(DecoderError::Bitstream(
                "stsc record index is out of range".to_string(),
            ));
        }
        let offset = checked_add(
            8,
            index.checked_mul(12).ok_or_else(|| {
                DecoderError::Bitstream("stsc offset overflows usize".to_string())
            })?,
            "stsc record offset",
        )?;
        Ok((
            read_u32(self.payload, offset)?,
            read_u32(self.payload, offset + 4)?,
            read_u32(self.payload, offset + 8)?,
        ))
    }

    fn validate_records(&self) -> Result<(), DecoderError> {
        let mut previous = 0u32;
        for index in 0..self.count {
            let (first_chunk, samples_per_chunk, description) = self.record(index)?;
            if first_chunk == 0 || samples_per_chunk == 0 || description == 0 {
                return Err(DecoderError::Bitstream(
                    "stsc record contains a zero field".to_string(),
                ));
            }
            if first_chunk <= previous {
                return Err(DecoderError::Bitstream(
                    "stsc first_chunk values are not strictly increasing".to_string(),
                ));
            }
            previous = first_chunk;
        }
        Ok(())
    }

    fn for_chunk(&self, chunk_number: u32) -> Result<(u32, u32), DecoderError> {
        let mut selected = None;
        for index in 0..self.count {
            let (first_chunk, samples, description) = self.record(index)?;
            if first_chunk > chunk_number {
                break;
            }
            selected = Some((samples, description));
        }
        selected.ok_or_else(|| DecoderError::Bitstream("stsc first_chunk is invalid".to_string()))
    }
}

/// Visits every sample mapping without allocating a table of offsets,
/// description indices, or per-chunk records.
pub(super) fn walk_samples<F>(
    chunks: &ChunkOffsetView<'_>,
    stsc: &StscView<'_>,
    sizes: &StszView<'_>,
    mut visit: F,
) -> Result<(), DecoderError>
where
    F: FnMut(usize, u64, usize, usize) -> Result<(), DecoderError>,
{
    let chunk_count = u32::try_from(chunks.len())
        .map_err(|_| DecoderError::Bitstream("chunk count is too large".to_string()))?;
    for index in 0..stsc.count {
        let (first_chunk, _, _) = stsc.record(index)?;
        if first_chunk > chunk_count {
            return Err(DecoderError::Bitstream(
                "stsc first_chunk is beyond the chunk table".to_string(),
            ));
        }
    }
    let mut sample_index = 0usize;
    for chunk_index in 0..chunks.len() {
        let chunk_number =
            u32::try_from(chunk_index.checked_add(1).ok_or_else(|| {
                DecoderError::Bitstream("chunk number overflows usize".to_string())
            })?)
            .map_err(|_| DecoderError::Bitstream("chunk number is too large".to_string()))?;
        let (samples_per_chunk, description_index) = stsc.for_chunk(chunk_number)?;
        let mut offset = chunks.offset(chunk_index)?;
        for _ in 0..samples_per_chunk {
            let size = sizes.size(sample_index)?;
            visit(
                sample_index,
                offset,
                size,
                usize::try_from(description_index).map_err(|_| {
                    DecoderError::Bitstream(
                        "AVIS sample description index is too large".to_string(),
                    )
                })? - 1,
            )?;
            offset = offset.checked_add(size as u64).ok_or_else(|| {
                DecoderError::Bitstream("AVIS sample offset overflows u64".to_string())
            })?;
            sample_index = sample_index.checked_add(1).ok_or_else(|| {
                DecoderError::Bitstream("AVIS sample index overflows usize".to_string())
            })?;
        }
    }
    if sample_index != sizes.len() {
        return Err(DecoderError::Bitstream(
            "stsz contains samples not covered by stsc".to_string(),
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AuxiliaryType {
    None,
    Alpha,
    Other,
}

fn auxiliary_type(description: &[u8]) -> AuxiliaryType {
    let Some(marker) = description.windows(4).position(|window| window == b"auxi") else {
        return AuxiliaryType::None;
    };
    let Some(payload) = description.get(marker.saturating_add(8)..) else {
        return AuxiliaryType::Other;
    };
    let end = payload
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(payload.len());
    if payload[..end].as_ref() == ALPHA_AUX_TYPE.as_bytes() {
        AuxiliaryType::Alpha
    } else {
        AuxiliaryType::Other
    }
}

fn stsd_auxiliary_type(stsd: &StsdView<'_>) -> Result<AuxiliaryType, DecoderError> {
    let mut offset = stsd.entries_offset;
    for _ in 0..stsd.entry_count {
        let header = read_box_header(stsd.payload, offset, stsd.payload.len())?;
        let end = checked_add(header.offset, header.size, "stsd auxiliary entry end")?;
        if header.box_type == *b"av01" {
            let kind = auxiliary_type(&stsd.payload[offset..end]);
            if kind != AuxiliaryType::None {
                return Ok(kind);
            }
        }
        offset = end;
    }
    Ok(AuxiliaryType::None)
}

fn validate_track_references(payload: &[u8]) -> Result<bool, DecoderError> {
    if payload.is_empty() || !payload.len().is_multiple_of(4) {
        return Err(DecoderError::NotEnoughData(
            "track reference payload is too short".to_string(),
        ));
    }
    for index in 0..payload.len() / 4 {
        let _ = read_u32(payload, index * 4)?;
    }
    Ok(true)
}

fn parse_edit_list_repetition_borrowed(
    payload: &[u8],
) -> Result<AvifRepetitionCount, DecoderError> {
    if payload.len() < 8 {
        return Err(DecoderError::NotEnoughData(
            "edts payload is too short".to_string(),
        ));
    }
    let Some(elst) = child_box(payload, b"elst")? else {
        return Err(DecoderError::Bitstream("edts is missing elst".to_string()));
    };
    let elst_payload = box_payload(payload, elst)?;
    if elst_payload.len() < 8 {
        return Err(DecoderError::NotEnoughData(
            "elst payload is too short".to_string(),
        ));
    }
    let version = elst_payload[0];
    if version > 1 {
        return Err(DecoderError::Bitstream(
            "elst version is unsupported".to_string(),
        ));
    }
    let count = usize::try_from(read_u32(elst_payload, 4)?)
        .map_err(|_| DecoderError::Bitstream("elst entry count is too large".to_string()))?;
    let entry_size = if version == 1 { 20 } else { 12 };
    let required = count
        .checked_mul(entry_size)
        .and_then(|size| size.checked_add(8))
        .ok_or_else(|| DecoderError::Bitstream("elst size overflows usize".to_string()))?;
    if required > elst_payload.len() {
        return Err(DecoderError::NotEnoughData(
            "elst entries are truncated".to_string(),
        ));
    }
    if count == 0 {
        return Ok(AvifRepetitionCount::Unknown);
    }
    let mut offset = 8usize;
    let mut first_duration = 0u64;
    let mut all_media_zero = true;
    let mut all_durations_equal = true;
    let mut infinite = false;
    for index in 0..count {
        let duration = if version == 1 {
            read_u64(elst_payload, offset)?
        } else {
            u64::from(read_u32(elst_payload, offset)?)
        };
        let media_time_offset = if version == 1 { 8 } else { 4 };
        let media_time = if version == 1 {
            i64::from_be_bytes(read_u64(elst_payload, offset + media_time_offset)?.to_be_bytes())
        } else {
            i64::from(i32::from_be_bytes(
                read_u32(elst_payload, offset + media_time_offset)?.to_be_bytes(),
            ))
        };
        if index == 0 {
            first_duration = duration;
        } else if duration != first_duration {
            all_durations_equal = false;
        }
        all_media_zero &= media_time == 0;
        infinite |= media_time == 0 && duration == u64::from(u32::MAX);
        offset = checked_add(offset, entry_size, "elst entry end")?;
    }
    if infinite {
        return Ok(AvifRepetitionCount::Infinite);
    }
    if all_media_zero && all_durations_equal {
        return Ok(AvifRepetitionCount::Finite(
            u32::try_from(count.saturating_sub(1)).unwrap_or(u32::MAX),
        ));
    }
    Ok(AvifRepetitionCount::Unknown)
}

fn sample_contains_sequence_header_borrowed(sample: &[u8]) -> Result<bool, DecoderError> {
    let mut found = false;
    for obu in ObuIter::new(sample) {
        if obu?.obu_type == ObuType::SequenceHeader {
            found = true;
        }
    }
    Ok(found)
}

fn validate_sequence_sample_description_borrowed(
    description: &[u8],
    first_description: &[u8],
    sample: &[u8],
) -> Result<(), DecoderError> {
    if description != first_description && !sample_contains_sequence_header_borrowed(sample)? {
        return Err(DecoderError::Unsupported(
            "AVIS differing sample descriptions require a sequence header in each changed sample"
                .to_string(),
        ));
    }
    Ok(())
}

fn auxiliary_reference_present(trak_payload: &[u8]) -> Result<bool, DecoderError> {
    let Some(tref) = child_box(trak_payload, b"tref")? else {
        return Ok(false);
    };
    let tref_payload = box_payload(trak_payload, tref)?;
    let Some(auxl) = child_box(tref_payload, b"auxl")? else {
        return Ok(false);
    };
    validate_track_references(box_payload(tref_payload, auxl)?)
}

fn copy_selected_sample(
    data: &[u8],
    offset: u64,
    size: usize,
    description_index: usize,
    stsd: &StsdView<'_>,
    first_description: &[u8],
    samples: &mut Vec<Vec<u8>>,
    context: &mut ParseContext<'_>,
) -> Result<(), DecoderError> {
    let start = usize::try_from(offset)
        .map_err(|_| DecoderError::Bitstream("AVIS sample offset is too large".to_string()))?;
    let end = start
        .checked_add(size)
        .ok_or_else(|| DecoderError::Bitstream("AVIS sample end overflows usize".to_string()))?;
    if end > data.len() {
        return Err(DecoderError::NotEnoughData(
            "AVIS sample extends beyond the file".to_string(),
        ));
    }
    let description = stsd.av01(description_index)?.ok_or_else(|| {
        DecoderError::Bitstream("AVIS sample description is not av01".to_string())
    })?;
    let sample_source = &data[start..end];
    validate_sequence_sample_description_borrowed(description, first_description, sample_source)?;
    let mut sample = Vec::new();
    let mut token = AllocationToken::new(AllocationClass::Payload);
    context.try_reserve_class_with_token(
        &mut sample,
        &mut token,
        sample_source.len(),
        AllocationClass::Payload,
        "AVIS sample payload",
    )?;
    sample.extend_from_slice(sample_source);
    context.retain_native_token(RetainedOwnerKind::SequencePayload, token)?;
    samples.push(sample);
    Ok(())
}

/// Strict native sequence parser for one `moov`.  Only the first selected
/// color/alpha track is materialized; every sibling still walks and validates
/// all tables and sample ranges.
pub(super) fn parse_sequence_tracks_native(
    data: &[u8],
    moov_payload: &[u8],
    context: &mut ParseContext<'_>,
    selected_color: &mut Option<SequenceTrack>,
    selected_alpha: &mut Option<SequenceTrack>,
) -> Result<(), DecoderError> {
    for header in child_box_iter(moov_payload) {
        header?;
    }
    for trak_result in child_box_iter(moov_payload) {
        let trak = trak_result?;
        if trak.box_type != *b"trak" {
            continue;
        }
        let trak_payload = box_payload(moov_payload, trak)?;
        let has_aux_reference = auxiliary_reference_present(trak_payload)?;
        let Some(mdia) = child_box(trak_payload, b"mdia")? else {
            continue;
        };
        let mdia_payload = box_payload(trak_payload, mdia)?;
        let Some(hdlr) = child_box(mdia_payload, b"hdlr")? else {
            continue;
        };
        let hdlr_payload = box_payload(mdia_payload, hdlr)?;
        if hdlr_payload.len() < 12
            || (&hdlr_payload[8..12] != b"vide"
                && &hdlr_payload[8..12] != b"pict"
                && &hdlr_payload[8..12] != b"auxv")
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
        let stsd_view = StsdView::parse(box_payload(stbl_payload, stsd)?)?;
        let Some(first_description) = stsd_view.first_av01()? else {
            continue;
        };
        let auxiliary = stsd_auxiliary_type(&stsd_view)?;
        let is_alpha = has_aux_reference && auxiliary == AuxiliaryType::Alpha;
        let is_color = !has_aux_reference && auxiliary == AuxiliaryType::None;
        let retain_payload =
            (is_color && selected_color.is_none()) || (is_alpha && selected_alpha.is_none());
        let Some(stsc) = child_box(stbl_payload, b"stsc")? else {
            continue;
        };
        let Some(stsz) = child_box(stbl_payload, b"stsz")? else {
            continue;
        };
        let (offset_header, width) = if let Some(header) = child_box(stbl_payload, b"stco")? {
            (Some(header), 4)
        } else if let Some(header) = child_box(stbl_payload, b"co64")? {
            (Some(header), 8)
        } else {
            (None, 0)
        };
        let Some(offset_header) = offset_header else {
            continue;
        };
        let sizes = StszView::parse(box_payload(stbl_payload, stsz)?)?;
        if let Some(limits) = context.limits() {
            limits.check_count(sizes.len(), limits.max_frames(), "frame")?;
        }
        let chunks = ChunkOffsetView::parse(box_payload(stbl_payload, offset_header)?, width)?;
        let stsc_view = StscView::parse(box_payload(stbl_payload, stsc)?)?;
        // Pass 1 validates the complete mapping and every sample.  No output
        // allocation is performed by the walker for a sibling track.
        walk_samples(
            &chunks,
            &stsc_view,
            &sizes,
            |_, offset, size, description| {
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
                let description = stsd_view.av01(description)?.ok_or_else(|| {
                    DecoderError::Bitstream("AVIS sample description is not av01".to_string())
                })?;
                validate_sequence_sample_description_borrowed(
                    description,
                    first_description,
                    &data[start..end],
                )
            },
        )?;
        // Pass 2 copies only the globally selected track.  Rewalking is
        // allocation-free and ensures no partially retained sample survives a
        // later table/range error.
        let mut samples = Vec::new();
        let sample_outer_class = if is_alpha {
            AllocationClass::Payload
        } else {
            AllocationClass::Metadata
        };
        let mut samples_token = AllocationToken::new(sample_outer_class);
        if retain_payload {
            context.try_reserve_class_with_token(
                &mut samples,
                &mut samples_token,
                sizes.len(),
                sample_outer_class,
                "AVIS sample list",
            )?;
            walk_samples(
                &chunks,
                &stsc_view,
                &sizes,
                |_, offset, size, description| {
                    copy_selected_sample(
                        data,
                        offset,
                        size,
                        description,
                        &stsd_view,
                        first_description,
                        &mut samples,
                        context,
                    )
                },
            )?;
        }
        let timescale = child_box(mdia_payload, b"mdhd")?
            .map(|mdhd| box_payload(mdia_payload, mdhd))
            .transpose()?
            .map(parse_mdhd_timescale)
            .transpose()?
            .unwrap_or(1);
        let (pts_in_timescales, durations_in_timescales, timing_owners) =
            if let Some(stts) = child_box(stbl_payload, b"stts")? {
                let payload = box_payload(stbl_payload, stts)?;
                if retain_payload {
                    parse_sample_timing_native(payload, sizes.len(), context)?
                } else {
                    validate_sample_timing(payload, sizes.len())?;
                    (Vec::new(), Vec::new(), SequenceTimingOwners::default())
                }
            } else if retain_payload {
                zero_sample_timing_native(sizes.len(), context)?
            } else {
                (Vec::new(), Vec::new(), SequenceTimingOwners::default())
            };
        let (durations_ms, timing_owners) = if retain_payload {
            let (values, token) =
                make_durations_ms_native(&durations_in_timescales, u64::from(timescale), context)?;
            (
                values,
                SequenceTimingOwners {
                    durations_ms: token,
                    ..timing_owners
                },
            )
        } else {
            (Vec::new(), timing_owners)
        };
        let repetition_count = child_box(trak_payload, b"edts")?
            .map(|edts| box_payload(trak_payload, edts))
            .transpose()?
            .map(parse_edit_list_repetition_borrowed)
            .transpose()?
            .unwrap_or(AvifRepetitionCount::Unknown);
        let track = SequenceTrack {
            samples,
            timescale: u64::from(timescale),
            pts_in_timescales,
            durations_in_timescales,
            durations_ms,
            timing_owners,
            repetition_count,
            is_alpha,
            is_color,
        };
        if retain_payload {
            if is_alpha {
                if selected_alpha.is_none() {
                    *selected_alpha = Some(track);
                    context
                        .retain_native_token(RetainedOwnerKind::SequencePayload, samples_token)?;
                }
            } else if is_color && selected_color.is_none() {
                *selected_color = Some(track);
                context.retain_native_token(RetainedOwnerKind::RichMetadata, samples_token)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_box_payload(body: &[u8]) -> Vec<u8> {
        let mut payload = vec![0; 4];
        payload.extend_from_slice(body);
        payload
    }

    #[test]
    fn borrowed_walk_has_no_heap_allocations() {
        let stsz = full_box_payload(&[
            0, 0, 0, 1, // fixed sample size
            0, 0, 0, 1, // one sample
        ]);
        let stco = full_box_payload(&[
            0, 0, 0, 1, // one chunk
            0, 0, 0, 0, // offset
        ]);
        let stsc = full_box_payload(&[
            0, 0, 0, 1, // one record
            0, 0, 0, 1, // first chunk
            0, 0, 0, 1, // samples per chunk
            0, 0, 0, 1, // description index
        ]);
        let (result, requests) = crate::test_allocation_observer::count_allocation_requests(|| {
            let sizes = StszView::parse(&stsz).unwrap();
            let chunks = ChunkOffsetView::parse(&stco, 4).unwrap();
            let mapping = StscView::parse(&stsc).unwrap();
            let mut visits = 0usize;
            walk_samples(&chunks, &mapping, &sizes, |_, _, _, _| {
                visits += 1;
                Ok(())
            })
            .map(|_| visits)
        });
        assert_eq!(result.unwrap(), 1);
        assert_eq!(requests, 0);
    }

    #[test]
    fn stsz_fixed_size_is_borrowed() {
        let payload = [0u8, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3];
        let view = StszView::parse(&payload).unwrap();
        assert_eq!(view.len(), 3);
        assert_eq!(view.size(2).unwrap(), 2);
    }

    #[test]
    fn stsc_rejects_zero_and_truncated_records() {
        let zero = [0u8, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0];
        assert!(StscView::parse(&zero).is_err());
        let truncated = [0u8, 0, 0, 0, 0, 0, 0, 1];
        assert!(StscView::parse(&truncated).is_err());
    }

    #[test]
    fn borrowed_views_reject_truncated_tables_without_partial_walk() {
        let stsz = full_box_payload(&[0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1]);
        assert!(StszView::parse(&stsz).is_err());
        let stco = full_box_payload(&[0, 0, 0, 2, 0, 0, 0, 0]);
        assert!(ChunkOffsetView::parse(&stco, 4).is_err());
    }
}
