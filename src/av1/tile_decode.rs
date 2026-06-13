use super::cdf::CdfContext;
use super::decode::{FrameDecodePlan, TileDecodePlan};
use super::entropy::EntropyDecoder;
use super::frame::FrameHeader;
use super::sequence::SequenceHeader;
use super::syntax::{BlockSize, Partition, PredictionMode, UvPredictionMode};
use super::tile_group::TileGroup;
use super::transform::{TransformBlock, plan_transform_blocks};
use crate::DecoderError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileEntropyState {
    pub tile_id: u32,
    pub payload_offset: usize,
    pub payload_len: usize,
    pub entropy_start_bits: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PartitionProbe {
    pub tile_id: u32,
    pub block_size: BlockSize,
    pub context: usize,
    pub symbol: usize,
    pub partition: Partition,
    pub bit_position_after: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockModeProbe {
    pub tile_id: u32,
    pub block_size: BlockSize,
    pub skip_context: usize,
    pub skip_symbol: usize,
    pub skip: bool,
    pub cdef_idx: Option<u32>,
    pub y_above_context: usize,
    pub y_left_context: usize,
    pub y_mode_symbol: usize,
    pub y_mode: PredictionMode,
    pub angle_delta_y: Option<i8>,
    pub uv_mode_symbol: Option<usize>,
    pub uv_mode: Option<UvPredictionMode>,
    pub angle_delta_uv: Option<i8>,
    pub bit_position_after: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidualProbe {
    pub tile_id: u32,
    pub block_size: BlockSize,
    pub skipped: bool,
    pub transform_count: usize,
    pub zero_transform_count: usize,
    pub first_tx_size: Option<super::syntax::TxSize>,
    pub txb_skip_context: Option<usize>,
    pub all_zero_symbol: Option<usize>,
    pub first_transform_all_zero: bool,
    pub bit_position_after: usize,
}

pub struct TileDecoder<'a> {
    reader: EntropyDecoder<'a>,
    cdf: CdfContext,
}

impl<'a> TileDecoder<'a> {
    pub fn new(payload: &'a [u8], frame: &FrameHeader) -> Result<Self, DecoderError> {
        Ok(Self {
            reader: EntropyDecoder::new(payload, frame.disable_cdf_update)?,
            cdf: CdfContext::new(frame.base_q_idx),
        })
    }

    pub fn read_root_partition(
        &mut self,
        tile: &TileDecodePlan,
    ) -> Result<PartitionProbe, DecoderError> {
        let block_size = BlockSize::Block128x128;
        let context = 0;
        let symbol = self.reader.read_symbol(
            self.cdf
                .partition_cdf_mut(block_size.width_mi_log2(), context),
        )?;
        let partition = Partition::from_symbol(block_size, symbol).ok_or_else(|| {
            DecoderError::Bitstream(format!(
                "AV1 partition symbol {symbol} is invalid for {block_size:?}"
            ))
        })?;
        Ok(PartitionProbe {
            tile_id: tile.tile_id,
            block_size,
            context,
            symbol,
            partition,
            bit_position_after: self.reader.bit_position(),
        })
    }

    pub fn read_intra_frame_block_mode(
        &mut self,
        sequence: &SequenceHeader,
        frame: &FrameHeader,
        tile: &TileDecodePlan,
        block_size: BlockSize,
    ) -> Result<BlockModeProbe, DecoderError> {
        if frame.delta_q.present {
            return Err(DecoderError::Unsupported(
                "AV1 delta-q block syntax is not supported yet".to_string(),
            ));
        }
        if frame.delta_lf.present {
            return Err(DecoderError::Unsupported(
                "AV1 delta loop-filter block syntax is not supported yet".to_string(),
            ));
        }
        if frame.allow_intrabc {
            return Err(DecoderError::Unsupported(
                "AV1 intrabc block syntax is not supported yet".to_string(),
            ));
        }

        let skip_context = 0;
        let skip_symbol = self
            .reader
            .read_symbol(self.cdf.skip_cdf_mut(skip_context))?;
        let skip = skip_symbol != 0;
        let cdef_idx = if !skip && frame.cdef.enabled && !frame.allow_intrabc && frame.cdef.bits > 0
        {
            Some(
                self.reader
                    .read_literal(frame.cdef.bits as usize)
                    .map_err(|err| DecoderError::Bitstream(format!("AV1 cdef_idx: {err}")))?,
            )
        } else {
            None
        };

        if sequence.enable_filter_intra && block_size.width() <= 32 && block_size.height() <= 32 {
            return Err(DecoderError::Unsupported(
                "AV1 filter-intra block syntax is not supported yet".to_string(),
            ));
        }

        let y_above_context = 0;
        let y_left_context = 0;
        let y_mode_symbol = self.reader.read_symbol(
            self.cdf
                .intra_frame_y_mode_cdf_mut(y_above_context, y_left_context),
        )?;
        let y_mode = PredictionMode::from_intra_symbol(y_mode_symbol).ok_or_else(|| {
            DecoderError::Bitstream(format!("AV1 y_mode symbol {y_mode_symbol} is invalid"))
        })?;
        let angle_delta_y = if y_mode.is_directional() {
            Some(self.read_angle_delta(y_mode.directional_index().unwrap())?)
        } else {
            None
        };

        let has_chroma = !sequence.color_config.monochrome;
        let (uv_mode_symbol, uv_mode, angle_delta_uv) = if has_chroma {
            let uv_symbol = self
                .reader
                .read_symbol(self.cdf.uv_mode_cfl_not_allowed_cdf_mut(y_mode_symbol))?;
            let uv_mode = UvPredictionMode::from_symbol(uv_symbol).ok_or_else(|| {
                DecoderError::Bitstream(format!("AV1 uv_mode symbol {uv_symbol} is invalid"))
            })?;
            if uv_mode == UvPredictionMode::Cfl {
                return Err(DecoderError::Unsupported(
                    "AV1 CFL chroma prediction is not supported yet".to_string(),
                ));
            }
            let angle_delta = if uv_mode.is_directional() {
                Some(self.read_angle_delta(uv_mode.directional_index().unwrap())?)
            } else {
                None
            };
            (Some(uv_symbol), Some(uv_mode), angle_delta)
        } else {
            (None, None, None)
        };

        Ok(BlockModeProbe {
            tile_id: tile.tile_id,
            block_size,
            skip_context,
            skip_symbol,
            skip,
            cdef_idx,
            y_above_context,
            y_left_context,
            y_mode_symbol,
            y_mode,
            angle_delta_y,
            uv_mode_symbol,
            uv_mode,
            angle_delta_uv,
            bit_position_after: self.reader.bit_position(),
        })
    }

    fn read_angle_delta(&mut self, directional_index: usize) -> Result<i8, DecoderError> {
        let symbol = self
            .reader
            .read_symbol(self.cdf.angle_delta_cdf_mut(directional_index))?;
        Ok(symbol as i8 - 3)
    }

    pub fn read_first_transform_residual(
        &mut self,
        tile_id: u32,
        block_mode: &BlockModeProbe,
        first_transform: Option<TransformBlock>,
        transform_count: usize,
    ) -> Result<ResidualProbe, DecoderError> {
        if block_mode.skip {
            return Ok(ResidualProbe {
                tile_id,
                block_size: block_mode.block_size,
                skipped: true,
                transform_count,
                zero_transform_count: transform_count,
                first_tx_size: first_transform.map(|transform| transform.tx_size),
                txb_skip_context: None,
                all_zero_symbol: None,
                first_transform_all_zero: true,
                bit_position_after: block_mode.bit_position_after,
            });
        }

        let Some(first_transform) = first_transform else {
            return Ok(ResidualProbe {
                tile_id,
                block_size: block_mode.block_size,
                skipped: false,
                transform_count,
                zero_transform_count: 0,
                first_tx_size: None,
                txb_skip_context: None,
                all_zero_symbol: None,
                first_transform_all_zero: false,
                bit_position_after: block_mode.bit_position_after,
            });
        };

        let txb_skip_context = first_txb_skip_context(block_mode.block_size, first_transform);
        let all_zero_symbol = self.reader.read_symbol(
            self.cdf
                .txb_skip_cdf_mut(first_transform.tx_size.coeff_cdf_index(), txb_skip_context),
        )?;
        let first_transform_all_zero = all_zero_symbol != 0;

        Ok(ResidualProbe {
            tile_id,
            block_size: block_mode.block_size,
            skipped: false,
            transform_count,
            zero_transform_count: usize::from(first_transform_all_zero),
            first_tx_size: Some(first_transform.tx_size),
            txb_skip_context: Some(txb_skip_context),
            all_zero_symbol: Some(all_zero_symbol),
            first_transform_all_zero,
            bit_position_after: self.reader.bit_position(),
        })
    }
}

fn first_txb_skip_context(block_size: BlockSize, transform: TransformBlock) -> usize {
    if block_size.width() == transform.tx_size.width()
        && block_size.height() == transform.tx_size.height()
    {
        0
    } else {
        1
    }
}

pub fn prepare_tile_entropy(
    data: &[u8],
    tile_group: &TileGroup,
    frame: &FrameHeader,
) -> Result<Vec<TileEntropyState>, DecoderError> {
    if tile_group.tiles.is_empty() {
        return Err(DecoderError::Bitstream(
            "AV1 tile group has no tile payloads".to_string(),
        ));
    }

    let mut states = Vec::with_capacity(tile_group.tiles.len());
    for tile in &tile_group.tiles {
        let end = tile
            .offset
            .checked_add(tile.len)
            .ok_or_else(|| DecoderError::Bitstream("AV1 tile payload end overflow".to_string()))?;
        let payload = data.get(tile.offset..end).ok_or_else(|| {
            DecoderError::NotEnoughData("AV1 tile payload extends beyond tile group".to_string())
        })?;
        let decoder = EntropyDecoder::new(payload, frame.disable_cdf_update)?;
        states.push(TileEntropyState {
            tile_id: tile.tile_id,
            payload_offset: tile.offset,
            payload_len: tile.len,
            entropy_start_bits: decoder.bit_position(),
        });
    }
    Ok(states)
}

pub fn probe_tile_partitions(
    data: &[u8],
    tile_group: &TileGroup,
    frame: &FrameHeader,
    plan: &FrameDecodePlan,
) -> Result<Vec<PartitionProbe>, DecoderError> {
    let mut probes = Vec::with_capacity(tile_group.tiles.len());
    for (index, tile_payload) in tile_group.tiles.iter().enumerate() {
        let end = tile_payload
            .offset
            .checked_add(tile_payload.len)
            .ok_or_else(|| DecoderError::Bitstream("AV1 tile payload end overflow".to_string()))?;
        let payload = data.get(tile_payload.offset..end).ok_or_else(|| {
            DecoderError::NotEnoughData("AV1 tile payload extends beyond tile group".to_string())
        })?;
        let tile_plan = plan.tiles.get(index).ok_or_else(|| {
            DecoderError::Bitstream("AV1 tile decode plan is missing a tile".to_string())
        })?;
        let mut decoder = TileDecoder::new(payload, frame)?;
        probes.push(decoder.read_root_partition(tile_plan)?);
    }
    Ok(probes)
}

pub fn probe_tile_block_modes(
    data: &[u8],
    tile_group: &TileGroup,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    plan: &FrameDecodePlan,
) -> Result<Vec<BlockModeProbe>, DecoderError> {
    let mut probes = Vec::with_capacity(tile_group.tiles.len());
    for (index, tile_payload) in tile_group.tiles.iter().enumerate() {
        let end = tile_payload
            .offset
            .checked_add(tile_payload.len)
            .ok_or_else(|| DecoderError::Bitstream("AV1 tile payload end overflow".to_string()))?;
        let payload = data.get(tile_payload.offset..end).ok_or_else(|| {
            DecoderError::NotEnoughData("AV1 tile payload extends beyond tile group".to_string())
        })?;
        let tile_plan = plan.tiles.get(index).ok_or_else(|| {
            DecoderError::Bitstream("AV1 tile decode plan is missing a tile".to_string())
        })?;
        let mut decoder = TileDecoder::new(payload, frame)?;
        let partition = decoder.read_root_partition(tile_plan)?;
        if partition.partition == Partition::None {
            probes.push(decoder.read_intra_frame_block_mode(
                sequence,
                frame,
                tile_plan,
                partition.block_size,
            )?);
        }
    }
    Ok(probes)
}

pub fn probe_first_block_residuals(
    data: &[u8],
    tile_group: &TileGroup,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    plan: &FrameDecodePlan,
) -> Result<Vec<ResidualProbe>, DecoderError> {
    let mut probes = Vec::with_capacity(tile_group.tiles.len());
    for (index, tile_payload) in tile_group.tiles.iter().enumerate() {
        let end = tile_payload
            .offset
            .checked_add(tile_payload.len)
            .ok_or_else(|| DecoderError::Bitstream("AV1 tile payload end overflow".to_string()))?;
        let payload = data.get(tile_payload.offset..end).ok_or_else(|| {
            DecoderError::NotEnoughData("AV1 tile payload extends beyond tile group".to_string())
        })?;
        let tile_plan = plan.tiles.get(index).ok_or_else(|| {
            DecoderError::Bitstream("AV1 tile decode plan is missing a tile".to_string())
        })?;
        let mut decoder = TileDecoder::new(payload, frame)?;
        let partition = decoder.read_root_partition(tile_plan)?;
        if partition.partition == Partition::None {
            let block_mode = decoder.read_intra_frame_block_mode(
                sequence,
                frame,
                tile_plan,
                partition.block_size,
            )?;
            let transforms =
                plan_transform_blocks(0, 0, 0, block_mode.block_size, plan.width, plan.height);
            probes.push(decoder.read_first_transform_residual(
                tile_plan.tile_id,
                &block_mode,
                transforms.first().copied(),
                transforms.len(),
            )?);
        }
    }
    Ok(probes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av1::{
        build_still_decode_plan, parse_frame_header, parse_sequence_header, parse_tile_group,
        plan_transform_blocks,
    };
    use crate::container::parse_avif;
    use crate::obu::{ObuType, find_obu_payload};

    #[test]
    fn prepares_sample_tile_entropy_state() {
        let data = std::fs::read(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("samples")
                .join("WML2Viewer.avif"),
        )
        .expect("sample AVIF should exist");
        let info = parse_avif(&data).unwrap();
        let sequence_payload =
            find_obu_payload(&info.primary_item_payload, ObuType::SequenceHeader)
                .unwrap()
                .expect("sequence header OBU should exist");
        let sequence = parse_sequence_header(sequence_payload).unwrap();
        let frame_payload = find_obu_payload(&info.primary_item_payload, ObuType::Frame)
            .unwrap()
            .expect("frame OBU should exist");
        let frame = parse_frame_header(frame_payload, &sequence).unwrap();
        let tile_group = parse_tile_group(
            frame_payload,
            frame.uncompressed_header_bits,
            &frame.tile_info,
        )
        .unwrap();

        let states = prepare_tile_entropy(frame_payload, &tile_group, &frame).unwrap();

        assert_eq!(states.len(), 1);
        assert_eq!(states[0].tile_id, 0);
        assert_eq!(states[0].entropy_start_bits, 15);
        assert!(states[0].payload_len > 0);
    }

    #[test]
    fn reads_sample_root_partition_symbol() {
        let data = std::fs::read(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("samples")
                .join("WML2Viewer.avif"),
        )
        .expect("sample AVIF should exist");
        let info = parse_avif(&data).unwrap();
        let sequence_payload =
            find_obu_payload(&info.primary_item_payload, ObuType::SequenceHeader)
                .unwrap()
                .expect("sequence header OBU should exist");
        let sequence = parse_sequence_header(sequence_payload).unwrap();
        let frame_payload = find_obu_payload(&info.primary_item_payload, ObuType::Frame)
            .unwrap()
            .expect("frame OBU should exist");
        let frame = parse_frame_header(frame_payload, &sequence).unwrap();
        let tile_group = parse_tile_group(
            frame_payload,
            frame.uncompressed_header_bits,
            &frame.tile_info,
        )
        .unwrap();
        let plan = build_still_decode_plan(&sequence, &frame, &tile_group).unwrap();
        let tile_payload = &tile_group.tiles[0];
        let payload = &frame_payload[tile_payload.offset..tile_payload.offset + tile_payload.len];
        let mut decoder = TileDecoder::new(payload, &frame).unwrap();

        let probe = decoder.read_root_partition(&plan.tiles[0]).unwrap();

        assert_eq!(probe.tile_id, 0);
        assert_eq!(probe.block_size, BlockSize::Block128x128);
        assert_eq!(probe.symbol, 0);
        assert_eq!(probe.partition, Partition::None);
        assert!(probe.bit_position_after >= 15);
    }

    #[test]
    fn reads_sample_first_block_mode_symbols() {
        let data = std::fs::read(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("samples")
                .join("WML2Viewer.avif"),
        )
        .expect("sample AVIF should exist");
        let info = parse_avif(&data).unwrap();
        let sequence_payload =
            find_obu_payload(&info.primary_item_payload, ObuType::SequenceHeader)
                .unwrap()
                .expect("sequence header OBU should exist");
        let sequence = parse_sequence_header(sequence_payload).unwrap();
        let frame_payload = find_obu_payload(&info.primary_item_payload, ObuType::Frame)
            .unwrap()
            .expect("frame OBU should exist");
        let frame = parse_frame_header(frame_payload, &sequence).unwrap();
        let tile_group = parse_tile_group(
            frame_payload,
            frame.uncompressed_header_bits,
            &frame.tile_info,
        )
        .unwrap();
        let plan = build_still_decode_plan(&sequence, &frame, &tile_group).unwrap();

        let probes =
            probe_tile_block_modes(frame_payload, &tile_group, &sequence, &frame, &plan).unwrap();

        assert_eq!(probes.len(), 1);
        assert_eq!(probes[0].tile_id, 0);
        assert_eq!(probes[0].block_size, BlockSize::Block128x128);
        assert!(probes[0].y_mode_symbol < 13);
        assert!(probes[0].uv_mode_symbol.is_some());
        assert!(probes[0].bit_position_after > 15);
    }

    #[test]
    fn plans_sample_first_block_transforms() {
        let data = std::fs::read(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("samples")
                .join("WML2Viewer.avif"),
        )
        .expect("sample AVIF should exist");
        let info = parse_avif(&data).unwrap();
        let sequence_payload =
            find_obu_payload(&info.primary_item_payload, ObuType::SequenceHeader)
                .unwrap()
                .expect("sequence header OBU should exist");
        let sequence = parse_sequence_header(sequence_payload).unwrap();
        let frame_payload = find_obu_payload(&info.primary_item_payload, ObuType::Frame)
            .unwrap()
            .expect("frame OBU should exist");
        let frame = parse_frame_header(frame_payload, &sequence).unwrap();
        let tile_group = parse_tile_group(
            frame_payload,
            frame.uncompressed_header_bits,
            &frame.tile_info,
        )
        .unwrap();
        let plan = build_still_decode_plan(&sequence, &frame, &tile_group).unwrap();
        let probes =
            probe_tile_block_modes(frame_payload, &tile_group, &sequence, &frame, &plan).unwrap();

        let transforms =
            plan_transform_blocks(0, 0, 0, probes[0].block_size, plan.width, plan.height);

        assert_eq!(transforms.len(), 16);
        assert!(transforms.iter().all(|tx| tx.plane == 0));
        assert!(
            transforms
                .iter()
                .all(|tx| tx.tx_size == super::super::syntax::TxSize::Tx32x32)
        );
    }

    #[test]
    fn probes_sample_first_block_residual_plan() {
        let data = std::fs::read(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("samples")
                .join("WML2Viewer.avif"),
        )
        .expect("sample AVIF should exist");
        let info = parse_avif(&data).unwrap();
        let sequence_payload =
            find_obu_payload(&info.primary_item_payload, ObuType::SequenceHeader)
                .unwrap()
                .expect("sequence header OBU should exist");
        let sequence = parse_sequence_header(sequence_payload).unwrap();
        let frame_payload = find_obu_payload(&info.primary_item_payload, ObuType::Frame)
            .unwrap()
            .expect("frame OBU should exist");
        let frame = parse_frame_header(frame_payload, &sequence).unwrap();
        let tile_group = parse_tile_group(
            frame_payload,
            frame.uncompressed_header_bits,
            &frame.tile_info,
        )
        .unwrap();
        let plan = build_still_decode_plan(&sequence, &frame, &tile_group).unwrap();
        let probes =
            probe_first_block_residuals(frame_payload, &tile_group, &sequence, &frame, &plan)
                .unwrap();

        assert_eq!(probes.len(), 1);
        assert_eq!(probes[0].tile_id, 0);
        assert_eq!(probes[0].block_size, BlockSize::Block128x128);
        assert_eq!(probes[0].transform_count, 16);
        if probes[0].skipped {
            assert_eq!(probes[0].zero_transform_count, 16);
            assert_eq!(probes[0].txb_skip_context, None);
            assert_eq!(probes[0].all_zero_symbol, None);
        } else {
            assert!(probes[0].txb_skip_context.unwrap() <= 1);
            assert!(probes[0].all_zero_symbol.unwrap() <= 1);
            assert_eq!(
                probes[0].zero_transform_count,
                usize::from(probes[0].first_transform_all_zero)
            );
        }
        assert_eq!(
            probes[0].first_tx_size,
            Some(super::super::syntax::TxSize::Tx32x32)
        );
    }
}
