use super::cdf::CdfContext;
use super::decode::{FrameDecodePlan, TileDecodePlan};
use super::entropy::EntropyDecoder;
use super::frame::FrameHeader;
use super::sequence::SequenceHeader;
use super::syntax::{BlockSize, Partition, PredictionMode, UvPredictionMode};
use super::tile_group::TileGroup;
use super::transform::{TransformBlock, plan_transform_blocks, zig_zag_scan};
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
    pub first_non_zero_transform_index: Option<usize>,
    pub first_non_zero_tx_size: Option<super::syntax::TxSize>,
    pub txb_skip_context: Option<usize>,
    pub all_zero_symbol: Option<usize>,
    pub first_transform_all_zero: bool,
    pub eob_multisize: Option<usize>,
    pub eob_pt_symbol: Option<usize>,
    pub eob_pt: Option<usize>,
    pub eob_base: Option<usize>,
    pub eob_extra_context: Option<usize>,
    pub eob_extra_symbol: Option<usize>,
    pub eob_extra_literal_bits: Option<usize>,
    pub eob: Option<usize>,
    pub coeff_base_eob_context: Option<usize>,
    pub coeff_base_eob_symbol: Option<usize>,
    pub coeff_base_eob_level: Option<usize>,
    pub regular_coeff_base_count: Option<usize>,
    pub regular_coeff_base_decoded_count: Option<usize>,
    pub coeff_base_non_zero_count: Option<usize>,
    pub coeff_base_range_count: Option<usize>,
    pub coeff_br_decoded_count: Option<usize>,
    pub first_coeff_br_scan_index: Option<usize>,
    pub first_coeff_br_position: Option<usize>,
    pub first_coeff_br_context: Option<usize>,
    pub first_coeff_br_symbol: Option<usize>,
    pub first_coeff_br_level: Option<usize>,
    pub sign_decoded_count: Option<usize>,
    pub dc_sign_context: Option<usize>,
    pub dc_sign_symbol: Option<usize>,
    pub first_ac_sign_scan_index: Option<usize>,
    pub first_ac_sign_bit: Option<usize>,
    pub golomb_decoded_count: Option<usize>,
    pub first_golomb_scan_index: Option<usize>,
    pub first_golomb_value: Option<usize>,
    pub first_coeff_base_scan_index: Option<usize>,
    pub first_coeff_base_position: Option<usize>,
    pub first_coeff_base_context: Option<usize>,
    pub first_coeff_base_reference_magnitude: Option<usize>,
    pub first_coeff_base_symbol: Option<usize>,
    pub first_coeff_base_level: Option<usize>,
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
        transforms: &[TransformBlock],
    ) -> Result<ResidualProbe, DecoderError> {
        let transform_count = transforms.len();
        let first_transform = transforms.first().copied();
        if block_mode.skip {
            return Ok(ResidualProbe {
                tile_id,
                block_size: block_mode.block_size,
                skipped: true,
                transform_count,
                zero_transform_count: transform_count,
                first_tx_size: first_transform.map(|transform| transform.tx_size),
                first_non_zero_transform_index: None,
                first_non_zero_tx_size: None,
                txb_skip_context: None,
                all_zero_symbol: None,
                first_transform_all_zero: true,
                eob_multisize: None,
                eob_pt_symbol: None,
                eob_pt: None,
                eob_base: None,
                eob_extra_context: None,
                eob_extra_symbol: None,
                eob_extra_literal_bits: None,
                eob: None,
                coeff_base_eob_context: None,
                coeff_base_eob_symbol: None,
                coeff_base_eob_level: None,
                regular_coeff_base_count: None,
                regular_coeff_base_decoded_count: None,
                coeff_base_non_zero_count: None,
                coeff_base_range_count: None,
                coeff_br_decoded_count: None,
                first_coeff_br_scan_index: None,
                first_coeff_br_position: None,
                first_coeff_br_context: None,
                first_coeff_br_symbol: None,
                first_coeff_br_level: None,
                sign_decoded_count: None,
                dc_sign_context: None,
                dc_sign_symbol: None,
                first_ac_sign_scan_index: None,
                first_ac_sign_bit: None,
                golomb_decoded_count: None,
                first_golomb_scan_index: None,
                first_golomb_value: None,
                first_coeff_base_scan_index: None,
                first_coeff_base_position: None,
                first_coeff_base_context: None,
                first_coeff_base_reference_magnitude: None,
                first_coeff_base_symbol: None,
                first_coeff_base_level: None,
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
                first_non_zero_transform_index: None,
                first_non_zero_tx_size: None,
                txb_skip_context: None,
                all_zero_symbol: None,
                first_transform_all_zero: false,
                eob_multisize: None,
                eob_pt_symbol: None,
                eob_pt: None,
                eob_base: None,
                eob_extra_context: None,
                eob_extra_symbol: None,
                eob_extra_literal_bits: None,
                eob: None,
                coeff_base_eob_context: None,
                coeff_base_eob_symbol: None,
                coeff_base_eob_level: None,
                regular_coeff_base_count: None,
                regular_coeff_base_decoded_count: None,
                coeff_base_non_zero_count: None,
                coeff_base_range_count: None,
                coeff_br_decoded_count: None,
                first_coeff_br_scan_index: None,
                first_coeff_br_position: None,
                first_coeff_br_context: None,
                first_coeff_br_symbol: None,
                first_coeff_br_level: None,
                sign_decoded_count: None,
                dc_sign_context: None,
                dc_sign_symbol: None,
                first_ac_sign_scan_index: None,
                first_ac_sign_bit: None,
                golomb_decoded_count: None,
                first_golomb_scan_index: None,
                first_golomb_value: None,
                first_coeff_base_scan_index: None,
                first_coeff_base_position: None,
                first_coeff_base_context: None,
                first_coeff_base_reference_magnitude: None,
                first_coeff_base_symbol: None,
                first_coeff_base_level: None,
                bit_position_after: block_mode.bit_position_after,
            });
        };

        let mut first_txb_skip_context_value = None;
        let mut first_all_zero_symbol = None;
        let mut first_transform_all_zero = true;
        let mut zero_transform_count = 0usize;
        let mut first_non_zero_transform = None;
        let mut first_non_zero_transform_index = None;

        for (index, transform) in transforms.iter().copied().enumerate() {
            let txb_skip_context = first_txb_skip_context(block_mode.block_size, transform);
            let all_zero_symbol = self.reader.read_symbol(
                self.cdf
                    .txb_skip_cdf_mut(transform.tx_size.coeff_cdf_index(), txb_skip_context),
            )?;
            if index == 0 {
                first_txb_skip_context_value = Some(txb_skip_context);
                first_all_zero_symbol = Some(all_zero_symbol);
                first_transform_all_zero = all_zero_symbol != 0;
            }
            if all_zero_symbol != 0 {
                zero_transform_count += 1;
                continue;
            }
            first_non_zero_transform = Some(transform);
            first_non_zero_transform_index = Some(index);
            break;
        }

        let (
            eob_multisize,
            eob_pt_symbol,
            eob_pt,
            eob_base,
            eob_extra_context,
            eob_extra_symbol,
            eob_extra_literal_bits,
            eob,
            coeff_base_eob_context,
            coeff_base_eob_symbol,
            coeff_base_eob_level,
            regular_coeff_base_count,
            regular_coeff_base_decoded_count,
            coeff_base_non_zero_count,
            coeff_base_range_count,
            coeff_br_decoded_count,
            first_coeff_br_scan_index,
            first_coeff_br_position,
            first_coeff_br_context,
            first_coeff_br_symbol,
            first_coeff_br_level,
            sign_decoded_count,
            dc_sign_context,
            dc_sign_symbol,
            first_ac_sign_scan_index,
            first_ac_sign_bit,
            golomb_decoded_count,
            first_golomb_scan_index,
            first_golomb_value,
            first_coeff_base_scan_index,
            first_coeff_base_position,
            first_coeff_base_context,
            first_coeff_base_reference_magnitude,
            first_coeff_base_symbol,
            first_coeff_base_level,
        ) = if let Some(non_zero_transform) = first_non_zero_transform {
            let eob_multisize = eob_multisize(non_zero_transform);
            if eob_multisize != 6 {
                return Err(DecoderError::Unsupported(format!(
                    "AV1 eob_pt decode for eobMultisize {eob_multisize} is not supported yet"
                )));
            }
            let plane_type = usize::from(non_zero_transform.plane > 0);
            let eob_pt_symbol = self
                .reader
                .read_symbol(self.cdf.eob_pt_1024_cdf_mut(plane_type))?;
            let eob_pt = eob_pt_symbol + 1;
            let eob_base = eob_base_from_pt(eob_pt);
            let (eob_extra_context, eob_extra_symbol, eob_extra_literal_bits, eob) =
                self.read_eob_extra(non_zero_transform, plane_type, eob_pt, eob_base)?;
            if non_zero_transform.tx_size != super::syntax::TxSize::Tx32x32 {
                return Err(DecoderError::Unsupported(format!(
                    "AV1 coeff_base_eob decode for {:?} is not supported yet",
                    non_zero_transform.tx_size
                )));
            }
            let coeff_base_eob_context =
                coeff_base_eob_context(non_zero_transform.tx_size, eob.saturating_sub(1));
            let coeff_base_eob_symbol = self.reader.read_symbol(
                self.cdf
                    .coeff_base_eob_tx32_cdf_mut(plane_type, coeff_base_eob_context),
            )?;
            let coeff_base_eob_level = coeff_base_eob_symbol + 1;
            let coeff_base_read = self.read_regular_coeff_bases(
                non_zero_transform.tx_size,
                plane_type,
                eob,
                coeff_base_eob_level,
            )?;
            debug_assert_eq!(
                coeff_base_read.base_levels.len(),
                non_zero_transform.tx_size.sample_count()
            );
            (
                Some(eob_multisize),
                Some(eob_pt_symbol),
                Some(eob_pt),
                Some(eob_base),
                eob_extra_context,
                eob_extra_symbol,
                eob_extra_literal_bits,
                Some(eob),
                Some(coeff_base_eob_context),
                Some(coeff_base_eob_symbol),
                Some(coeff_base_eob_level),
                Some(coeff_base_read.probe.remaining_count),
                Some(coeff_base_read.probe.decoded_count),
                Some(coeff_base_read.non_zero_count),
                Some(coeff_base_read.base_range_count),
                Some(coeff_base_read.coeff_br_symbol_count),
                coeff_base_read.first_coeff_br.map(|first| first.scan_index),
                coeff_base_read.first_coeff_br.map(|first| first.position),
                coeff_base_read.first_coeff_br.map(|first| first.context),
                coeff_base_read.first_coeff_br.map(|first| first.symbol),
                coeff_base_read
                    .first_coeff_br
                    .map(|first| first.level_after_symbol),
                Some(coeff_base_read.signs.sign_count),
                coeff_base_read.signs.dc_sign_context,
                coeff_base_read.signs.dc_sign_symbol,
                coeff_base_read.signs.first_ac_sign_scan_index,
                coeff_base_read.signs.first_ac_sign_bit,
                Some(coeff_base_read.signs.golomb_count),
                coeff_base_read.signs.first_golomb_scan_index,
                coeff_base_read.signs.first_golomb_value,
                coeff_base_read.probe.scan_index,
                coeff_base_read.probe.position,
                coeff_base_read.probe.context,
                coeff_base_read.probe.reference_magnitude,
                coeff_base_read.probe.symbol,
                coeff_base_read.probe.level,
            )
        } else {
            (
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None,
            )
        };

        Ok(ResidualProbe {
            tile_id,
            block_size: block_mode.block_size,
            skipped: false,
            transform_count,
            zero_transform_count,
            first_tx_size: Some(first_transform.tx_size),
            first_non_zero_transform_index,
            first_non_zero_tx_size: first_non_zero_transform.map(|transform| transform.tx_size),
            txb_skip_context: first_txb_skip_context_value,
            all_zero_symbol: first_all_zero_symbol,
            first_transform_all_zero,
            eob_multisize,
            eob_pt_symbol,
            eob_pt,
            eob_base,
            eob_extra_context,
            eob_extra_symbol,
            eob_extra_literal_bits,
            eob,
            coeff_base_eob_context,
            coeff_base_eob_symbol,
            coeff_base_eob_level,
            regular_coeff_base_count,
            regular_coeff_base_decoded_count,
            coeff_base_non_zero_count,
            coeff_base_range_count,
            coeff_br_decoded_count,
            first_coeff_br_scan_index,
            first_coeff_br_position,
            first_coeff_br_context,
            first_coeff_br_symbol,
            first_coeff_br_level,
            sign_decoded_count,
            dc_sign_context,
            dc_sign_symbol,
            first_ac_sign_scan_index,
            first_ac_sign_bit,
            golomb_decoded_count,
            first_golomb_scan_index,
            first_golomb_value,
            first_coeff_base_scan_index,
            first_coeff_base_position,
            first_coeff_base_context,
            first_coeff_base_reference_magnitude,
            first_coeff_base_symbol,
            first_coeff_base_level,
            bit_position_after: self.reader.bit_position(),
        })
    }

    fn read_eob_extra(
        &mut self,
        transform: TransformBlock,
        plane_type: usize,
        eob_pt: usize,
        eob_base: usize,
    ) -> Result<(Option<usize>, Option<usize>, Option<usize>, usize), DecoderError> {
        if eob_pt < 3 {
            return Ok((None, None, Some(0), eob_base));
        }
        if transform.tx_size != super::syntax::TxSize::Tx32x32 {
            return Err(DecoderError::Unsupported(format!(
                "AV1 eob_extra decode for {:?} is not supported yet",
                transform.tx_size
            )));
        }

        let context = eob_pt - 3;
        let eob_extra_symbol = self
            .reader
            .read_symbol(self.cdf.eob_extra_tx32_cdf_mut(plane_type, context))?;
        let mut eob = eob_base;
        let eob_shift = eob_pt - 3;
        if eob_extra_symbol != 0 {
            eob += 1usize << eob_shift;
        }

        let literal_bits = eob_pt.saturating_sub(3);
        for index in 0..literal_bits {
            let shift = literal_bits - 1 - index;
            let bit = self
                .reader
                .read_literal(1)
                .map_err(|err| DecoderError::Bitstream(format!("AV1 eob_extra_bit: {err}")))?;
            if bit != 0 {
                eob += 1usize << shift;
            }
        }

        Ok((
            Some(context),
            Some(eob_extra_symbol),
            Some(literal_bits),
            eob,
        ))
    }

    fn read_regular_coeff_bases(
        &mut self,
        tx_size: super::syntax::TxSize,
        plane_type: usize,
        eob: usize,
        eob_level: usize,
    ) -> Result<CoeffBaseRead, DecoderError> {
        let sample_count = tx_size.sample_count();
        if eob == 0 || eob > sample_count {
            return Err(DecoderError::Bitstream(format!(
                "AV1 eob {eob} is invalid for {tx_size:?}"
            )));
        }
        let remaining_count = eob - 1;
        let scan = zig_zag_scan(tx_size);
        let mut quant = vec![0i32; sample_count];
        let mut base_range_count = 0usize;
        let mut coeff_br_symbol_count = 0usize;
        let mut first_coeff_br = None;

        let eob_position = scan[eob - 1];
        let eob_level = self.read_coeff_br_range(
            tx_size,
            plane_type,
            eob - 1,
            eob_position,
            eob_level,
            &quant,
            &mut base_range_count,
            &mut coeff_br_symbol_count,
            &mut first_coeff_br,
        )?;
        quant[eob_position] = eob_level as i32;
        if remaining_count == 0 {
            let non_zero_count = coeff_base_non_zero_count(&quant);
            let signs =
                self.read_coeff_signs_and_golomb(tx_size, plane_type, eob, &scan, &mut quant)?;
            return Ok(CoeffBaseRead {
                probe: CoeffBaseProbe {
                    remaining_count,
                    decoded_count: 0,
                    scan_index: None,
                    position: None,
                    context: None,
                    reference_magnitude: None,
                    symbol: None,
                    level: None,
                },
                base_levels: quant,
                non_zero_count,
                base_range_count,
                coeff_br_symbol_count,
                first_coeff_br,
                signs,
            });
        }

        let mut first = None;
        let mut decoded_count = 0usize;

        for scan_index in (0..eob - 1).rev() {
            let position = scan[scan_index];
            let (context, reference_magnitude) = coeff_base_context_2d(tx_size, position, &quant)?;
            if context >= 42 {
                return Err(DecoderError::Unsupported(format!(
                    "AV1 coeff_base decode for Tx32x32 context {context} is not supported yet"
                )));
            }
            let symbol = self
                .reader
                .read_symbol(self.cdf.coeff_base_tx32_cdf_mut(plane_type, context))?;
            let level = self.read_coeff_br_range(
                tx_size,
                plane_type,
                scan_index,
                position,
                symbol,
                &quant,
                &mut base_range_count,
                &mut coeff_br_symbol_count,
                &mut first_coeff_br,
            )?;
            quant[position] = level as i32;
            decoded_count += 1;

            if first.is_none() {
                first = Some((scan_index, position, context, reference_magnitude, symbol));
            }
        }

        let (scan_index, position, context, reference_magnitude, symbol) =
            first.expect("remaining_count > 0 should decode at least one coeff_base");
        let non_zero_count = coeff_base_non_zero_count(&quant);
        Ok(CoeffBaseRead {
            probe: CoeffBaseProbe {
                remaining_count,
                decoded_count,
                scan_index: Some(scan_index),
                position: Some(position),
                context: Some(context),
                reference_magnitude: Some(reference_magnitude),
                symbol: Some(symbol),
                level: Some(symbol),
            },
            signs: self.read_coeff_signs_and_golomb(tx_size, plane_type, eob, &scan, &mut quant)?,
            base_levels: quant,
            non_zero_count,
            base_range_count,
            coeff_br_symbol_count,
            first_coeff_br,
        })
    }

    fn read_coeff_br_range(
        &mut self,
        tx_size: super::syntax::TxSize,
        plane_type: usize,
        scan_index: usize,
        position: usize,
        base_level: usize,
        quant: &[i32],
        base_range_count: &mut usize,
        coeff_br_symbol_count: &mut usize,
        first_coeff_br: &mut Option<CoeffBrProbe>,
    ) -> Result<usize, DecoderError> {
        if base_level <= NUM_BASE_LEVELS {
            return Ok(base_level);
        }
        *base_range_count += 1;
        let mut level = base_level;
        for _ in 0..COEFF_BR_CDF_ROUNDS {
            let context = coeff_br_context_2d(tx_size, position, quant)?;
            let symbol = self
                .reader
                .read_symbol(self.cdf.coeff_br_tx32_cdf_mut(plane_type, context))?;
            level += symbol;
            *coeff_br_symbol_count += 1;
            if first_coeff_br.is_none() {
                *first_coeff_br = Some(CoeffBrProbe {
                    scan_index,
                    position,
                    context,
                    symbol,
                    level_after_symbol: level,
                });
            }
            if symbol < BR_CDF_SIZE - 1 {
                break;
            }
        }
        Ok(level)
    }

    fn read_coeff_signs_and_golomb(
        &mut self,
        _tx_size: super::syntax::TxSize,
        plane_type: usize,
        eob: usize,
        scan: &[usize],
        levels: &mut [i32],
    ) -> Result<CoeffSignRead, DecoderError> {
        if eob == 0 || eob > scan.len() {
            return Err(DecoderError::InvalidParam(
                "AV1 coefficient sign eob exceeds scan".to_string(),
            ));
        }

        let mut sign_count = 0usize;
        let mut dc_sign_context = None;
        let mut dc_sign_symbol = None;
        let mut first_ac_sign_scan_index = None;
        let mut first_ac_sign_bit = None;
        let mut golomb_count = 0usize;
        let mut first_golomb_scan_index = None;
        let mut first_golomb_value = None;

        for scan_index in 0..eob {
            let position = scan[scan_index];
            let mut level = levels[position].unsigned_abs() as usize;
            if level == 0 {
                continue;
            }

            let sign = if scan_index == 0 {
                let context = 0;
                let symbol = self
                    .reader
                    .read_symbol(self.cdf.dc_sign_cdf_mut(plane_type, context))?;
                dc_sign_context = Some(context);
                dc_sign_symbol = Some(symbol);
                symbol != 0
            } else {
                let bit =
                    self.reader.read_literal(1).map_err(|err| {
                        DecoderError::Bitstream(format!("AV1 coeff_sign_bit: {err}"))
                    })? as usize;
                if first_ac_sign_scan_index.is_none() {
                    first_ac_sign_scan_index = Some(scan_index);
                    first_ac_sign_bit = Some(bit);
                }
                bit != 0
            };
            sign_count += 1;

            if level >= MAX_BASE_BR_RANGE {
                let golomb = self.read_golomb()?;
                level += golomb;
                golomb_count += 1;
                if first_golomb_scan_index.is_none() {
                    first_golomb_scan_index = Some(scan_index);
                    first_golomb_value = Some(golomb);
                }
            }

            levels[position] = if sign { -(level as i32) } else { level as i32 };
        }

        Ok(CoeffSignRead {
            sign_count,
            dc_sign_context,
            dc_sign_symbol,
            first_ac_sign_scan_index,
            first_ac_sign_bit,
            golomb_count,
            first_golomb_scan_index,
            first_golomb_value,
        })
    }

    fn read_golomb(&mut self) -> Result<usize, DecoderError> {
        let mut value = 1usize;
        let mut length = 0usize;
        loop {
            length += 1;
            if length > 20 {
                return Err(DecoderError::Bitstream(
                    "AV1 coeff golomb length exceeds 20 bits".to_string(),
                ));
            }
            let bit = self.reader.read_literal(1).map_err(|err| {
                DecoderError::Bitstream(format!("AV1 coeff_golomb_prefix: {err}"))
            })?;
            if bit != 0 {
                break;
            }
        }
        for _ in 0..length - 1 {
            value <<= 1;
            value |=
                self.reader.read_literal(1).map_err(|err| {
                    DecoderError::Bitstream(format!("AV1 coeff_golomb_suffix: {err}"))
                })? as usize;
        }
        Ok(value - 1)
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

fn eob_multisize(transform: TransformBlock) -> usize {
    usize::from(transform.tx_size.width_log2().min(5) + transform.tx_size.height_log2().min(5) - 4)
}

fn eob_base_from_pt(eob_pt: usize) -> usize {
    if eob_pt < 2 {
        eob_pt
    } else {
        (1 << (eob_pt - 2)) + 1
    }
}

fn coeff_base_eob_context(tx_size: super::syntax::TxSize, scan_index: usize) -> usize {
    let samples = tx_size.sample_count();
    if scan_index == 0 {
        0
    } else if scan_index <= samples / 8 {
        1
    } else if scan_index <= samples / 4 {
        2
    } else {
        3
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CoeffBaseProbe {
    remaining_count: usize,
    decoded_count: usize,
    scan_index: Option<usize>,
    position: Option<usize>,
    context: Option<usize>,
    reference_magnitude: Option<usize>,
    symbol: Option<usize>,
    level: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CoeffBaseRead {
    probe: CoeffBaseProbe,
    base_levels: Vec<i32>,
    non_zero_count: usize,
    base_range_count: usize,
    coeff_br_symbol_count: usize,
    first_coeff_br: Option<CoeffBrProbe>,
    signs: CoeffSignRead,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CoeffBrProbe {
    scan_index: usize,
    position: usize,
    context: usize,
    symbol: usize,
    level_after_symbol: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CoeffSignRead {
    sign_count: usize,
    dc_sign_context: Option<usize>,
    dc_sign_symbol: Option<usize>,
    first_ac_sign_scan_index: Option<usize>,
    first_ac_sign_bit: Option<usize>,
    golomb_count: usize,
    first_golomb_scan_index: Option<usize>,
    first_golomb_value: Option<usize>,
}

fn coeff_base_non_zero_count(base_levels: &[i32]) -> usize {
    let mut non_zero_count = 0usize;
    for level in base_levels.iter().copied() {
        let magnitude = level.unsigned_abs() as usize;
        if magnitude != 0 {
            non_zero_count += 1;
        }
    }
    non_zero_count
}

const NUM_BASE_LEVELS: usize = 2;
const COEFF_BASE_RANGE: usize = 12;
const BR_CDF_SIZE: usize = 4;
const COEFF_BR_CDF_ROUNDS: usize = COEFF_BASE_RANGE / (BR_CDF_SIZE - 1);
const MAX_BASE_BR_RANGE: usize = NUM_BASE_LEVELS + COEFF_BASE_RANGE + 1;
const BR_LEVEL_CAP: usize = COEFF_BASE_RANGE + NUM_BASE_LEVELS + 1;
const MAG_REF_OFFSET_WITH_TX_CLASS_2D: [(usize, usize); 3] = [(0, 1), (1, 0), (1, 1)];
const SIG_REF_DIFF_OFFSET_2D: [(usize, usize); 5] = [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0)];
const COEFF_BASE_CTX_OFFSET_SQUARE: [[[usize; 5]; 5]; 5] = [
    [
        [0, 1, 6, 6, 0],
        [1, 6, 6, 21, 0],
        [6, 6, 21, 21, 0],
        [6, 21, 21, 21, 0],
        [0, 0, 0, 0, 0],
    ],
    [
        [0, 1, 6, 6, 21],
        [1, 6, 6, 21, 21],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 1, 6, 6, 21],
        [1, 6, 6, 21, 21],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 1, 6, 6, 21],
        [1, 6, 6, 21, 21],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 1, 6, 6, 21],
        [1, 6, 6, 21, 21],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
];

fn coeff_base_context_2d(
    tx_size: super::syntax::TxSize,
    position: usize,
    quant: &[i32],
) -> Result<(usize, usize), DecoderError> {
    if quant.len() != tx_size.sample_count() {
        return Err(DecoderError::InvalidParam(
            "AV1 coeff_base context quant buffer size does not match transform size".to_string(),
        ));
    }
    if position >= quant.len() {
        return Err(DecoderError::InvalidParam(
            "AV1 coeff_base context position exceeds transform size".to_string(),
        ));
    }

    let width = tx_size.width();
    let height = tx_size.height();
    let row = position / width;
    let col = position % width;
    let mut magnitude = 0usize;
    for (row_offset, col_offset) in SIG_REF_DIFF_OFFSET_2D {
        let ref_row = row + row_offset;
        let ref_col = col + col_offset;
        if ref_row < height && ref_col < width {
            magnitude += quant[ref_row * width + ref_col].unsigned_abs().min(3) as usize;
        }
    }

    if row == 0 && col == 0 {
        return Ok((0, magnitude));
    }

    let context_delta = ((magnitude + 1) >> 1).min(4);
    let offset = COEFF_BASE_CTX_OFFSET_SQUARE[tx_size.coeff_cdf_index()][row.min(4)][col.min(4)];
    Ok((context_delta + offset, magnitude))
}

fn coeff_br_context_2d(
    tx_size: super::syntax::TxSize,
    position: usize,
    quant: &[i32],
) -> Result<usize, DecoderError> {
    if quant.len() != tx_size.sample_count() {
        return Err(DecoderError::InvalidParam(
            "AV1 coeff_br context quant buffer size does not match transform size".to_string(),
        ));
    }
    if position >= quant.len() {
        return Err(DecoderError::InvalidParam(
            "AV1 coeff_br context position exceeds transform size".to_string(),
        ));
    }

    let width = tx_size.width();
    let height = tx_size.height();
    let row = position / width;
    let col = position % width;
    let mut magnitude = 0usize;
    for (row_offset, col_offset) in MAG_REF_OFFSET_WITH_TX_CLASS_2D {
        let ref_row = row + row_offset;
        let ref_col = col + col_offset;
        if ref_row < height && ref_col < width {
            magnitude +=
                (quant[ref_row * width + ref_col].unsigned_abs() as usize).min(BR_LEVEL_CAP);
        }
    }

    let magnitude_context = ((magnitude + 1) >> 1).min(6);
    if position == 0 {
        Ok(magnitude_context)
    } else if row < 2 && col < 2 {
        Ok(magnitude_context + 7)
    } else {
        Ok(magnitude_context + 14)
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
                &transforms,
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
            assert_eq!(probes[0].first_non_zero_transform_index, None);
            assert_eq!(probes[0].first_non_zero_tx_size, None);
            assert_eq!(probes[0].coeff_base_eob_context, None);
            assert_eq!(probes[0].coeff_base_eob_symbol, None);
            assert_eq!(probes[0].coeff_base_eob_level, None);
            assert_eq!(probes[0].regular_coeff_base_count, None);
            assert_eq!(probes[0].regular_coeff_base_decoded_count, None);
            assert_eq!(probes[0].coeff_base_non_zero_count, None);
            assert_eq!(probes[0].coeff_base_range_count, None);
            assert_eq!(probes[0].coeff_br_decoded_count, None);
            assert_eq!(probes[0].first_coeff_br_scan_index, None);
            assert_eq!(probes[0].first_coeff_br_context, None);
            assert_eq!(probes[0].first_coeff_br_symbol, None);
            assert_eq!(probes[0].first_coeff_br_level, None);
            assert_eq!(probes[0].first_coeff_base_scan_index, None);
            assert_eq!(probes[0].first_coeff_base_context, None);
            assert_eq!(probes[0].first_coeff_base_symbol, None);
            assert_eq!(probes[0].first_coeff_base_level, None);
        } else {
            assert!(probes[0].txb_skip_context.unwrap() <= 1);
            assert!(probes[0].all_zero_symbol.unwrap() <= 1);
            assert_eq!(
                probes[0].zero_transform_count,
                probes[0].first_non_zero_transform_index.unwrap_or(16)
            );
            if probes[0].first_non_zero_transform_index.is_none() {
                assert_eq!(probes[0].eob_multisize, None);
                assert_eq!(probes[0].eob_pt_symbol, None);
                assert_eq!(probes[0].eob_base, None);
                assert_eq!(probes[0].eob_extra_symbol, None);
                assert_eq!(probes[0].eob, None);
                assert_eq!(probes[0].coeff_base_eob_context, None);
                assert_eq!(probes[0].coeff_base_eob_symbol, None);
                assert_eq!(probes[0].coeff_base_eob_level, None);
                assert_eq!(probes[0].regular_coeff_base_count, None);
                assert_eq!(probes[0].regular_coeff_base_decoded_count, None);
                assert_eq!(probes[0].coeff_base_non_zero_count, None);
                assert_eq!(probes[0].coeff_base_range_count, None);
                assert_eq!(probes[0].coeff_br_decoded_count, None);
                assert_eq!(probes[0].first_coeff_br_scan_index, None);
                assert_eq!(probes[0].first_coeff_br_context, None);
                assert_eq!(probes[0].first_coeff_br_symbol, None);
                assert_eq!(probes[0].first_coeff_br_level, None);
                assert_eq!(probes[0].first_coeff_base_scan_index, None);
                assert_eq!(probes[0].first_coeff_base_context, None);
                assert_eq!(probes[0].first_coeff_base_symbol, None);
                assert_eq!(probes[0].first_coeff_base_level, None);
            } else {
                assert!(probes[0].first_non_zero_transform_index.unwrap() < 16);
                assert_eq!(
                    probes[0].first_non_zero_tx_size,
                    Some(super::super::syntax::TxSize::Tx32x32)
                );
                assert_eq!(probes[0].eob_multisize, Some(6));
                assert!(probes[0].eob_pt_symbol.unwrap() < 11);
                assert_eq!(
                    probes[0].eob_pt.unwrap(),
                    probes[0].eob_pt_symbol.unwrap() + 1
                );
                assert!(probes[0].eob_base.unwrap() > 0);
                assert_eq!(
                    probes[0].eob_extra_context,
                    probes[0].eob_pt.filter(|pt| *pt >= 3).map(|pt| pt - 3)
                );
                assert!(probes[0].eob_extra_symbol.unwrap_or(0) <= 1);
                assert_eq!(
                    probes[0].eob_extra_literal_bits,
                    Some(probes[0].eob_pt.unwrap().saturating_sub(3))
                );
                assert!(probes[0].eob.unwrap() >= probes[0].eob_base.unwrap());
                assert!(probes[0].eob.unwrap() <= 1024);
                assert!(probes[0].coeff_base_eob_context.unwrap() < 4);
                assert!(probes[0].coeff_base_eob_symbol.unwrap() < 3);
                assert_eq!(
                    probes[0].coeff_base_eob_level.unwrap(),
                    probes[0].coeff_base_eob_symbol.unwrap() + 1
                );
                assert_eq!(
                    probes[0].regular_coeff_base_count,
                    Some(probes[0].eob.unwrap() - 1)
                );
                assert_eq!(
                    probes[0].regular_coeff_base_decoded_count,
                    probes[0].regular_coeff_base_count
                );
                assert!(probes[0].coeff_base_non_zero_count.unwrap() >= 1);
                assert!(probes[0].coeff_base_non_zero_count.unwrap() <= probes[0].eob.unwrap());
                assert!(
                    probes[0].coeff_base_range_count.unwrap()
                        <= probes[0].coeff_base_non_zero_count.unwrap()
                );
                assert!(
                    probes[0].coeff_br_decoded_count.unwrap()
                        >= probes[0].coeff_base_range_count.unwrap()
                );
                if probes[0].coeff_base_range_count.unwrap() > 0 {
                    assert!(probes[0].first_coeff_br_scan_index.unwrap() < probes[0].eob.unwrap());
                    assert!(probes[0].first_coeff_br_position.unwrap() < 1024);
                    assert!(probes[0].first_coeff_br_context.unwrap() < 21);
                    assert!(probes[0].first_coeff_br_symbol.unwrap() < 4);
                    assert!(probes[0].first_coeff_br_level.unwrap() >= 3);
                } else {
                    assert_eq!(probes[0].first_coeff_br_scan_index, None);
                    assert_eq!(probes[0].first_coeff_br_context, None);
                    assert_eq!(probes[0].first_coeff_br_symbol, None);
                    assert_eq!(probes[0].first_coeff_br_level, None);
                }
                if probes[0].regular_coeff_base_count.unwrap() > 0 {
                    assert_eq!(
                        probes[0].first_coeff_base_scan_index,
                        Some(probes[0].eob.unwrap() - 2)
                    );
                    assert!(probes[0].first_coeff_base_position.unwrap() < 1024);
                    assert!(probes[0].first_coeff_base_context.unwrap() < 42);
                    assert!(probes[0].first_coeff_base_reference_magnitude.unwrap() <= 15);
                    assert!(probes[0].first_coeff_base_symbol.unwrap() < 4);
                    assert_eq!(
                        probes[0].first_coeff_base_level,
                        probes[0].first_coeff_base_symbol
                    );
                }
            }
        }
        assert_eq!(
            probes[0].first_tx_size,
            Some(super::super::syntax::TxSize::Tx32x32)
        );
    }

    #[test]
    fn coeff_base_context_2d_matches_square_offset_rules() {
        let mut quant = vec![0; super::super::syntax::TxSize::Tx32x32.sample_count()];

        assert_eq!(
            coeff_base_context_2d(super::super::syntax::TxSize::Tx32x32, 0, &quant).unwrap(),
            (0, 0)
        );

        quant[2] = 3;
        assert_eq!(
            coeff_base_context_2d(super::super::syntax::TxSize::Tx32x32, 1, &quant).unwrap(),
            (3, 3)
        );

        assert_eq!(
            coeff_base_context_2d(super::super::syntax::TxSize::Tx32x32, 4 * 32 + 4, &quant)
                .unwrap(),
            (21, 0)
        );
    }

    #[test]
    fn coeff_br_context_2d_matches_square_tx_rules() {
        let mut quant = vec![0; super::super::syntax::TxSize::Tx32x32.sample_count()];

        assert_eq!(
            coeff_br_context_2d(super::super::syntax::TxSize::Tx32x32, 0, &quant).unwrap(),
            0
        );

        quant[1] = 3;
        assert_eq!(
            coeff_br_context_2d(super::super::syntax::TxSize::Tx32x32, 0, &quant).unwrap(),
            2
        );

        assert_eq!(
            coeff_br_context_2d(super::super::syntax::TxSize::Tx32x32, 32 + 1, &quant).unwrap(),
            7
        );

        assert_eq!(
            coeff_br_context_2d(super::super::syntax::TxSize::Tx32x32, 4 * 32 + 4, &quant).unwrap(),
            14
        );
    }
}
