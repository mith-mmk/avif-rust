use super::coefficient::{EntropyCoefficientSource, decode_coefficients};
use super::coefficient_context::TxbContext;
use super::diagnostic::ResidualProbe;
use super::residual_preview::build_residual_preview;
use super::{BlockModeProbe, DecodedTransform, TileDecoder, coefficient_entropy_context};
use crate::DecoderError;
use crate::av1::frame::FrameHeader;
use crate::av1::quant::QuantState;
use crate::av1::syntax::{TxSize, TxType};
use crate::av1::tile_decode::coefficient::CoefficientRead;
use crate::av1::transform::TransformBlock;

fn empty_residual_probe(
    tile_id: u32,
    block_mode: &BlockModeProbe,
    skipped: bool,
    transform_count: usize,
    zero_transform_count: usize,
    first_tx_size: Option<TxSize>,
    first_transform_all_zero: bool,
) -> ResidualProbe {
    ResidualProbe {
        tile_id,
        block_size: block_mode.block_size,
        skipped,
        transform_count,
        zero_transform_count,
        first_tx_size,
        first_non_zero_transform_index: None,
        first_non_zero_transform: None,
        first_non_zero_tx_size: None,
        tx_type_read: false,
        tx_type_set: None,
        tx_type_symbol: None,
        tx_type: None,
        txb_skip_context: None,
        all_zero_symbol: None,
        first_transform_all_zero,
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
        signed_coeff_non_zero_count: None,
        first_signed_coeff_scan_index: None,
        first_signed_coeff_position: None,
        first_signed_coeff_value: None,
        dequant_non_zero_count: None,
        first_dequant_coeff_position: None,
        first_dequant_coeff_value: None,
        residual_preview_tx_type: None,
        residual_preview_sample_count: None,
        first_residual_preview_sample: None,
        first_coeff_base_scan_index: None,
        first_coeff_base_position: None,
        first_coeff_base_context: None,
        first_coeff_base_reference_magnitude: None,
        first_coeff_base_symbol: None,
        first_coeff_base_level: None,
        first_quantized_coefficients: None,
        bit_position_after: block_mode.bit_position_after,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FirstNonZeroTransformScan {
    txb_skip_context: Option<usize>,
    all_zero_symbol: Option<usize>,
    first_transform_all_zero: bool,
    zero_transform_count: usize,
    first_non_zero_transform: Option<TransformBlock>,
    first_non_zero_transform_index: Option<usize>,
    first_non_zero_txb_context: Option<TxbContext>,
}

impl<'a> TileDecoder<'a> {
    pub fn read_first_transform_residual(
        &mut self,
        tile_id: u32,
        frame: &FrameHeader,
        block_mode: &BlockModeProbe,
        transforms: &[TransformBlock],
        quant_state: QuantState,
        bit_depth: u8,
    ) -> Result<ResidualProbe, DecoderError> {
        let transform_count = transforms.len();
        let first_transform = transforms.first().copied();
        if block_mode.skip {
            for transform in transforms.iter().copied() {
                self.set_txb_entropy_context(transform, 0);
            }
            return Ok(empty_residual_probe(
                tile_id,
                block_mode,
                true,
                transform_count,
                transform_count,
                first_transform.map(|transform| transform.tx_size),
                true,
            ));
        }

        let Some(first_transform) = first_transform else {
            return Ok(empty_residual_probe(
                tile_id,
                block_mode,
                false,
                transform_count,
                0,
                None,
                false,
            ));
        };

        let first_non_zero_scan = self.read_first_non_zero_transform(block_mode, transforms)?;

        let (
            eob_multisize,
            eob_pt_symbol,
            eob_pt,
            eob_base,
            eob_extra_context,
            eob_extra_symbol,
            eob_extra_literal_bits,
            eob,
            tx_type_read,
            tx_type_set,
            tx_type_symbol,
            tx_type,
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
            signed_coeff_non_zero_count,
            first_signed_coeff_scan_index,
            first_signed_coeff_position,
            first_signed_coeff_value,
            dequant_non_zero_count,
            first_dequant_coeff_position,
            first_dequant_coeff_value,
            residual_preview_tx_type,
            residual_preview_sample_count,
            first_residual_preview_sample,
            first_coeff_base_scan_index,
            first_coeff_base_position,
            first_coeff_base_context,
            first_coeff_base_reference_magnitude,
            first_coeff_base_symbol,
            first_coeff_base_level,
            first_quantized_coefficients,
        ) = if let Some(non_zero_transform) = first_non_zero_scan.first_non_zero_transform {
            let tx_type_probe = self.read_intra_tx_type(frame, block_mode, non_zero_transform)?;
            let plane_type = usize::from(non_zero_transform.plane > 0);
            let coefficient_read = self.read_coefficient_state(
                non_zero_transform.tx_size,
                tx_type_probe.tx_type,
                plane_type,
                first_non_zero_scan
                    .first_non_zero_txb_context
                    .expect("non-zero transform should retain its txb context")
                    .dc_sign,
            )?;
            let coeff_base_read = coefficient_read.base;
            self.set_txb_entropy_context(
                non_zero_transform,
                coefficient_entropy_context(&coeff_base_read.base_levels),
            );
            debug_assert_eq!(
                coeff_base_read.base_levels.len(),
                non_zero_transform.tx_size.sample_count()
            );
            let residual_preview = build_residual_preview(
                non_zero_transform,
                &coeff_base_read.base_levels,
                quant_state,
                bit_depth,
                tx_type_probe.tx_type,
            )?;
            (
                Some(coefficient_read.eob_multisize),
                Some(coefficient_read.eob_pt_symbol),
                Some(coefficient_read.eob_pt),
                Some(coefficient_read.eob_base),
                coefficient_read.eob_extra_context,
                coefficient_read.eob_extra_symbol,
                Some(coefficient_read.eob_extra_literal_bits),
                Some(coefficient_read.eob),
                tx_type_probe.read,
                tx_type_probe.set,
                tx_type_probe.symbol,
                Some(tx_type_probe.tx_type),
                Some(coefficient_read.coeff_base_eob_context),
                Some(coefficient_read.coeff_base_eob_symbol),
                Some(coefficient_read.coeff_base_eob_level),
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
                Some(coeff_base_read.signed_non_zero_count),
                coeff_base_read
                    .first_signed_coeff
                    .map(|first| first.scan_index),
                coeff_base_read
                    .first_signed_coeff
                    .map(|first| first.position),
                coeff_base_read.first_signed_coeff.map(|first| first.value),
                residual_preview
                    .as_ref()
                    .map(|preview| preview.dequant_non_zero_count),
                residual_preview
                    .as_ref()
                    .and_then(|preview| preview.first_dequant_coeff)
                    .map(|first| first.position),
                residual_preview
                    .as_ref()
                    .and_then(|preview| preview.first_dequant_coeff)
                    .map(|first| first.value),
                residual_preview.as_ref().map(|preview| preview.tx_type),
                residual_preview
                    .as_ref()
                    .map(|preview| preview.residual_sample_count),
                residual_preview
                    .as_ref()
                    .and_then(|preview| preview.first_residual_sample),
                coeff_base_read.probe.scan_index,
                coeff_base_read.probe.position,
                coeff_base_read.probe.context,
                coeff_base_read.probe.reference_magnitude,
                coeff_base_read.probe.symbol,
                coeff_base_read.probe.level,
                Some(coeff_base_read.base_levels),
            )
        } else {
            (
                None, None, None, None, None, None, None, None, false, None, None, None, None,
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None, None,
            )
        };

        Ok(ResidualProbe {
            tile_id,
            block_size: block_mode.block_size,
            skipped: false,
            transform_count,
            zero_transform_count: first_non_zero_scan.zero_transform_count,
            first_tx_size: Some(first_transform.tx_size),
            first_non_zero_transform_index: first_non_zero_scan.first_non_zero_transform_index,
            first_non_zero_transform: first_non_zero_scan.first_non_zero_transform,
            first_non_zero_tx_size: first_non_zero_scan
                .first_non_zero_transform
                .map(|transform| transform.tx_size),
            tx_type_read,
            tx_type_set,
            tx_type_symbol,
            tx_type,
            txb_skip_context: first_non_zero_scan.txb_skip_context,
            all_zero_symbol: first_non_zero_scan.all_zero_symbol,
            first_transform_all_zero: first_non_zero_scan.first_transform_all_zero,
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
            signed_coeff_non_zero_count,
            first_signed_coeff_scan_index,
            first_signed_coeff_position,
            first_signed_coeff_value,
            dequant_non_zero_count,
            first_dequant_coeff_position,
            first_dequant_coeff_value,
            residual_preview_tx_type,
            residual_preview_sample_count,
            first_residual_preview_sample,
            first_coeff_base_scan_index,
            first_coeff_base_position,
            first_coeff_base_context,
            first_coeff_base_reference_magnitude,
            first_coeff_base_symbol,
            first_coeff_base_level,
            first_quantized_coefficients,
            bit_position_after: self.reader.bit_position(),
        })
    }

    fn read_first_non_zero_transform(
        &mut self,
        block_mode: &BlockModeProbe,
        transforms: &[TransformBlock],
    ) -> Result<FirstNonZeroTransformScan, DecoderError> {
        let mut scan = FirstNonZeroTransformScan {
            txb_skip_context: None,
            all_zero_symbol: None,
            first_transform_all_zero: true,
            zero_transform_count: 0,
            first_non_zero_transform: None,
            first_non_zero_transform_index: None,
            first_non_zero_txb_context: None,
        };

        for (index, transform) in transforms.iter().copied().enumerate() {
            let txb_context = self.txb_context(block_mode.block_size, transform);
            let all_zero_symbol = self.reader.read_symbol(
                self.cdf
                    .txb_skip_cdf_mut(transform.tx_size.coeff_cdf_index(), txb_context.skip),
            )?;
            if index == 0 {
                scan.txb_skip_context = Some(txb_context.skip);
                scan.all_zero_symbol = Some(all_zero_symbol);
                scan.first_transform_all_zero = all_zero_symbol != 0;
            }
            if all_zero_symbol != 0 {
                self.set_txb_entropy_context(transform, 0);
                scan.zero_transform_count += 1;
                continue;
            }
            scan.first_non_zero_transform = Some(transform);
            scan.first_non_zero_transform_index = Some(index);
            scan.first_non_zero_txb_context = Some(txb_context);
            break;
        }

        Ok(scan)
    }

    pub(super) fn read_decoded_transform(
        &mut self,
        frame: &FrameHeader,
        block_mode: &BlockModeProbe,
        transform: TransformBlock,
        dc_sign_context: usize,
    ) -> Result<DecodedTransform, DecoderError> {
        let tx_type = self
            .read_intra_tx_type(frame, block_mode, transform)?
            .tx_type;
        let plane_type = usize::from(transform.plane > 0);
        let coefficient_read =
            self.read_coefficient_state(transform.tx_size, tx_type, plane_type, dc_sign_context)?;
        Ok(DecodedTransform {
            transform,
            tx_type,
            coefficients: coefficient_read.base.base_levels,
        })
    }

    pub(super) fn read_coefficient_state(
        &mut self,
        tx_size: TxSize,
        tx_type: TxType,
        plane_type: usize,
        dc_sign_context: usize,
    ) -> Result<CoefficientRead, DecoderError> {
        let mut source = EntropyCoefficientSource::new(&mut self.reader, &mut self.cdf);
        decode_coefficients(&mut source, tx_size, tx_type, plane_type, dc_sign_context)
    }
}
