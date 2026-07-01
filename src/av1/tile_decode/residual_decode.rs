use super::coefficient::{EntropyCoefficientSource, decode_coefficients};
use super::diagnostic::ResidualProbe;
use super::residual_preview::build_residual_preview;
use super::residual_probe::{
    FirstNonZeroTransformScan, ResidualProbeContext, ResidualProbeFields, empty_residual_probe,
    scanned_residual_probe,
};
use super::{BlockModeProbe, DecodedTransform, TileDecoder, coefficient_entropy_context};
use crate::DecoderError;
use crate::av1::frame::FrameHeader;
use crate::av1::quant::QuantState;
use crate::av1::syntax::{TxSize, TxType};
use crate::av1::tile_decode::coefficient::CoefficientRead;
use crate::av1::transform::TransformBlock;

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

        self.read_scanned_residual_probe(
            ResidualProbeContext {
                tile_id,
                skipped: false,
                transform_count,
                first_tx_size: Some(first_transform.tx_size),
                bit_position_after: self.reader.bit_position(),
            },
            frame,
            block_mode,
            first_non_zero_scan,
            quant_state,
            bit_depth,
        )
    }

    fn read_scanned_residual_probe(
        &mut self,
        context: ResidualProbeContext,
        frame: &FrameHeader,
        block_mode: &BlockModeProbe,
        first_non_zero_scan: FirstNonZeroTransformScan,
        quant_state: QuantState,
        bit_depth: u8,
    ) -> Result<ResidualProbe, DecoderError> {
        let fields = if let Some(non_zero_transform) = first_non_zero_scan.first_non_zero_transform
        {
            self.read_non_zero_residual_probe_fields(
                frame,
                block_mode,
                non_zero_transform,
                first_non_zero_scan
                    .first_non_zero_txb_context
                    .expect("non-zero transform should retain its txb context")
                    .dc_sign,
                quant_state,
                bit_depth,
            )?
        } else {
            ResidualProbeFields::default()
        };

        Ok(scanned_residual_probe(
            ResidualProbeContext {
                bit_position_after: self.reader.bit_position(),
                ..context
            },
            block_mode,
            first_non_zero_scan,
            fields,
        ))
    }

    fn read_non_zero_residual_probe_fields(
        &mut self,
        frame: &FrameHeader,
        block_mode: &BlockModeProbe,
        non_zero_transform: TransformBlock,
        dc_sign_context: usize,
        quant_state: QuantState,
        bit_depth: u8,
    ) -> Result<ResidualProbeFields, DecoderError> {
        let tx_type_probe = self.read_intra_tx_type(frame, block_mode, non_zero_transform)?;
        let plane_type = usize::from(non_zero_transform.plane > 0);
        let coefficient_read = self.read_coefficient_state(
            non_zero_transform.tx_size,
            tx_type_probe.tx_type,
            plane_type,
            dc_sign_context,
        )?;
        let coeff_base_read = &coefficient_read.base;
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

        Ok(ResidualProbeFields::from_reads(
            tx_type_probe,
            coefficient_read,
            residual_preview,
        ))
    }

    fn read_first_non_zero_transform(
        &mut self,
        block_mode: &BlockModeProbe,
        transforms: &[TransformBlock],
    ) -> Result<FirstNonZeroTransformScan, DecoderError> {
        let mut scan = FirstNonZeroTransformScan::scanning();

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
