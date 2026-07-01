use super::coefficient::CoefficientRead;
use super::diagnostic::TxTypeProbe;
use super::residual_preview::build_probe_residual_preview;
use super::residual_probe::ResidualProbeFields;
use super::{BlockModeProbe, TileDecoder, coefficient_entropy_context};
use crate::DecoderError;
use crate::av1::frame::FrameHeader;
use crate::av1::quant::QuantState;
use crate::av1::transform::TransformBlock;

pub(super) struct NonZeroCoefficientState {
    tx_type_probe: TxTypeProbe,
    coefficient_read: CoefficientRead,
}

impl<'a> TileDecoder<'a> {
    pub(super) fn read_non_zero_residual_probe_fields(
        &mut self,
        frame: &FrameHeader,
        block_mode: &BlockModeProbe,
        non_zero_transform: TransformBlock,
        dc_sign_context: usize,
        quant_state: QuantState,
        bit_depth: u8,
    ) -> Result<ResidualProbeFields, DecoderError> {
        let non_zero_state = self.read_non_zero_coefficient_state(
            frame,
            block_mode,
            non_zero_transform,
            dc_sign_context,
        )?;
        self.finish_non_zero_residual_probe_fields(
            non_zero_transform,
            non_zero_state,
            quant_state,
            bit_depth,
        )
    }

    fn finish_non_zero_residual_probe_fields(
        &mut self,
        non_zero_transform: TransformBlock,
        non_zero_state: NonZeroCoefficientState,
        quant_state: QuantState,
        bit_depth: u8,
    ) -> Result<ResidualProbeFields, DecoderError> {
        self.update_non_zero_txb_entropy_context(
            non_zero_transform,
            &non_zero_state.coefficient_read,
        );
        let residual_preview = build_probe_residual_preview(
            non_zero_transform,
            &non_zero_state.coefficient_read,
            quant_state,
            bit_depth,
            non_zero_state.tx_type_probe.tx_type,
        )?;

        Ok(ResidualProbeFields::from_reads(
            non_zero_state.tx_type_probe,
            non_zero_state.coefficient_read,
            residual_preview,
        ))
    }

    fn read_non_zero_coefficient_state(
        &mut self,
        frame: &FrameHeader,
        block_mode: &BlockModeProbe,
        non_zero_transform: TransformBlock,
        dc_sign_context: usize,
    ) -> Result<NonZeroCoefficientState, DecoderError> {
        let tx_type_probe = self.read_intra_tx_type(frame, block_mode, non_zero_transform)?;
        let plane_type = usize::from(non_zero_transform.plane > 0);
        let coefficient_read = self.read_coefficient_state(
            non_zero_transform.tx_size,
            tx_type_probe.tx_type,
            plane_type,
            dc_sign_context,
        )?;
        Ok(NonZeroCoefficientState {
            tx_type_probe,
            coefficient_read,
        })
    }

    fn update_non_zero_txb_entropy_context(
        &mut self,
        transform: TransformBlock,
        coefficient_read: &CoefficientRead,
    ) {
        self.set_txb_entropy_context(
            transform,
            coefficient_entropy_context(&coefficient_read.base.base_levels),
        );
    }
}
