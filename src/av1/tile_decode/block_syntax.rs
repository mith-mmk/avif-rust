use super::{BlockModeProbe, CflParams, TileDecoder};
use crate::DecoderError;
use crate::av1::decode::TileDecodePlan;
use crate::av1::frame::{FrameHeader, TxMode};
use crate::av1::sequence::SequenceHeader;
use crate::av1::syntax::{BlockSize, PredictionMode, TxSize, UvPredictionMode};
use crate::av1::tile_decode::context_grid::{
    fill_mi_grid, has_smooth_neighbour, intra_mode_context,
};

impl<'a> TileDecoder<'a> {
    pub fn read_intra_frame_block_mode(
        &mut self,
        sequence: &SequenceHeader,
        frame: &FrameHeader,
        tile: &TileDecodePlan,
        block_size: BlockSize,
        x: usize,
        y: usize,
    ) -> Result<BlockModeProbe, DecoderError> {
        self.current_cfl = None;
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

        let skip_context = self.skip_context(x, y);
        let skip_symbol = self
            .reader
            .read_symbol(self.cdf.skip_cdf_mut(skip_context))?;
        let skip = skip_symbol != 0;
        let cdef_idx = self.read_cdef_index(sequence, frame, skip, x, y)?;
        let trace_stage =
            std::env::var_os("AVIF_TRACE_WML2_STAGES").is_some() && x == 80 && y == 28;
        if trace_stage {
            eprintln!(
                "Rust stage skip ctx={skip_context} symbol={skip_symbol} state={:?}",
                self.reader.trace_state()
            );
        }

        let y_above_context = self.above_y_mode_context(x, y);
        let y_left_context = self.left_y_mode_context(x, y);
        let y_mode_symbol = self.reader.read_symbol(
            self.cdf
                .intra_frame_y_mode_cdf_mut(y_above_context, y_left_context),
        )?;
        let y_mode = PredictionMode::from_intra_symbol(y_mode_symbol).ok_or_else(|| {
            DecoderError::Bitstream(format!("AV1 y_mode symbol {y_mode_symbol} is invalid"))
        })?;
        let use_angle_delta = use_angle_delta(block_size);
        if trace_stage {
            eprintln!(
                "Rust stage y_mode symbol={y_mode_symbol} mode={y_mode:?} state={:?}",
                self.reader.trace_state()
            );
        }
        let angle_delta_y = if use_angle_delta && y_mode.is_directional() {
            Some(self.read_angle_delta(y_mode.directional_index().unwrap())?)
        } else {
            None
        };
        let has_chroma = !sequence.color_config.monochrome;
        let (uv_mode_symbol, uv_mode, angle_delta_uv) = if has_chroma {
            let cfl_allowed = cfl_is_allowed(frame.quantization.coded_lossless(), block_size);
            let uv_symbol = if cfl_allowed {
                self.reader
                    .read_symbol(self.cdf.uv_mode_cfl_allowed_cdf_mut(y_mode_symbol))?
            } else {
                self.reader
                    .read_symbol(self.cdf.uv_mode_cfl_not_allowed_cdf_mut(y_mode_symbol))?
            };
            let uv_mode = UvPredictionMode::from_symbol(uv_symbol).ok_or_else(|| {
                DecoderError::Bitstream(format!("AV1 uv_mode symbol {uv_symbol} is invalid"))
            })?;
            if trace_stage {
                eprintln!(
                    "Rust stage uv symbol={uv_symbol} mode={uv_mode:?} state={:?}",
                    self.reader.trace_state()
                );
            }
            if uv_mode == UvPredictionMode::Cfl {
                self.current_cfl = Some(self.read_cfl_params()?);
            }
            let angle_delta = if use_angle_delta && uv_mode.is_directional() {
                Some(self.read_angle_delta(uv_mode.directional_index().unwrap())?)
            } else {
                None
            };
            (Some(uv_symbol), Some(uv_mode), angle_delta)
        } else {
            (None, None, None)
        };
        if std::env::var_os("AVIF_TRACE_WML2_MODES").is_some() && x == 88 && y == 16 {
            eprintln!("Rust x88 after uv state={:?}", self.reader.trace_state());
        }
        let palette =
            self.read_palette_mode_info(sequence, frame, block_size, x, y, y_mode, uv_mode)?;
        if std::env::var_os("AVIF_TRACE_WML2_MODES").is_some() && x == 88 && y == 16 {
            eprintln!(
                "Rust x88 palette y={:?} uv={:?} state={:?}",
                palette.y.as_ref().map(|value| value.colors.len()),
                palette.uv.as_ref().map(|value| value.colors.len() / 2),
                self.reader.trace_state()
            );
        }
        let mut filter_intra_mode = None;
        if sequence.enable_filter_intra
            && block_size.width() <= 32
            && block_size.height() <= 32
            && y_mode == PredictionMode::Dc
            && palette.y.is_none()
        {
            let use_filter_intra = self.reader.read_symbol(
                self.cdf
                    .use_filter_intra_cdf_mut(block_size.filter_intra_cdf_index()),
            )? != 0;
            if use_filter_intra {
                filter_intra_mode = Some(
                    self.reader
                        .read_symbol(self.cdf.filter_intra_mode_cdf_mut())?,
                );
            }
        }
        let y_smooth_neighbour = self.has_smooth_intra_neighbour(0, x, y);
        let uv_smooth_neighbour = self.has_smooth_intra_neighbour(1, x, y);
        self.set_y_mode_context(x, y, block_size, intra_mode_context(y_mode));
        self.set_smooth_context(
            x,
            y,
            block_size,
            y_mode.is_smooth(),
            matches!(uv_mode, Some(UvPredictionMode::Intra(mode)) if mode.is_smooth()),
        );
        self.set_skip_context(x, y, block_size, skip);
        let mut palette = palette;
        self.read_palette_tokens(sequence, block_size, x, y, &mut palette)?;
        if std::env::var_os("AVIF_TRACE_WML2_MODES").is_some() && x == 88 && y == 16 {
            eprintln!(
                "Rust x88 palette tokens state={:?}",
                self.reader.trace_state()
            );
        }
        let (tx_size_context, tx_size_symbol, tx_size) =
            self.read_intra_tx_size(frame, block_size, skip, x, y)?;

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
            y_smooth_neighbour,
            filter_intra_mode,
            uv_mode_symbol,
            uv_mode,
            angle_delta_uv,
            uv_smooth_neighbour,
            palette,
            tx_size_context,
            tx_size_symbol,
            tx_size,
            bit_position_after: self.reader.bit_position(),
        })
    }

    fn read_intra_tx_size(
        &mut self,
        frame: &FrameHeader,
        block_size: BlockSize,
        skip: bool,
        x: usize,
        y: usize,
    ) -> Result<(Option<usize>, Option<usize>, TxSize), DecoderError> {
        let tx_size = match frame.tx_mode {
            TxMode::Only4x4 => TxSize::Tx4x4,
            TxMode::Largest => block_size.largest_supported_tx_size(),
            TxMode::Select if block_size.signals_tx_size() && !skip => {
                let context = self.tx_size_context(x, y, block_size);
                let category = block_size.tx_size_category();
                let symbol = self
                    .reader
                    .read_symbol(self.cdf.tx_size_cdf_mut(category, context))?;
                if symbol > block_size.max_tx_size_depth() {
                    return Err(DecoderError::Bitstream(format!(
                        "AV1 tx_size symbol {symbol} exceeds max depth for {block_size:?}"
                    )));
                }
                let tx_size = block_size.tx_size_from_depth(symbol);
                self.set_txfm_context(x, y, block_size, tx_size);
                return Ok((Some(context), Some(symbol), tx_size));
            }
            TxMode::Select => TxSize::Tx4x4,
        };
        self.set_txfm_context(x, y, block_size, tx_size);
        Ok((None, None, tx_size))
    }

    fn read_cdef_index(
        &mut self,
        sequence: &SequenceHeader,
        frame: &FrameHeader,
        skip: bool,
        x: usize,
        y: usize,
    ) -> Result<Option<u32>, DecoderError> {
        if !frame.cdef.enabled || frame.allow_intrabc {
            return Ok(None);
        }
        let superblock_mi = if sequence.use_128x128_superblock {
            32
        } else {
            16
        };
        let mi_col = x >> 2;
        let mi_row = y >> 2;
        if mi_col % superblock_mi == 0 && mi_row % superblock_mi == 0 {
            self.cdef_transmitted = [false; 4];
        }
        let index = if sequence.use_128x128_superblock {
            usize::from((mi_col & 16) != 0) + usize::from((mi_row & 16) != 0) * 2
        } else {
            0
        };
        if self.cdef_transmitted[index] || skip {
            return Ok(None);
        }
        let value = self
            .reader
            .read_literal(frame.cdef.bits as usize)
            .map_err(|err| DecoderError::Bitstream(format!("AV1 cdef_idx: {err}")))?;
        self.cdef_transmitted[index] = true;
        Ok(Some(value))
    }

    fn read_angle_delta(&mut self, directional_index: usize) -> Result<i8, DecoderError> {
        let symbol = self
            .reader
            .read_symbol(self.cdf.angle_delta_cdf_mut(directional_index))?;
        Ok(symbol as i8 - 3)
    }

    fn read_cfl_params(&mut self) -> Result<CflParams, DecoderError> {
        let joint_sign = self.reader.read_symbol(self.cdf.cfl_sign_cdf_mut())?;
        let (sign_u, sign_v) = cfl_signs(joint_sign)?;
        let alpha_u_q3 = self.read_cfl_alpha(joint_sign, sign_u, true)?;
        let alpha_v_q3 = self.read_cfl_alpha(joint_sign, sign_v, false)?;
        Ok(CflParams {
            alpha_u_q3,
            alpha_v_q3,
        })
    }

    fn read_cfl_alpha(
        &mut self,
        joint_sign: usize,
        sign: usize,
        is_u: bool,
    ) -> Result<i8, DecoderError> {
        if sign == 0 {
            return Ok(0);
        }
        let context = if is_u {
            joint_sign.saturating_sub(2)
        } else {
            let (sign_u, sign_v) = cfl_signs(joint_sign)?;
            sign_v * 3 + sign_u - 3
        };
        let magnitude = self
            .reader
            .read_symbol(self.cdf.cfl_alpha_cdf_mut(context))?
            + 1;
        Ok(if sign == 1 {
            -(magnitude as i8)
        } else {
            magnitude as i8
        })
    }

    fn above_y_mode_context(&self, x: usize, y: usize) -> usize {
        if y < 4 {
            return 0;
        }
        self.y_mode_at_mi(x >> 2, (y >> 2).saturating_sub(1))
            .unwrap_or(0)
    }

    fn left_y_mode_context(&self, x: usize, y: usize) -> usize {
        if x < 4 {
            return 0;
        }
        self.y_mode_at_mi((x >> 2).saturating_sub(1), y >> 2)
            .unwrap_or(0)
    }

    fn y_mode_at_mi(&self, mi_col: usize, mi_row: usize) -> Option<usize> {
        if mi_col >= self.mi_cols || mi_row >= self.mi_rows {
            return None;
        }
        self.y_mode_grid[mi_row * self.mi_cols + mi_col]
    }

    fn set_y_mode_context(&mut self, x: usize, y: usize, block_size: BlockSize, symbol: usize) {
        fill_mi_grid(
            &mut self.y_mode_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            symbol,
        );
    }

    fn set_smooth_context(
        &mut self,
        x: usize,
        y: usize,
        block_size: BlockSize,
        y_smooth: bool,
        uv_smooth: bool,
    ) {
        fill_mi_grid(
            &mut self.y_smooth_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            y_smooth,
        );
        fill_mi_grid(
            &mut self.uv_smooth_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            uv_smooth,
        );
    }

    fn has_smooth_intra_neighbour(&self, plane: usize, x: usize, y: usize) -> bool {
        let grid = if plane == 0 {
            &self.y_smooth_grid
        } else {
            &self.uv_smooth_grid
        };
        has_smooth_neighbour(grid, self.mi_cols, self.mi_rows, x, y)
    }
}

pub(super) fn cfl_is_allowed(coded_lossless: bool, block_size: BlockSize) -> bool {
    if coded_lossless {
        block_size == BlockSize::Block4x4
    } else {
        block_size.width() <= 32 && block_size.height() <= 32
    }
}

pub(super) fn use_angle_delta(block_size: BlockSize) -> bool {
    !matches!(
        block_size,
        BlockSize::Block4x4 | BlockSize::Block4x8 | BlockSize::Block8x4
    )
}

pub(super) fn cfl_signs(joint_sign: usize) -> Result<(usize, usize), DecoderError> {
    if joint_sign >= 8 {
        return Err(DecoderError::Bitstream(format!(
            "AV1 CFL joint sign {joint_sign} is invalid"
        )));
    }
    let sign_u = ((joint_sign + 1) * 11) >> 5;
    let sign_v = joint_sign + 1 - 3 * sign_u;
    Ok((sign_u, sign_v))
}
