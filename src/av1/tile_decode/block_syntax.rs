use super::{BlockModeProbe, CflParams, CompoundMask, InterIntraMode, MotionMode, TileDecoder};
use crate::DecoderError;
use crate::av1::decode::TileDecodePlan;
use crate::av1::frame::{FrameHeader, FrameType, InterpolationFilter, TxMode};
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
        self.read_intra_frame_block_mode_with_chroma_reference(
            sequence, frame, tile, block_size, x, y, true,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "arguments map directly to the AV1 block-mode syntax context"
    )]
    pub fn read_intra_frame_block_mode_with_chroma_reference(
        &mut self,
        sequence: &SequenceHeader,
        frame: &FrameHeader,
        tile: &TileDecodePlan,
        block_size: BlockSize,
        x: usize,
        y: usize,
        chroma_reference: bool,
    ) -> Result<BlockModeProbe, DecoderError> {
        self.configure_plane_entropy_contexts(sequence);
        self.current_cfl = None;

        let skip_context = self.skip_context(x, y);
        let (skip_symbol, segment_id) = if frame.segmentation.preskip {
            let segment_id = self.read_segmentation_id(frame, block_size, x, y, false)?;
            let skip_symbol = if frame.segmentation.segment_skip[usize::from(segment_id)] {
                1usize
            } else {
                self.reader
                    .read_symbol(self.cdf.skip_cdf_mut(skip_context))?
            };
            (skip_symbol, segment_id)
        } else {
            let skip_symbol = self
                .reader
                .read_symbol(self.cdf.skip_cdf_mut(skip_context))?;
            let skip = skip_symbol != 0;
            let segment_id = self.read_segmentation_id(frame, block_size, x, y, skip)?;
            (skip_symbol, segment_id)
        };
        let skip = skip_symbol != 0;
        let cdef_idx = self.read_cdef_index(sequence, frame, skip, x, y)?;
        let qindex = self.read_delta_qindex(sequence, frame, block_size, skip, x, y)?;
        let delta_lf = self.read_delta_lf(sequence, frame, block_size, skip, x, y)?;
        let is_inter = if matches!(frame.frame_type, FrameType::Inter | FrameType::Switch) {
            let context = self.intra_inter_context(x, y);
            let symbol = self
                .reader
                .read_symbol(self.cdf.intra_inter_cdf_mut(context))?;
            symbol != 0
        } else {
            false
        };
        if is_inter {
            return self.read_inter_frame_block_mode(
                sequence,
                frame,
                tile,
                block_size,
                x,
                y,
                chroma_reference,
                segment_id,
                qindex,
                delta_lf,
                skip_context,
                skip_symbol,
                skip,
                cdef_idx,
            );
        }

        let (use_intrabc, intra_block_copy_mv) = if frame.allow_intrabc {
            let use_intrabc = self.reader.read_symbol(self.cdf.intrabc_cdf_mut())? != 0;
            if use_intrabc {
                let predictor = self
                    .intra_bc_mv_predictor(x, y, block_size)
                    .unwrap_or_else(|| default_intrabc_mv(sequence, frame, tile, y));
                let mv = self.read_intrabc_mv(predictor)?;
                validate_intrabc_mv(sequence, frame, tile, block_size, x, y, mv)?;
                (true, Some(mv))
            } else {
                (false, None)
            }
        } else {
            (false, None)
        };

        let y_above_context = self.above_y_mode_context(x, y);
        let y_left_context = self.left_y_mode_context(x, y);
        let use_angle_delta = use_angle_delta(block_size);
        let (y_mode_symbol, y_mode, angle_delta_y) = if use_intrabc {
            (0, PredictionMode::Dc, None)
        } else {
            let y_mode_symbol = if matches!(frame.frame_type, FrameType::Inter | FrameType::Switch)
            {
                self.reader
                    .read_symbol(self.cdf.y_mode_cdf_mut(block_size.size_group()))?
            } else {
                self.reader.read_symbol(
                    self.cdf
                        .intra_frame_y_mode_cdf_mut(y_above_context, y_left_context),
                )?
            };
            let y_mode = PredictionMode::from_intra_symbol(y_mode_symbol).ok_or_else(|| {
                DecoderError::Bitstream(format!("AV1 y_mode symbol {y_mode_symbol} is invalid"))
            })?;
            let angle_delta_y = if use_angle_delta && y_mode.is_directional() {
                Some(self.read_angle_delta(y_mode.directional_index().unwrap())?)
            } else {
                None
            };
            (y_mode_symbol, y_mode, angle_delta_y)
        };
        let has_chroma = !sequence.color_config.monochrome && chroma_reference;
        let (uv_mode_symbol, uv_mode, angle_delta_uv) = if has_chroma {
            if use_intrabc {
                (
                    Some(0),
                    Some(UvPredictionMode::Intra(PredictionMode::Dc)),
                    None,
                )
            } else {
                let cfl_allowed = cfl_is_allowed(
                    frame.coded_lossless(),
                    block_size,
                    sequence.color_config.subsampling_x,
                    sequence.color_config.subsampling_y,
                );
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
                if uv_mode == UvPredictionMode::Cfl {
                    self.current_cfl = Some(self.read_cfl_params()?);
                }
                let angle_delta = if use_angle_delta && uv_mode.is_directional() {
                    Some(self.read_angle_delta(uv_mode.directional_index().unwrap())?)
                } else {
                    None
                };
                (Some(uv_symbol), Some(uv_mode), angle_delta)
            }
        } else {
            (None, None, None)
        };
        let palette = if use_intrabc {
            super::PaletteBlockInfo { y: None, uv: None }
        } else {
            self.read_palette_mode_info(sequence, frame, block_size, x, y, y_mode, uv_mode)?
        };
        let mut filter_intra_mode = None;
        if !use_intrabc
            && sequence.enable_filter_intra
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
        let (tx_size_context, tx_size_symbol, tx_size) =
            self.read_intra_tx_size(frame, block_size, skip, use_intrabc, x, y)?;
        self.clear_inter_mv(x, y, block_size);

        Ok(BlockModeProbe {
            tile_id: tile.tile_id,
            block_size,
            segment_id,
            qindex,
            delta_lf,
            skip_context,
            skip_symbol,
            skip,
            is_inter: false,
            reference_frame: None,
            reference_frame_secondary: None,
            motion_vector: None,
            motion_vector_secondary: None,
            motion_mode: MotionMode::Simple,
            interintra_mode: None,
            interintra_wedge_index: None,
            local_warp_neighbors: [None; 4],
            interpolation_filter: None,
            compound_weight: None,
            compound_mask: None,
            use_intrabc,
            intra_block_copy_mv,
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

    #[expect(
        clippy::too_many_arguments,
        reason = "AV1 inter syntax state is explicit"
    )]
    fn read_inter_frame_block_mode(
        &mut self,
        sequence: &SequenceHeader,
        frame: &FrameHeader,
        tile: &TileDecodePlan,
        block_size: BlockSize,
        x: usize,
        y: usize,
        chroma_reference: bool,
        segment_id: u8,
        qindex: u8,
        delta_lf: [i8; 4],
        skip_context: usize,
        skip_symbol: usize,
        skip: bool,
        cdef_idx: Option<u32>,
    ) -> Result<BlockModeProbe, DecoderError> {
        let ref_context = self.intra_inter_context(x, y).min(4);
        let is_compound = frame.reference_select
            && self
                .reader
                .read_symbol(self.cdf.comp_inter_cdf_mut(ref_context))?
                != 0;
        let (reference_type, reference_type_secondary) = if is_compound {
            self.read_compound_reference_types(ref_context, frame)?
        } else {
            (self.read_single_reference_type(ref_context)?, None)
        };
        let reference_frame = *frame
            .reference_frame_indices
            .get(reference_type)
            .ok_or_else(|| DecoderError::Bitstream("AV1 reference type is invalid".to_string()))?;
        let reference_frame_secondary = reference_type_secondary
            .map(|reference_type| frame.reference_frame_indices.get(reference_type).copied())
            .flatten();
        if is_compound && reference_frame_secondary.is_none() {
            return Err(DecoderError::Bitstream(
                "AV1 compound reference type is invalid".to_string(),
            ));
        }

        // Inter mode is a small decision tree.  New-MV is decoded against the
        // Decode MVs against the nearest usable same-reference neighbours.
        // Keeping the predictor in the MI grid is required for ordinary
        // NEWMV/NEARMV inter blocks and avoids treating every delta as an
        // absolute vector.
        let mode_context = self.intra_inter_context(x, y).min(5);
        let primary_predictor = self.inter_mv_predictor(x, y, block_size, reference_frame);
        let primary_candidate_count =
            self.inter_mv_candidate_count(x, y, block_size, reference_frame, false);
        let secondary_predictor = reference_frame_secondary
            .map(|reference| self.inter_mv_predictor_secondary(x, y, block_size, reference));
        let (motion_vector, motion_vector_secondary) = if is_compound {
            let mode = if skip {
                1
            } else {
                self.reader
                    .read_symbol(self.cdf.inter_compound_mode_cdf_mut(mode_context))?
            };
            let ref_mv_index = if matches!(mode, 1 | 4 | 5) {
                self.read_drl_index(1, primary_candidate_count)?
            } else if mode == 7 {
                self.read_drl_index(0, primary_candidate_count)?
            } else {
                0
            };
            let first_new = matches!(mode, 3 | 5 | 7);
            let second_new = matches!(mode, 2 | 4 | 7);
            let first = if first_new {
                let predictor = if matches!(mode, 5 | 7) {
                    self.inter_mv_candidate(x, y, block_size, reference_frame, ref_mv_index, false)
                } else {
                    primary_predictor
                };
                self.read_new_mv(predictor, frame)?
            } else {
                self.inter_mv_candidate(x, y, block_size, reference_frame, ref_mv_index, false)
            };
            let second = if second_new {
                let predictor = if matches!(mode, 4 | 7) {
                    self.inter_mv_candidate(
                        x,
                        y,
                        block_size,
                        reference_frame_secondary.unwrap_or(reference_frame),
                        ref_mv_index,
                        true,
                    )
                } else {
                    secondary_predictor.unwrap_or((0, 0))
                };
                self.read_new_mv(predictor, frame)?
            } else {
                if mode == 6 {
                    frame.global_motion.motion_vector(
                        reference_type_secondary.unwrap_or(reference_type),
                        block_size,
                        x,
                        y,
                        frame.allow_high_precision_mv,
                        frame.force_integer_mv == 1,
                    )?
                } else {
                    secondary_predictor
                        .map(|_| {
                            self.inter_mv_candidate(
                                x,
                                y,
                                block_size,
                                reference_frame_secondary.unwrap_or(reference_frame),
                                ref_mv_index,
                                true,
                            )
                        })
                        .unwrap_or((0, 0))
                }
            };
            let first = if mode == 6 {
                frame.global_motion.motion_vector(
                    reference_type,
                    block_size,
                    x,
                    y,
                    frame.allow_high_precision_mv,
                    frame.force_integer_mv == 1,
                )?
            } else {
                first
            };
            (first, Some(second))
        } else {
            let new_mv = self
                .reader
                .read_symbol(self.cdf.newmv_cdf_mut(mode_context))?
                == 0;
            let motion_vector = if new_mv {
                let ref_mv_index = self.read_drl_index(0, primary_candidate_count)?;
                self.read_new_mv(
                    self.inter_mv_candidate(x, y, block_size, reference_frame, ref_mv_index, false),
                    frame,
                )?
            } else {
                let zero_mv = self
                    .reader
                    .read_symbol(self.cdf.zeromv_cdf_mut(mode_context.min(1)))?
                    == 0;
                if !zero_mv {
                    let nearest_or_near = self
                        .reader
                        .read_symbol(self.cdf.refmv_cdf_mut(mode_context))?;
                    let ref_mv_index = if nearest_or_near == 0 {
                        0
                    } else {
                        self.read_drl_index(1, primary_candidate_count)?
                    };
                    self.inter_mv_candidate(x, y, block_size, reference_frame, ref_mv_index, false)
                } else {
                    frame.global_motion.motion_vector(
                        reference_type,
                        block_size,
                        x,
                        y,
                        frame.allow_high_precision_mv,
                        frame.force_integer_mv == 1,
                    )?
                }
            };
            (motion_vector, None)
        };
        let (interintra_mode, interintra_wedge_index) = if sequence.enable_interintra_compound
            && !is_compound
            && !skip
            && block_size.width() >= 8
            && block_size.height() >= 8
        {
            let interintra = self
                .reader
                .read_symbol(self.cdf.interintra_cdf_mut(block_size.size_group()))?;
            if interintra == 0 {
                (None, None)
            } else {
                let mode = match self
                    .reader
                    .read_symbol(self.cdf.interintra_mode_cdf_mut(block_size.size_group()))?
                {
                    0 => InterIntraMode::Dc,
                    1 => InterIntraMode::Vertical,
                    2 => InterIntraMode::Horizontal,
                    3 => InterIntraMode::Smooth,
                    symbol => {
                        return Err(DecoderError::Bitstream(format!(
                            "AV1 inter-intra mode symbol {symbol} is invalid"
                        )));
                    }
                };
                let use_wedge = self
                    .reader
                    .read_symbol(self.cdf.wedge_interintra_cdf_mut(block_size as usize))?
                    != 0;
                let wedge_index = use_wedge.then(|| {
                    self.reader
                        .read_symbol(self.cdf.wedge_idx_cdf_mut(block_size as usize))
                        .map(|symbol| symbol as u8)
                });
                let wedge_index = wedge_index.transpose()?;
                (Some(mode), wedge_index)
            }
        } else {
            (None, None)
        };
        let motion_mode = if interintra_mode.is_some() {
            MotionMode::Simple
        } else if frame.is_motion_mode_switchable
            && block_size.width() >= 8
            && block_size.height() >= 8
        {
            match self.reader.read_symbol(
                self.cdf
                    .motion_mode_cdf_mut(block_size.motion_mode_cdf_index()),
            )? {
                0 => MotionMode::Simple,
                1 => MotionMode::Obmc,
                2 => MotionMode::LocalWarp,
                symbol => {
                    return Err(DecoderError::Bitstream(format!(
                        "AV1 motion mode symbol {symbol} is invalid"
                    )));
                }
            }
        } else {
            MotionMode::Simple
        };
        let local_warp_neighbors = if motion_mode == MotionMode::LocalWarp {
            let neighbors =
                self.inter_mv_neighbor_candidates(x, y, block_size, reference_frame, false);
            [neighbors[0], neighbors[1], neighbors[2], neighbors[3]]
        } else {
            [None; 4]
        };
        let mut compound_weight = None;
        let mut compound_mask = None;
        if is_compound && !skip {
            let masked_compound_used = sequence.enable_masked_compound
                && block_size.width() >= 8
                && block_size.height() >= 8;
            let comp_group_idx = if masked_compound_used {
                self.reader
                    .read_symbol(self.cdf.comp_group_idx_cdf_mut(0))?
            } else {
                0
            };
            if comp_group_idx == 0 && sequence.enable_dist_wtd_comp {
                let compound_idx = self.reader.read_symbol(self.cdf.compound_idx_cdf_mut(0))?;
                if compound_idx == 0 {
                    compound_weight = Some(distance_weighted_compound_weight(
                        sequence,
                        frame,
                        reference_type,
                        reference_type_secondary.unwrap_or(reference_type),
                    ));
                }
            } else if comp_group_idx == 1 {
                let compound_type = self
                    .reader
                    .read_symbol(self.cdf.compound_type_cdf_mut(block_size as usize))?;
                match compound_type {
                    0 => {
                        let wedge_index = self
                            .reader
                            .read_symbol(self.cdf.wedge_idx_cdf_mut(block_size as usize))?
                            as u8;
                        let inverse = self.reader.read_bool()? != 0;
                        compound_mask = Some(CompoundMask::Wedge {
                            index: wedge_index,
                            inverse,
                        });
                    }
                    1 => {
                        let inverse = self.reader.read_bool()? != 0;
                        compound_mask = Some(CompoundMask::DifferenceWeighted { inverse });
                    }
                    _ => unreachable!("compound_type CDF has only two symbols"),
                }
            }
        }
        let interpolation_filter = if frame.is_filter_switchable {
            let filter_context = mode_context.min(15);
            let vertical = InterpolationFilter::from_switchable_symbol(
                self.reader
                    .read_symbol(self.cdf.switchable_interp_cdf_mut(filter_context))?,
            )?;
            let horizontal = if sequence.enable_dual_filter {
                InterpolationFilter::from_switchable_symbol(
                    self.reader
                        .read_symbol(self.cdf.switchable_interp_cdf_mut(filter_context))?,
                )?
            } else {
                vertical
            };
            // Keep the pair in (horizontal, vertical) order for sampling.
            Some((horizontal, vertical))
        } else {
            Some((frame.interpolation_filter, frame.interpolation_filter))
        };

        self.set_inter_context(x, y, block_size, true);
        self.set_inter_mv(
            x,
            y,
            block_size,
            reference_frame,
            motion_vector,
            reference_frame_secondary,
            motion_vector_secondary,
        );
        self.set_smooth_context(x, y, block_size, false, false);
        self.set_skip_context(x, y, block_size, skip);
        let (tx_size_context, tx_size_symbol, tx_size) =
            self.read_intra_tx_size(frame, block_size, skip, false, x, y)?;
        let has_chroma = !sequence.color_config.monochrome && chroma_reference;
        let (uv_mode_symbol, uv_mode) = if has_chroma {
            (Some(0), Some(UvPredictionMode::Intra(PredictionMode::Dc)))
        } else {
            (None, None)
        };
        Ok(BlockModeProbe {
            tile_id: tile.tile_id,
            block_size,
            segment_id,
            qindex,
            delta_lf,
            skip_context,
            skip_symbol,
            skip,
            is_inter: true,
            reference_frame: Some(reference_frame),
            reference_frame_secondary,
            motion_vector: Some(motion_vector),
            motion_vector_secondary,
            motion_mode,
            interintra_mode,
            interintra_wedge_index,
            local_warp_neighbors,
            interpolation_filter,
            compound_weight,
            compound_mask,
            use_intrabc: false,
            intra_block_copy_mv: None,
            cdef_idx,
            y_above_context: 0,
            y_left_context: 0,
            y_mode_symbol: 0,
            y_mode: PredictionMode::Dc,
            angle_delta_y: None,
            y_smooth_neighbour: false,
            filter_intra_mode: None,
            uv_mode_symbol,
            uv_mode,
            angle_delta_uv: None,
            uv_smooth_neighbour: false,
            palette: super::PaletteBlockInfo { y: None, uv: None },
            tx_size_context,
            tx_size_symbol,
            tx_size,
            bit_position_after: self.reader.bit_position(),
        })
    }

    fn read_single_reference_type(&mut self, context: usize) -> Result<usize, DecoderError> {
        Ok(
            if self
                .reader
                .read_symbol(self.cdf.single_ref_cdf_mut(context, 0))?
                != 0
            {
                if self
                    .reader
                    .read_symbol(self.cdf.single_ref_cdf_mut(context, 1))?
                    == 0
                {
                    if self
                        .reader
                        .read_symbol(self.cdf.single_ref_cdf_mut(context, 5))?
                        != 0
                    {
                        5
                    } else {
                        4
                    }
                } else {
                    6
                }
            } else if self
                .reader
                .read_symbol(self.cdf.single_ref_cdf_mut(context, 2))?
                != 0
            {
                if self
                    .reader
                    .read_symbol(self.cdf.single_ref_cdf_mut(context, 4))?
                    != 0
                {
                    3
                } else {
                    2
                }
            } else if self
                .reader
                .read_symbol(self.cdf.single_ref_cdf_mut(context, 3))?
                != 0
            {
                1
            } else {
                0
            },
        )
    }

    fn read_compound_reference_types(
        &mut self,
        context: usize,
        _frame: &FrameHeader,
    ) -> Result<(usize, Option<usize>), DecoderError> {
        let comp_context = context.min(4);
        let reference_type = self
            .reader
            .read_symbol(self.cdf.comp_ref_type_cdf_mut(comp_context))?;
        let (first, second) = if reference_type == 0 {
            let bit = self
                .reader
                .read_symbol(self.cdf.uni_comp_ref_cdf_mut(0, 0))?;
            if bit != 0 {
                (4, 5)
            } else {
                let bit1 = self
                    .reader
                    .read_symbol(self.cdf.uni_comp_ref_cdf_mut(0, 1))?;
                if bit1 != 0 {
                    let bit2 = self
                        .reader
                        .read_symbol(self.cdf.uni_comp_ref_cdf_mut(0, 2))?;
                    if bit2 != 0 { (0, 3) } else { (0, 2) }
                } else {
                    (0, 1)
                }
            }
        } else {
            let bit = self.reader.read_symbol(self.cdf.comp_ref_cdf_mut(0, 0))?;
            let first = if bit == 0 {
                let bit1 = self.reader.read_symbol(self.cdf.comp_ref_cdf_mut(0, 1))?;
                if bit1 != 0 { 1 } else { 0 }
            } else {
                let bit2 = self.reader.read_symbol(self.cdf.comp_ref_cdf_mut(0, 2))?;
                if bit2 != 0 { 3 } else { 2 }
            };
            let bit_bwd = self
                .reader
                .read_symbol(self.cdf.comp_bwdref_cdf_mut(0, 0))?;
            let second = if bit_bwd == 0 {
                let bit1_bwd = self
                    .reader
                    .read_symbol(self.cdf.comp_bwdref_cdf_mut(0, 1))?;
                if bit1_bwd != 0 { 6 } else { 4 }
            } else {
                5
            };
            (first, second)
        };
        Ok((first, Some(second)))
    }

    fn read_new_mv(
        &mut self,
        predictor: (i32, i32),
        frame: &FrameHeader,
    ) -> Result<(i32, i32), DecoderError> {
        let joint = self.reader.read_symbol(self.cdf.motion_joint_cdf_mut())?;
        let mut delta = [0i32; 2];
        if matches!(joint, 2 | 3) {
            delta[0] = self.read_mv_component(0, frame)?;
        }
        if matches!(joint, 1 | 3) {
            delta[1] = self.read_mv_component(1, frame)?;
        }
        Ok((
            predictor
                .0
                .checked_add(delta[0])
                .ok_or_else(|| DecoderError::Bitstream("AV1 inter row MV overflows".to_string()))?,
            predictor.1.checked_add(delta[1]).ok_or_else(|| {
                DecoderError::Bitstream("AV1 inter column MV overflows".to_string())
            })?,
        ))
    }

    fn read_drl_index(
        &mut self,
        start: usize,
        candidate_count: usize,
    ) -> Result<usize, DecoderError> {
        let mut index = start;
        for context in 0..2 {
            if start + context >= candidate_count {
                break;
            }
            let symbol = self.reader.read_symbol(self.cdf.drl_cdf_mut(context))?;
            index += symbol;
            if symbol == 0 {
                break;
            }
        }
        Ok(index)
    }

    fn read_mv_component(
        &mut self,
        component: usize,
        frame: &FrameHeader,
    ) -> Result<i32, DecoderError> {
        let cdf = self.cdf.motion_component_cdf_mut(component);
        let sign = self.reader.read_symbol(&mut cdf.sign)?;
        let mv_class = self.reader.read_symbol(&mut cdf.classes)?;
        let integer_mv = frame.force_integer_mv == 1;
        let magnitude = if mv_class == 0 {
            let class0_bit = self.reader.read_symbol(&mut cdf.class0)?;
            let fractional = if integer_mv {
                3
            } else {
                self.reader.read_symbol(&mut cdf.class0_fp[class0_bit])?
            };
            let high_precision = if integer_mv || !frame.allow_high_precision_mv {
                1
            } else {
                self.reader.read_symbol(&mut cdf.class0_hp)?
            };
            ((class0_bit << 3) | (fractional << 1) | high_precision) + 1
        } else {
            let mut offset = 0usize;
            for bit in 0..mv_class {
                offset |= self.reader.read_symbol(&mut cdf.bits[bit])? << bit;
            }
            let fractional = if integer_mv {
                3
            } else {
                self.reader.read_symbol(&mut cdf.fp)?
            };
            let high_precision = if integer_mv || !frame.allow_high_precision_mv {
                1
            } else {
                self.reader.read_symbol(&mut cdf.hp)?
            };
            (1usize << (mv_class + 2))
                .saturating_add((offset << 3) | (fractional << 1) | high_precision)
                .saturating_add(1)
        };
        let magnitude = i32::try_from(magnitude)
            .map_err(|_| DecoderError::Bitstream("AV1 inter MV magnitude overflows".to_string()))?;
        Ok(if sign == 0 { magnitude } else { -magnitude })
    }

    fn read_intrabc_mv(&mut self, predictor: (i32, i32)) -> Result<(i32, i32), DecoderError> {
        let joint = self
            .reader
            .read_symbol(self.cdf.intrabc_mv_joint_cdf_mut())?;
        let mut mv = predictor;
        if matches!(joint, 2 | 3) {
            mv.0 = mv
                .0
                .checked_add(self.read_intrabc_mv_component(0)?)
                .ok_or_else(|| DecoderError::Bitstream("intrabc row MV overflows".to_string()))?;
        }
        if matches!(joint, 1 | 3) {
            mv.1 =
                mv.1.checked_add(self.read_intrabc_mv_component(1)?)
                    .ok_or_else(|| {
                        DecoderError::Bitstream("intrabc column MV overflows".to_string())
                    })?;
        }
        Ok(mv)
    }

    fn read_intrabc_mv_component(&mut self, component: usize) -> Result<i32, DecoderError> {
        let cdf = self.cdf.intrabc_mv_component_cdf_mut(component);
        let sign = self.reader.read_symbol(&mut cdf.sign)?;
        let class = self.reader.read_symbol(&mut cdf.class)?;
        let magnitude = if class == 0 {
            let class0_bit = self.reader.read_symbol(&mut cdf.class0)?;
            (class0_bit << 3) + 8
        } else {
            let mut d = 0usize;
            for bit in 0..class {
                d |= self.reader.read_symbol(&mut cdf.bits[bit])? << bit;
            }
            (2usize << (class + 2)) + (d << 3) + 8
        };
        let magnitude = i32::try_from(magnitude)
            .map_err(|_| DecoderError::Bitstream("intrabc MV magnitude overflows".to_string()))?;
        Ok(if sign == 0 { magnitude } else { -magnitude })
    }

    fn read_delta_qindex(
        &mut self,
        sequence: &SequenceHeader,
        frame: &FrameHeader,
        block_size: BlockSize,
        skip: bool,
        x: usize,
        y: usize,
    ) -> Result<u8, DecoderError> {
        if !frame.delta_q.present {
            return Ok(self.current_qindex);
        }

        let superblock_size = if sequence.use_128x128_superblock {
            128
        } else {
            64
        };
        let at_superblock_origin =
            x.is_multiple_of(superblock_size) && y.is_multiple_of(superblock_size);
        let is_full_superblock =
            block_size.width() == superblock_size && block_size.height() == superblock_size;
        if !at_superblock_origin || (is_full_superblock && skip) {
            return Ok(self.current_qindex);
        }

        let abs_symbol = self.reader.read_symbol(self.cdf.delta_q_cdf_mut())?;
        let abs = if abs_symbol == 3 {
            let rem_bits = self.reader.read_literal(3)? as usize + 1;
            let abs_bits = self.reader.read_literal(rem_bits)? as i32;
            abs_bits + (1i32 << rem_bits) + 1
        } else {
            abs_symbol as i32
        };
        if abs != 0 {
            let sign = self.reader.read_bool()?;
            let signed = if sign != 0 { -abs } else { abs };
            let scaled = signed << frame.delta_q.res;
            self.current_qindex = (i32::from(self.current_qindex) + scaled).clamp(1, 255) as u8;
        }
        Ok(self.current_qindex)
    }

    fn read_delta_lf(
        &mut self,
        sequence: &SequenceHeader,
        frame: &FrameHeader,
        block_size: BlockSize,
        skip: bool,
        x: usize,
        y: usize,
    ) -> Result<[i8; 4], DecoderError> {
        if !frame.delta_lf.present {
            return Ok(self.current_delta_lf);
        }

        let superblock_size = if sequence.use_128x128_superblock {
            128
        } else {
            64
        };
        let at_superblock_origin =
            x.is_multiple_of(superblock_size) && y.is_multiple_of(superblock_size);
        let is_full_superblock =
            block_size.width() == superblock_size && block_size.height() == superblock_size;
        if !at_superblock_origin || (is_full_superblock && skip) {
            return Ok(self.current_delta_lf);
        }

        let count = if !frame.delta_lf.multi {
            1
        } else if sequence.color_config.monochrome {
            2
        } else {
            4
        };
        for index in 0..count {
            let abs_symbol = if frame.delta_lf.multi {
                self.reader
                    .read_symbol(self.cdf.delta_lf_multi_cdf_mut(index))?
            } else {
                self.reader.read_symbol(self.cdf.delta_lf_cdf_mut())?
            };
            let abs = if abs_symbol == 3 {
                let rem_bits = self.reader.read_literal(3)? as usize + 1;
                let abs_bits = self.reader.read_literal(rem_bits)? as i32;
                abs_bits + (1i32 << rem_bits) + 1
            } else {
                abs_symbol as i32
            };
            if abs != 0 {
                let sign = self.reader.read_bool()?;
                let signed = if sign != 0 { -abs } else { abs };
                let scaled = signed << frame.delta_lf.res;
                self.current_delta_lf[index] =
                    (i32::from(self.current_delta_lf[index]) + scaled).clamp(-63, 63) as i8;
            }
        }
        Ok(self.current_delta_lf)
    }

    fn read_intra_tx_size(
        &mut self,
        frame: &FrameHeader,
        block_size: BlockSize,
        skip: bool,
        use_intrabc: bool,
        x: usize,
        y: usize,
    ) -> Result<(Option<usize>, Option<usize>, TxSize), DecoderError> {
        let tx_size = match frame.tx_mode {
            TxMode::Only4x4 => TxSize::Tx4x4,
            TxMode::Largest => block_size.largest_supported_tx_size(),
            // Intra blocks signal the selected transform size even when
            // skip_txfm suppresses coefficient payloads. IntrABC is parsed as
            // an inter block by the AV1 syntax, so skipped IntrABC blocks do
            // not carry this transform-size symbol.
            TxMode::Select if block_size.signals_tx_size() && !(use_intrabc && skip) => {
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
        if skip {
            let (width, height) = block_size.largest_supported_tx_dimensions();
            self.set_txfm_context_dimensions(x, y, block_size, width, height);
        } else {
            self.set_txfm_context(x, y, block_size, tx_size);
        }
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
        if mi_col.is_multiple_of(superblock_mi) && mi_row.is_multiple_of(superblock_mi) {
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
        if (y >> 2) <= self.tile_mi_row_start {
            return 0;
        }
        self.y_mode_at_mi(x >> 2, (y >> 2).saturating_sub(1))
            .unwrap_or(0)
    }

    fn left_y_mode_context(&self, x: usize, y: usize) -> usize {
        if (x >> 2) <= self.tile_mi_col_start {
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

    fn is_inter_at_mi(&self, mi_col: usize, mi_row: usize) -> Option<bool> {
        if mi_col >= self.mi_cols || mi_row >= self.mi_rows {
            return None;
        }
        self.is_inter_grid[mi_row * self.mi_cols + mi_col]
    }

    fn intra_inter_context(&self, x: usize, y: usize) -> usize {
        let mi_col = x >> 2;
        let mi_row = y >> 2;
        let above = (mi_row > self.tile_mi_row_start)
            .then(|| self.is_inter_at_mi(mi_col, mi_row - 1))
            .flatten();
        let left = (mi_col > self.tile_mi_col_start)
            .then(|| self.is_inter_at_mi(mi_col - 1, mi_row))
            .flatten();
        match (above, left) {
            (Some(above), Some(left)) => match (above, left) {
                (false, false) => 3,
                (false, true) | (true, false) => 1,
                (true, true) => 0,
            },
            (Some(above), None) | (None, Some(above)) => usize::from(!above) * 2,
            (None, None) => 0,
        }
    }

    fn set_inter_context(&mut self, x: usize, y: usize, block_size: BlockSize, is_inter: bool) {
        fill_mi_grid(
            &mut self.is_inter_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            is_inter,
        );
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

pub(super) fn cfl_is_allowed(
    coded_lossless: bool,
    block_size: BlockSize,
    subsampling_x: bool,
    subsampling_y: bool,
) -> bool {
    if coded_lossless {
        let plane_width = ceil_shift(block_size.width(), usize::from(subsampling_x)).max(4);
        let plane_height = ceil_shift(block_size.height(), usize::from(subsampling_y)).max(4);
        plane_width == 4 && plane_height == 4
    } else {
        block_size.width() <= 32 && block_size.height() <= 32
    }
}

fn ceil_shift(value: usize, shift: usize) -> usize {
    (value + ((1usize << shift) - 1)) >> shift
}

pub(super) fn use_angle_delta(block_size: BlockSize) -> bool {
    !matches!(
        block_size,
        BlockSize::Block4x4 | BlockSize::Block4x8 | BlockSize::Block8x4
    )
}

fn default_intrabc_mv(
    sequence: &SequenceHeader,
    _frame: &FrameHeader,
    tile: &TileDecodePlan,
    y: usize,
) -> (i32, i32) {
    let mib_size = if sequence.use_128x128_superblock {
        32
    } else {
        16
    };
    let mi_row = y / 4;
    if mi_row < mib_size + tile.mi_row_start as usize {
        (0, -((mib_size * 4 + 256) as i32 * 8))
    } else {
        (-((mib_size * 4) as i32 * 8), 0)
    }
}

fn validate_intrabc_mv(
    _sequence: &SequenceHeader,
    _frame: &FrameHeader,
    tile: &TileDecodePlan,
    block_size: BlockSize,
    x: usize,
    y: usize,
    mv: (i32, i32),
) -> Result<(), DecoderError> {
    if mv.0 % 8 != 0 || mv.1 % 8 != 0 {
        return Err(DecoderError::Bitstream(
            "AV1 intrabc DV must use integer-pixel offsets".to_string(),
        ));
    }
    let source_x = i64::try_from(x)
        .ok()
        .and_then(|value| value.checked_add(i64::from(mv.1 / 8)))
        .ok_or_else(|| DecoderError::Bitstream("AV1 intrabc source x overflows".to_string()))?;
    let source_y = i64::try_from(y)
        .ok()
        .and_then(|value| value.checked_add(i64::from(mv.0 / 8)))
        .ok_or_else(|| DecoderError::Bitstream("AV1 intrabc source y overflows".to_string()))?;
    let tile_left = i64::from(tile.mi_col_start) * 4;
    let tile_top = i64::from(tile.mi_row_start) * 4;
    let tile_right = i64::from(tile.mi_col_end) * 4;
    let tile_bottom = i64::from(tile.mi_row_end) * 4;
    let width = i64::try_from(block_size.width()).unwrap_or(i64::MAX);
    let height = i64::try_from(block_size.height()).unwrap_or(i64::MAX);
    let source_right = source_x
        .checked_add(width)
        .ok_or_else(|| DecoderError::Bitstream("AV1 intrabc source width overflows".to_string()))?;
    let source_bottom = source_y.checked_add(height).ok_or_else(|| {
        DecoderError::Bitstream("AV1 intrabc source height overflows".to_string())
    })?;
    if source_x < tile_left
        || source_y < tile_top
        || source_right > tile_right
        || source_bottom > tile_bottom
    {
        return Err(DecoderError::Bitstream(format!(
            "AV1 intrabc DV points outside the tile: dst=({x},{y}) mv=({},{}) src=({source_x},{source_y}) block={}x{} tile=({tile_left},{tile_top})..({tile_right},{tile_bottom})",
            mv.0,
            mv.1,
            block_size.width(),
            block_size.height(),
        )));
    }
    Ok(())
}

fn distance_weighted_compound_weight(
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    primary_type: usize,
    secondary_type: usize,
) -> u8 {
    let current = frame.order_hint;
    let primary = frame.reference_order_hints[primary_type].unwrap_or(current);
    let secondary = frame.reference_order_hints[secondary_type].unwrap_or(current);
    let primary_distance =
        relative_order_distance(sequence.order_hint_bits, primary, current).abs();
    let secondary_distance =
        relative_order_distance(sequence.order_hint_bits, secondary, current).abs();
    let total = primary_distance.saturating_add(secondary_distance);
    if total == 0 {
        return 32;
    }
    u8::try_from((i64::from(secondary_distance) * 64 + i64::from(total / 2)) / i64::from(total))
        .unwrap_or(32)
        .min(64)
}

fn relative_order_distance(bits: u8, reference: u32, current: u32) -> i32 {
    if bits == 0 {
        return 0;
    }
    let modulo = 1i32 << bits;
    let mask = modulo - 1;
    let mut distance = ((reference as i32 - current as i32) & mask) as i32;
    if distance & (modulo >> 1) != 0 {
        distance -= modulo;
    }
    distance
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
