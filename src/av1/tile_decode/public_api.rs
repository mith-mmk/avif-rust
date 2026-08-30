use super::decode_flow::{decode_luma_block_tree, decode_luma_root_block};
use super::post_filter_state::PostFilterState;
use super::{
    BlockModeProbe, DecodedBlockPrefix, DecodedLumaBlock, DecodedTransform, MotionField,
    PartitionProbe, ResidualProbe, TileDecoder, TileEntropyState,
};
use crate::DecoderError;
use crate::allocation::AllocationClass;
use crate::allocation::AllocationTicket;
use crate::av1::cdf::CdfContext;
use crate::av1::decode::{FrameBuffers, FrameDecodePlan};
use crate::av1::entropy::EntropyDecoder;
use crate::av1::frame::FrameHeader;
use crate::av1::predict::{IntraEdges, predict_intra};
use crate::av1::quant::QuantState;
use crate::av1::sequence::SequenceHeader;
use crate::av1::syntax::Partition;
use crate::av1::tile_decode::partition_syntax::root_block_size;
use crate::av1::tile_group::{TileGroup, TilePayload};
use crate::av1::transform::{
    QuantizedTransform, plan_transform_blocks_with_tx_size, reconstruct_lossless_transform_block,
    reconstruct_transform_block,
};
use crate::container::DecodeBudget;
use std::sync::Arc;

/// Owns one tile decoder and, on the strict-native path, its bounded
/// coefficient-scratch ticket.  The explicit owner makes the drop order
/// observable: all decoder-local vectors are destroyed before the budget
/// ticket is released, including early-return and `?` error paths.
struct TileDecoderOwner<'a, 'b> {
    decoder: Option<TileDecoder<'a>>,
    budget: Option<&'b mut DecodeBudget>,
    tickets: Option<[AllocationTicket; 6]>,
    memory_ticket: Option<AllocationTicket>,
    #[cfg(test)]
    synthetic_scratch: Option<Vec<i32>>,
    #[cfg(test)]
    synthetic_ticket: Option<AllocationTicket>,
}

/// Post-filter vectors detached from a strict tile decoder.  Their tickets
/// follow the vectors until the caller has completed all filters; this keeps
/// accounting valid after the decoder owner itself is dropped.
#[derive(Debug)]
pub(crate) struct BudgetedPostFilterState {
    state: PostFilterState,
    tickets: [AllocationTicket; 5],
}

impl std::ops::Deref for BudgetedPostFilterState {
    type Target = PostFilterState;

    fn deref(&self) -> &Self::Target {
        &self.state
    }
}

impl std::ops::DerefMut for BudgetedPostFilterState {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.state
    }
}

impl BudgetedPostFilterState {
    fn empty() -> Self {
        Self {
            state: PostFilterState::default(),
            tickets: std::array::from_fn(|_| AllocationTicket::new(AllocationClass::Frame)),
        }
    }

    fn merge_into(
        self,
        destination: &mut Self,
        budget: &mut DecodeBudget,
    ) -> Result<(), DecoderError> {
        let destination_empty = destination.state.is_empty()
            && destination
                .state
                .cdef_units
                .capacity()
                .checked_mul(std::mem::size_of::<super::post_filter_state::CdefUnit>())
                .is_some_and(|bytes| bytes == destination.tickets[0].charged_capacity_bytes)
            && destination
                .state
                .cdef_blocks
                .capacity()
                .checked_mul(std::mem::size_of::<super::post_filter_state::CdefBlockIndex>())
                .is_some_and(|bytes| bytes == destination.tickets[1].charged_capacity_bytes)
            && destination
                .state
                .transform_boundaries
                .capacity()
                .checked_mul(std::mem::size_of::<super::post_filter_state::TransformBoundary>())
                .is_some_and(|bytes| bytes == destination.tickets[2].charged_capacity_bytes)
            && destination
                .state
                .restoration_units
                .capacity()
                .checked_mul(std::mem::size_of::<super::post_filter_state::RestorationUnit>())
                .is_some_and(|bytes| bytes == destination.tickets[3].charged_capacity_bytes)
            && destination
                .state
                .block_filter_states
                .capacity()
                .checked_mul(std::mem::size_of::<super::post_filter_state::BlockFilterState>())
                .is_some_and(|bytes| bytes == destination.tickets[4].charged_capacity_bytes)
            && destination.tickets.iter().all(|ticket| ticket.charged_capacity_bytes == 0);
        if destination_empty {
            destination.state = self.state;
            destination.tickets = self.tickets;
            return Ok(());
        }
        let BudgetedPostFilterState {
            state,
            mut tickets,
        } = self;
        let lengths = [
            state.cdef_units.len(),
            state.cdef_blocks.len(),
            state.transform_boundaries.len(),
            state.restoration_units.len(),
            state.block_filter_states.len(),
        ];
        let mut source_state = Some(state);
        let labels = [
            "native AV1 merged CDEF units",
            "native AV1 merged CDEF blocks",
            "native AV1 merged transform boundaries",
            "native AV1 merged restoration units",
            "native AV1 merged block filter states",
        ];
        macro_rules! reserve_destination {
            ($field:ident, $index:expr) => {
                crate::allocation::replace_vec(
                    budget,
                    &mut destination.state.$field,
                    &mut destination.tickets[$index],
                    lengths[$index],
                    AllocationClass::Frame,
                    labels[$index],
                )?;
            };
        }
        let result = (|| {
            reserve_destination!(cdef_units, 0);
            reserve_destination!(cdef_blocks, 1);
            reserve_destination!(transform_boundaries, 2);
            reserve_destination!(restoration_units, 3);
            reserve_destination!(block_filter_states, 4);
            destination
                .state
                .merge(source_state.take().expect("source state is live"));
            Ok(())
        })();
        if let Err(error) = result {
            // The source vectors must die before their tickets are released;
            // successful destination replacements remain valid and charged.
            drop(source_state.take());
            for ticket in &mut tickets {
                budget.release_token(ticket)?;
            }
            return Err(error);
        }
        for ticket in &mut tickets {
            budget.release_token(ticket)?;
        }
        Ok(())
    }

    pub(crate) fn release(mut self, budget: &mut DecodeBudget) -> Result<(), DecoderError> {
        drop(self.state);
        for ticket in &mut self.tickets {
            budget.release_token(ticket)?;
        }
        Ok(())
    }

    fn into_state(self) -> PostFilterState {
        self.state
    }
}

fn merge_post_filter_state(
    destination: &mut BudgetedPostFilterState,
    source: BudgetedPostFilterState,
    budget: Option<&mut DecodeBudget>,
) -> Result<(), DecoderError> {
    if let Some(budget) = budget {
        match source.merge_into(destination, budget) {
            Ok(()) => Ok(()),
            Err(error) => {
                // The accumulator may already own one or more successful
                // replacements.  It is not returned on this error path, so
                // consume it and release every destination ticket after its
                // vectors have become unreachable.
                let retained = std::mem::replace(destination, BudgetedPostFilterState::empty());
                retained.release(budget)?;
                Err(error)
            }
        }
    } else {
        destination.state.merge(source.state);
        Ok(())
    }
}

impl<'a, 'b> TileDecoderOwner<'a, 'b> {
    fn legacy(decoder: TileDecoder<'a>) -> Self {
        Self {
            decoder: Some(decoder),
            budget: None,
            tickets: None,
            memory_ticket: None,
            #[cfg(test)]
            synthetic_scratch: None,
            #[cfg(test)]
            synthetic_ticket: None,
        }
    }

    fn strict(
        decoder: TileDecoder<'a>,
        tickets: [AllocationTicket; 6],
        budget: &'b mut DecodeBudget,
    ) -> Self {
        Self {
            decoder: Some(decoder),
            budget: Some(budget),
            tickets: Some(tickets),
            memory_ticket: None,
            #[cfg(test)]
            synthetic_scratch: None,
            #[cfg(test)]
            synthetic_ticket: None,
        }
    }

    fn strict_with_memory(
        decoder: TileDecoder<'a>,
        tickets: [AllocationTicket; 6],
        memory_ticket: AllocationTicket,
        budget: &'b mut DecodeBudget,
    ) -> Self {
        Self {
            decoder: Some(decoder),
            budget: Some(budget),
            tickets: Some(tickets),
            memory_ticket: Some(memory_ticket),
            #[cfg(test)]
            synthetic_scratch: None,
            #[cfg(test)]
            synthetic_ticket: None,
        }
    }

    #[cfg(test)]
    fn synthetic_scratch(
        scratch: Vec<i32>,
        ticket: AllocationTicket,
        budget: &'b mut DecodeBudget,
    ) -> TileDecoderOwner<'static, 'b> {
        TileDecoderOwner {
            decoder: None,
            budget: Some(budget),
            tickets: None,
            memory_ticket: None,
            synthetic_scratch: Some(scratch),
            synthetic_ticket: Some(ticket),
        }
    }
}

impl<'a, 'b> std::ops::Deref for TileDecoderOwner<'a, 'b> {
    type Target = TileDecoder<'a>;

    fn deref(&self) -> &Self::Target {
        self.decoder.as_ref().expect("tile decoder owner is live")
    }
}

impl<'a, 'b> std::ops::DerefMut for TileDecoderOwner<'a, 'b> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.decoder.as_mut().expect("tile decoder owner is live")
    }
}

impl<'a, 'b> TileDecoderOwner<'a, 'b> {
    fn reconcile_dynamic_ticket(&mut self) -> Result<(), DecoderError> {
        let Some(decoder) = self.decoder.as_ref() else {
            return Ok(());
        };
        if !decoder.strict_dynamic_enabled() {
            return Ok(());
        }
        let reserved = decoder.strict_dynamic_reserved_bytes();
        let actual = decoder.strict_dynamic_bytes()?;
        if actual > reserved {
            return Err(DecoderError::InvalidParam(
                "native AV1 dynamic tile capacity exceeds admission".to_string(),
            ));
        }
        let unused = reserved - actual;
        if let Some(memory_ticket) = self.memory_ticket.as_ref() {
            if memory_ticket.charged_capacity_bytes < unused {
                return Err(DecoderError::InvalidParam(
                    "native AV1 dynamic tile ticket is inconsistent".to_string(),
                ));
            }
        } else if unused != 0 {
            return Err(DecoderError::InvalidParam(
                "native AV1 dynamic tile ticket is missing".to_string(),
            ));
        }
        if let Some(budget) = self.budget.as_deref_mut() {
            budget.release_class_bytes(AllocationClass::Frame, unused)?;
        }
        if let Some(memory_ticket) = self.memory_ticket.as_mut() {
            memory_ticket.charged_capacity_bytes -= unused;
        }
        if let Some(decoder) = self.decoder.as_mut() {
            decoder.disable_strict_dynamic();
        }
        Ok(())
    }

    fn take_post_filter_state(mut self) -> Result<PostFilterState, DecoderError> {
        self.reconcile_dynamic_ticket()?;
        Ok(self
            .decoder
            .take()
            .expect("tile decoder owner is live")
            .take_post_filter_state())
    }

    fn take_budgeted_post_filter_state(mut self) -> Result<BudgetedPostFilterState, DecoderError> {
        self.reconcile_dynamic_ticket()?;
        let decoder = self
            .decoder
            .take()
            .expect("tile decoder owner is live");
        let state = decoder.take_post_filter_state();
        let mut result = BudgetedPostFilterState::empty();
        result.state = state;
        let capacities = [
            result.state.cdef_units.capacity(),
            result.state.cdef_blocks.capacity(),
            result.state.transform_boundaries.capacity(),
            result.state.restoration_units.capacity(),
            result.state.block_filter_states.capacity(),
        ];
        let sizes = [
            std::mem::size_of::<super::post_filter_state::CdefUnit>(),
            std::mem::size_of::<super::post_filter_state::CdefBlockIndex>(),
            std::mem::size_of::<super::post_filter_state::TransformBoundary>(),
            std::mem::size_of::<super::post_filter_state::RestorationUnit>(),
            std::mem::size_of::<super::post_filter_state::BlockFilterState>(),
        ];
        let mut memory_ticket = self.memory_ticket.take();
        // The compatibility strict constructor has no aggregate storage
        // ticket.  Admit its retained post-filter vectors at transfer time,
        // but keep the historical zero-block path allocation-free in the
        // accounting ledger.
        if memory_ticket.is_none() && !result.state.is_empty() {
            let mut total = 0usize;
            for index in 0..5 {
                let bytes = capacities[index].checked_mul(sizes[index]).ok_or_else(|| {
                    DecoderError::InvalidParam("AV1 post-filter ticket overflows".to_string())
                })?;
                total = total.checked_add(bytes).ok_or_else(|| {
                    DecoderError::InvalidParam("AV1 post-filter ticket overflows".to_string())
                })?;
            }
            let budget = self.budget.as_deref_mut().ok_or_else(|| {
                DecoderError::InvalidParam(
                    "strict post-filter state has no budget owner".to_string(),
                )
            })?;
            memory_ticket = Some(budget.reserve_existing_bytes(
                AllocationClass::Frame,
                total,
                "native AV1 strict post-filter state",
            )?);
        }
        if let Some(mut memory_ticket) = memory_ticket {
            let mut total = 0usize;
            for index in 0..5 {
                let bytes = match capacities[index].checked_mul(sizes[index]) {
                    Some(bytes) => bytes,
                    None => {
                        self.memory_ticket = Some(memory_ticket);
                        return Err(DecoderError::InvalidParam(
                            "AV1 post-filter ticket overflows".to_string(),
                        ));
                    }
                };
                result.tickets[index].charged_capacity_bytes = bytes;
                total = match total.checked_add(bytes) {
                    Some(total) => total,
                    None => {
                        self.memory_ticket = Some(memory_ticket);
                        return Err(DecoderError::InvalidParam(
                            "AV1 post-filter ticket overflows".to_string(),
                        ));
                    }
                };
            }
            if total > memory_ticket.charged_capacity_bytes {
                self.memory_ticket = Some(memory_ticket);
                return Err(DecoderError::InvalidParam(
                    "AV1 post-filter ticket exceeds tile ticket".to_string(),
                ));
            }
            memory_ticket.charged_capacity_bytes -= total;
            self.memory_ticket = Some(memory_ticket);
        }
        Ok(result)
    }
}

impl Drop for TileDecoderOwner<'_, '_> {
    fn drop(&mut self) {
        // Drop cannot report a reconciliation error.  On an error path the
        // full admission remains attached to `memory_ticket`, so releasing it
        // after dropping the decoder is lossless and keeps the destructor
        // infallible.  Successful handoff paths reconcile explicitly before
        // taking ownership of post-filter state.
        drop(self.decoder.take());
        #[cfg(test)]
        drop(self.synthetic_scratch.take());
        let budget = self.budget.take();
        let tickets = self.tickets.take();
        let memory_ticket = self.memory_ticket.take();
        #[cfg(test)]
        let synthetic_ticket = self.synthetic_ticket.take();
        if let Some(budget) = budget {
            if let Some(tickets) = tickets {
                for mut ticket in tickets {
                    let _ = budget.release_token(&mut ticket);
                }
            }
            if let Some(mut ticket) = memory_ticket {
                let _ = budget.release_token(&mut ticket);
            }
            #[cfg(test)]
            if let Some(mut ticket) = synthetic_ticket {
                let _ = budget.release_token(&mut ticket);
            }
        }
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
        let payload = tile_payload_bytes(data, tile)?;
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
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    plan: &FrameDecodePlan,
) -> Result<Vec<PartitionProbe>, DecoderError> {
    let mut probes = Vec::with_capacity(tile_group.tiles.len());
    for (index, tile_payload) in tile_group.tiles.iter().enumerate() {
        let payload = tile_payload_bytes(data, tile_payload)?;
        let tile_plan = plan.tiles.get(index).ok_or_else(|| {
            DecoderError::Bitstream("AV1 tile decode plan is missing a tile".to_string())
        })?;
        let mut decoder = TileDecoder::new(payload, frame)?;
        decoder.set_tile_bounds(tile_plan);
        probes.push(decoder.read_root_partition(tile_plan, sequence)?);
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
        let payload = tile_payload_bytes(data, tile_payload)?;
        let tile_plan = plan.tiles.get(index).ok_or_else(|| {
            DecoderError::Bitstream("AV1 tile decode plan is missing a tile".to_string())
        })?;
        let mut decoder = TileDecoder::new(payload, frame)?;
        decoder.set_tile_bounds(tile_plan);
        let partition = decoder.read_first_leaf_partition(tile_plan, sequence)?;
        if partition.partition == Partition::None {
            probes.push(decoder.read_intra_frame_block_mode(
                sequence,
                frame,
                tile_plan,
                partition.block_size,
                tile_plan.pixel_x,
                tile_plan.pixel_y,
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
        let payload = tile_payload_bytes(data, tile_payload)?;
        let tile_plan = plan.tiles.get(index).ok_or_else(|| {
            DecoderError::Bitstream("AV1 tile decode plan is missing a tile".to_string())
        })?;
        let mut decoder = TileDecoder::new(payload, frame)?;
        decoder.set_tile_bounds(tile_plan);
        let partition = decoder.read_first_leaf_partition(tile_plan, sequence)?;
        if partition.partition == Partition::None {
            let block_mode = decoder.read_intra_frame_block_mode(
                sequence,
                frame,
                tile_plan,
                partition.block_size,
                tile_plan.pixel_x,
                tile_plan.pixel_y,
            )?;
            let quant_state = QuantState::from_qindex(
                &frame.quantization,
                block_mode.qindex,
                sequence.color_config.bit_depth,
            )?;
            let transforms = plan_transform_blocks_with_tx_size(
                0,
                0,
                0,
                block_mode.block_size,
                block_mode.tx_size,
                plan.width,
                plan.height,
            );
            probes.push(decoder.read_first_transform_residual(
                tile_plan.tile_id,
                frame,
                &block_mode,
                &transforms,
                quant_state,
                sequence.color_config.bit_depth,
            )?);
        }
    }
    Ok(probes)
}

pub fn decode_first_luma_transform(
    data: &[u8],
    tile_group: &TileGroup,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    plan: &FrameDecodePlan,
    buffers: &mut FrameBuffers,
) -> Result<ResidualProbe, DecoderError> {
    let tile_payload = tile_group.tiles.first().ok_or_else(|| {
        DecoderError::Bitstream("AV1 tile group has no tile payloads".to_string())
    })?;
    let payload = tile_payload_bytes(data, tile_payload)?;
    let tile_plan = plan
        .tiles
        .first()
        .ok_or_else(|| DecoderError::Bitstream("AV1 tile decode plan is missing".to_string()))?;
    let mut decoder = TileDecoder::new(payload, frame)?;
    decoder.set_tile_bounds(tile_plan);
    let partition = decoder.read_first_leaf_partition(tile_plan, sequence)?;

    let block_mode = decoder.read_intra_frame_block_mode(
        sequence,
        frame,
        tile_plan,
        partition.block_size,
        tile_plan.pixel_x,
        tile_plan.pixel_y,
    )?;
    let quant_state = QuantState::from_qindex(
        &frame.quantization,
        block_mode.qindex,
        sequence.color_config.bit_depth,
    )?;
    let transforms = plan_transform_blocks_with_tx_size(
        0,
        tile_plan.pixel_x,
        tile_plan.pixel_y,
        block_mode.block_size,
        block_mode.tx_size,
        plan.width,
        plan.height,
    );
    let residual = decoder.read_first_transform_residual(
        tile_plan.tile_id,
        frame,
        &block_mode,
        &transforms,
        quant_state,
        sequence.color_config.bit_depth,
    )?;

    if residual.skipped || residual.first_non_zero_transform.is_none() {
        return Ok(residual);
    }
    let transform = residual
        .first_non_zero_transform
        .expect("checked first_non_zero_transform");
    let tx_type = residual
        .tx_type
        .ok_or_else(|| DecoderError::Bitstream("AV1 residual tx_type is missing".to_string()))?;
    let coefficients = residual
        .first_quantized_coefficients
        .as_ref()
        .ok_or_else(|| {
            DecoderError::Bitstream("AV1 residual quantized coefficients are missing".to_string())
        })?;
    let mid = 1u16 << (sequence.color_config.bit_depth - 1);
    let above = vec![mid; transform.tx_size.width()];
    let left = vec![mid; transform.tx_size.height()];
    let prediction = predict_intra(
        block_mode.y_mode,
        block_mode.angle_delta_y,
        transform.tx_size.width(),
        transform.tx_size.height(),
        IntraEdges {
            above: Some(&above),
            left: Some(&left),
            above_left: Some(mid),
            bit_depth: sequence.color_config.bit_depth,
        },
    )?;
    let quantized = QuantizedTransform {
        block: transform,
        tx_type,
        coefficients: coefficients.clone(),
    };
    let luma = buffers
        .planes
        .get_mut(0)
        .ok_or_else(|| DecoderError::Bitstream("AV1 luma plane is missing".to_string()))?;
    if frame.coded_lossless() {
        reconstruct_lossless_transform_block(
            luma,
            &quantized,
            quant_state.plane(transform.plane),
            &prediction,
            sequence.color_config.bit_depth,
        )?;
    } else {
        reconstruct_transform_block(
            luma,
            &quantized,
            quant_state.plane(transform.plane),
            &prediction,
            sequence.color_config.bit_depth,
        )?;
    }

    Ok(residual)
}

pub fn decode_first_luma_block(
    data: &[u8],
    tile_group: &TileGroup,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    plan: &FrameDecodePlan,
    buffers: &mut FrameBuffers,
) -> Result<Vec<DecodedTransform>, DecoderError> {
    let tile_payload = tile_group.tiles.first().ok_or_else(|| {
        DecoderError::Bitstream("AV1 tile group has no tile payloads".to_string())
    })?;
    let payload = tile_payload_bytes(data, tile_payload)?;
    let tile_plan = plan
        .tiles
        .first()
        .ok_or_else(|| DecoderError::Bitstream("AV1 tile decode plan is missing".to_string()))?;
    let mut decoder = TileDecoder::new(payload, frame)?;
    decoder.set_tile_bounds(tile_plan);
    let block = decode_luma_root_block(
        &mut decoder,
        sequence,
        frame,
        tile_plan,
        plan,
        buffers,
        tile_plan.pixel_x,
        tile_plan.pixel_y,
        true,
    )?;
    Ok(block.transforms)
}

pub fn decode_luma_root_blocks(
    data: &[u8],
    tile_group: &TileGroup,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    plan: &FrameDecodePlan,
    buffers: &mut FrameBuffers,
    max_blocks: usize,
) -> Result<Vec<DecodedLumaBlock>, DecoderError> {
    Ok(
        decode_luma_root_block_prefix(
            data, tile_group, sequence, frame, plan, buffers, max_blocks,
        )?
        .blocks,
    )
}

pub fn decode_luma_root_block_prefix(
    data: &[u8],
    tile_group: &TileGroup,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    plan: &FrameDecodePlan,
    buffers: &mut FrameBuffers,
    max_blocks: usize,
) -> Result<DecodedBlockPrefix, DecoderError> {
    // Keep the diagnostic prefix path's entropy traversal independent from
    // post-filter state aggregation. The public/test prefix oracle relies on
    // returning at the same block boundary as the historical decoder.
    if tile_group.tiles.is_empty() {
        return Err(DecoderError::Bitstream(
            "AV1 tile group has no tile payloads".to_string(),
        ));
    }
    let mut blocks = Vec::new();
    let mut block_budget = max_blocks;
    let mut decoded_block_count = 0;
    for (tile_index, tile_payload) in tile_group.tiles.iter().enumerate() {
        let payload = tile_payload_bytes(data, tile_payload)?;
        let tile_plan = plan.tiles.get(tile_index).ok_or_else(|| {
            DecoderError::Bitstream("AV1 tile decode plan is missing a tile".to_string())
        })?;
        let mut decoder = TileDecoder::new(payload, frame)?;
        decoder.set_tile_bounds(tile_plan);
        for sb_row in tile_plan.sb_row_start..tile_plan.sb_row_end {
            decoder.reset_left_superblock_contexts();
            for sb_col in tile_plan.sb_col_start..tile_plan.sb_col_end {
                if block_budget == 0 {
                    return Ok(DecodedBlockPrefix {
                        blocks,
                        next_unsupported: None,
                    });
                }
                let x = (sb_col as usize * plan.superblock_size).min(plan.width);
                let y = (sb_row as usize * plan.superblock_size).min(plan.height);
                decoder.read_restoration_units(sequence, x, y)?;
                let block_start = decoded_block_count;
                let result = decode_luma_block_tree(
                    &mut decoder,
                    sequence,
                    frame,
                    tile_plan,
                    plan,
                    buffers,
                    root_block_size(sequence),
                    x,
                    y,
                    &mut block_budget,
                    &mut blocks,
                    &mut decoded_block_count,
                    true,
                );
                match result {
                    Ok(()) => {}
                    Err(err @ DecoderError::Unsupported(_))
                        if decoded_block_count > block_start =>
                    {
                        return Ok(DecodedBlockPrefix {
                            blocks,
                            next_unsupported: Some(err),
                        });
                    }
                    Err(err) => return Err(err),
                }
            }
        }
    }
    Ok(DecodedBlockPrefix {
        blocks,
        next_unsupported: None,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "internal prefix decode exposes each independently testable pipeline input"
)]
#[cfg(test)]
pub(crate) fn decode_luma_root_block_prefix_with_post_filter_state_and_entropy(
    data: &[u8],
    tile_group: &TileGroup,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    plan: &FrameDecodePlan,
    buffers: &mut FrameBuffers,
    max_blocks: usize,
    validate_entropy: bool,
) -> Result<(DecodedBlockPrefix, PostFilterState), DecoderError> {
    decode_luma_root_block_prefix_with_post_filter_state_and_entropy_options(
        data,
        tile_group,
        sequence,
        frame,
        plan,
        buffers,
        max_blocks,
        validate_entropy,
        true,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "internal prefix decode exposes each independently testable pipeline input"
)]
#[cfg(test)]
pub(crate) fn decode_luma_root_block_prefix_with_post_filter_state_and_entropy_options(
    data: &[u8],
    tile_group: &TileGroup,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    plan: &FrameDecodePlan,
    buffers: &mut FrameBuffers,
    max_blocks: usize,
    validate_entropy: bool,
    collect_diagnostics: bool,
) -> Result<(DecodedBlockPrefix, PostFilterState), DecoderError> {
    let (prefix, state, _) = decode_luma_root_block_prefix_with_post_filter_state_and_entropy_options_with_references_and_cdf(
        data,
        tile_group,
        sequence,
        frame,
        plan,
        buffers,
        max_blocks,
        validate_entropy,
        collect_diagnostics,
        std::array::from_fn(|_| None),
        None,
        true,
    )?;
    Ok((prefix, state))
}

#[expect(
    clippy::too_many_arguments,
    reason = "internal prefix decode exposes each independently testable pipeline input"
)]
#[cfg(test)]
fn decode_luma_root_block_prefix_with_post_filter_state_and_entropy_options_with_references_and_cdf(
    data: &[u8],
    tile_group: &TileGroup,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    plan: &FrameDecodePlan,
    buffers: &mut FrameBuffers,
    max_blocks: usize,
    validate_entropy: bool,
    collect_diagnostics: bool,
    reference_buffers: [Option<Arc<FrameBuffers>>; 8],
    initial_cdfs: Option<&[CdfContext]>,
    collect_cdf: bool,
) -> Result<(DecodedBlockPrefix, PostFilterState, Vec<CdfContext>), DecoderError> {
    let (prefix, state, cdfs, _) =
        decode_luma_root_block_prefix_with_post_filter_state_and_entropy_options_with_references_and_cdf_and_motion(
            data,
            tile_group,
            sequence,
            frame,
            plan,
            buffers,
            max_blocks,
            validate_entropy,
            collect_diagnostics,
            reference_buffers,
            initial_cdfs,
            collect_cdf,
            true,
            None,
        )?;
    Ok((prefix, state, cdfs))
}

#[expect(
    clippy::too_many_arguments,
    reason = "internal prefix decode exposes each independently testable pipeline input"
)]
pub(crate) fn decode_luma_root_block_prefix_with_post_filter_state_and_entropy_options_with_references_and_cdf_and_motion(
    data: &[u8],
    tile_group: &TileGroup,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    plan: &FrameDecodePlan,
    buffers: &mut FrameBuffers,
    max_blocks: usize,
    validate_entropy: bool,
    collect_diagnostics: bool,
    reference_buffers: [Option<Arc<FrameBuffers>>; 8],
    initial_cdfs: Option<&[CdfContext]>,
    collect_cdf: bool,
    collect_motion: bool,
    temporal_motion_field: Option<Arc<MotionField>>,
) -> Result<
    (
        DecodedBlockPrefix,
        PostFilterState,
        Vec<CdfContext>,
        Option<MotionField>,
    ),
    DecoderError,
> {
    let (prefix, state, cdfs, motion) = decode_luma_root_block_prefix_with_budget(
        data,
        tile_group,
        sequence,
        frame,
        plan,
        buffers,
        max_blocks,
        validate_entropy,
        collect_diagnostics,
        reference_buffers,
        initial_cdfs,
        collect_cdf,
        collect_motion,
        temporal_motion_field,
        None,
    )?;
    Ok((prefix, state.into_state(), cdfs, motion))
}

#[expect(
    clippy::too_many_arguments,
    reason = "strict prefix decode additionally carries its bounded allocation ledger"
)]
pub(crate) fn decode_luma_root_block_prefix_with_post_filter_state_and_entropy_options_with_references_and_cdf_and_motion_and_budget(
    data: &[u8],
    tile_group: &TileGroup,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    plan: &FrameDecodePlan,
    buffers: &mut FrameBuffers,
    max_blocks: usize,
    validate_entropy: bool,
    collect_diagnostics: bool,
    reference_buffers: [Option<Arc<FrameBuffers>>; 8],
    initial_cdfs: Option<&[CdfContext]>,
    collect_cdf: bool,
    collect_motion: bool,
    temporal_motion_field: Option<Arc<MotionField>>,
    budget: &mut DecodeBudget,
) -> Result<
    (
        DecodedBlockPrefix,
        BudgetedPostFilterState,
        Vec<CdfContext>,
        Option<MotionField>,
    ),
    DecoderError,
> {
    decode_luma_root_block_prefix_with_budget(
        data,
        tile_group,
        sequence,
        frame,
        plan,
        buffers,
        max_blocks,
        validate_entropy,
        collect_diagnostics,
        reference_buffers,
        initial_cdfs,
        collect_cdf,
        collect_motion,
        temporal_motion_field,
        Some(budget),
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "internal prefix decode exposes each independently testable pipeline input"
)]
fn decode_luma_root_block_prefix_with_budget(
    data: &[u8],
    tile_group: &TileGroup,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    plan: &FrameDecodePlan,
    buffers: &mut FrameBuffers,
    max_blocks: usize,
    validate_entropy: bool,
    collect_diagnostics: bool,
    reference_buffers: [Option<Arc<FrameBuffers>>; 8],
    initial_cdfs: Option<&[CdfContext]>,
    collect_cdf: bool,
    collect_motion: bool,
    temporal_motion_field: Option<Arc<MotionField>>,
    mut budget: Option<&mut DecodeBudget>,
) -> Result<
    (
        DecodedBlockPrefix,
        BudgetedPostFilterState,
        Vec<CdfContext>,
        Option<MotionField>,
    ),
    DecoderError,
> {
    if tile_group.tiles.is_empty() {
        return Err(DecoderError::Bitstream(
            "AV1 tile group has no tile payloads".to_string(),
        ));
    }
    let mut blocks = Vec::new();
    let mut block_budget = max_blocks;
    let mut decoded_block_count = 0;
    let mut post_filter_state = BudgetedPostFilterState::empty();
    let budgeted_post_filter = budget.is_some();
    // Strict native still decode does not retain entropy or temporal-reference
    // state.  Keep the legacy/sequence collection behavior behind explicit
    // switches so the no-state path never reserves either owner.
    let mut final_cdfs = if collect_cdf {
        Vec::with_capacity(tile_group.tiles.len())
    } else {
        Vec::new()
    };
    let mut motion_field = None;

    for (tile_index, tile_payload) in tile_group.tiles.iter().enumerate() {
        let payload = tile_payload_bytes(data, tile_payload)?;
        let tile_plan = plan.tiles.get(tile_index).ok_or_else(|| {
            DecoderError::Bitstream("AV1 tile decode plan is missing a tile".to_string())
        })?;
        let initial_cdf = initial_cdfs.and_then(|cdfs| cdfs.get(tile_index)).cloned();
        let mut decoder = if let Some(native_budget) = budget.as_deref_mut() {
            if collect_cdf || collect_motion {
                let (decoder, tickets, memory_ticket) =
                    TileDecoder::new_with_references_and_cdf_with_budget_for_tile(
                        payload,
                        frame,
                        tile_plan,
                        reference_buffers.clone(),
                        initial_cdf,
                        native_budget,
                    )?;
                TileDecoderOwner::strict_with_memory(
                    decoder,
                    tickets,
                    memory_ticket,
                    native_budget,
                )
            } else {
                let (decoder, tickets) = TileDecoder::new_with_references_and_cdf_with_budget(
                    payload,
                    frame,
                    reference_buffers.clone(),
                    initial_cdf,
                    native_budget,
                )?;
                TileDecoderOwner::strict(decoder, tickets, native_budget)
            }
        } else {
            TileDecoderOwner::legacy(TileDecoder::new_with_references_and_cdf(
                payload,
                frame,
                reference_buffers.clone(),
                initial_cdf,
            )?)
        };
        decoder.set_tile_bounds(tile_plan);
        decoder.set_order_hint_bits(sequence.order_hint_bits);
        if frame.use_ref_frame_mvs {
            decoder.set_temporal_motion_field(temporal_motion_field.clone());
        }
        for sb_row in tile_plan.sb_row_start..tile_plan.sb_row_end {
            decoder.reset_left_superblock_contexts();
            for sb_col in tile_plan.sb_col_start..tile_plan.sb_col_end {
                if block_budget == 0 {
                    if collect_cdf {
                        final_cdfs.push(decoder.cdf_snapshot());
                    }
                    let motion_field = collect_motion
                        .then(|| decoder.motion_field(frame, sequence.order_hint_bits));
                    let tile_state = if budgeted_post_filter {
                        decoder.take_budgeted_post_filter_state()?
                    } else {
                        BudgetedPostFilterState {
                            state: decoder.take_post_filter_state()?,
                            tickets: std::array::from_fn(|_| {
                                AllocationTicket::new(AllocationClass::Frame)
                            }),
                        }
                    };
                    merge_post_filter_state(&mut post_filter_state, tile_state, budget.as_deref_mut())?;
                    return Ok((
                        DecodedBlockPrefix {
                            blocks,
                            next_unsupported: None,
                        },
                        post_filter_state,
                        final_cdfs,
                        motion_field,
                    ));
                }
                let x = (sb_col as usize * plan.superblock_size).min(plan.width);
                let y = (sb_row as usize * plan.superblock_size).min(plan.height);
                decoder.read_restoration_units(sequence, x, y)?;
                let block_start = decoded_block_count;
                let result = decode_luma_block_tree(
                    &mut decoder,
                    sequence,
                    frame,
                    tile_plan,
                    plan,
                    buffers,
                    root_block_size(sequence),
                    x,
                    y,
                    &mut block_budget,
                    &mut blocks,
                    &mut decoded_block_count,
                    collect_diagnostics,
                );
                match result {
                    Ok(()) => {}
                    Err(err @ DecoderError::Unsupported(_))
                        if decoded_block_count > block_start =>
                    {
                        if collect_cdf {
                            final_cdfs.push(decoder.cdf_snapshot());
                        }
                        let motion_field = collect_motion
                            .then(|| decoder.motion_field(frame, sequence.order_hint_bits));
                        let tile_state = if budgeted_post_filter {
                            decoder.take_budgeted_post_filter_state()?
                        } else {
                            BudgetedPostFilterState {
                                state: decoder.take_post_filter_state()?,
                                tickets: std::array::from_fn(|_| {
                                    AllocationTicket::new(AllocationClass::Frame)
                                }),
                            }
                        };
                        merge_post_filter_state(&mut post_filter_state, tile_state, budget.as_deref_mut())?;
                        return Ok((
                            DecodedBlockPrefix {
                                blocks,
                                next_unsupported: Some(err),
                            },
                            post_filter_state,
                            final_cdfs,
                            motion_field,
                        ));
                    }
                    Err(err) => return Err(err),
                }
                if collect_diagnostics {
                    post_filter_state.record_luma_blocks(&blocks[block_start..]);
                }
            }
        }
        if validate_entropy {
            decoder.finish_entropy().map_err(|err| {
                DecoderError::Bitstream(format!(
                    "AV1 tile {} entropy validation failed: {err}",
                    tile_payload.tile_id
                ))
            })?;
        }
        if collect_cdf {
            final_cdfs.push(decoder.cdf_snapshot());
        }
        if collect_motion {
            let tile_motion_field = decoder.motion_field(frame, sequence.order_hint_bits);
            let mut frame_motion_field = motion_field.take().unwrap_or_else(|| {
                MotionField::empty(tile_motion_field.mi_cols, tile_motion_field.mi_rows)
            });
            frame_motion_field.merge(tile_motion_field);
            motion_field = Some(frame_motion_field);
        }
        let tile_state = if budgeted_post_filter {
            decoder.take_budgeted_post_filter_state()?
        } else {
            BudgetedPostFilterState {
                state: decoder.take_post_filter_state()?,
                tickets: std::array::from_fn(|_| AllocationTicket::new(AllocationClass::Frame)),
            }
        };
        merge_post_filter_state(&mut post_filter_state, tile_state, budget.as_deref_mut())?;
    }

    Ok((
        DecodedBlockPrefix {
            blocks,
            next_unsupported: None,
        },
        post_filter_state,
        final_cdfs,
        motion_field,
    ))
}

fn tile_payload_bytes<'a>(
    data: &'a [u8],
    tile_payload: &TilePayload,
) -> Result<&'a [u8], DecoderError> {
    let end = tile_payload
        .offset
        .checked_add(tile_payload.len)
        .ok_or_else(|| DecoderError::Bitstream("AV1 tile payload end overflow".to_string()))?;
    data.get(tile_payload.offset..end).ok_or_else(|| {
        DecoderError::NotEnoughData("AV1 tile payload extends beyond tile group".to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocation::AllocationLedger;
    use crate::av1::{
        TxSize, alloc_frame_buffers, build_still_decode_plan, parse_frame_header,
        parse_sequence_header, parse_tile_group,
    };
    use crate::container::parse_avif;
    use crate::obu::{ObuType, find_obu_payload};

    #[test]
    fn strict_budgeted_owner_drops_fixture_free_scratch_before_ticket_on_question_mark_error() {
        let scratch_bytes = 16usize
            .checked_mul(std::mem::size_of::<i32>())
            .expect("synthetic scratch size must fit");
        let mut budget = DecodeBudget::new(Some(scratch_bytes));
        budget
            .charge(AllocationClass::Frame, scratch_bytes)
            .expect("synthetic scratch must be admitted");
        let scratch = Vec::<i32>::with_capacity(16);
        let ticket = AllocationTicket {
            class: AllocationClass::Frame,
            charged_capacity_bytes: scratch_bytes,
        };
        let observation = crate::test_allocation_observer::Observation::begin(usize::MAX, false);
        observation.track(&scratch);
        observation.track_raw_slot(0, scratch.as_ptr().cast(), scratch_bytes);

        let result: Result<(), DecoderError> = (|| {
            let _owner = TileDecoderOwner::synthetic_scratch(scratch, ticket, &mut budget);
            Err(DecoderError::InvalidParam(
                "synthetic post-construction error".to_string(),
            ))?;
            Ok(())
        })();
        assert!(
            matches!(result, Err(DecoderError::InvalidParam(message)) if message.contains("post-construction"))
        );
        assert_eq!(budget.accounting().aggregate_live, 0);
        assert_eq!(observation.release_count(), 1);
        assert_eq!(
            observation.release_drop_snapshots()[0],
            1,
            "scratch must be dropped before the ticket release snapshot"
        );
        drop(observation);

        // The same exact accounting can be admitted again after the `?`
        // error, proving that no ticket or live-byte state leaked.
        let mut retry_budget = DecodeBudget::new(Some(scratch_bytes));
        retry_budget
            .charge(AllocationClass::Frame, scratch_bytes)
            .expect("retry scratch must be admitted");
        let retry_scratch = Vec::<i32>::with_capacity(16);
        let retry_ticket = AllocationTicket {
            class: AllocationClass::Frame,
            charged_capacity_bytes: scratch_bytes,
        };
        drop(TileDecoderOwner::synthetic_scratch(
            retry_scratch,
            retry_ticket,
            &mut retry_budget,
        ));
        assert_eq!(retry_budget.accounting().aggregate_live, 0);
    }

    #[test]
    fn tile_payload_bytes_checks_bounds_for_each_tile() {
        let data = b"0123456789";
        let first = TilePayload {
            tile_id: 0,
            offset: 2,
            len: 3,
        };
        assert_eq!(tile_payload_bytes(data, &first).unwrap(), b"234");

        let truncated = TilePayload {
            tile_id: 1,
            offset: 8,
            len: 4,
        };
        assert!(matches!(
            tile_payload_bytes(data, &truncated),
            Err(DecoderError::NotEnoughData(_))
        ));
    }

    #[test]
    fn prepares_sample_tile_entropy_state() {
        let Some(data) = crate::test_support::wml2viewer_avif() else {
            return;
        };
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

    fn sample_decode_inputs() -> Option<(
        Vec<u8>,
        SequenceHeader,
        FrameHeader,
        TileGroup,
        FrameDecodePlan,
    )> {
        let data = crate::test_support::wml2viewer_avif()?;
        let info = parse_avif(&data).ok()?;
        let sequence_payload =
            find_obu_payload(&info.primary_item_payload, ObuType::SequenceHeader).ok()??;
        let sequence = parse_sequence_header(sequence_payload).ok()?;
        let frame_payload = find_obu_payload(&info.primary_item_payload, ObuType::Frame)
            .ok()??
            .to_vec();
        let frame = parse_frame_header(&frame_payload, &sequence).ok()?;
        let tile_group = parse_tile_group(
            &frame_payload,
            frame.uncompressed_header_bits,
            &frame.tile_info,
        )
        .ok()?;
        let plan = build_still_decode_plan(&sequence, &frame, &tile_group).ok()?;
        Some((frame_payload, sequence, frame, tile_group, plan))
    }

    fn decode_sample_with_state_collection(
        frame_payload: &[u8],
        sequence: &SequenceHeader,
        frame: &FrameHeader,
        tile_group: &TileGroup,
        plan: &FrameDecodePlan,
        collect_cdf: bool,
        collect_motion: bool,
    ) -> Result<(usize, bool), DecoderError> {
        let mut buffers = alloc_frame_buffers(plan)?;
        let (_, _, cdfs, motion_field) =
            decode_luma_root_block_prefix_with_post_filter_state_and_entropy_options_with_references_and_cdf_and_motion(
                frame_payload,
                tile_group,
                sequence,
                frame,
                plan,
                &mut buffers,
                usize::MAX,
                true,
                false,
                std::array::from_fn(|_| None),
                None,
                collect_cdf,
                collect_motion,
                None,
            )?;
        Ok((cdfs.len(), motion_field.is_some()))
    }

    #[test]
    fn strict_still_root_omits_cdf_and_motion_state_while_legacy_collection_keeps_it() {
        let Some((frame_payload, sequence, frame, tile_group, plan)) = sample_decode_inputs()
        else {
            return;
        };

        let strict = decode_sample_with_state_collection(
            &frame_payload,
            &sequence,
            &frame,
            &tile_group,
            &plan,
            false,
            false,
        )
        .expect("strict no-state sample decode must succeed");
        let legacy = decode_sample_with_state_collection(
            &frame_payload,
            &sequence,
            &frame,
            &tile_group,
            &plan,
            true,
            true,
        )
        .expect("legacy state-collecting sample decode must succeed");

        assert_eq!(strict, (0, false));
        assert_eq!(legacy.0, tile_group.tiles.len());
        assert!(legacy.1, "legacy/sequence state must retain motion");
    }

    #[test]
    fn strict_tile_coefficient_scratch_is_admitted_and_released_transactionally() {
        let Some((frame_payload, sequence, frame, tile_group, plan)) = sample_decode_inputs()
        else {
            return;
        };
        let samples = crate::av1::TxSize::Tx64x64.sample_count();
        let scratch_bytes = (2 * samples * std::mem::size_of::<i32>())
            .checked_add(3 * samples * std::mem::size_of::<u16>())
            .and_then(|bytes| bytes.checked_add(samples * std::mem::size_of::<i32>()))
            .expect("tile scratch size must fit");

        let mut exact_budget = DecodeBudget::new(Some(scratch_bytes));
        let mut exact_buffers = alloc_frame_buffers(&plan).expect("sample buffers must allocate");
        decode_luma_root_block_prefix_with_post_filter_state_and_entropy_options_with_references_and_cdf_and_motion_and_budget(
            &frame_payload,
            &tile_group,
            &sequence,
            &frame,
            &plan,
            &mut exact_buffers,
            0,
            true,
            false,
            std::array::from_fn(|_| None),
            None,
            false,
            false,
            None,
            &mut exact_budget,
        )
        .expect("exact coefficient scratch budget must succeed");
        assert_eq!(exact_budget.accounting().aggregate_live, 0);
        assert!(exact_budget.accounting().aggregate_peak >= scratch_bytes);

        let mut under_budget = DecodeBudget::new(Some(scratch_bytes - 1));
        let mut under_buffers = alloc_frame_buffers(&plan).expect("sample buffers must allocate");
        let error = decode_luma_root_block_prefix_with_post_filter_state_and_entropy_options_with_references_and_cdf_and_motion_and_budget(
            &frame_payload,
            &tile_group,
            &sequence,
            &frame,
            &plan,
            &mut under_buffers,
            0,
            true,
            false,
            std::array::from_fn(|_| None),
            None,
            false,
            false,
            None,
            &mut under_budget,
        )
        .expect_err("one byte below coefficient scratch must fail before construction");
        assert!(
            matches!(error, DecoderError::InvalidParam(message) if message.contains("live allocation"))
        );
        assert_eq!(under_budget.accounting().aggregate_live, 0);
    }

    #[test]
    fn strict_tile_coefficient_scratch_rejects_actual_capacity_then_retries() {
        let Some((frame_payload, sequence, frame, tile_group, plan)) = sample_decode_inputs()
        else {
            return;
        };
        let label = "native AV1 tile reconstruction scratch";
        let _extra = crate::test_allocation_observer::force_fresh_capacity_extra(label, 1 << 20);
        let mut reference_budget = DecodeBudget::new(Some(usize::MAX / 2));
        let mut reference_buffers =
            alloc_frame_buffers(&plan).expect("sample buffers must allocate");
        decode_luma_root_block_prefix_with_post_filter_state_and_entropy_options_with_references_and_cdf_and_motion_and_budget(
            &frame_payload,
            &tile_group,
            &sequence,
            &frame,
            &plan,
            &mut reference_buffers,
            0,
            true,
            false,
            std::array::from_fn(|_| None),
            None,
            false,
            false,
            None,
            &mut reference_budget,
        )
        .expect("reference coefficient scratch allocation must succeed");
        let actual = reference_budget.accounting().aggregate_peak;
        assert!(actual > 0);
        assert_eq!(reference_budget.accounting().aggregate_live, 0);

        let mut under_budget = DecodeBudget::new(Some(actual - 1));
        let mut under_buffers = alloc_frame_buffers(&plan).expect("sample buffers must allocate");
        let observation = crate::test_allocation_observer::Observation::begin(1, false);
        let error = decode_luma_root_block_prefix_with_post_filter_state_and_entropy_options_with_references_and_cdf_and_motion_and_budget(
            &frame_payload,
            &tile_group,
            &sequence,
            &frame,
            &plan,
            &mut under_buffers,
            0,
            true,
            false,
            std::array::from_fn(|_| None),
            None,
            false,
            false,
            None,
            &mut under_budget,
        )
        .expect_err("actual coefficient capacity must reject one-byte-under");
        assert!(
            matches!(error, DecoderError::InvalidParam(message) if message.contains("live allocation"))
        );
        assert_eq!(under_budget.accounting().aggregate_live, 0);
        assert_eq!(
            observation.drops(),
            1,
            "the rejected scratch candidate must be physically dropped exactly once"
        );
        assert_eq!(
            observation.restore_count(),
            1,
            "the rejected scratch transaction must restore exactly once"
        );
        assert_eq!(
            observation.restore_drop_snapshots()[0],
            1,
            "scratch must be physically dropped before the budget checkpoint is restored"
        );
        drop(observation);

        let mut retry_budget = DecodeBudget::new(Some(actual));
        let mut retry_buffers = alloc_frame_buffers(&plan).expect("sample buffers must allocate");
        decode_luma_root_block_prefix_with_post_filter_state_and_entropy_options_with_references_and_cdf_and_motion_and_budget(
            &frame_payload,
            &tile_group,
            &sequence,
            &frame,
            &plan,
            &mut retry_buffers,
            0,
            true,
            false,
            std::array::from_fn(|_| None),
            None,
            false,
            false,
            None,
            &mut retry_budget,
        )
        .expect("exact actual capacity retry must succeed");
        assert_eq!(retry_budget.accounting().aggregate_live, 0);
    }

    #[test]
    fn strict_tile_scratch_drops_all_prefix_owners_before_releasing_tickets() {
        let samples = super::super::TILE_SCRATCH_SAMPLES;
        let prefix_bytes = 2usize
            .checked_mul(samples * std::mem::size_of::<i32>())
            .and_then(|bytes| bytes.checked_add(2 * samples * std::mem::size_of::<u16>()))
            .expect("tile scratch prefix size must fit");
        let fifth_bytes = samples * std::mem::size_of::<u16>();
        let mut budget = DecodeBudget::new(Some(prefix_bytes + fifth_bytes));
        let observation = crate::test_allocation_observer::Observation::begin(1, false);
        let _all_candidates = crate::test_allocation_observer::track_all_candidates();
        let _extra = crate::test_allocation_observer::force_fresh_capacity_extra(
            "native AV1 tile reconstruction scratch",
            1 << 20,
        );

        let (error, scratch, tickets) = match super::super::TileScratch::strict(&mut budget) {
            Ok(_) => panic!("the fifth scratch owner must fail its actual-capacity check"),
            Err(failure) => (failure.error, failure.scratch, failure.tickets),
        };
        assert!(
            matches!(error, DecoderError::InvalidParam(message) if message.contains("live allocation"))
        );

        // The fifth candidate is dropped by its admission transaction before
        // its checkpoint restore. The four successful prefix candidates stay
        // owned by the failure value until all of them can be dropped before
        // their tickets release.
        drop(scratch);
        for mut ticket in tickets.into_iter().flatten() {
            budget
                .release_token(&mut ticket)
                .expect("prefix ticket release must succeed");
        }
        assert_eq!(budget.accounting().aggregate_live, 0);
        assert_eq!(observation.restore_count(), 1);
        assert_eq!(observation.restore_drop_snapshots()[0], 1);
        assert_eq!(observation.release_count(), 4);
        let release_snapshots = observation.release_drop_snapshots();
        assert_eq!(&release_snapshots[..4], &[5, 5, 5, 5]);
        assert_eq!(observation.registered_drops()[..5], [1, 1, 1, 1, 1]);
        drop(observation);

        drop(_extra);
        // The exact accounting remains reusable after the failed fifth owner.
        let mut retry_budget = DecodeBudget::new(Some(prefix_bytes + fifth_bytes));
        let (retry_scratch, retry_tickets) = super::super::TileScratch::strict(&mut retry_budget)
            .expect("exact scratch capacity retry must succeed");
        drop(retry_scratch);
        for mut ticket in retry_tickets {
            retry_budget
                .release_token(&mut ticket)
                .expect("retry ticket release must succeed");
        }
        assert_eq!(retry_budget.accounting().aggregate_live, 0);
    }

    #[test]
    fn strict_constructor_drops_coefficient_before_releasing_ticket_on_tile_failure() {
        let Some((frame_payload, _sequence, frame, tile_group, _plan)) = sample_decode_inputs()
        else {
            return;
        };
        let tile = tile_group
            .tiles
            .first()
            .expect("sample must contain one tile");
        let tile_payload = frame_payload
            .get(tile.offset..tile.offset + tile.len)
            .expect("sample tile payload must be in range");
        let samples = super::super::TILE_SCRATCH_SAMPLES;
        let coefficient_bytes = TxSize::Tx64x64.sample_count() * std::mem::size_of::<i32>();
        let prefix_bytes =
            2 * samples * std::mem::size_of::<i32>() + 2 * samples * std::mem::size_of::<u16>();
        let fifth_bytes = samples * std::mem::size_of::<u16>();
        let capacity = coefficient_bytes + prefix_bytes + fifth_bytes;
        let mut budget = DecodeBudget::new(Some(capacity));
        let observation = crate::test_allocation_observer::Observation::begin(1, false);
        let _all_candidates = crate::test_allocation_observer::track_all_candidates();
        let _extra = crate::test_allocation_observer::force_fresh_capacity_extra(
            "native AV1 tile reconstruction scratch",
            1 << 20,
        );

        let error = match super::super::TileDecoder::new_with_references_and_cdf_with_budget(
            tile_payload,
            &frame,
            std::array::from_fn(|_| None),
            None,
            &mut budget,
        ) {
            Ok(_) => panic!("the fifth tile scratch owner must fail actual capacity admission"),
            Err(error) => error,
        };
        assert!(
            matches!(error, DecoderError::InvalidParam(message) if message.contains("live allocation"))
        );
        assert_eq!(budget.accounting().aggregate_live, 0);
        assert_eq!(observation.restore_count(), 1);
        assert_eq!(observation.release_count(), 5);
        let releases = observation.release_drop_snapshots();
        assert_eq!(
            releases[0], 6,
            "all owners must drop before the first ticket"
        );
        assert_eq!(&releases[..5], &[6, 6, 6, 6, 6]);
        assert_eq!(observation.registered_drops()[..6], [1, 1, 1, 1, 1, 1]);
        drop(observation);
        drop(_extra);

        let mut retry_budget = DecodeBudget::new(Some(capacity));
        let (retry_decoder, retry_tickets) =
            super::super::TileDecoder::new_with_references_and_cdf_with_budget(
                tile_payload,
                &frame,
                std::array::from_fn(|_| None),
                None,
                &mut retry_budget,
            )
            .expect("exact constructor capacity must be reusable");
        drop(retry_decoder);
        for mut ticket in retry_tickets {
            retry_budget
                .release_token(&mut ticket)
                .expect("retry ticket release must succeed");
        }
        assert_eq!(retry_budget.accounting().aggregate_live, 0);
    }

    #[test]
    fn strict_budgeted_decoder_drops_scratch_before_ticket_on_post_construction_error() {
        let Some((frame_payload, sequence, frame, mut tile_group, plan)) = sample_decode_inputs()
        else {
            return;
        };
        let tile = tile_group
            .tiles
            .first_mut()
            .expect("sample must contain a tile");
        // Keep the payload offset valid but truncate it after the strict
        // TileDecoder has been constructed.  This drives the fallible `?`
        // path after the scratch ticket has been admitted.
        tile.len = 1;

        let mut buffers = alloc_frame_buffers(&plan).expect("sample buffers must allocate");
        let mut budget = DecodeBudget::new(None);
        let observation = crate::test_allocation_observer::Observation::begin(usize::MAX, false);
        let _all_candidates = crate::test_allocation_observer::track_all_candidates();
        let error = decode_luma_root_block_prefix_with_post_filter_state_and_entropy_options_with_references_and_cdf_and_motion_and_budget(
            &frame_payload,
            &tile_group,
            &sequence,
            &frame,
            &plan,
            &mut buffers,
            usize::MAX,
            true,
            false,
            std::array::from_fn(|_| None),
            None,
            false,
            false,
            None,
            &mut budget,
        )
        .expect_err("truncated post-construction tile must fail");
        assert!(matches!(error, DecoderError::NotEnoughData(_)));
        assert_eq!(budget.accounting().aggregate_live, 0);
        assert_eq!(observation.release_count(), 6);
        assert_eq!(
            observation.release_drop_snapshots()[0],
            6,
            "scratch must be physically dropped before its ticket is released"
        );
        drop(observation);

        // The same strict route remains usable after the error; this catches
        // leaked ticket/accounting state without relying on a second fixture.
        let Some((frame_payload, sequence, frame, tile_group, plan)) = sample_decode_inputs()
        else {
            return;
        };
        let mut retry_buffers = alloc_frame_buffers(&plan).expect("sample buffers must allocate");
        let mut retry_budget = DecodeBudget::new(None);
        decode_luma_root_block_prefix_with_post_filter_state_and_entropy_options_with_references_and_cdf_and_motion_and_budget(
            &frame_payload,
            &tile_group,
            &sequence,
            &frame,
            &plan,
            &mut retry_buffers,
            0,
            true,
            false,
            std::array::from_fn(|_| None),
            None,
            false,
            false,
            None,
            &mut retry_budget,
        )
        .expect("strict route must retry after post-construction error");
        assert_eq!(retry_budget.accounting().aggregate_live, 0);
    }

    #[test]
    fn strict_no_state_path_preserves_source_on_early_tile_error() {
        let Some((frame_payload, sequence, frame, mut tile_group, plan)) = sample_decode_inputs()
        else {
            return;
        };
        let source_snapshot = frame_payload.clone();
        let first = tile_group
            .tiles
            .first_mut()
            .expect("sample must have one tile");
        first.offset = frame_payload.len();
        first.len = 1;

        let mut buffers = alloc_frame_buffers(&plan).expect("sample buffers must allocate");
        let error = decode_luma_root_block_prefix_with_post_filter_state_and_entropy_options_with_references_and_cdf_and_motion(
            &frame_payload,
            &tile_group,
            &sequence,
            &frame,
            &plan,
            &mut buffers,
            usize::MAX,
            true,
            false,
            std::array::from_fn(|_| None),
            None,
            false,
            false,
            None,
        )
        .expect_err("out-of-range tile must fail before state collection");

        assert!(matches!(error, DecoderError::NotEnoughData(_)));
        assert_eq!(frame_payload, source_snapshot);
    }

    #[test]
    fn strict_one_tile_state_omission_reduces_heap_requests() {
        let Some((frame_payload, sequence, frame, tile_group, plan)) = sample_decode_inputs()
        else {
            return;
        };
        if tile_group.tiles.len() != 1 {
            return;
        }

        let (strict, strict_requests) =
            crate::test_allocation_observer::count_allocation_requests(|| {
                decode_sample_with_state_collection(
                    &frame_payload,
                    &sequence,
                    &frame,
                    &tile_group,
                    &plan,
                    false,
                    false,
                )
                .expect("strict no-state sample decode must succeed")
            });
        let (legacy, legacy_requests) =
            crate::test_allocation_observer::count_allocation_requests(|| {
                decode_sample_with_state_collection(
                    &frame_payload,
                    &sequence,
                    &frame,
                    &tile_group,
                    &plan,
                    true,
                    true,
                )
                .expect("legacy state-collecting sample decode must succeed")
            });

        assert_eq!(strict, (0, false));
        assert_eq!(legacy.0, 1);
        assert!(legacy.1);
        assert!(
            strict_requests < legacy_requests,
            "strict no-state decode must make fewer allocation requests: strict={strict_requests}, legacy={legacy_requests}"
        );
    }

    #[test]
    fn strict_tile_memory_plan_rejects_one_byte_under_before_allocation() {
        let Some((_payload, _sequence, frame, _group, plan)) = sample_decode_inputs() else {
            return;
        };
        let tile = plan.tiles.first().expect("sample must contain a tile");
        let memory_plan = super::super::TileDecoderMemoryPlan::for_frame(&frame, Some(tile))
            .expect("sample tile memory plan must be representable");
        assert!(memory_plan.requested_bytes > 0);
        assert!(memory_plan.tile_mi_cols > 0);
        assert!(memory_plan.tile_mi_rows > 0);
        let mut budget = DecodeBudget::new(Some(memory_plan.requested_bytes - 1));
        let error = match super::super::TileDecoder::new_with_references_and_cdf_with_budget_for_tile(
            &[],
            &frame,
            tile,
            std::array::from_fn(|_| None),
            None,
            &mut budget,
        ) {
            Ok(_) => panic!("one byte below the complete tile plan must fail before allocation"),
            Err(error) => error,
        };
        assert!(matches!(error, DecoderError::InvalidParam(message) if message.contains("live allocation")));
        assert_eq!(budget.accounting().aggregate_live, 0);
    }

    #[test]
    fn strict_tile_memory_plan_rejects_actual_capacity_and_retries_exactly() {
        let Some((payload, _sequence, frame, _group, plan)) = sample_decode_inputs() else {
            return;
        };
        let tile = plan.tiles.first().expect("sample must contain a tile");
        let memory_plan = super::super::TileDecoderMemoryPlan::for_frame(&frame, Some(tile))
            .expect("sample tile memory plan must be representable");
        let forced_extra = 1usize << 20;
        let mut budget = DecodeBudget::new(Some(
            memory_plan
                .requested_bytes
                .checked_add(forced_extra - 1)
                .expect("test budget must fit"),
        ));
        let observation = crate::test_allocation_observer::Observation::begin(128, false);
        let _extra = crate::test_allocation_observer::force_fresh_capacity_extra(
            "native AV1 y mode grid",
            forced_extra,
        );
        let error = match super::super::TileDecoder::new_with_references_and_cdf_with_budget_for_tile(
            &payload,
            &frame,
            tile,
            std::array::from_fn(|_| None),
            None,
            &mut budget,
        ) {
            Ok(_) => panic!("actual capacity above the plan must fail transactionally"),
            Err(error) => error,
        };
        assert!(matches!(error, DecoderError::InvalidParam(message) if message.contains("live allocation")));
        assert_eq!(budget.accounting().aggregate_live, 0);
        assert!(observation.drops() >= 1);
        drop(observation);
        drop(_extra);

        let mut retry_budget = DecodeBudget::new(Some(memory_plan.requested_bytes));
        let tile_payload = payload
            .get(tile.payload_offset..tile.payload_offset + tile.payload_len)
            .expect("sample tile payload must be in range");
        let (decoder, tickets, mut memory_ticket) =
            super::super::TileDecoder::new_with_references_and_cdf_with_budget_for_tile(
                tile_payload,
                &frame,
                tile,
                std::array::from_fn(|_| None),
                None,
                &mut retry_budget,
            )
            .expect("exact planned capacity must be reusable after failure");
        drop(decoder);
        for mut ticket in tickets {
            retry_budget
                .release_token(&mut ticket)
                .expect("tile scratch ticket release must succeed");
        }
        retry_budget
            .release_token(&mut memory_ticket)
            .expect("tile memory ticket release must succeed");
        assert_eq!(retry_budget.accounting().aggregate_live, 0);
    }

    #[test]
    fn budgeted_post_filter_merge_releases_partial_destination_on_failure() {
        let mut destination_units = Vec::with_capacity(1);
        destination_units.push(super::super::post_filter_state::CdefUnit {
            x: 0,
            y: 0,
            index: 0,
        });
        let mut destination = BudgetedPostFilterState {
            state: PostFilterState {
                cdef_units: destination_units,
                ..PostFilterState::default()
            },
            tickets: std::array::from_fn(|_| AllocationTicket::new(AllocationClass::Frame)),
        };
        let destination_bytes = std::mem::size_of::<super::super::post_filter_state::CdefUnit>();
        let second_requested =
            std::mem::size_of::<super::super::post_filter_state::CdefBlockIndex>();
        let first_requested = destination_bytes;
        let mut budget = DecodeBudget::new(Some(
            destination_bytes
                .checked_add(first_requested)
                .and_then(|bytes| bytes.checked_add(second_requested))
                .expect("merge test budget must fit"),
        ));
        budget
            .charge(AllocationClass::Frame, destination_bytes)
            .expect("destination must be admitted");
        destination.tickets[0].charged_capacity_bytes = destination_bytes;
        let source = BudgetedPostFilterState {
            state: PostFilterState {
                cdef_units: vec![super::super::post_filter_state::CdefUnit {
                    x: 4,
                    y: 0,
                    index: 1,
                }],
                cdef_blocks: vec![super::super::post_filter_state::CdefBlockIndex {
                    x: 0,
                    y: 0,
                    index: 0,
                }],
                ..PostFilterState::default()
            },
            tickets: std::array::from_fn(|_| AllocationTicket::new(AllocationClass::Frame)),
        };
        let observation = crate::test_allocation_observer::Observation::begin(1, false);
        let _extra = crate::test_allocation_observer::force_fresh_capacity_extra(
            "native AV1 merged CDEF blocks",
            1 << 20,
        );
        let error = merge_post_filter_state(&mut destination, source, Some(&mut budget))
            .expect_err("second destination replacement must fail");
        assert!(matches!(error, DecoderError::InvalidParam(message) if message.contains("live allocation")));
        assert_eq!(budget.accounting().aggregate_live, 0);
        assert!(observation.drops() >= 1);
        drop(observation);
    }
}
