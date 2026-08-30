use super::frame::{FrameHeader, TxMode};
use super::sequence::SequenceHeader;
use super::syntax::mi_dimension;
use super::tile_group::TileGroup;
use crate::DecoderError;
use crate::allocation::{AllocationClass, AllocationTicket, admit_fresh_with, fresh_replacement};
use crate::container::DecodeBudget;
use crate::limits::NativeDecodeLimits;

const MAX_PLANE_SAMPLE_ALLOCATION: usize = 1 << 28;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaneLayout {
    pub plane: u8,
    pub width: usize,
    pub height: usize,
    pub subsampling_x: u8,
    pub subsampling_y: u8,
    pub sample_count: usize,
}

impl PlaneLayout {
    pub fn stride(&self) -> usize {
        self.width
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileDecodePlan {
    pub tile_id: u32,
    pub tile_col: u32,
    pub tile_row: u32,
    pub sb_col_start: u32,
    pub sb_col_end: u32,
    pub sb_row_start: u32,
    pub sb_row_end: u32,
    pub mi_col_start: u32,
    pub mi_col_end: u32,
    pub mi_row_start: u32,
    pub mi_row_end: u32,
    pub pixel_x: usize,
    pub pixel_y: usize,
    pub pixel_width: usize,
    pub pixel_height: usize,
    pub payload_offset: usize,
    pub payload_len: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameDecodePlan {
    pub width: usize,
    pub height: usize,
    pub upscaled_width: usize,
    pub render_width: usize,
    pub render_height: usize,
    pub bit_depth: u8,
    pub base_q_idx: u8,
    pub tx_mode: TxMode,
    pub superblock_size: usize,
    pub superblock_cols: u32,
    pub superblock_rows: u32,
    pub uses_cdef: bool,
    pub uses_restoration: bool,
    pub planes: Vec<PlaneLayout>,
    pub tiles: Vec<TileDecodePlan>,
}

/// Ownership tickets for the dynamic decode-plan tables created by the
/// strict-native `OBU_FRAME` assembler.  The plan itself remains the public
/// compatibility type; these tickets exist only until its private header is
/// dropped.
pub(crate) struct NativeDecodePlanAllocation {
    planes: AllocationTicket,
    tiles: AllocationTicket,
}

impl NativeDecodePlanAllocation {
    fn new() -> Self {
        Self {
            planes: AllocationTicket::new(AllocationClass::Frame),
            tiles: AllocationTicket::new(AllocationClass::Frame),
        }
    }

    pub(crate) fn release(&mut self, budget: &mut DecodeBudget) -> Result<(), DecoderError> {
        budget.release_token(&mut self.planes)?;
        budget.release_token(&mut self.tiles)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaneBuffer {
    pub layout: PlaneLayout,
    pub samples: Vec<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameBuffers {
    pub width: usize,
    pub height: usize,
    pub planes: Vec<PlaneBuffer>,
}

/// Private ownership tickets for a frame allocated by the bounded native
/// still decoder.  The public `FrameBuffers` layout remains unchanged; this
/// sidecar exists only until the strict decode returns.
pub(crate) struct NativeFrameAllocation {
    outer: AllocationTicket,
    samples: [Option<AllocationTicket>; 4],
}

impl NativeFrameAllocation {
    fn new() -> Self {
        Self {
            outer: AllocationTicket::new(AllocationClass::Frame),
            samples: std::array::from_fn(|_| None),
        }
    }

    pub(crate) fn release(&mut self, budget: &mut DecodeBudget) -> Result<(), DecoderError> {
        budget.release_token(&mut self.outer)?;
        for ticket in self.samples.iter_mut().flatten() {
            budget.release_token(ticket)?;
        }
        Ok(())
    }

    pub(crate) fn take_plane_ticket(
        &mut self,
        index: usize,
    ) -> Result<AllocationTicket, DecoderError> {
        self.samples
            .get_mut(index)
            .and_then(Option::take)
            .ok_or_else(|| {
                DecoderError::InvalidParam("native AVIF plane ticket is missing".to_string())
            })
    }

    pub(crate) fn release_outer(&mut self, budget: &mut DecodeBudget) -> Result<(), DecoderError> {
        budget.release_token(&mut self.outer)
    }

    pub(crate) fn adopt_alpha(&mut self, ticket: AllocationTicket) -> Result<(), DecoderError> {
        let slot = self.samples.get_mut(3).ok_or_else(|| {
            DecoderError::InvalidParam("native AVIF alpha ticket slot is missing".to_string())
        })?;
        if slot.is_some() {
            return Err(DecoderError::InvalidParam(
                "native AVIF alpha ticket is already present".to_string(),
            ));
        }
        *slot = Some(ticket);
        Ok(())
    }

    pub(crate) fn grow_outer_for_alpha(
        &mut self,
        planes: &mut Vec<PlaneBuffer>,
        budget: &mut DecodeBudget,
    ) -> Result<(), DecoderError> {
        self.grow_outer_for_alpha_with_maker(planes, budget, fresh_replacement::<PlaneBuffer>)
    }

    fn grow_outer_for_alpha_with_maker<F>(
        &mut self,
        planes: &mut Vec<PlaneBuffer>,
        budget: &mut DecodeBudget,
        make: F,
    ) -> Result<(), DecoderError>
    where
        F: FnOnce(
            usize,
            &str,
        )
            -> Result<crate::allocation::FreshReplacement<PlaneBuffer>, DecoderError>,
    {
        crate::allocation::replace_vec_with_maker(
            budget,
            planes,
            &mut self.outer,
            1,
            AllocationClass::Frame,
            "native frame plane owner",
            make,
        )
    }
}

pub fn build_still_decode_plan(
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    tile_group: &TileGroup,
) -> Result<FrameDecodePlan, DecoderError> {
    validate_complete_tile_group(
        frame.tile_info.tile_cols,
        frame.tile_info.tile_rows,
        tile_group,
    )?;
    if frame.frame_width == 0 || frame.frame_height == 0 {
        return Err(DecoderError::Bitstream(
            "AV1 frame dimensions must be non-zero".to_string(),
        ));
    }

    let width = usize::try_from(frame.frame_width)
        .map_err(|_| DecoderError::InvalidParam("AV1 frame width is too large".to_string()))?;
    let height = usize::try_from(frame.frame_height)
        .map_err(|_| DecoderError::InvalidParam("AV1 frame height is too large".to_string()))?;
    let upscaled_width = usize::try_from(frame.upscaled_width)
        .map_err(|_| DecoderError::InvalidParam("AV1 upscaled width is too large".to_string()))?;
    let superblock_size: usize = if sequence.use_128x128_superblock {
        128
    } else {
        64
    };
    let superblock_mi: usize = superblock_size / 4;
    let mi_cols = usize::try_from(mi_dimension(frame.frame_width))
        .map_err(|_| DecoderError::InvalidParam("AV1 frame width is too large".to_string()))?;
    let mi_rows = usize::try_from(mi_dimension(frame.frame_height))
        .map_err(|_| DecoderError::InvalidParam("AV1 frame height is too large".to_string()))?;
    let superblock_cols = round_shift_usize(mi_cols, superblock_mi.trailing_zeros() as u8) as u32;
    let superblock_rows = round_shift_usize(mi_rows, superblock_mi.trailing_zeros() as u8) as u32;
    let planes = build_plane_layouts(sequence, width, height)?;
    let tiles = build_tile_plans(frame, tile_group, width, height, superblock_mi as u32)?;

    Ok(FrameDecodePlan {
        width,
        height,
        upscaled_width,
        render_width: usize::try_from(frame.render_width)
            .map_err(|_| DecoderError::InvalidParam("AV1 render width is too large".to_string()))?,
        render_height: usize::try_from(frame.render_height).map_err(|_| {
            DecoderError::InvalidParam("AV1 render height is too large".to_string())
        })?,
        bit_depth: sequence.color_config.bit_depth,
        base_q_idx: frame.base_q_idx,
        tx_mode: frame.tx_mode,
        superblock_size,
        superblock_cols,
        superblock_rows,
        uses_cdef: frame.cdef.enabled,
        uses_restoration: frame.restoration.uses_lr,
        planes,
        tiles,
    })
}

/// Strict-native counterpart of [`build_still_decode_plan`].  The final
/// `planes` and `tiles` tables are admitted before they are populated, rather
/// than being made by an untracked legacy helper and moved into the plan.
pub(crate) fn build_still_decode_plan_with_budget(
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    tile_group: &TileGroup,
    budget: &mut DecodeBudget,
) -> Result<(FrameDecodePlan, NativeDecodePlanAllocation), DecoderError> {
    validate_complete_tile_group(
        frame.tile_info.tile_cols,
        frame.tile_info.tile_rows,
        tile_group,
    )?;
    if frame.frame_width == 0 || frame.frame_height == 0 {
        return Err(DecoderError::Bitstream(
            "AV1 frame dimensions must be non-zero".to_string(),
        ));
    }

    let width = usize::try_from(frame.frame_width)
        .map_err(|_| DecoderError::InvalidParam("AV1 frame width is too large".to_string()))?;
    let height = usize::try_from(frame.frame_height)
        .map_err(|_| DecoderError::InvalidParam("AV1 frame height is too large".to_string()))?;
    let upscaled_width = usize::try_from(frame.upscaled_width)
        .map_err(|_| DecoderError::InvalidParam("AV1 upscaled width is too large".to_string()))?;
    let superblock_size: usize = if sequence.use_128x128_superblock {
        128
    } else {
        64
    };
    let superblock_mi = superblock_size / 4;
    let mi_cols = usize::try_from(mi_dimension(frame.frame_width))
        .map_err(|_| DecoderError::InvalidParam("AV1 frame width is too large".to_string()))?;
    let mi_rows = usize::try_from(mi_dimension(frame.frame_height))
        .map_err(|_| DecoderError::InvalidParam("AV1 frame height is too large".to_string()))?;
    let superblock_cols = round_shift_usize(mi_cols, superblock_mi.trailing_zeros() as u8) as u32;
    let superblock_rows = round_shift_usize(mi_rows, superblock_mi.trailing_zeros() as u8) as u32;
    let plane_count = if sequence.color_config.monochrome {
        1
    } else {
        3
    };
    let tile_count = tile_group.tiles.len();

    let mut allocation = NativeDecodePlanAllocation::new();
    let result = (|| {
        let (mut plane_candidate, plane_ticket) = admit_fresh_with(
            budget,
            plane_count,
            AllocationClass::Frame,
            "native AV1 decode plan planes",
            fresh_replacement::<PlaneLayout>,
        )?;
        allocation.planes = plane_ticket;
        for plane in 0..plane_count {
            let layout =
                plane_layout_for_geometry(sequence, width, height, plane)?.ok_or_else(|| {
                    DecoderError::InvalidParam("AV1 decode plan plane is missing".to_string())
                })?;
            plane_candidate.values_mut().push(layout);
        }

        let (mut tile_candidate, tile_ticket) = admit_fresh_with(
            budget,
            tile_count,
            AllocationClass::Frame,
            "native AV1 decode plan tiles",
            fresh_replacement::<TileDecodePlan>,
        )?;
        allocation.tiles = tile_ticket;
        populate_tile_plans(
            tile_candidate.values_mut(),
            frame,
            tile_group,
            width,
            height,
            superblock_mi as u32,
        )?;

        Ok(FrameDecodePlan {
            width,
            height,
            upscaled_width,
            render_width: usize::try_from(frame.render_width).map_err(|_| {
                DecoderError::InvalidParam("AV1 render width is too large".to_string())
            })?,
            render_height: usize::try_from(frame.render_height).map_err(|_| {
                DecoderError::InvalidParam("AV1 render height is too large".to_string())
            })?,
            bit_depth: sequence.color_config.bit_depth,
            base_q_idx: frame.base_q_idx,
            tx_mode: frame.tx_mode,
            superblock_size,
            superblock_cols,
            superblock_rows,
            uses_cdef: frame.cdef.enabled,
            uses_restoration: frame.restoration.uses_lr,
            planes: plane_candidate.into_vec(),
            tiles: tile_candidate.into_vec(),
        })
    })();
    match result {
        Ok(plan) => Ok((plan, allocation)),
        Err(error) => {
            allocation.release(budget)?;
            Err(error)
        }
    }
}

#[rustfmt::skip]
const SUPERRES_FILTER: [[i8; 8]; 64] = [
    [0, 0, 0, -128, 0, 0, 0, 0], [0, 0, 1, -128, -2, 1, 0, 0],
    [0, -1, 3, -127, -4, 2, -1, 0], [0, -1, 4, -127, -6, 3, -1, 0],
    [0, -2, 6, -126, -8, 3, -1, 0], [0, -2, 7, -125, -11, 4, -1, 0],
    [1, -2, 8, -125, -13, 5, -2, 0], [1, -3, 9, -124, -15, 6, -2, 0],
    [1, -3, 10, -123, -18, 6, -2, 1], [1, -3, 11, -122, -20, 7, -3, 1],
    [1, -4, 12, -121, -22, 8, -3, 1], [1, -4, 13, -120, -25, 9, -3, 1],
    [1, -4, 14, -118, -28, 9, -3, 1], [1, -4, 15, -117, -30, 10, -4, 1],
    [1, -5, 16, -116, -32, 11, -4, 1], [1, -5, 16, -114, -35, 12, -4, 1],
    [1, -5, 17, -112, -38, 12, -4, 1], [1, -5, 18, -111, -40, 13, -5, 1],
    [1, -5, 18, -109, -43, 14, -5, 1], [1, -6, 19, -107, -45, 14, -5, 1],
    [1, -6, 19, -105, -48, 15, -5, 1], [1, -6, 19, -103, -51, 16, -5, 1],
    [1, -6, 20, -101, -53, 16, -6, 1], [1, -6, 20, -99, -56, 17, -6, 1],
    [1, -6, 20, -97, -58, 17, -6, 1], [1, -6, 20, -95, -61, 18, -6, 1],
    [2, -7, 20, -93, -64, 18, -6, 2], [2, -7, 20, -91, -66, 19, -6, 1],
    [2, -7, 20, -88, -69, 19, -6, 1], [2, -7, 20, -86, -71, 19, -6, 1],
    [2, -7, 20, -84, -74, 20, -7, 2], [2, -7, 20, -81, -76, 20, -7, 1],
    [2, -7, 20, -79, -79, 20, -7, 2], [1, -7, 20, -76, -81, 20, -7, 2],
    [2, -7, 20, -74, -84, 20, -7, 2], [1, -6, 19, -71, -86, 20, -7, 2],
    [1, -6, 19, -69, -88, 20, -7, 2], [1, -6, 19, -66, -91, 20, -7, 2],
    [2, -6, 18, -64, -93, 20, -7, 2], [1, -6, 18, -61, -95, 20, -6, 1],
    [1, -6, 17, -58, -97, 20, -6, 1], [1, -6, 17, -56, -99, 20, -6, 1],
    [1, -6, 16, -53, -101, 20, -6, 1], [1, -5, 16, -51, -103, 19, -6, 1],
    [1, -5, 15, -48, -105, 19, -6, 1], [1, -5, 14, -45, -107, 19, -6, 1],
    [1, -5, 14, -43, -109, 18, -5, 1], [1, -5, 13, -40, -111, 18, -5, 1],
    [1, -4, 12, -38, -112, 17, -5, 1], [1, -4, 12, -35, -114, 16, -5, 1],
    [1, -4, 11, -32, -116, 16, -5, 1], [1, -4, 10, -30, -117, 15, -4, 1],
    [1, -3, 9, -28, -118, 14, -4, 1], [1, -3, 9, -25, -120, 13, -4, 1],
    [1, -3, 8, -22, -121, 12, -4, 1], [1, -3, 7, -20, -122, 11, -3, 1],
    [1, -2, 6, -18, -123, 10, -3, 1], [0, -2, 6, -15, -124, 9, -3, 1],
    [0, -2, 5, -13, -125, 8, -2, 1], [0, -1, 4, -11, -125, 7, -2, 0],
    [0, -1, 3, -8, -126, 6, -2, 0], [0, -1, 3, -6, -127, 4, -1, 0],
    [0, -1, 2, -4, -127, 3, -1, 0], [0, 0, 1, -2, -128, 1, 0, 0],
];

pub(crate) fn apply_superres_horizontal(
    buffers: &mut FrameBuffers,
    upscaled_width: usize,
    bit_depth: u8,
) -> Result<(), DecoderError> {
    if buffers.width >= upscaled_width {
        return Ok(());
    }
    for plane in &mut buffers.planes {
        let target_width = round_shift_usize(upscaled_width, plane.layout.subsampling_x);
        if target_width <= plane.layout.width {
            continue;
        }
        let source_width = plane.layout.width;
        let step =
            (((source_width as u64) << 14) + (target_width as u64 / 2)) / target_width as u64;
        let err = (target_width as i64 * step as i64) - ((source_width as i64) << 14);
        let start = (((-((target_width as i64 - source_width as i64) << 13)
            + (target_width as i64 >> 1))
            / target_width as i64)
            + 128
            - err / 2)
            & 0x3fff;
        let mut resized = vec![0; target_width * plane.layout.height];
        let max_x = source_width.saturating_sub(1) as i64;
        for y in 0..plane.layout.height {
            let row = &plane.samples[y * source_width..(y + 1) * source_width];
            for x in 0..target_width {
                let phase = (start + x as i64 * step as i64) as usize;
                let filter = &SUPERRES_FILTER[(phase >> 8) & 63];
                let src_x = -4 + ((phase as i64) >> 14);
                let mut sum = 0i32;
                for (tap, coeff) in filter.iter().enumerate() {
                    let index = (src_x + tap as i64).clamp(0, max_x) as usize;
                    sum += i32::from(*coeff) * i32::from(row[index]);
                }
                let max_sample = (1i32 << bit_depth) - 1;
                resized[y * target_width + x] = ((-sum + 64) >> 7).clamp(0, max_sample) as u16;
            }
        }
        plane.layout.width = target_width;
        plane.layout.sample_count = resized.len();
        plane.samples = resized;
    }
    buffers.width = upscaled_width;
    Ok(())
}

fn validate_complete_tile_group(
    tile_cols: u32,
    tile_rows: u32,
    tile_group: &TileGroup,
) -> Result<(), DecoderError> {
    let tile_count = tile_cols.checked_mul(tile_rows).ok_or_else(|| {
        DecoderError::InvalidParam("AV1 tile count overflows the decoder limit".to_string())
    })?;
    if tile_count == 0
        || tile_group.start_tile != 0
        || tile_group.end_tile != tile_count - 1
        || tile_group.tiles.len() != usize::try_from(tile_count).unwrap_or(usize::MAX)
    {
        return Err(DecoderError::Unsupported(
            "AV1 partial tile groups are not supported for still-image decode".to_string(),
        ));
    }
    Ok(())
}

pub fn alloc_frame_buffers(plan: &FrameDecodePlan) -> Result<FrameBuffers, DecoderError> {
    alloc_frame_buffers_for_layouts(plan.bit_depth, plan.width, plan.height, &plan.planes)
}

fn alloc_frame_buffers_for_layouts(
    bit_depth: u8,
    width: usize,
    height: usize,
    layouts: &[PlaneLayout],
) -> Result<FrameBuffers, DecoderError> {
    if bit_depth > 16 {
        return Err(DecoderError::Unsupported(format!(
            "AV1 {bit_depth}-bit output buffers are not supported"
        )));
    }
    let mut planes = Vec::with_capacity(layouts.len());
    for layout in layouts {
        if layout.sample_count > MAX_PLANE_SAMPLE_ALLOCATION {
            return Err(DecoderError::InvalidParam(format!(
                "AV1 plane sample count {} exceeds decoder resource limit",
                layout.sample_count
            )));
        }
        planes.push(PlaneBuffer {
            layout: *layout,
            samples: vec![0; layout.sample_count],
        });
    }
    Ok(FrameBuffers {
        width,
        height,
        planes,
    })
}

pub(crate) fn alloc_coded_frame_buffers(
    plan: &FrameDecodePlan,
) -> Result<FrameBuffers, DecoderError> {
    let coded_width =
        usize::try_from(mi_dimension(u32::try_from(plan.width).map_err(|_| {
            DecoderError::InvalidParam("AV1 frame width is too large".to_string())
        })?))
        .map_err(|_| DecoderError::InvalidParam("AV1 frame width is too large".to_string()))?
            << 2;
    let coded_height =
        usize::try_from(mi_dimension(u32::try_from(plan.height).map_err(|_| {
            DecoderError::InvalidParam("AV1 frame height is too large".to_string())
        })?))
        .map_err(|_| DecoderError::InvalidParam("AV1 frame height is too large".to_string()))?
            << 2;
    let mut coded_layouts = plan.planes.clone();
    for layout in &mut coded_layouts {
        layout.width = round_shift_usize(coded_width, layout.subsampling_x);
        layout.height = round_shift_usize(coded_height, layout.subsampling_y);
        layout.sample_count = layout.width.checked_mul(layout.height).ok_or_else(|| {
            DecoderError::InvalidParam("AV1 coded plane dimensions are too large".to_string())
        })?;
    }
    // Keep the public frame dimensions visible while the plane layouts retain
    // aligned coded dimensions until the post-filter crop stage.
    alloc_frame_buffers_for_layouts(plan.bit_depth, plan.width, plan.height, &coded_layouts)
}

/// Bounded variant of coded-frame allocation used only by strict native
/// decode.  Actual vector capacities, not just requested samples, are
/// admitted into the parser's moved budget before the frame is materialized.
pub(crate) fn alloc_coded_frame_buffers_with_budget(
    plan: &FrameDecodePlan,
    budget: &mut DecodeBudget,
    limits: &NativeDecodeLimits,
) -> Result<(FrameBuffers, NativeFrameAllocation), DecoderError> {
    let coded_width =
        usize::try_from(mi_dimension(u32::try_from(plan.width).map_err(|_| {
            DecoderError::InvalidParam("AV1 frame width is too large".to_string())
        })?))
        .map_err(|_| DecoderError::InvalidParam("AV1 frame width is too large".to_string()))?
            << 2;
    let coded_height =
        usize::try_from(mi_dimension(u32::try_from(plan.height).map_err(|_| {
            DecoderError::InvalidParam("AV1 frame height is too large".to_string())
        })?))
        .map_err(|_| DecoderError::InvalidParam("AV1 frame height is too large".to_string()))?
            << 2;
    if plan.planes.len() > 3 {
        return Err(DecoderError::InvalidParam(
            "native AVIF coded frame has too many planes".to_string(),
        ));
    }
    let mut layouts = plan.planes.clone();
    for layout in &mut layouts {
        layout.width = round_shift_usize(coded_width, layout.subsampling_x);
        layout.height = round_shift_usize(coded_height, layout.subsampling_y);
        layout.sample_count = layout.width.checked_mul(layout.height).ok_or_else(|| {
            DecoderError::InvalidParam("AV1 coded plane dimensions are too large".to_string())
        })?;
    }

    let mut allocation = NativeFrameAllocation::new();
    let result = (|| {
        let mut planes = Vec::new();
        budget.try_reserve_class_with_token(
            &mut planes,
            &mut allocation.outer,
            layouts.len(),
            AllocationClass::Frame,
            "native frame plane owner",
        )?;
        for (index, layout) in layouts.into_iter().enumerate() {
            if layout.sample_count > MAX_PLANE_SAMPLE_ALLOCATION {
                return Err(DecoderError::InvalidParam(format!(
                    "AV1 plane sample count {} exceeds decoder resource limit",
                    layout.sample_count
                )));
            }
            let requested_bytes = layout
                .sample_count
                .checked_mul(std::mem::size_of::<u16>())
                .ok_or_else(|| {
                    DecoderError::InvalidParam("native AVIF plane size overflows".to_string())
                })?;
            limits.check_plane_bytes(requested_bytes)?;
            let mut samples = Vec::new();
            let mut ticket = AllocationTicket::new(AllocationClass::Frame);
            budget.try_reserve_class_with_token(
                &mut samples,
                &mut ticket,
                layout.sample_count,
                AllocationClass::Frame,
                "native frame samples",
            )?;
            samples.resize(layout.sample_count, 0);
            allocation.samples[index] = Some(ticket);
            planes.push(PlaneBuffer { layout, samples });
        }
        Ok(FrameBuffers {
            width: plan.width,
            height: plan.height,
            planes,
        })
    })();
    match result {
        Ok(buffers) => Ok((buffers, allocation)),
        Err(error) => {
            allocation.release(budget)?;
            Err(error)
        }
    }
}

/// Crops native coded planes while their old allocations remain live. Each
/// visible candidate is budgeted at its actual capacity before its coded owner
/// is dropped, so the caller-observable live cap covers the crop peak.
pub(crate) fn crop_native_frame_buffers_to_plan(
    buffers: &mut FrameBuffers,
    allocation: &mut NativeFrameAllocation,
    plan: &FrameDecodePlan,
    budget: &mut DecodeBudget,
    limits: &NativeDecodeLimits,
) -> Result<(), DecoderError> {
    crop_native_frame_buffers_to_plan_with_maker(
        buffers,
        allocation,
        plan,
        budget,
        limits,
        fresh_replacement::<u16>,
    )
}

fn crop_native_frame_buffers_to_plan_with_maker<F>(
    buffers: &mut FrameBuffers,
    allocation: &mut NativeFrameAllocation,
    plan: &FrameDecodePlan,
    budget: &mut DecodeBudget,
    limits: &NativeDecodeLimits,
    mut make: F,
) -> Result<(), DecoderError>
where
    F: FnMut(usize, &str) -> Result<crate::allocation::FreshReplacement<u16>, DecoderError>,
{
    if buffers.planes.len() != plan.planes.len() {
        return Err(DecoderError::InvalidParam(
            "AV1 frame buffer plane count does not match decode plan".to_string(),
        ));
    }
    if buffers
        .planes
        .iter()
        .zip(&plan.planes)
        .all(|(plane, target)| plane.layout == *target)
    {
        return Ok(());
    }
    for (index, (plane, target)) in buffers.planes.iter_mut().zip(&plan.planes).enumerate() {
        if plane.layout.width < target.width || plane.layout.height < target.height {
            return Err(DecoderError::InvalidParam(
                "AV1 coded plane is smaller than the visible frame".to_string(),
            ));
        }
        let requested_bytes = target
            .sample_count
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| {
                DecoderError::InvalidParam("native AVIF visible plane size overflows".to_string())
            })?;
        limits.check_plane_bytes(requested_bytes)?;
        if allocation.samples.get(index).is_none_or(Option::is_none) {
            return Err(DecoderError::InvalidParam(
                "native AVIF coded frame plane ticket is missing".to_string(),
            ));
        }
        let (mut visible, candidate) = admit_fresh_with(
            budget,
            target.sample_count,
            AllocationClass::Frame,
            "native visible frame samples",
            &mut make,
        )?;
        visible.values_mut().resize(target.sample_count, 0);
        for row in 0..target.height {
            let source = row * plane.layout.width;
            let destination = row * target.width;
            visible.values_mut()[destination..destination + target.width]
                .copy_from_slice(&plane.samples[source..source + target.width]);
        }
        let previous = std::mem::replace(&mut plane.samples, visible.into_vec());
        let previous_ticket = allocation.take_plane_ticket(index)?;
        drop(previous);
        let mut previous_ticket = previous_ticket;
        budget.release_token(&mut previous_ticket)?;
        allocation.samples[index] = Some(candidate);
        plane.layout = *target;
    }
    Ok(())
}

pub(crate) fn crop_frame_buffers_to_plan(
    buffers: &mut FrameBuffers,
    plan: &FrameDecodePlan,
) -> Result<(), DecoderError> {
    if buffers.planes.len() != plan.planes.len() {
        return Err(DecoderError::InvalidParam(
            "AV1 frame buffer plane count does not match decode plan".to_string(),
        ));
    }
    if buffers
        .planes
        .iter()
        .zip(&plan.planes)
        .all(|(plane, target_layout)| plane.layout == *target_layout)
    {
        return Ok(());
    }
    for (plane, target_layout) in buffers.planes.iter_mut().zip(&plan.planes) {
        if plane.layout.width < target_layout.width || plane.layout.height < target_layout.height {
            return Err(DecoderError::InvalidParam(
                "AV1 coded plane is smaller than the visible frame".to_string(),
            ));
        }
        let mut visible = vec![0; target_layout.sample_count];
        for row in 0..target_layout.height {
            let source = row * plane.layout.width;
            let target = row * target_layout.width;
            visible[target..target + target_layout.width]
                .copy_from_slice(&plane.samples[source..source + target_layout.width]);
        }
        plane.layout = *target_layout;
        plane.samples = visible;
    }
    Ok(())
}

fn build_plane_layouts(
    sequence: &SequenceHeader,
    width: usize,
    height: usize,
) -> Result<Vec<PlaneLayout>, DecoderError> {
    build_plane_layouts_for_geometry(sequence, width, height)
}

/// Computes the visible plane geometry without allocating any sample buffers.
///
/// Native bounded decoding uses the same checked arithmetic as the ordinary
/// decode-plan builder before it resumes parsing or copies tile payloads.
fn build_plane_layouts_for_geometry(
    sequence: &SequenceHeader,
    width: usize,
    height: usize,
) -> Result<Vec<PlaneLayout>, DecoderError> {
    let plane_count = if sequence.color_config.monochrome {
        1
    } else {
        3
    };
    let mut planes = Vec::with_capacity(plane_count);
    for plane in 0..plane_count {
        if let Some(layout) = plane_layout_for_geometry(sequence, width, height, plane)? {
            planes.push(layout);
        }
    }
    Ok(planes)
}

/// Returns one plane's checked geometry without allocating a container.  The
/// native prefix preflight uses this fixed-slot form; the legacy decode-plan
/// builder above remains a Vec-producing adapter.
pub(crate) fn plane_layout_for_geometry(
    sequence: &SequenceHeader,
    width: usize,
    height: usize,
    plane: usize,
) -> Result<Option<PlaneLayout>, DecoderError> {
    let plane_count = if sequence.color_config.monochrome {
        1
    } else {
        3
    };
    if plane >= plane_count {
        return Ok(None);
    }
    let subsampling_x = if plane == 0 {
        0
    } else {
        sequence.color_config.subsampling_x as u8
    };
    let subsampling_y = if plane == 0 {
        0
    } else {
        sequence.color_config.subsampling_y as u8
    };
    let plane_width = round_shift_usize(width, subsampling_x);
    let plane_height = round_shift_usize(height, subsampling_y);
    let sample_count = plane_width
        .checked_mul(plane_height)
        .ok_or_else(|| DecoderError::InvalidParam("AV1 plane sample count overflow".to_string()))?;
    Ok(Some(PlaneLayout {
        plane: plane as u8,
        width: plane_width,
        height: plane_height,
        subsampling_x,
        subsampling_y,
        sample_count,
    }))
}

fn build_tile_plans(
    frame: &FrameHeader,
    tile_group: &TileGroup,
    width: usize,
    height: usize,
    superblock_mi: u32,
) -> Result<Vec<TileDecodePlan>, DecoderError> {
    let tile_cols = frame.tile_info.tile_cols;
    if tile_cols == 0 || frame.tile_info.tile_rows == 0 {
        return Err(DecoderError::Bitstream(
            "AV1 tile grid must be non-zero".to_string(),
        ));
    }

    let mut plans = Vec::with_capacity(tile_group.tiles.len());
    populate_tile_plans(&mut plans, frame, tile_group, width, height, superblock_mi)?;
    Ok(plans)
}

fn populate_tile_plans(
    plans: &mut Vec<TileDecodePlan>,
    frame: &FrameHeader,
    tile_group: &TileGroup,
    width: usize,
    height: usize,
    superblock_mi: u32,
) -> Result<(), DecoderError> {
    if plans.capacity().saturating_sub(plans.len()) < tile_group.tiles.len() {
        return Err(DecoderError::InvalidParam(
            "AV1 tile decode plan capacity is too small".to_string(),
        ));
    }
    let tile_cols = frame.tile_info.tile_cols;
    if tile_cols == 0 || frame.tile_info.tile_rows == 0 {
        return Err(DecoderError::Bitstream(
            "AV1 tile grid must be non-zero".to_string(),
        ));
    }
    for payload in &tile_group.tiles {
        let tile_col = payload.tile_id % tile_cols;
        let tile_row = payload.tile_id / tile_cols;
        let mi_col_start = *frame
            .tile_info
            .mi_col_starts
            .get(tile_col as usize)
            .ok_or_else(|| DecoderError::Bitstream("AV1 tile column is invalid".to_string()))?;
        let mi_col_end = *frame
            .tile_info
            .mi_col_starts
            .get(tile_col as usize + 1)
            .ok_or_else(|| DecoderError::Bitstream("AV1 tile column end is invalid".to_string()))?;
        let mi_row_start = *frame
            .tile_info
            .mi_row_starts
            .get(tile_row as usize)
            .ok_or_else(|| DecoderError::Bitstream("AV1 tile row is invalid".to_string()))?;
        let mi_row_end = *frame
            .tile_info
            .mi_row_starts
            .get(tile_row as usize + 1)
            .ok_or_else(|| DecoderError::Bitstream("AV1 tile row end is invalid".to_string()))?;
        let pixel_x = ((mi_col_start as usize) << 2).min(width);
        let pixel_y = ((mi_row_start as usize) << 2).min(height);
        let pixel_end_x = ((mi_col_end as usize) << 2).min(width);
        let pixel_end_y = ((mi_row_end as usize) << 2).min(height);
        plans.push(TileDecodePlan {
            tile_id: payload.tile_id,
            tile_col,
            tile_row,
            sb_col_start: mi_col_start / superblock_mi,
            sb_col_end: round_shift_u32(mi_col_end, superblock_mi.trailing_zeros() as u8),
            sb_row_start: mi_row_start / superblock_mi,
            sb_row_end: round_shift_u32(mi_row_end, superblock_mi.trailing_zeros() as u8),
            mi_col_start,
            mi_col_end,
            mi_row_start,
            mi_row_end,
            pixel_x,
            pixel_y,
            pixel_width: pixel_end_x.saturating_sub(pixel_x),
            pixel_height: pixel_end_y.saturating_sub(pixel_y),
            payload_offset: payload.offset,
            payload_len: payload.len,
        });
    }
    Ok(())
}

fn round_shift_usize(value: usize, shift: u8) -> usize {
    (value + ((1usize << shift) - 1)) >> shift
}

fn round_shift_u32(value: u32, shift: u8) -> u32 {
    (value + ((1u32 << shift) - 1)) >> shift
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocation::test_fresh_replacement;
    use crate::av1::tile_group::TilePayload;
    use crate::limits::NativeDecodeLimits;

    fn native_limits(live: usize) -> NativeDecodeLimits {
        NativeDecodeLimits::new(
            1 << 20,
            64,
            64,
            4096,
            1 << 20,
            1 << 20,
            1 << 20,
            8,
            8,
            1,
            1,
            1,
        )
        .with_max_live_allocation_bytes(live)
        .unwrap()
    }

    fn native_crop_plan(width: usize, height: usize) -> FrameDecodePlan {
        FrameDecodePlan {
            width,
            height,
            upscaled_width: width,
            render_width: width,
            render_height: height,
            bit_depth: 8,
            base_q_idx: 0,
            tx_mode: TxMode::Largest,
            superblock_size: 64,
            superblock_cols: 1,
            superblock_rows: 1,
            uses_cdef: false,
            uses_restoration: false,
            planes: vec![PlaneLayout {
                plane: 0,
                width,
                height,
                subsampling_x: 0,
                subsampling_y: 0,
                sample_count: width * height,
            }],
            tiles: Vec::new(),
        }
    }

    #[test]
    fn native_crop_accounts_coded_and_visible_peak_then_retires_coded_owner() {
        let plan = native_crop_plan(5, 5);
        let limits = native_limits(1 << 20);
        let mut budget = DecodeBudget::new(limits.max_live_allocation_bytes());
        let (mut buffers, mut owners) =
            alloc_coded_frame_buffers_with_budget(&plan, &mut budget, &limits).unwrap();
        let coded_live = budget.accounting().aggregate_live;
        assert_eq!(buffers.planes[0].layout.width, 8);

        crop_native_frame_buffers_to_plan(&mut buffers, &mut owners, &plan, &mut budget, &limits)
            .unwrap();

        let final_live = budget.accounting().aggregate_live;
        assert_eq!(buffers.planes[0].layout, plan.planes[0]);
        assert!(budget.accounting().aggregate_peak > coded_live);
        assert!(budget.accounting().aggregate_peak > final_live);
        owners.release(&mut budget).unwrap();
        assert_eq!(budget.accounting().frame_live, 0);
    }

    #[test]
    fn native_crop_skips_replacement_when_coded_and_visible_layouts_match() {
        let plan = native_crop_plan(8, 8);
        let limits = native_limits(1 << 20);
        let mut budget = DecodeBudget::new(limits.max_live_allocation_bytes());
        let (mut buffers, mut owners) =
            alloc_coded_frame_buffers_with_budget(&plan, &mut budget, &limits).unwrap();
        let pointer = buffers.planes[0].samples.as_ptr();
        let live = budget.accounting().aggregate_live;

        crop_native_frame_buffers_to_plan(&mut buffers, &mut owners, &plan, &mut budget, &limits)
            .unwrap();

        assert_eq!(buffers.planes[0].samples.as_ptr(), pointer);
        assert_eq!(budget.accounting().aggregate_live, live);
        owners.release(&mut budget).unwrap();
    }

    #[test]
    fn native_crop_actual_capacity_keeps_coded_owner_live_one_under_then_retries() {
        let plan = native_crop_plan(5, 5);
        let limits = native_limits(1 << 20);
        let mut budget = DecodeBudget::new(None);
        let (mut master, mut master_owners) =
            alloc_coded_frame_buffers_with_budget(&plan, &mut budget, &limits).unwrap();
        let master_live = budget.accounting().aggregate_live;

        let mut exact_candidate = Some(Vec::<u16>::with_capacity(26));
        let exact_bytes = exact_candidate.as_ref().unwrap().capacity() * std::mem::size_of::<u16>();
        budget.set_max_live_bytes_for_test(master_live + exact_bytes);
        crop_native_frame_buffers_to_plan_with_maker(
            &mut master,
            &mut master_owners,
            &plan,
            &mut budget,
            &limits,
            move |_, label| test_fresh_replacement(exact_candidate.take().unwrap(), label),
        )
        .unwrap();
        assert_eq!(master.planes[0].layout, plan.planes[0]);
        master_owners.release(&mut budget).unwrap();
        assert_eq!(budget.accounting().frame_live, 0);

        let (mut master, mut master_owners) =
            alloc_coded_frame_buffers_with_budget(&plan, &mut budget, &limits).unwrap();
        let master_pointer = master.planes[0].samples.as_ptr();
        let master_ticket = master_owners.samples[0]
            .as_ref()
            .unwrap()
            .charged_capacity_bytes;
        let master_live = budget.accounting().aggregate_live;

        let mut under_candidate = Some(Vec::<u16>::with_capacity(26));
        let under_bytes = under_candidate.as_ref().unwrap().capacity() * std::mem::size_of::<u16>();
        assert!(under_bytes > plan.planes[0].sample_count * std::mem::size_of::<u16>());
        budget.set_max_live_bytes_for_test(master_live + under_bytes - 1);
        let observation = crate::test_allocation_observer::Observation::begin(1, false);
        observation.track(under_candidate.as_ref().unwrap());
        observation.track_raw_slot(
            0,
            under_candidate.as_ref().unwrap().as_ptr().cast(),
            under_bytes,
        );
        let error = crop_native_frame_buffers_to_plan_with_maker(
            &mut master,
            &mut master_owners,
            &plan,
            &mut budget,
            &limits,
            move |_, label| test_fresh_replacement(under_candidate.take().unwrap(), label),
        )
        .unwrap_err();
        assert!(
            matches!(error, DecoderError::InvalidParam(message) if message.contains("live allocation"))
        );
        assert_eq!(budget.accounting().aggregate_live, master_live);
        assert_eq!(master.planes[0].samples.as_ptr(), master_pointer);
        assert_eq!(
            master_owners.samples[0]
                .as_ref()
                .unwrap()
                .charged_capacity_bytes,
            master_ticket
        );
        assert_eq!(observation.drops(), 1);
        assert_eq!(observation.restore_count(), 1);
        assert!(observation.restore_drop_snapshots()[0] >= 1);
        drop(observation);

        let mut retry_candidate = Some(Vec::<u16>::with_capacity(26));
        let retry_bytes = retry_candidate.as_ref().unwrap().capacity() * std::mem::size_of::<u16>();
        budget.set_max_live_bytes_for_test(master_live + retry_bytes);
        crop_native_frame_buffers_to_plan_with_maker(
            &mut master,
            &mut master_owners,
            &plan,
            &mut budget,
            &limits,
            move |_, label| test_fresh_replacement(retry_candidate.take().unwrap(), label),
        )
        .unwrap();
        master_owners.release(&mut budget).unwrap();
        assert_eq!(budget.accounting().frame_live, 0);
    }

    #[test]
    fn native_alpha_outer_growth_uses_combined_peak_and_preserves_both_frames_on_one_under() {
        fn setup() -> (
            FrameBuffers,
            NativeFrameAllocation,
            FrameBuffers,
            NativeFrameAllocation,
            DecodeBudget,
            NativeDecodeLimits,
        ) {
            let plan = native_crop_plan(8, 8);
            let limits = native_limits(1 << 20);
            let mut budget = DecodeBudget::new(None);
            let (master, master_owners) =
                alloc_coded_frame_buffers_with_budget(&plan, &mut budget, &limits).unwrap();
            let (alpha, alpha_owners) =
                alloc_coded_frame_buffers_with_budget(&plan, &mut budget, &limits).unwrap();
            (master, master_owners, alpha, alpha_owners, budget, limits)
        }

        let (mut master, mut master_owners, alpha, mut alpha_owners, mut budget, _limits) = setup();
        let baseline = budget.accounting().aggregate_live;
        let exact_candidate = Vec::<PlaneBuffer>::with_capacity(3);
        let exact_bytes = exact_candidate.capacity() * std::mem::size_of::<PlaneBuffer>();
        budget.set_max_live_bytes_for_test(baseline + exact_bytes);
        master_owners
            .grow_outer_for_alpha_with_maker(&mut master.planes, &mut budget, move |_, label| {
                test_fresh_replacement(exact_candidate, label)
            })
            .unwrap();
        master_owners.release(&mut budget).unwrap();
        alpha_owners.release(&mut budget).unwrap();
        drop(master);
        drop(alpha);
        assert_eq!(budget.accounting().frame_live, 0);

        let (mut master, mut master_owners, alpha, mut alpha_owners, mut budget, _limits) = setup();
        let baseline = budget.accounting().aggregate_live;
        let master_plane_pointer = master.planes.as_ptr();
        let master_sample_pointer = master.planes[0].samples.as_ptr();
        let alpha_plane_pointer = alpha.planes.as_ptr();
        let alpha_sample_pointer = alpha.planes[0].samples.as_ptr();
        let old_outer_bytes = master_owners.outer.charged_capacity_bytes;
        let old_outer_class = master_owners.outer.class;
        let alpha_outer_bytes = alpha_owners.outer.charged_capacity_bytes;
        let alpha_outer_class = alpha_owners.outer.class;
        let under_candidate = Vec::<PlaneBuffer>::with_capacity(3);
        let under_bytes = under_candidate.capacity() * std::mem::size_of::<PlaneBuffer>();
        assert!(under_bytes > 2 * std::mem::size_of::<PlaneBuffer>());
        budget.set_max_live_bytes_for_test(baseline + under_bytes - 1);
        let observation = crate::test_allocation_observer::Observation::begin(1, false);
        observation.track(&under_candidate);
        observation.track_raw_slot(0, under_candidate.as_ptr().cast(), under_bytes);
        let error = master_owners.grow_outer_for_alpha_with_maker(
            &mut master.planes,
            &mut budget,
            move |_, label| test_fresh_replacement(under_candidate, label),
        );
        let error = error.unwrap_err();
        assert!(
            matches!(error, DecoderError::InvalidParam(message) if message.contains("live allocation"))
        );
        assert_eq!(budget.accounting().aggregate_live, baseline);
        assert_eq!(master.planes.as_ptr(), master_plane_pointer);
        assert_eq!(master.planes[0].samples.as_ptr(), master_sample_pointer);
        assert_eq!(alpha.planes.as_ptr(), alpha_plane_pointer);
        assert_eq!(alpha.planes[0].samples.as_ptr(), alpha_sample_pointer);
        assert_eq!(master_owners.outer.charged_capacity_bytes, old_outer_bytes);
        assert_eq!(master_owners.outer.class, old_outer_class);
        assert_eq!(alpha_owners.outer.charged_capacity_bytes, alpha_outer_bytes);
        assert_eq!(alpha_owners.outer.class, alpha_outer_class);
        assert_eq!(observation.drops(), 1);
        assert_eq!(observation.restore_count(), 1);
        assert!(observation.restore_drop_snapshots()[0] >= 1);
        drop(observation);

        let retry_candidate = Vec::<PlaneBuffer>::with_capacity(3);
        let retry_bytes = retry_candidate.capacity() * std::mem::size_of::<PlaneBuffer>();
        budget.set_max_live_bytes_for_test(baseline + retry_bytes);
        master_owners
            .grow_outer_for_alpha_with_maker(&mut master.planes, &mut budget, move |_, label| {
                test_fresh_replacement(retry_candidate, label)
            })
            .unwrap();
        master_owners.release(&mut budget).unwrap();
        alpha_owners.release(&mut budget).unwrap();
        drop(master);
        drop(alpha);
        assert_eq!(budget.accounting().frame_live, 0);
    }

    #[test]
    fn rejects_frame_buffer_allocation_above_resource_limit() {
        let plan = FrameDecodePlan {
            width: 1,
            height: 1,
            upscaled_width: 1,
            render_width: 1,
            render_height: 1,
            bit_depth: 8,
            base_q_idx: 0,
            tx_mode: TxMode::Largest,
            superblock_size: 64,
            superblock_cols: 1,
            superblock_rows: 1,
            uses_cdef: false,
            uses_restoration: false,
            planes: vec![PlaneLayout {
                plane: 0,
                width: 1,
                height: 1,
                subsampling_x: 0,
                subsampling_y: 0,
                sample_count: MAX_PLANE_SAMPLE_ALLOCATION + 1,
            }],
            tiles: Vec::new(),
        };

        let err = alloc_frame_buffers(&plan).unwrap_err();

        assert!(
            matches!(err, DecoderError::InvalidParam(message) if message.contains("resource limit"))
        );
    }

    #[test]
    fn rejects_partial_tile_group_for_still_decode() {
        let tile_group = TileGroup {
            start_tile: 1,
            end_tile: 1,
            data_start_offset: 0,
            tiles: vec![TilePayload {
                tile_id: 1,
                offset: 0,
                len: 0,
            }],
        };

        let err = validate_complete_tile_group(2, 1, &tile_group).unwrap_err();

        assert!(
            matches!(err, DecoderError::Unsupported(message) if message.contains("partial tile"))
        );
    }

    #[test]
    fn coded_buffers_preserve_aligned_mi_padding_until_crop() {
        let plan = FrameDecodePlan {
            width: 900,
            height: 900,
            upscaled_width: 900,
            render_width: 900,
            render_height: 900,
            bit_depth: 8,
            base_q_idx: 0,
            tx_mode: TxMode::Largest,
            superblock_size: 64,
            superblock_cols: 15,
            superblock_rows: 15,
            uses_cdef: false,
            uses_restoration: false,
            planes: vec![PlaneLayout {
                plane: 0,
                width: 900,
                height: 900,
                subsampling_x: 0,
                subsampling_y: 0,
                sample_count: 900 * 900,
            }],
            tiles: Vec::new(),
        };
        let mut buffers = alloc_coded_frame_buffers(&plan).unwrap();
        assert_eq!(buffers.planes[0].layout.width, 904);
        assert_eq!(buffers.planes[0].layout.height, 904);
        buffers.planes[0].samples[899] = 7;
        buffers.planes[0].samples[900] = 9;

        crop_frame_buffers_to_plan(&mut buffers, &plan).unwrap();

        assert_eq!(buffers.planes[0].layout, plan.planes[0]);
        assert_eq!(buffers.planes[0].samples.len(), 900 * 900);
        assert_eq!(buffers.planes[0].samples[899], 7);
    }

    #[test]
    fn crop_skips_allocation_when_coded_layout_is_already_visible() {
        let plan = FrameDecodePlan {
            width: 2,
            height: 1,
            upscaled_width: 2,
            render_width: 2,
            render_height: 1,
            bit_depth: 8,
            base_q_idx: 0,
            tx_mode: TxMode::Largest,
            superblock_size: 64,
            superblock_cols: 1,
            superblock_rows: 1,
            uses_cdef: false,
            uses_restoration: false,
            planes: vec![PlaneLayout {
                plane: 0,
                width: 2,
                height: 1,
                subsampling_x: 0,
                subsampling_y: 0,
                sample_count: 2,
            }],
            tiles: Vec::new(),
        };
        let mut buffers = FrameBuffers {
            width: 2,
            height: 1,
            planes: vec![PlaneBuffer {
                layout: plan.planes[0],
                samples: vec![11, 29],
            }],
        };
        let samples = buffers.planes[0].samples.as_ptr();

        crop_frame_buffers_to_plan(&mut buffers, &plan).unwrap();

        assert_eq!(buffers.planes[0].samples.as_ptr(), samples);
        assert_eq!(buffers.planes[0].samples, vec![11, 29]);
    }

    #[test]
    fn superres_horizontal_resizes_plane_and_preserves_range() {
        let mut buffers = FrameBuffers {
            width: 4,
            height: 1,
            planes: vec![PlaneBuffer {
                layout: PlaneLayout {
                    plane: 0,
                    width: 4,
                    height: 1,
                    subsampling_x: 0,
                    subsampling_y: 0,
                    sample_count: 4,
                },
                samples: vec![0, 256, 512, 768],
            }],
        };
        apply_superres_horizontal(&mut buffers, 8, 10).unwrap();
        assert_eq!(buffers.width, 8);
        assert_eq!(buffers.planes[0].layout.width, 8);
        assert_eq!(buffers.planes[0].samples.len(), 8);
        assert!(
            buffers.planes[0]
                .samples
                .iter()
                .all(|sample| *sample <= 1023)
        );
        assert!(
            buffers.planes[0]
                .samples
                .first()
                .copied()
                .unwrap_or_default()
                < 256
        );
        assert!(
            buffers.planes[0]
                .samples
                .last()
                .copied()
                .unwrap_or_default()
                > 512
        );
    }
}
