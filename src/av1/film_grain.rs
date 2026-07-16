use super::decode::FrameBuffers;
use super::frame::FilmGrainParams;
use super::sequence::ColorConfig;

const AR_PAD: usize = 3;
const GRAIN_WIDTH: usize = 82;
const GRAIN_HEIGHT: usize = 73;
const SUB_GRAIN_WIDTH: usize = 44;
const SUB_GRAIN_HEIGHT: usize = 38;
const BLOCK_SIZE: usize = 32;

// The AV1 Gaussian sequence is the normative 11-bit grain source. The table
// is filled below from the reference sequence; keeping it in this module
// avoids a runtime dependency on an external AV1 implementation.
const GAUSSIAN_SEQUENCE: [i16; 2048] = [
    56, 568, -180, 172, 124, -84, 172, -64, -900, 24, 820, 224, 1248, 996, 272, -8, -916, -388,
    -732, -104, -188, 800, 112, -652, -320, -376, 140, -252, 492, -168, 44, -788, 588, -584, 500,
    -228, 12, 680, 272, -476, 972, -100, 652, 368, 432, -196, -720, -192, 1000, -332, 652, -136,
    -552, -604, -4, 192, -220, -136, 1000, -52, 372, -96, -624, 124, -24, 396, 540, -12, -104, 640,
    464, 244, -208, -84, 368, -528, -740, 248, -968, -848, 608, 376, -60, -292, -40, -156, 252,
    -292, 248, 224, -280, 400, -244, 244, -60, 76, -80, 212, 532, 340, 128, -36, 824, -352, -60,
    -264, -96, -612, 416, -704, 220, -204, 640, -160, 1220, -408, 900, 336, 20, -336, -96, -792,
    304, 48, -28, -1232, -1172, -448, 104, -292, -520, 244, 60, -948, 0, -708, 268, 108, 356, -548,
    488, -344, -136, 488, -196, -224, 656, -236, -1128, 60, 4, 140, 276, -676, -376, 168, -108,
    464, 8, 564, 64, 240, 308, -300, -400, -456, -136, 56, 120, -408, -116, 436, 504, -232, 328,
    844, -164, -84, 784, -168, 232, -224, 348, -376, 128, 568, 96, -1244, -288, 276, 848, 832,
    -360, 656, 464, -384, -332, -356, 728, -388, 160, -192, 468, 296, 224, 140, -776, -100, 280, 4,
    196, 44, -36, -648, 932, 16, 1428, 28, 528, 808, 772, 20, 268, 88, -332, -284, 124, -384, -448,
    208, -228, -1044, -328, 660, 380, -148, -300, 588, 240, 540, 28, 136, -88, -436, 256, 296,
    -1000, 1400, 0, -48, 1056, -136, 264, -528, -1108, 632, -484, -592, -344, 796, 124, -668, -768,
    388, 1296, -232, -188, -200, -288, -4, 308, 100, -168, 256, -500, 204, -508, 648, -136, 372,
    -272, -120, -1004, -552, -548, -384, 548, -296, 428, -108, -8, -912, -324, -224, -88, -112,
    -220, -100, 996, -796, 548, 360, -216, 180, 428, -200, -212, 148, 96, 148, 284, 216, -412,
    -320, 120, -300, -384, -604, -572, -332, -8, -180, -176, 696, 116, -88, 628, 76, 44, -516, 240,
    -208, -40, 100, -592, 344, -308, -452, -228, 20, 916, -1752, -136, -340, -804, 140, 40, 512,
    340, 248, 184, -492, 896, -156, 932, -628, 328, -688, -448, -616, -752, -100, 560, -1020, 180,
    -800, -64, 76, 576, 1068, 396, 660, 552, -108, -28, 320, -628, 312, -92, -92, -472, 268, 16,
    560, 516, -672, -52, 492, -100, 260, 384, 284, 292, 304, -148, 88, -152, 1012, 1064, -228, 164,
    -376, -684, 592, -392, 156, 196, -524, -64, -884, 160, -176, 636, 648, 404, -396, -436, 864,
    424, -728, 988, -604, 904, -592, 296, -224, 536, -176, -920, 436, -48, 1176, -884, 416, -776,
    -824, -884, 524, -548, -564, -68, -164, -96, 692, 364, -692, -1012, -68, 260, -480, 876, -1116,
    452, -332, -352, 892, -1088, 1220, -676, 12, -292, 244, 496, 372, -32, 280, 200, 112, -440,
    -96, 24, -644, -184, 56, -432, 224, -980, 272, -260, 144, -436, 420, 356, 364, -528, 76, 172,
    -744, -368, 404, -752, -416, 684, -688, 72, 540, 416, 92, 444, 480, -72, -1416, 164, -1172,
    -68, 24, 424, 264, 1040, 128, -912, -524, -356, 64, 876, -12, 4, -88, 532, 272, -524, 320, 276,
    -508, 940, 24, -400, -120, 756, 60, 236, -412, 100, 376, -484, 400, -100, -740, -108, -260,
    328, -268, 224, -200, -416, 184, -604, -564, -20, 296, 60, 892, -888, 60, 164, 68, -760, 216,
    -296, 904, -336, -28, 404, -356, -568, -208, -1480, -512, 296, 328, -360, -164, -1560, -776,
    1156, -428, 164, -504, -112, 120, -216, -148, -264, 308, 32, 64, -72, 72, 116, 176, -64, -272,
    460, -536, -784, -280, 348, 108, -752, -132, 524, -540, -776, 116, -296, -1196, -288, -560,
    1040, -472, 116, -848, -1116, 116, 636, 696, 284, -176, 1016, 204, -864, -648, -248, 356, 972,
    -584, -204, 264, 880, 528, -24, -184, 116, 448, -144, 828, 524, 212, -212, 52, 12, 200, 268,
    -488, -404, -880, 824, -672, -40, 908, -248, 500, 716, -576, 492, -576, 16, 720, -108, 384,
    124, 344, 280, 576, -500, 252, 104, -308, 196, -188, -8, 1268, 296, 1032, -1196, 436, 316, 372,
    -432, -200, -660, 704, -224, 596, -132, 268, 32, -452, 884, 104, -1008, 424, -1348, -280, 4,
    -1168, 368, 476, 696, 300, -8, 24, 180, -592, -196, 388, 304, 500, 724, -160, 244, -84, 272,
    -256, -420, 320, 208, -144, -156, 156, 364, 452, 28, 540, 316, 220, -644, -248, 464, 72, 360,
    32, -388, 496, -680, -48, 208, -116, -408, 60, -604, -392, 548, -840, 784, -460, 656, -544,
    -388, -264, 908, -800, -628, -612, -568, 572, -220, 164, 288, -16, -308, 308, -112, -636, -760,
    280, -668, 432, 364, 240, -196, 604, 340, 384, 196, 592, -44, -500, 432, -580, -132, 636, -76,
    392, 4, -412, 540, 508, 328, -356, -36, 16, -220, -64, -248, -60, 24, -192, 368, 1040, 92, -24,
    -1044, -32, 40, 104, 148, 192, -136, -520, 56, -816, -224, 732, 392, 356, 212, -80, -424,
    -1008, -324, 588, -1496, 576, 460, -816, -848, 56, -580, -92, -1372, -112, -496, 200, 364, 52,
    -140, 48, -48, -60, 84, 72, 40, 132, -356, -268, -104, -284, -404, 732, -520, 164, -304, -540,
    120, 328, -76, -460, 756, 388, 588, 236, -436, -72, -176, -404, -316, -148, 716, -604, 404,
    -72, -88, -888, -68, 944, 88, -220, -344, 960, 472, 460, -232, 704, 120, 832, -228, 692, -508,
    132, -476, 844, -748, -364, -44, 1116, -1104, -1056, 76, 428, 552, -692, 60, 356, 96, -384,
    -188, -612, -576, 736, 508, 892, 352, -1132, 504, -24, -352, 324, 332, -600, -312, 292, 508,
    -144, -8, 484, 48, 284, -260, -240, 256, -100, -292, -204, -44, 472, -204, 908, -188, -1000,
    -256, 92, 1164, -392, 564, 356, 652, -28, -884, 256, 484, -192, 760, -176, 376, -524, -452,
    -436, 860, -736, 212, 124, 504, -476, 468, 76, -472, 552, -692, -944, -620, 740, -240, 400,
    132, 20, 192, -196, 264, -668, -1012, -60, 296, -316, -828, 76, -156, 284, -768, -448, -832,
    148, 248, 652, 616, 1236, 288, -328, -400, -124, 588, 220, 520, -696, 1032, 768, -740, -92,
    -272, 296, 448, -464, 412, -200, 392, 440, -200, 264, -152, -260, 320, 1032, 216, 320, -8, -64,
    156, -1016, 1084, 1172, 536, 484, -432, 132, 372, -52, -256, 84, 116, -352, 48, 116, 304, -384,
    412, 924, -300, 528, 628, 180, 648, 44, -980, -220, 1320, 48, 332, 748, 524, -268, -720, 540,
    -276, 564, -344, -208, -196, 436, 896, 88, -392, 132, 80, -964, -288, 568, 56, -48, -456, 888,
    8, 552, -156, -292, 948, 288, 128, -716, -292, 1192, -152, 876, 352, -600, -260, -812, -468,
    -28, -120, -32, -44, 1284, 496, 192, 464, 312, -76, -516, -380, -456, -1012, -48, 308, -156,
    36, 492, -156, -808, 188, 1652, 68, -120, -116, 316, 160, -140, 352, 808, -416, 592, 316, -480,
    56, 528, -204, -568, 372, -232, 752, -344, 744, -4, 324, -416, -600, 768, 268, -248, -88, -132,
    -420, -432, 80, -288, 404, -316, -1216, -588, 520, -108, 92, -320, 368, -480, -216, -92, 1688,
    -300, 180, 1020, -176, 820, -68, -228, -260, 436, -904, 20, 40, -508, 440, -736, 312, 332, 204,
    760, -372, 728, 96, -20, -632, -520, -560, 336, 1076, -64, -532, 776, 584, 192, 396, -728,
    -520, 276, -188, 80, -52, -612, -252, -48, 648, 212, -688, 228, -52, -260, 428, -412, -272,
    -404, 180, 816, -796, 48, 152, 484, -88, -216, 988, 696, 188, -528, 648, -116, -180, 316, 476,
    12, -564, 96, 476, -252, -364, -376, -392, 556, -256, -576, 260, -352, 120, -16, -136, -260,
    -492, 72, 556, 660, 580, 616, 772, 436, 424, -32, -324, -1268, 416, -324, -80, 920, 160, 228,
    724, 32, -516, 64, 384, 68, -128, 136, 240, 248, -204, -68, 252, -932, -120, -480, -628, -84,
    192, 852, -404, -288, -132, 204, 100, 168, -68, -196, -868, 460, 1080, 380, -80, 244, 0, 484,
    -888, 64, 184, 352, 600, 460, 164, 604, -196, 320, -64, 588, -184, 228, 12, 372, 48, -848,
    -344, 224, 208, -200, 484, 128, -20, 272, -468, -840, 384, 256, -720, -520, -464, -580, 112,
    -120, 644, -356, -208, -608, -528, 704, 560, -424, 392, 828, 40, 84, 200, -152, 0, -144, 584,
    280, -120, 80, -556, -972, -196, -472, 724, 80, 168, -32, 88, 160, -688, 0, 160, 356, 372,
    -776, 740, -128, 676, -248, -480, 4, -364, 96, 544, 232, -1032, 956, 236, 356, 20, -40, 300,
    24, -676, -596, 132, 1120, -104, 532, -1096, 568, 648, 444, 508, 380, 188, -376, -604, 1488,
    424, 24, 756, -220, -192, 716, 120, 920, 688, 168, 44, -460, 568, 284, 1144, 1160, 600, 424,
    888, 656, -356, -320, 220, 316, -176, -724, -188, -816, -628, -348, -228, -380, 1012, -452,
    -660, 736, 928, 404, -696, -72, -268, -892, 128, 184, -344, -780, 360, 336, 400, 344, 428, 548,
    -112, 136, -228, -216, -820, -516, 340, 92, -136, 116, -300, 376, -244, 100, -316, -520, -284,
    -12, 824, 164, -548, -180, -128, 116, -924, -828, 268, -368, -580, 620, 192, 160, 0, -1676,
    1068, 424, -56, -360, 468, -156, 720, 288, -528, 556, -364, 548, -148, 504, 316, 152, -648,
    -620, -684, -24, -376, -384, -108, -920, -1032, 768, 180, -264, -508, -1268, -260, -60, 300,
    -240, 988, 724, -376, -576, -212, -736, 556, 192, 1092, -620, -880, 376, -56, -4, -216, -32,
    836, 268, 396, 1332, 864, -600, 100, 56, -412, -92, 356, 180, 884, -468, -436, 292, -388, -804,
    -704, -840, 368, -348, 140, -724, 1536, 940, 372, 112, -372, 436, -480, 1136, 296, -32, -228,
    132, -48, -220, 868, -1016, -60, -1044, -464, 328, 916, 244, 12, -736, -296, 360, 468, -376,
    -108, -92, 788, 368, -56, 544, 400, -672, -420, 728, 16, 320, 44, -284, -380, -796, 488, 132,
    204, -596, -372, 88, -152, -908, -636, -572, -624, -116, -692, -200, -56, 276, -88, 484, -324,
    948, 864, 1000, -456, -184, -276, 292, -296, 156, 676, 320, 160, 908, -84, -1236, -288, -116,
    260, -372, -644, 732, -756, -96, 84, 344, -520, 348, -688, 240, -84, 216, -1044, -136, -676,
    -396, -1500, 960, -40, 176, 168, 1516, 420, -504, -344, -364, -360, 1216, -940, -380, -212,
    252, -660, -708, 484, -444, -152, 928, -120, 1112, 476, -260, 560, -148, -344, 108, -196, 228,
    -288, 504, 560, -328, -88, 288, -1008, 460, -228, 468, -836, -196, 76, 388, 232, 412, -1168,
    -716, -644, 756, -172, -356, -504, 116, 432, 528, 48, 476, -168, -608, 448, 160, -532, -272,
    28, -676, -12, 828, 980, 456, 520, 104, -104, 256, -344, -4, -28, -368, -52, -524, -572, -556,
    -200, 768, 1124, -208, -512, 176, 232, 248, -148, -888, 604, -600, -304, 804, -156, -212, 488,
    -192, -804, -256, 368, -360, -916, -328, 228, -240, -448, -472, 856, -556, -364, 572, -12,
    -156, -368, -340, 432, 252, -752, -152, 288, 268, -580, -848, -592, 108, -76, 244, 312, -716,
    592, -80, 436, 360, 4, -248, 160, 516, 584, 732, 44, -468, -280, -292, -156, -588, 28, 308,
    912, 24, 124, 156, 180, -252, 944, -924, -772, -520, -428, -624, 300, -212, -1144, 32, -724,
    800, -1128, -212, -1288, -848, 180, -416, 440, 192, -576, -792, -76, -1080, 80, -532, -352,
    -132, 380, -820, 148, 1112, 128, 164, 456, 700, -924, 144, -668, -384, 648, -832, 508, 552,
    -52, -100, -656, 208, -568, 748, -88, 680, 232, 300, 192, -408, -1012, -152, -252, -268, 272,
    -876, -664, -648, -332, -136, 16, 12, 1152, -28, 332, -536, 320, -672, -460, -316, 532, -260,
    228, -40, 1052, -816, 180, 88, -496, -556, -672, -368, 428, 92, 356, 404, -408, 252, 196, -176,
    -556, 792, 268, 32, 372, 40, 96, -332, 328, 120, 372, -900, -40, 472, -264, -592, 952, 128,
    656, 112, 664, -232, 420, 4, -344, -464, 556, 244, -416, -32, 252, 0, -412, 188, -696, 508,
    -476, 324, -1096, 656, -312, 560, 264, -136, 304, 160, -64, -580, 248, 336, -720, 560, -348,
    -288, -276, -196, -500, 852, -544, -236, -1128, -992, -776, 116, 56, 52, 860, 884, 212, -12,
    168, 1020, 512, -552, 924, -148, 716, 188, 164, -340, -520, -184, 880, -152, -680, -208, -1156,
    -300, -528, -472, 364, 100, -744, -1056, -32, 540, 280, 144, -676, -32, -232, -280, -224, 96,
    568, -76, 172, 148, 148, 104, 32, -296, -32, 788, -80, 32, -16, 280, 288, 944, 428, -484,
];

pub(crate) fn apply(buffers: &mut FrameBuffers, color: &ColorConfig, params: &FilmGrainParams) {
    if buffers.planes.is_empty() {
        return;
    }
    let bit_depth = color.bit_depth;
    let luma_source = buffers.planes[0].samples.clone();
    let luma_width = buffers.planes[0].layout.width;
    let luma_height = buffers.planes[0].layout.height;
    let y_lut = generate_luma_lut(params, bit_depth);
    let y_scaling = generate_scaling(
        &params.scaling_points_y[..usize::from(params.num_y_points)],
        bit_depth,
    );
    apply_luma_plane(
        &mut buffers.planes[0].samples,
        luma_width,
        luma_height,
        &y_scaling,
        &y_lut,
        params,
        bit_depth,
    );

    if color.monochrome || buffers.planes.len() < 3 {
        return;
    }
    let sub_x = color.subsampling_x;
    let sub_y = color.subsampling_y;
    let is_identity = color
        .color_description
        .is_some_and(|description| description.matrix_coefficients == 0);
    for (plane_index, points, coeffs, mult, luma_mult, offset) in [
        (
            1usize,
            &params.scaling_points_cb,
            &params.ar_coeffs_cb,
            params.cb_mult,
            params.cb_luma_mult,
            params.cb_offset,
        ),
        (
            2usize,
            &params.scaling_points_cr,
            &params.ar_coeffs_cr,
            params.cr_mult,
            params.cr_luma_mult,
            params.cr_offset,
        ),
    ] {
        let point_count = if plane_index == 1 {
            params.num_cb_points
        } else {
            params.num_cr_points
        };
        if point_count == 0 && !params.chroma_scaling_from_luma {
            continue;
        }
        let lut = generate_chroma_lut(
            params,
            coeffs,
            bit_depth,
            plane_index == 2,
            sub_x,
            sub_y,
            &y_lut,
        );
        let scaling = if params.chroma_scaling_from_luma {
            &y_scaling
        } else {
            // Keep the owned vector alive for the duration of this iteration.
            // The branch is split below so the borrow does not escape.
            let scaling = generate_scaling(&points[..usize::from(point_count)], bit_depth);
            let plane_width = buffers.planes[plane_index].layout.width;
            let plane_height = buffers.planes[plane_index].layout.height;
            apply_chroma_plane(
                &mut buffers.planes[plane_index].samples,
                plane_width,
                plane_height,
                &luma_source,
                luma_width,
                luma_height,
                &scaling,
                &lut,
                params,
                bit_depth,
                sub_x,
                sub_y,
                is_identity,
                mult,
                luma_mult,
                offset,
            );
            continue;
        };
        let plane_width = buffers.planes[plane_index].layout.width;
        let plane_height = buffers.planes[plane_index].layout.height;
        apply_chroma_plane(
            &mut buffers.planes[plane_index].samples,
            plane_width,
            plane_height,
            &luma_source,
            luma_width,
            luma_height,
            scaling,
            &lut,
            params,
            bit_depth,
            sub_x,
            sub_y,
            is_identity,
            mult,
            luma_mult,
            offset,
        );
    }
}

fn generate_scaling(points: &[[u8; 2]], bit_depth: u8) -> Vec<i32> {
    let size = 1usize << bit_depth;
    let mut scaling = vec![0; size];
    if points.is_empty() {
        return scaling;
    }
    let shift = usize::from(bit_depth.saturating_sub(8));
    let first_x = usize::from(points[0][0]) << shift;
    scaling[..first_x.min(size)].fill(i32::from(points[0][1]));
    for pair in points.windows(2) {
        let [start, end] = [pair[0], pair[1]];
        let bx = usize::from(start[0]);
        let ex = usize::from(end[0]);
        let dx = ex.saturating_sub(bx);
        if dx == 0 {
            continue;
        }
        let delta = (i32::from(end[1]) - i32::from(start[1])) * ((0x10000 + (dx >> 1)) / dx) as i32;
        let mut d = 0x8000i32;
        for x in 0..dx {
            let index = ((bx + x) << shift).min(size - 1);
            scaling[index] = i32::from(start[1]) + (d >> 16);
            d += delta;
        }
    }
    let last_x = (usize::from(points[points.len() - 1][0]) << shift).min(size);
    scaling[last_x..].fill(i32::from(points[points.len() - 1][1]));
    if shift > 0 {
        let pad = 1usize << shift;
        let rnd = pad >> 1;
        for pair in points.windows(2) {
            let bx = usize::from(pair[0][0]) << shift;
            let ex = usize::from(pair[1][0]) << shift;
            let dx = ex.saturating_sub(bx);
            for x in (0..dx).step_by(pad) {
                let left = (bx + x).min(size - 1);
                let right = (bx + x + pad).min(size - 1);
                let range = scaling[right] - scaling[left];
                let mut r = rnd as i32;
                for n in 1..pad {
                    let index = bx + x + n;
                    if index >= size {
                        break;
                    }
                    r += range;
                    scaling[index] = scaling[left] + (r >> shift);
                }
            }
        }
    }
    scaling
}

fn random_number(bits: u8, state: &mut u32) -> i32 {
    let bit = (*state ^ (*state >> 1) ^ (*state >> 3) ^ (*state >> 12)) & 1;
    *state = (*state >> 1) | (bit << 15);
    ((*state >> (16 - bits)) & ((1 << bits) - 1)) as i32
}

fn round_shift(value: i32, shift: u8) -> i32 {
    if shift == 0 {
        value
    } else {
        (value + (1i32 << (shift - 1))) >> shift
    }
}

fn generate_luma_lut(params: &FilmGrainParams, bit_depth: u8) -> Vec<Vec<i32>> {
    let shift = 4i32 - i32::from(bit_depth.saturating_sub(8)) + i32::from(params.grain_scale_shift);
    let grain_ctr = 128i32 << bit_depth.saturating_sub(8);
    let mut lut = vec![vec![0; GRAIN_WIDTH]; GRAIN_HEIGHT];
    let mut seed = u32::from(params.random_seed);
    for row in &mut lut {
        for value in row {
            let random = random_number(11, &mut seed) as usize;
            *value = round_shift(i32::from(GAUSSIAN_SEQUENCE[random]), shift.max(0) as u8)
                .clamp(-grain_ctr, grain_ctr - 1);
        }
    }
    let lag = usize::from(params.ar_coeff_lag & 3);
    for y in 0..GRAIN_HEIGHT - AR_PAD {
        for x in 0..GRAIN_WIDTH - 2 * AR_PAD {
            let mut coeff_index = 0;
            let mut sum = 0i32;
            for dy in 0..=lag {
                for dx in 0..=(2 * lag) {
                    if dy == lag && dx == lag {
                        break;
                    }
                    let sample = lut[y + AR_PAD - lag + dy][x + AR_PAD - lag + dx];
                    sum += i32::from(params.ar_coeffs_y[coeff_index]) * sample;
                    coeff_index += 1;
                }
            }
            let index_y = y + AR_PAD;
            let index_x = x + AR_PAD;
            lut[index_y][index_x] = (lut[index_y][index_x]
                + round_shift(sum, params.ar_coeff_shift))
            .clamp(-grain_ctr, grain_ctr - 1);
        }
    }
    lut
}

fn generate_chroma_lut(
    params: &FilmGrainParams,
    coeffs: &[i16; 25],
    bit_depth: u8,
    is_cr: bool,
    sub_x: bool,
    sub_y: bool,
    luma_lut: &[Vec<i32>],
) -> Vec<Vec<i32>> {
    let width = if sub_x { SUB_GRAIN_WIDTH } else { GRAIN_WIDTH };
    let height = if sub_y {
        SUB_GRAIN_HEIGHT
    } else {
        GRAIN_HEIGHT
    };
    let shift = 4i32 - i32::from(bit_depth.saturating_sub(8)) + i32::from(params.grain_scale_shift);
    let grain_ctr = 128i32 << bit_depth.saturating_sub(8);
    let mut lut = vec![vec![0; width]; height];
    let mut seed = u32::from(params.random_seed) ^ if is_cr { 0x49d8 } else { 0xb524 };
    for row in &mut lut {
        for value in row {
            let random = random_number(11, &mut seed) as usize;
            *value = round_shift(i32::from(GAUSSIAN_SEQUENCE[random]), shift.max(0) as u8)
                .clamp(-grain_ctr, grain_ctr - 1);
        }
    }
    let lag = usize::from(params.ar_coeff_lag & 3);
    let limit_y = height - AR_PAD;
    let limit_x = width - 2 * AR_PAD;
    for y in 0..limit_y {
        for x in 0..limit_x {
            let mut coeff_index = 0;
            let mut sum = 0i32;
            for dy in 0..=lag {
                for dx in 0..=(2 * lag) {
                    if dy == lag && dx == lag {
                        let ly = (y << usize::from(sub_y)) + AR_PAD;
                        let lx = (x << usize::from(sub_x)) + AR_PAD;
                        let mut luma =
                            luma_lut[ly.min(luma_lut.len() - 1)][lx.min(luma_lut[0].len() - 1)];
                        if sub_x {
                            luma += luma_lut[ly.min(luma_lut.len() - 1)]
                                [(lx + 1).min(luma_lut[0].len() - 1)];
                        }
                        if sub_y {
                            luma += luma_lut[(ly + 1).min(luma_lut.len() - 1)]
                                [lx.min(luma_lut[0].len() - 1)];
                            if sub_x {
                                luma += luma_lut[(ly + 1).min(luma_lut.len() - 1)]
                                    [(lx + 1).min(luma_lut[0].len() - 1)];
                            }
                        }
                        let count = 1i32 << (usize::from(sub_x) + usize::from(sub_y));
                        sum += i32::from(coeffs[coeff_index]) * (luma / count);
                        break;
                    }
                    let sample = lut[y + AR_PAD - lag + dy][x + AR_PAD - lag + dx];
                    sum += i32::from(coeffs[coeff_index]) * sample;
                    coeff_index += 1;
                }
            }
            let index_y = y + AR_PAD;
            let index_x = x + AR_PAD;
            lut[index_y][index_x] = (lut[index_y][index_x]
                + round_shift(sum, params.ar_coeff_shift))
            .clamp(-grain_ctr, grain_ctr - 1);
        }
    }
    lut
}

fn sample_lut(
    lut: &[Vec<i32>],
    offsets: &[[i32; 2]; 2],
    sub_x: bool,
    sub_y: bool,
    block_x: bool,
    block_y: bool,
    x: usize,
    y: usize,
) -> i32 {
    let sx = usize::from(sub_x);
    let sy = usize::from(sub_y);
    let bx = usize::from(block_x);
    let by = usize::from(block_y);
    let random = offsets[bx][by] as usize;
    let offset_x = 3 + (2 >> sx) * (3 + (random >> 4));
    let offset_y = 3 + (2 >> sy) * (3 + (random & 0x0f));
    let block_width = BLOCK_SIZE >> sx;
    let block_height = BLOCK_SIZE >> sy;
    lut[offset_y + y + block_height * by][offset_x + x + block_width * bx]
}

fn row_seeds(row: usize, params: &FilmGrainParams) -> [u32; 2] {
    let rows = 1 + usize::from(params.overlap_flag && row > 0);
    let mut seeds = [0; 2];
    for (index, seed) in seeds.iter_mut().take(rows).enumerate() {
        let source_row = row - index;
        *seed = u32::from(params.random_seed);
        *seed ^= (((source_row * 37 + 178) & 0xff) as u32) << 8;
        *seed ^= ((source_row * 173 + 105) & 0xff) as u32;
    }
    seeds
}

fn blend_grain(old: i32, current: i32, subsampled: bool, index: usize) -> i32 {
    let [old_weight, current_weight] = if subsampled {
        [[23, 22], [0, 0]][index]
    } else {
        [[27, 17], [17, 27]][index]
    };
    round_shift(old * old_weight + current * current_weight, 5)
}

fn apply_luma_plane(
    samples: &mut Vec<u16>,
    width: usize,
    height: usize,
    scaling: &[i32],
    lut: &[Vec<i32>],
    params: &FilmGrainParams,
    bit_depth: u8,
) {
    apply_plane(
        samples, width, height, false, false, scaling, lut, params, bit_depth, None,
    );
}

fn apply_chroma_plane(
    samples: &mut Vec<u16>,
    width: usize,
    height: usize,
    luma: &[u16],
    luma_width: usize,
    luma_height: usize,
    scaling: &[i32],
    lut: &[Vec<i32>],
    params: &FilmGrainParams,
    bit_depth: u8,
    sub_x: bool,
    sub_y: bool,
    is_identity: bool,
    mult: u8,
    luma_mult: u8,
    offset: u16,
) {
    apply_plane(
        samples,
        width,
        height,
        sub_x,
        sub_y,
        scaling,
        lut,
        params,
        bit_depth,
        Some((
            luma,
            luma_width,
            luma_height,
            is_identity,
            mult,
            luma_mult,
            offset,
        )),
    );
}

fn apply_plane(
    samples: &mut Vec<u16>,
    width: usize,
    height: usize,
    sub_x: bool,
    sub_y: bool,
    scaling: &[i32],
    lut: &[Vec<i32>],
    params: &FilmGrainParams,
    bit_depth: u8,
    chroma: Option<(&[u16], usize, usize, bool, u8, u8, u16)>,
) {
    let source = samples.clone();
    let mut output = source.clone();
    let sx = usize::from(sub_x);
    let sy = usize::from(sub_y);
    let grain_ctr = 128i32 << bit_depth.saturating_sub(8);
    let grain_min = -grain_ctr;
    let grain_max = grain_ctr - 1;
    let max_value = (1i32 << bit_depth) - 1;
    let (clip_min, clip_max) = if params.clip_to_restricted_range {
        let max = if chroma.is_some_and(|(_, _, _, is_identity, _, _, _)| !is_identity) {
            240
        } else {
            235
        };
        (
            16i32 << bit_depth.saturating_sub(8),
            max << bit_depth.saturating_sub(8),
        )
    } else {
        (0, max_value)
    };
    let block_width = BLOCK_SIZE >> sx;
    let block_height = BLOCK_SIZE >> sy;
    let rows = height.div_ceil(block_height);
    for row_num in 0..rows {
        let mut seeds = row_seeds(row_num, params);
        let mut offsets = [[0i32; 2]; 2];
        for bx in (0..width).step_by(block_width) {
            let current_width = block_width.min(width - bx);
            let current_height = block_height.min(height - row_num * block_height);
            let row_count = 1 + usize::from(params.overlap_flag && row_num > 0);
            if params.overlap_flag && bx != 0 {
                for index in 0..row_count {
                    offsets[1][index] = offsets[0][index];
                }
            }
            for (index, seed) in seeds.iter_mut().enumerate().take(row_count) {
                offsets[0][index] = random_number(8, seed);
            }
            let x_start = if params.overlap_flag && bx != 0 {
                (2 >> sx).min(current_width)
            } else {
                0
            };
            let y_start = if params.overlap_flag && row_num != 0 {
                (2 >> sy).min(current_height)
            } else {
                0
            };
            for y in 0..current_height {
                for x in 0..current_width {
                    let mut grain = sample_lut(lut, &offsets, sub_x, sub_y, false, false, x, y);
                    if x < x_start && y < y_start {
                        let top = blend_grain(
                            sample_lut(lut, &offsets, sub_x, sub_y, true, true, x, y),
                            sample_lut(lut, &offsets, sub_x, sub_y, false, true, x, y),
                            sub_x,
                            x,
                        )
                        .clamp(grain_min, grain_max);
                        let left = blend_grain(
                            sample_lut(lut, &offsets, sub_x, sub_y, true, false, x, y),
                            grain,
                            sub_x,
                            x,
                        )
                        .clamp(grain_min, grain_max);
                        grain = blend_grain(top, left, sub_y, y).clamp(grain_min, grain_max);
                    } else if x < x_start {
                        grain = blend_grain(
                            sample_lut(lut, &offsets, sub_x, sub_y, true, false, x, y),
                            grain,
                            sub_x,
                            x,
                        )
                        .clamp(grain_min, grain_max);
                    } else if y < y_start {
                        grain = blend_grain(
                            sample_lut(lut, &offsets, sub_x, sub_y, false, true, x, y),
                            grain,
                            sub_y,
                            y,
                        )
                        .clamp(grain_min, grain_max);
                    }
                    grain = grain.clamp(grain_min, grain_max);
                    let index = (row_num * block_height + y) * width + bx + x;
                    let src = i32::from(source[index]);
                    let scale_index = if let Some((
                        luma,
                        luma_width,
                        luma_height,
                        _is_identity,
                        mult,
                        luma_mult,
                        offset,
                    )) = chroma
                    {
                        let ly = (row_num * block_height + y) << sy;
                        let lx = (bx + x) << sx;
                        let mut avg = i32::from(
                            luma[ly.min(luma_height - 1) * luma_width + lx.min(luma_width - 1)],
                        );
                        if sub_x {
                            let right = lx.saturating_add(1).min(luma_width - 1);
                            avg = (avg
                                + i32::from(luma[ly.min(luma_height - 1) * luma_width + right])
                                + 1)
                                >> 1;
                        }
                        if sub_y {
                            let next = (ly + 1).min(luma_height - 1);
                            let mut next_avg =
                                i32::from(luma[next * luma_width + lx.min(luma_width - 1)]);
                            if sub_x {
                                let right = lx.saturating_add(1).min(luma_width - 1);
                                next_avg =
                                    (next_avg + i32::from(luma[next * luma_width + right]) + 1)
                                        >> 1;
                            }
                            avg = (avg + next_avg + 1) >> 1;
                        }
                        let value = if params.chroma_scaling_from_luma {
                            avg
                        } else {
                            let combined = avg * i32::from(luma_mult) + src * i32::from(mult);
                            ((combined >> 6)
                                + (i32::from(offset) - 256) * (1 << bit_depth.saturating_sub(8)))
                            .clamp(0, max_value)
                        };
                        value as usize
                    } else {
                        src as usize
                    };
                    let noise = round_shift(
                        scaling[scale_index.min(scaling.len() - 1)] * grain,
                        params.scaling_shift,
                    );
                    output[index] = (src + noise).clamp(clip_min, clip_max) as u16;
                }
            }
        }
    }
    *samples = output;
}

#[cfg(test)]
mod tests {
    use super::{blend_grain, row_seeds};
    use crate::av1::decode::{PlaneBuffer, PlaneLayout};
    use crate::av1::{ColorConfig, ColorRange, FilmGrainParams, FrameBuffers};

    fn params(overlap_flag: bool) -> FilmGrainParams {
        FilmGrainParams {
            random_seed: 45231,
            num_y_points: 0,
            scaling_points_y: [[0; 2]; 14],
            chroma_scaling_from_luma: false,
            num_cb_points: 0,
            scaling_points_cb: [[0; 2]; 10],
            num_cr_points: 0,
            scaling_points_cr: [[0; 2]; 10],
            scaling_shift: 11,
            ar_coeff_lag: 0,
            ar_coeffs_y: [0; 24],
            ar_coeffs_cb: [0; 25],
            ar_coeffs_cr: [0; 25],
            ar_coeff_shift: 8,
            grain_scale_shift: 0,
            cb_mult: 0,
            cb_luma_mult: 0,
            cb_offset: 0,
            cr_mult: 0,
            cr_luma_mult: 0,
            cr_offset: 0,
            overlap_flag,
            clip_to_restricted_range: false,
        }
    }

    #[test]
    fn overlap_weights_match_av1_rounding() {
        assert_eq!(blend_grain(100, 200, false, 0), 191);
        assert_eq!(blend_grain(100, 200, false, 1), 222);
        assert_eq!(blend_grain(100, 200, true, 0), 209);
    }

    #[test]
    fn overlap_row_seed_keeps_current_and_previous_rows() {
        let no_overlap = row_seeds(0, &params(false));
        assert_eq!(no_overlap[1], 0);
        let overlap = row_seeds(3, &params(true));
        assert_ne!(overlap[0], overlap[1]);
        assert_eq!(
            overlap,
            [
                row_seeds(3, &params(true))[0],
                row_seeds(2, &params(true))[0]
            ]
        );
    }

    #[test]
    fn overlap_path_changes_block_boundaries_without_changing_dimensions() {
        let layout = PlaneLayout {
            plane: 0,
            width: 64,
            height: 64,
            subsampling_x: 0,
            subsampling_y: 0,
            sample_count: 64 * 64,
        };
        let mut params = params(false);
        params.num_y_points = 2;
        params.scaling_points_y[0] = [0, 64];
        params.scaling_points_y[1] = [255, 64];
        let color = ColorConfig {
            high_bitdepth: false,
            twelve_bit: false,
            bit_depth: 8,
            monochrome: true,
            color_description: None,
            color_range: ColorRange::Full,
            subsampling_x: false,
            subsampling_y: false,
            chroma_sample_position: None,
            separate_uv_delta_q: false,
        };
        let mut no_overlap = FrameBuffers {
            width: 64,
            height: 64,
            planes: vec![PlaneBuffer {
                layout,
                samples: vec![128; 64 * 64],
            }],
        };
        let mut overlap = no_overlap.clone();
        super::apply(&mut no_overlap, &color, &params);
        params.overlap_flag = true;
        super::apply(&mut overlap, &color, &params);
        assert_eq!(overlap.planes[0].samples.len(), 64 * 64);
        assert!(overlap.planes[0].samples != no_overlap.planes[0].samples);
    }
}
