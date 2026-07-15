use super::frame::QuantizationParams;
use crate::DecoderError;

pub const DC_QLOOKUP_8: [i32; 256] = [
    4, 8, 8, 9, 10, 11, 12, 12, 13, 14, 15, 16, 17, 18, 19, 19, 20, 21, 22, 23, 24, 25, 26, 26, 27,
    28, 29, 30, 31, 32, 32, 33, 34, 35, 36, 37, 38, 38, 39, 40, 41, 42, 43, 43, 44, 45, 46, 47, 48,
    48, 49, 50, 51, 52, 53, 53, 54, 55, 56, 57, 57, 58, 59, 60, 61, 62, 62, 63, 64, 65, 66, 66, 67,
    68, 69, 70, 70, 71, 72, 73, 74, 74, 75, 76, 77, 78, 78, 79, 80, 81, 81, 82, 83, 84, 85, 85, 87,
    88, 90, 92, 93, 95, 96, 98, 99, 101, 102, 104, 105, 107, 108, 110, 111, 113, 114, 116, 117,
    118, 120, 121, 123, 125, 127, 129, 131, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154,
    156, 158, 161, 164, 166, 169, 172, 174, 177, 180, 182, 185, 187, 190, 192, 195, 199, 202, 205,
    208, 211, 214, 217, 220, 223, 226, 230, 233, 237, 240, 243, 247, 250, 253, 257, 261, 265, 269,
    272, 276, 280, 284, 288, 292, 296, 300, 304, 309, 313, 317, 322, 326, 330, 335, 340, 344, 349,
    354, 359, 364, 369, 374, 379, 384, 389, 395, 400, 406, 411, 417, 423, 429, 435, 441, 447, 454,
    461, 467, 475, 482, 489, 497, 505, 513, 522, 530, 539, 549, 559, 569, 579, 590, 602, 614, 626,
    640, 654, 668, 684, 700, 717, 736, 755, 775, 796, 819, 843, 869, 896, 925, 955, 988, 1022,
    1058, 1098, 1139, 1184, 1232, 1282, 1336,
];

pub const AC_QLOOKUP_8: [i32; 256] = [
    4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
    55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
    79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
    102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138,
    140, 142, 144, 146, 148, 150, 152, 155, 158, 161, 164, 167, 170, 173, 176, 179, 182, 185, 188,
    191, 194, 197, 200, 203, 207, 211, 215, 219, 223, 227, 231, 235, 239, 243, 247, 251, 255, 260,
    265, 270, 275, 280, 285, 290, 295, 300, 305, 311, 317, 323, 329, 335, 341, 347, 353, 359, 366,
    373, 380, 387, 394, 401, 408, 416, 424, 432, 440, 448, 456, 465, 474, 483, 492, 501, 510, 520,
    530, 540, 550, 560, 571, 582, 593, 604, 615, 627, 639, 651, 663, 676, 689, 702, 715, 729, 743,
    757, 771, 786, 801, 816, 832, 848, 864, 881, 898, 915, 933, 951, 969, 988, 1007, 1026, 1046,
    1066, 1087, 1108, 1129, 1151, 1173, 1196, 1219, 1243, 1267, 1292, 1317, 1343, 1369, 1396, 1423,
    1451, 1479, 1508, 1537, 1567, 1597, 1628, 1660, 1692, 1725, 1759, 1793, 1828,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaneQuant {
    pub dc: i32,
    pub ac: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantState {
    pub y: PlaneQuant,
    pub u: PlaneQuant,
    pub v: PlaneQuant,
}

impl QuantState {
    pub fn from_params(params: &QuantizationParams, bit_depth: u8) -> Result<Self, DecoderError> {
        if !matches!(bit_depth, 8 | 10 | 12) {
            return Err(DecoderError::Unsupported(format!(
                "AV1 {bit_depth}-bit quantization is not supported yet"
            )));
        }
        let base = i32::from(params.base_q_idx);
        Ok(Self {
            y: PlaneQuant {
                dc: dc_q(base + i32::from(params.delta_q_y_dc), bit_depth),
                ac: ac_q(base, bit_depth),
            },
            u: PlaneQuant {
                dc: dc_q(base + i32::from(params.delta_q_u_dc), bit_depth),
                ac: ac_q(base + i32::from(params.delta_q_u_ac), bit_depth),
            },
            v: PlaneQuant {
                dc: dc_q(base + i32::from(params.delta_q_v_dc), bit_depth),
                ac: ac_q(base + i32::from(params.delta_q_v_ac), bit_depth),
            },
        })
    }

    pub fn plane(self, plane: usize) -> PlaneQuant {
        match plane {
            0 => self.y,
            1 => self.u,
            _ => self.v,
        }
    }
}

pub fn dequantize_coefficients(
    quant: &[i32],
    plane_quant: PlaneQuant,
    bit_depth: u8,
    dq_denom: i32,
) -> Vec<i32> {
    let limit = 1i32 << (7 + bit_depth);
    quant
        .iter()
        .enumerate()
        .map(|(index, coeff)| {
            let q = if index == 0 {
                plane_quant.dc
            } else {
                plane_quant.ac
            };
            let dq = coeff.saturating_mul(q);
            let sign = if dq < 0 { -1 } else { 1 };
            let dq2 = sign * ((dq.abs() & 0x00ff_ffff) / dq_denom.max(1));
            dq2.clamp(-limit, limit - 1)
        })
        .collect()
}

fn dc_q(qindex: i32, bit_depth: u8) -> i32 {
    scale_quant(DC_QLOOKUP_8[clip_qindex(qindex)], bit_depth)
}

fn ac_q(qindex: i32, bit_depth: u8) -> i32 {
    scale_quant(AC_QLOOKUP_8[clip_qindex(qindex)], bit_depth)
}

fn scale_quant(value: i32, bit_depth: u8) -> i32 {
    value << bit_depth.saturating_sub(8)
}

fn clip_qindex(qindex: i32) -> usize {
    qindex.clamp(0, 255) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qlookup_8_matches_spec_endpoints() {
        assert_eq!(dc_q(-1, 8), 4);
        assert_eq!(dc_q(0, 8), 4);
        assert_eq!(dc_q(255, 8), 1336);
        assert_eq!(dc_q(256, 8), 1336);
        assert_eq!(ac_q(0, 8), 4);
        assert_eq!(ac_q(255, 8), 1828);
    }

    #[test]
    fn qlookup_scales_with_bit_depth() {
        assert_eq!(dc_q(255, 10), 1336 * 4);
        assert_eq!(ac_q(255, 12), 1828 * 16);
    }

    #[test]
    fn quant_state_applies_plane_deltas() {
        let params = QuantizationParams {
            base_q_idx: 100,
            delta_q_y_dc: 1,
            delta_q_u_dc: 2,
            delta_q_u_ac: 3,
            delta_q_v_dc: -2,
            delta_q_v_ac: -3,
            using_qmatrix: false,
        };

        let state = QuantState::from_params(&params, 8).unwrap();

        assert_eq!(state.y.dc, dc_q(101, 8));
        assert_eq!(state.y.ac, ac_q(100, 8));
        assert_eq!(state.u.dc, dc_q(102, 8));
        assert_eq!(state.u.ac, ac_q(103, 8));
        assert_eq!(state.v.dc, dc_q(98, 8));
        assert_eq!(state.v.ac, ac_q(97, 8));
    }

    #[test]
    fn dequantize_uses_dc_then_ac_and_clips() {
        let dequant = dequantize_coefficients(&[2, -3, 100_000], PlaneQuant { dc: 4, ac: 8 }, 8, 1);

        assert_eq!(dequant[0], 8);
        assert_eq!(dequant[1], -24);
        assert_eq!(dequant[2], (1 << 15) - 1);
    }

    #[test]
    fn dequantize_applies_large_transform_shift() {
        let quant = [4, -8];
        let plane = PlaneQuant { dc: 8, ac: 4 };

        assert_eq!(dequantize_coefficients(&quant, plane, 8, 1), [32, -32]);
        assert_eq!(dequantize_coefficients(&quant, plane, 8, 2), [16, -16]);
        assert_eq!(dequantize_coefficients(&quant, plane, 8, 4), [8, -8]);
    }
}
