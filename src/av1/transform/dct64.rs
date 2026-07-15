use super::{clamp_signed, cospi, half_btf};

const COS_BIT: u8 = 12;

pub(super) fn inverse_dct64(input: [i32; 64], range: u8) -> [i32; 64] {
    let mut a = std::array::from_fn(|index| input[index.reverse_bits() >> (usize::BITS - 6)]);
    let mut b = a;

    // Stage 2: rotate the odd-frequency half.
    for (offset, angle) in [63, 31, 47, 15, 55, 23, 39, 7, 59, 27, 43, 11, 51, 19, 35, 3]
        .into_iter()
        .enumerate()
    {
        rotate_pair(&a, &mut b, 32 + offset, 63 - offset, angle);
    }
    std::mem::swap(&mut a, &mut b);

    // Stage 3.
    b = a;
    for (offset, angle) in [62, 30, 46, 14, 54, 22, 38, 6].into_iter().enumerate() {
        rotate_pair(&a, &mut b, 16 + offset, 31 - offset, angle);
    }
    for start in (32..64).step_by(4) {
        combine_adjacent_pairs(&a, &mut b, start, range);
    }
    std::mem::swap(&mut a, &mut b);

    // Stage 4.
    b = a;
    for (offset, angle) in [60, 28, 44, 12].into_iter().enumerate() {
        rotate_pair(&a, &mut b, 8 + offset, 15 - offset, angle);
    }
    for start in (16..32).step_by(4) {
        combine_adjacent_pairs(&a, &mut b, start, range);
    }
    apply_mix_table(
        &a,
        &mut b,
        &[
            (33, 33, 62, -4, 60),
            (34, 34, 61, -60, -4),
            (37, 37, 58, -36, 28),
            (38, 38, 57, -28, -36),
            (41, 41, 54, -20, 44),
            (42, 42, 53, -44, -20),
            (45, 45, 50, -52, 12),
            (46, 46, 49, -12, -52),
            (49, 46, 49, -52, 12),
            (50, 45, 50, 12, 52),
            (53, 42, 53, -20, 44),
            (54, 41, 54, 44, 20),
            (57, 38, 57, -36, 28),
            (58, 37, 58, 28, 36),
            (61, 34, 61, -4, 60),
            (62, 33, 62, 60, 4),
        ],
    );
    std::mem::swap(&mut a, &mut b);

    // Stage 5.
    b = a;
    for (offset, angle) in [56, 24].into_iter().enumerate() {
        rotate_pair(&a, &mut b, 4 + offset, 7 - offset, angle);
    }
    for start in (8..16).step_by(4) {
        combine_adjacent_pairs(&a, &mut b, start, range);
    }
    apply_mix_table(
        &a,
        &mut b,
        &[
            (17, 17, 30, -8, 56),
            (18, 18, 29, -56, -8),
            (21, 21, 26, -40, 24),
            (22, 22, 25, -24, -40),
            (25, 22, 25, -40, 24),
            (26, 21, 26, 24, 40),
            (29, 18, 29, -8, 56),
            (30, 17, 30, 56, 8),
        ],
    );
    for start in (32..64).step_by(4) {
        combine_reverse(&a, &mut b, start, 4, (start / 4) % 2 == 1, range);
    }
    std::mem::swap(&mut a, &mut b);

    // Stage 6.
    b = a;
    mix(&a, &mut b, 0, 0, 1, 32, 32);
    mix(&a, &mut b, 1, 0, 1, 32, -32);
    mix(&a, &mut b, 2, 2, 3, 48, -16);
    mix(&a, &mut b, 3, 2, 3, 16, 48);
    combine_adjacent_pairs(&a, &mut b, 4, range);
    apply_mix_table(
        &a,
        &mut b,
        &[
            (9, 9, 14, -16, 48),
            (10, 10, 13, -48, -16),
            (13, 10, 13, -16, 48),
            (14, 9, 14, 48, 16),
        ],
    );
    for start in (16..32).step_by(4) {
        combine_reverse(&a, &mut b, start, 4, (start / 4) % 2 == 1, range);
    }
    apply_mix_table(
        &a,
        &mut b,
        &[
            (34, 34, 61, -8, 56),
            (35, 35, 60, -8, 56),
            (36, 36, 59, -56, -8),
            (37, 37, 58, -56, -8),
            (42, 42, 53, -40, 24),
            (43, 43, 52, -40, 24),
            (44, 44, 51, -24, -40),
            (45, 45, 50, -24, -40),
            (50, 45, 50, -40, 24),
            (51, 44, 51, -40, 24),
            (52, 43, 52, 24, 40),
            (53, 42, 53, 24, 40),
            (58, 37, 58, -8, 56),
            (59, 36, 59, -8, 56),
            (60, 35, 60, 56, 8),
            (61, 34, 61, 56, 8),
        ],
    );
    std::mem::swap(&mut a, &mut b);

    // Stage 7.
    b = a;
    combine_reverse(&a, &mut b, 0, 4, false, range);
    mix(&a, &mut b, 5, 5, 6, -32, 32);
    mix(&a, &mut b, 6, 5, 6, 32, 32);
    for start in (8..16).step_by(4) {
        combine_reverse(&a, &mut b, start, 4, (start / 4) % 2 == 1, range);
    }
    apply_mix_table(
        &a,
        &mut b,
        &[
            (18, 18, 29, -16, 48),
            (19, 19, 28, -16, 48),
            (20, 20, 27, -48, -16),
            (21, 21, 26, -48, -16),
            (26, 21, 26, -16, 48),
            (27, 20, 27, -16, 48),
            (28, 19, 28, 48, 16),
            (29, 18, 29, 48, 16),
        ],
    );
    for start in (32..64).step_by(8) {
        combine_reverse(&a, &mut b, start, 8, (start / 8) % 2 == 1, range);
    }
    std::mem::swap(&mut a, &mut b);

    // Stage 8.
    b = a;
    combine_reverse(&a, &mut b, 0, 8, false, range);
    mix(&a, &mut b, 10, 10, 13, -32, 32);
    mix(&a, &mut b, 11, 11, 12, -32, 32);
    mix(&a, &mut b, 12, 11, 12, 32, 32);
    mix(&a, &mut b, 13, 10, 13, 32, 32);
    for start in (16..32).step_by(8) {
        combine_reverse(&a, &mut b, start, 8, (start / 8) % 2 == 1, range);
    }
    apply_mix_table(
        &a,
        &mut b,
        &[
            (36, 36, 59, -16, 48),
            (37, 37, 58, -16, 48),
            (38, 38, 57, -16, 48),
            (39, 39, 56, -16, 48),
            (40, 40, 55, -48, -16),
            (41, 41, 54, -48, -16),
            (42, 42, 53, -48, -16),
            (43, 43, 52, -48, -16),
            (52, 43, 52, -16, 48),
            (53, 42, 53, -16, 48),
            (54, 41, 54, -16, 48),
            (55, 40, 55, -16, 48),
            (56, 39, 56, 48, 16),
            (57, 38, 57, 48, 16),
            (58, 37, 58, 48, 16),
            (59, 36, 59, 48, 16),
        ],
    );
    std::mem::swap(&mut a, &mut b);

    // Stage 9.
    b = a;
    combine_reverse(&a, &mut b, 0, 16, false, range);
    for offset in 0..4 {
        mix(&a, &mut b, 20 + offset, 20 + offset, 27 - offset, -32, 32);
        mix(&a, &mut b, 27 - offset, 20 + offset, 27 - offset, 32, 32);
    }
    for start in (32..64).step_by(16) {
        combine_reverse(&a, &mut b, start, 16, (start / 16) % 2 == 1, range);
    }
    std::mem::swap(&mut a, &mut b);

    // Stage 10.
    b = a;
    combine_reverse(&a, &mut b, 0, 32, false, range);
    for offset in 0..8 {
        mix(&a, &mut b, 40 + offset, 40 + offset, 55 - offset, -32, 32);
        mix(&a, &mut b, 55 - offset, 40 + offset, 55 - offset, 32, 32);
    }
    std::mem::swap(&mut a, &mut b);

    // Stage 11.
    combine_reverse(&a, &mut b, 0, 64, false, range);
    b
}

fn rotate_pair(input: &[i32; 64], output: &mut [i32; 64], low: usize, high: usize, angle: usize) {
    mix(
        input,
        output,
        low,
        low,
        high,
        angle as i32,
        -((64 - angle) as i32),
    );
    mix(
        input,
        output,
        high,
        low,
        high,
        (64 - angle) as i32,
        angle as i32,
    );
}

fn apply_mix_table(
    input: &[i32; 64],
    output: &mut [i32; 64],
    table: &[(usize, usize, usize, i32, i32)],
) {
    for &(out, first, second, first_angle, second_angle) in table {
        mix(input, output, out, first, second, first_angle, second_angle);
    }
}

fn mix(
    input: &[i32; 64],
    output: &mut [i32; 64],
    out: usize,
    first: usize,
    second: usize,
    first_angle: i32,
    second_angle: i32,
) {
    output[out] = half_btf(
        signed_cospi(first_angle),
        input[first],
        signed_cospi(second_angle),
        input[second],
        COS_BIT,
    );
}

fn signed_cospi(angle: i32) -> i32 {
    if angle < 0 {
        -cospi((-angle) as usize)
    } else {
        cospi(angle as usize)
    }
}

fn combine_reverse(
    input: &[i32; 64],
    output: &mut [i32; 64],
    start: usize,
    length: usize,
    negative_first: bool,
    range: u8,
) {
    let half = length / 2;
    for offset in 0..half {
        let first = input[start + offset];
        let last = input[start + length - 1 - offset];
        output[start + offset] = clamp_signed(
            if negative_first {
                -first + last
            } else {
                first + last
            },
            range,
        );

        let middle_left = input[start + half - 1 - offset];
        let middle_right = input[start + half + offset];
        output[start + half + offset] = clamp_signed(
            if negative_first {
                middle_left + middle_right
            } else {
                middle_left - middle_right
            },
            range,
        );
    }
}

fn combine_adjacent_pairs(input: &[i32; 64], output: &mut [i32; 64], start: usize, range: u8) {
    output[start] = clamp_signed(input[start] + input[start + 1], range);
    output[start + 1] = clamp_signed(input[start] - input[start + 1], range);
    output[start + 2] = clamp_signed(-input[start + 2] + input[start + 3], range);
    output[start + 3] = clamp_signed(input[start + 2] + input[start + 3], range);
}
