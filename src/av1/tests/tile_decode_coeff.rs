use super::{EntropyDecoder, eob_base_from_pt, read_golomb};

#[test]
fn golomb_values_match_aom_bitwriter_vectors() {
    const CASES: &[(usize, &[u8])] = &[
        (0, &[192, 0]),
        (1, &[72, 0, 0, 0, 0]),
        (2, &[104, 0, 0, 0, 0]),
        (5, &[52, 0, 0, 0, 0, 0]),
        (14, &[31, 0, 0, 0, 0, 0, 0, 0]),
        (31, &[4, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ];

    for &(expected, payload) in CASES {
        let mut reader = EntropyDecoder::new(payload, true).unwrap();
        assert_eq!(read_golomb(&mut reader).unwrap(), expected);
    }
}

#[test]
fn eob_point_groups_match_av1_group_starts() {
    assert_eq!(
        (1..=11).map(eob_base_from_pt).collect::<Vec<_>>(),
        vec![1, 2, 3, 5, 9, 17, 33, 65, 129, 257, 513]
    );
}
