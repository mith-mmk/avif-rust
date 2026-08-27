use avif_rust::container::parse_avif;
use avif_rust::{ColorInformationSet, NclxColorInformation, parse_rich_info};

fn boxed(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let size = u32::try_from(payload.len() + 8).expect("test box fits in u32");
    let mut output = Vec::with_capacity(payload.len() + 8);
    output.extend_from_slice(&size.to_be_bytes());
    output.extend_from_slice(kind);
    output.extend_from_slice(payload);
    output
}

fn full_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut full_payload = vec![0, 0, 0, 0];
    full_payload.extend_from_slice(payload);
    boxed(kind, &full_payload)
}

fn color_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut color_payload = kind.to_vec();
    color_payload.extend_from_slice(payload);
    boxed(b"colr", &color_payload)
}

fn nclx_payload() -> Vec<u8> {
    vec![0, 1, 0, 13, 0, 6, 0x80]
}

fn infe(item_id: u16, item_type: &[u8; 4], name: &[u8]) -> Vec<u8> {
    let mut payload = vec![2, 0, 0, 0];
    payload.extend_from_slice(&item_id.to_be_bytes());
    payload.extend_from_slice(&0u16.to_be_bytes());
    payload.extend_from_slice(item_type);
    payload.extend_from_slice(name);
    payload.push(0);
    boxed(b"infe", &payload)
}

fn build_rich_file(color_order: &[&[u8; 4]], malformed_nclx: bool) -> Vec<u8> {
    let mut ftyp_payload = Vec::new();
    ftyp_payload.extend_from_slice(b"avif");
    ftyp_payload.extend_from_slice(&0u32.to_be_bytes());
    ftyp_payload.extend_from_slice(b"avif");
    let ftyp = boxed(b"ftyp", &ftyp_payload);

    let mut properties = Vec::new();
    let mut ispe = vec![0, 0, 0, 0];
    ispe.extend_from_slice(&2u32.to_be_bytes());
    ispe.extend_from_slice(&1u32.to_be_bytes());
    properties.push(boxed(b"ispe", &ispe));
    properties.push(full_box(b"pixi", &[3, 8, 8, 8]));
    properties.push(boxed(b"av1C", &[0x81, 0, 0, 0]));

    let mut color_indices = Vec::new();
    for color_type in color_order {
        let (payload, index) = match *color_type {
            b"nclx" => (nclx_payload(), properties.len() + 1),
            b"prof" => (vec![1, 2, 3, 4], properties.len() + 1),
            b"rICC" => (vec![5, 6, 7], properties.len() + 1),
            b"zzzz" => (vec![9, 8, 7], properties.len() + 1),
            _ => panic!("unsupported test color type"),
        };
        let payload = if **color_type == *b"nclx" && malformed_nclx {
            vec![0, 1, 0]
        } else {
            payload
        };
        properties.push(color_box(color_type, &payload));
        color_indices.push(index as u8);
    }

    // Give the alternate item a different colour description.  Rich parsing
    // must scope colour collection to the effective primary item rather than
    // accidentally inheriting an item-local property from another image.
    let competing_color_index = properties.len() + 1;
    properties.push(color_box(b"nclx", &[0, 2, 0, 16, 0, 1, 0x80]));

    let ipco_payload = properties.into_iter().flatten().collect::<Vec<_>>();
    let ipco = boxed(b"ipco", &ipco_payload);

    // Split the primary item's associations across two ipma boxes.  This
    // exercises the parser's merge path and keeps the test on public parsing.
    let mut ipma_one_payload = vec![0, 0, 0, 0];
    ipma_one_payload.extend_from_slice(&2u32.to_be_bytes());
    ipma_one_payload.extend_from_slice(&1u16.to_be_bytes());
    ipma_one_payload.push(3 + u8::try_from(color_indices.len().min(1)).unwrap());
    ipma_one_payload.extend_from_slice(&[1, 2, 3]);
    if let Some(index) = color_indices.first() {
        ipma_one_payload.push(*index);
    }
    ipma_one_payload.extend_from_slice(&2u16.to_be_bytes());
    ipma_one_payload.push(2);
    ipma_one_payload.push(1);
    ipma_one_payload.push(competing_color_index as u8);
    let ipma_one = boxed(b"ipma", &ipma_one_payload);

    let mut ipma_two_payload = vec![0, 0, 0, 0];
    ipma_two_payload.extend_from_slice(&1u32.to_be_bytes());
    ipma_two_payload.extend_from_slice(&1u16.to_be_bytes());
    ipma_two_payload.push(u8::try_from(color_indices.len().saturating_sub(1)).unwrap());
    for index in color_indices.iter().skip(1) {
        ipma_two_payload.push(*index);
    }
    let ipma_two = boxed(b"ipma", &ipma_two_payload);

    let mut iprp_payload = ipco;
    iprp_payload.extend_from_slice(&ipma_one);
    iprp_payload.extend_from_slice(&ipma_two);
    let iprp = boxed(b"iprp", &iprp_payload);

    let mut pitm_payload = vec![0, 0, 0, 0];
    pitm_payload.extend_from_slice(&1u16.to_be_bytes());
    let pitm = boxed(b"pitm", &pitm_payload);

    let mut iinf_payload = vec![0, 0, 0, 0];
    iinf_payload.extend_from_slice(&2u16.to_be_bytes());
    iinf_payload.extend_from_slice(&infe(1, b"av01", b"primary"));
    iinf_payload.extend_from_slice(&infe(2, b"av01", b"alternate"));
    let iinf = boxed(b"iinf", &iinf_payload);

    // Two one-byte item extents are stored in mdat.  The 4-byte offset and
    // length fields make the generated file independent of platform layout.
    let meta_without_iloc = {
        let mut payload = vec![0, 0, 0, 0];
        payload.extend_from_slice(&pitm);
        payload.extend_from_slice(&iinf);
        payload.extend_from_slice(&iprp);
        payload
    };
    let provisional_meta = boxed(b"meta", &meta_without_iloc);
    let iloc_size = 8 + 8 + 2 * 14;
    let primary_offset = ftyp.len() + provisional_meta.len() + iloc_size + 8;
    let mut iloc_payload = vec![0, 0, 0, 0, 0x44, 0];
    iloc_payload.extend_from_slice(&2u16.to_be_bytes());
    for (item_id, offset) in [(1u16, primary_offset), (2u16, primary_offset + 1)] {
        iloc_payload.extend_from_slice(&item_id.to_be_bytes());
        iloc_payload.extend_from_slice(&0u16.to_be_bytes());
        iloc_payload.extend_from_slice(&1u16.to_be_bytes());
        iloc_payload.extend_from_slice(&(offset as u32).to_be_bytes());
        iloc_payload.extend_from_slice(&1u32.to_be_bytes());
    }
    let iloc = boxed(b"iloc", &iloc_payload);

    let mut meta_payload = vec![0, 0, 0, 0];
    meta_payload.extend_from_slice(&pitm);
    meta_payload.extend_from_slice(&iinf);
    meta_payload.extend_from_slice(&iloc);
    meta_payload.extend_from_slice(&iprp);
    let meta = boxed(b"meta", &meta_payload);

    let mut output = ftyp;
    output.extend_from_slice(&meta);
    output.extend_from_slice(&boxed(b"mdat", &[0x12, 0x34]));
    output
}

fn assert_rich_color_set(set: &ColorInformationSet, icc_type: &[u8; 4]) {
    assert_eq!(
        set.nclx,
        Some(NclxColorInformation {
            color_primaries: 1,
            transfer_characteristics: 13,
            matrix_coefficients: 6,
            full_range_flag: true,
        })
    );
    assert_eq!(set.icc_color_type, Some(*icc_type));
    assert!(set.icc_profile.is_some());
}

#[test]
fn public_rich_parser_retains_nclx_and_prof_in_both_orders_and_legacy_projection() {
    for order in [[b"nclx", b"prof"], [b"prof", b"nclx"]] {
        let file = build_rich_file(&[order[0], order[1]], false);
        let rich = parse_rich_info(&file).expect("synthetic rich AVIF parses");
        assert_rich_color_set(&rich.color_information, b"prof");
        assert_eq!(rich.info.primary_item_payload, vec![0x12]);
        let legacy = parse_avif(&file).expect("legacy parser remains compatible");
        assert_eq!(
            legacy.color_information.and_then(|v| v.nclx()),
            rich.color_information.nclx
        );
    }
}

#[test]
fn public_rich_parser_retains_ricc_and_unknown_colr() {
    let file = build_rich_file(&[b"rICC", b"nclx", b"zzzz"], false);
    let rich = parse_rich_info(&file).expect("synthetic rICC AVIF parses");
    assert_rich_color_set(&rich.color_information, b"rICC");
    assert_eq!(rich.color_information.unknown_colr.len(), 1);
    assert_eq!(rich.color_information.unknown_colr[0].color_type, *b"zzzz");
}

#[test]
fn public_rich_parser_rejects_truncated_nclx() {
    let file = build_rich_file(&[b"nclx", b"prof"], true);
    let error = parse_rich_info(&file).expect_err("truncated nclx must be rejected");
    assert!(
        matches!(error, avif_rust::DecoderError::Bitstream(message) if message.contains("nclx"))
    );
}
