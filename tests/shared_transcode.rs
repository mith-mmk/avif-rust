#![cfg(feature = "encode")]
use avif_rust::{buffer::*, encoder as enc, transcode::*, *};

fn limits() -> NativeDecodeLimits {
    NativeDecodeLimits::new(
        1 << 24,
        8192,
        8192,
        1 << 24,
        1 << 29,
        1 << 24,
        1 << 24,
        128,
        512,
        128,
        16,
        1,
    )
}
fn options(bits: u8, model: NativeModel) -> enc::NativeEncodeOptions {
    enc::NativeEncodeOptions {
        bit_depth: bits,
        subsampling: match model {
            NativeModel::Monochrome => enc::NativeSubsampling::Cs400,
            NativeModel::Yuv420 => enc::NativeSubsampling::Cs420,
            NativeModel::Yuv422 => enc::NativeSubsampling::Cs422,
            _ => enc::NativeSubsampling::Cs444,
        },
        encoder: enc::EncoderOptions {
            lossless: true,
            color_information: None,
            ..Default::default()
        },
        color_information: enc::NativeColorInformation::from_nclx(enc::NclxColorInformation {
            color_primaries: 1,
            transfer_characteristics: 13,
            matrix_coefficients: if model == NativeModel::IdentityGbr {
                0
            } else {
                6
            },
            full_range_flag: true,
        }),
    }
}
fn source(
    bits: u8,
    model: NativeModel,
    alpha: bool,
    opts: &enc::NativeEncodeOptions,
) -> (Vec<u8>, enc::FrameBuffers) {
    let (w, h) = (5usize, 3usize);
    let mut planes = Vec::new();
    for role in 0..4usize {
        if role == 3 && !alpha || (role == 1 || role == 2) && model == NativeModel::Monochrome {
            continue;
        }
        let (sx, sy) = if role == 1 || role == 2 {
            model.chroma_subsampling().unwrap_or((0, 0))
        } else {
            (0, 0)
        };
        let (pw, ph) = (w.div_ceil(1 << sx), h.div_ceil(1 << sy));
        let max = sample_max(bits).unwrap();
        planes.push(enc::PlaneBuffer {
            layout: enc::PlaneLayout {
                plane: role as u8,
                width: pw,
                height: ph,
                subsampling_x: sx as u8,
                subsampling_y: sy as u8,
                sample_count: pw * ph,
            },
            samples: (0..pw * ph)
                .map(|i| {
                    if role == 3 && i % 3 == 0 {
                        0
                    } else {
                        ((i * 41 + role * 137) as u32 % (max + 1)) as u16
                    }
                })
                .collect(),
        });
    }
    let buffers = enc::FrameBuffers {
        width: w,
        height: h,
        planes,
    };
    (enc::encode_native_bytes(&buffers, opts).unwrap(), buffers)
}

#[test]
fn both_codecs_use_identical_shared_types_and_native_lossless_samples() {
    // This assignment cannot compile if decoder/encoder resolve different model packages.
    fn same_type(v: avif_rust::buffer::ImageViewEx<'_>) -> enc::buffer::ImageViewEx<'_> {
        v
    }
    for bits in [8, 10, 12] {
        for model in [
            NativeModel::Monochrome,
            NativeModel::IdentityGbr,
            NativeModel::Yuv444,
            NativeModel::Yuv422,
            NativeModel::Yuv420,
        ] {
            for alpha in [false, true] {
                let opts = options(bits, model);
                let (bytes, expected) = source(bits, model, alpha, &opts);
                if let Some(directory) = std::env::var_os("AVIF_SHARED_ARTIFACTS") {
                    let name = format!("{bits}-{model:?}-{alpha}-source.avif");
                    std::fs::write(std::path::Path::new(&directory).join(name), &bytes).unwrap();
                }
                let image = decode_native_ex(&bytes, &limits()).unwrap();
                let ImageViewEx::NativePlanar(input_view) = image.view().unwrap() else {
                    panic!()
                };
                for p in &expected.planes {
                    let SamplesRef::U16(actual) = input_view.planes[usize::from(p.layout.plane)]
                        .unwrap()
                        .samples
                    else {
                        panic!()
                    };
                    assert_eq!(
                        actual, p.samples,
                        "initial source decode: bits={bits} model={model:?} role={}",
                        p.layout.plane
                    );
                }
                let original = image.clone();
                same_type(image.view().unwrap());
                let preview =
                    preflight_native_transcode(&image, &opts, &PreservationPolicy::default())
                        .unwrap();
                assert!(!preview.encoded);
                assert!(preview.dropped.is_empty());
                assert_eq!(preview.retained_alpha_item_id, alpha.then_some(2));
                assert!(preview.dropped_alpha_item_ids.is_empty());
                assert_eq!(preview.dropped_av1_metadata_obus, 0);
                let out =
                    transcode_native(&image, &opts, &PreservationPolicy::default(), &limits())
                        .unwrap();
                if let Some(directory) = std::env::var_os("AVIF_SHARED_ARTIFACTS") {
                    let name = format!("{bits}-{model:?}-{alpha}-output.avif");
                    std::fs::write(std::path::Path::new(&directory).join(name), &out.bytes)
                        .unwrap();
                }
                assert!(out.report.encoded && out.report.container_signalling_verified);
                assert!(!out.report.pixel_roundtrip_verified);
                assert_eq!(image, original);
                let actual = decode_frame_bytes_strict_with_limits(&out.bytes, &limits()).unwrap();
                assert_eq!(actual.frame().bit_depth, bits);
                for p in &expected.planes {
                    let q = actual
                        .frame()
                        .buffers
                        .planes
                        .iter()
                        .find(|q| q.layout.plane == p.layout.plane)
                        .unwrap();
                    assert_eq!(
                        q.samples, p.samples,
                        "bits={bits} model={model:?} role={}",
                        p.layout.plane
                    );
                    assert_eq!(
                        (q.layout.width, q.layout.height),
                        (p.layout.width, p.layout.height)
                    );
                }
            }
        }
    }
}

#[test]
fn icc_bytes_and_original_prof_or_ricc_property_type_are_preserved_without_cmm() {
    for kind in [*b"prof", *b"rICC"] {
        let mut opts = options(12, NativeModel::IdentityGbr);
        opts.color_information.icc_profile =
            Some(b"opaque ICC bytes; never a color transform".to_vec());
        opts.color_information.icc_color_type = Some(kind);
        let (bytes, _) = source(12, NativeModel::IdentityGbr, true, &opts);
        let image = decode_native_ex(&bytes, &limits()).unwrap();
        let out =
            transcode_native(&image, &opts, &PreservationPolicy::default(), &limits()).unwrap();
        let parsed = parse_native_info(&out.bytes, &limits()).unwrap();
        assert_eq!(
            parsed.color_information().icc_profile,
            opts.color_information.icc_profile
        );
        assert_eq!(parsed.color_information().icc_color_type, Some(kind));
        assert!(out.report.dropped.is_empty());
        let mut wrong = opts.clone();
        wrong.color_information.icc_color_type =
            Some(if kind == *b"prof" { *b"rICC" } else { *b"prof" });
        assert!(matches!(
            preflight_native_transcode(&image, &wrong, &PreservationPolicy::default()),
            Err(PreservationError::PreservationUnsupported(
                PreservationFeature::IccPropertyType
            ))
        ));
        wrong.color_information.icc_profile = None;
        wrong.color_information.icc_color_type = None;
        assert!(
            preflight_native_transcode(&image, &wrong, &PreservationPolicy::default()).is_err()
        );
        let policy = PreservationPolicy {
            allow_drop: vec![PreservationFeature::IccProfile],
            ..Default::default()
        };
        let out = transcode_native(&image, &wrong, &policy, &limits()).unwrap();
        assert_eq!(out.report.dropped, [PreservationFeature::IccProfile]);
    }
}

fn boxed(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = ((payload.len() + 8) as u32).to_be_bytes().to_vec();
    out.extend_from_slice(kind);
    out.extend_from_slice(payload);
    out
}
fn full(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    boxed(kind, &[[0; 4].as_slice(), payload].concat())
}
fn nclx(matrix: u16, full_range: bool) -> Vec<u8> {
    boxed(
        b"colr",
        &[
            b"nclx".as_slice(),
            &1u16.to_be_bytes(),
            &13u16.to_be_bytes(),
            &matrix.to_be_bytes(),
            &[if full_range { 128 } else { 0 }],
        ]
        .concat(),
    )
}
fn container(
    original: &[u8],
    colors: Vec<Vec<u8>>,
    extra_props: Vec<Vec<u8>>,
    extra_meta: &[u8],
) -> Vec<u8> {
    let info = container::parse_avif(original).unwrap();
    let mut properties = vec![
        full(
            b"ispe",
            &[
                info.width.unwrap().to_be_bytes(),
                info.height.unwrap().to_be_bytes(),
            ]
            .concat(),
        ),
        full(
            b"pixi",
            &[
                &[3u8][..],
                &info.pixel_information.unwrap().bits_per_channel,
            ]
            .concat(),
        ),
        boxed(b"av1C", &info.av1_config.unwrap()),
    ];
    properties.extend(colors);
    properties.extend(extra_props);
    let mut ipma = 1u32.to_be_bytes().to_vec();
    ipma.extend_from_slice(&1u16.to_be_bytes());
    ipma.push(properties.len() as u8);
    ipma.extend((1..=properties.len()).map(|v| v as u8));
    let iprp = boxed(
        b"iprp",
        &[
            boxed(
                b"ipco",
                &properties.into_iter().flatten().collect::<Vec<_>>(),
            ),
            full(b"ipma", &ipma),
        ]
        .concat(),
    );
    let infe = boxed(
        b"infe",
        &[
            &[2, 0, 0, 0][..],
            &1u16.to_be_bytes(),
            &[0, 0],
            b"av01Color\0",
        ]
        .concat(),
    );
    let iinf = full(b"iinf", &[1u16.to_be_bytes().as_slice(), &infe].concat());
    let mut meta = vec![0; 4];
    meta.extend_from_slice(&full(b"pitm", &1u16.to_be_bytes()));
    meta.extend_from_slice(&iinf);
    meta.extend_from_slice(&iprp);
    meta.extend_from_slice(extra_meta);
    let ftyp = boxed(b"ftyp", b"avif\0\0\0\0avif");
    let offset = (ftyp.len() + 8 + meta.len() + 30 + 8) as u32;
    let iloc = full(
        b"iloc",
        &[
            &[0x44, 0][..],
            &1u16.to_be_bytes(),
            &1u16.to_be_bytes(),
            &[0, 0, 0, 1],
            &offset.to_be_bytes(),
            &(info.primary_item_payload.len() as u32).to_be_bytes(),
        ]
        .concat(),
    );
    assert_eq!(iloc.len(), 30);
    meta.extend_from_slice(&iloc);
    [
        ftyp,
        boxed(b"meta", &meta),
        boxed(b"mdat", &info.primary_item_payload),
    ]
    .concat()
}

#[test]
fn missing_and_conflicting_color_signalling_require_explicit_authority() {
    let opts = options(8, NativeModel::IdentityGbr);
    let (original, _) = source(8, NativeModel::IdentityGbr, false, &opts);
    let conflicting = container(&original, vec![nclx(6, true)], vec![], &[]);
    assert!(matches!(decode_native_ex(&conflicting, &limits()),
        Err(DecoderError::Bitstream(message)) if message.contains("does not match AV1")));
    // Existing strict decode already rejects conflicting descriptions. Missing
    // nclx reaches planning and needs an explicit AV1 authority selection.
    let bytes = container(&original, vec![], vec![], &[]);
    let image = decode_native_ex(&bytes, &limits()).unwrap();
    assert!(matches!(
        preflight_native_transcode(&image, &opts, &PreservationPolicy::default()),
        Err(PreservationError::ColorAuthorityConflict(_))
    ));
    let policy = PreservationPolicy {
        color_authority: ColorAuthority::Av1,
        ..Default::default()
    };
    let out = transcode_native(&image, &opts, &policy, &limits()).unwrap();
    assert!(
        out.report
            .signalling_changes
            .contains(&SignallingChange::ContainerNclx)
    );
}

#[test]
fn geometry_unknown_properties_and_colr_drop_require_individual_permissions() {
    let opts = options(8, NativeModel::IdentityGbr);
    let (original, _) = source(8, NativeModel::IdentityGbr, false, &opts);
    for (prop, feature) in [
        (boxed(b"irot", &[1]), PreservationFeature::Rotation),
        (boxed(b"imir", &[0]), PreservationFeature::Mirror),
        (
            boxed(b"pasp", &[2u32.to_be_bytes(), 1u32.to_be_bytes()].concat()),
            PreservationFeature::PixelAspectRatio,
        ),
        (
            boxed(b"zzzz", b"not retained as payload"),
            PreservationFeature::UnknownProperty(*b"zzzz"),
        ),
        (
            boxed(b"colr", b"zzzzopaque"),
            PreservationFeature::UnknownColr(*b"zzzz"),
        ),
    ] {
        let bytes = container(&original, vec![nclx(0, true)], vec![prop], &[]);
        let image = decode_native_ex(&bytes, &limits()).unwrap();
        let original = image.clone();
        assert!(
            matches!(preflight_native_transcode(&image,&opts,&PreservationPolicy::default()),Err(PreservationError::PreservationUnsupported(v)) if v==feature)
        );
        let policy = PreservationPolicy {
            allow_drop: vec![feature],
            ..Default::default()
        };
        let out = transcode_native(&image, &opts, &policy, &limits()).unwrap();
        assert_eq!(out.report.dropped, [feature]);
        assert_eq!(image, original);
    }
}

#[test]
fn unmodelled_container_information_is_inventoried_before_it_can_be_dropped() {
    let opts = options(8, NativeModel::IdentityGbr);
    let (original, _) = source(8, NativeModel::IdentityGbr, false, &opts);
    let extra = boxed(b"zzzz", b"opaque metadata");
    let bytes = container(&original, vec![nclx(0, true)], vec![], &extra);
    let image = decode_native_ex(&bytes, &limits()).unwrap();
    assert!(
        image
            .context()
            .information()
            .source_inventory()
            .other_metadata
    );
    assert!(matches!(
        preflight_native_transcode(&image, &opts, &PreservationPolicy::default()),
        Err(PreservationError::PreservationUnsupported(
            PreservationFeature::OtherMetadata
        ))
    ));
    let policy = PreservationPolicy {
        allow_drop: vec![PreservationFeature::OtherMetadata],
        ..Default::default()
    };
    let out = transcode_native(&image, &opts, &policy, &limits()).unwrap();
    assert_eq!(out.report.dropped, [PreservationFeature::OtherMetadata]);
    let mut top = original;
    top.extend_from_slice(&extra);
    let image = decode_native_ex(&top, &limits()).unwrap();
    assert!(matches!(
        preflight_native_transcode(&image, &opts, &PreservationPolicy::default()),
        Err(PreservationError::PreservationUnsupported(
            PreservationFeature::OtherTopLevel
        ))
    ));
}

#[test]
fn strict_failures_do_not_convert_or_mutate_inputs_and_output_limits_are_transactional() {
    let opts = options(12, NativeModel::IdentityGbr);
    let (bytes, _) = source(12, NativeModel::IdentityGbr, true, &opts);
    let image = decode_native_ex(&bytes, &limits()).unwrap();
    let original = image.clone();
    let mut wrong = opts.clone();
    wrong.bit_depth = 8;
    assert!(matches!(
        preflight_native_transcode(&image, &wrong, &PreservationPolicy::default()),
        Err(PreservationError::ConversionRequired(_))
    ));
    let policy = PreservationPolicy {
        require_container_structure: true,
        ..Default::default()
    };
    assert!(matches!(
        preflight_native_transcode(&image, &opts, &policy),
        Err(PreservationError::UnsupportedStructure(
            PreservationFeature::ContainerStructure
        ))
    ));
    let short = NativeDecodeLimits::new(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
    assert!(transcode_native(&image, &opts, &PreservationPolicy::default(), &short).is_err());
    assert_eq!(image, original);
}

#[test]
fn packed16_explicit_policy_is_available_through_optional_delegation() {
    let legacy = Rgba16ImageBuffer {
        width: 1,
        height: 1,
        rgba: vec![1, 32768, 65535, 0],
    };
    let color = ColorMetadataRef {
        icc: None,
        cicp: Some(Cicp {
            primaries: 1,
            transfer: 13,
            matrix: 0,
        }),
    };
    let view = legacy.view_ex(color).unwrap();
    let mut options = EncodeOptionsEx {
        native: options(12, NativeModel::IdentityGbr),
        precision: PrecisionPolicy::Exact,
        color: enc::ColorPolicyEx::Preserve,
    };
    assert!(encode_ex(view, &options).is_err());
    options.precision = PrecisionPolicy::Nearest {
        ties: TieRule::AwayFromZero,
    };
    let encoded = encode_ex(view, &options).unwrap();
    assert_eq!(encoded.report.input_depth, 16);
    assert_eq!(encoded.report.output_depth, 12);
    let decoded = decode_frame_bytes_strict_with_limits(&encoded.bytes, &limits()).unwrap();
    let p = &decoded.frame().buffers.planes;
    assert_eq!(p[0].samples, [2048]);
    assert_eq!(p[1].samples, [4095]);
    assert_eq!(p[2].samples, [0]);
    assert_eq!(p[3].samples, [0]);
}

// Build item graphs with recomputed lengths/offsets so preservation tests do not
// depend on malformed containers or opaque binary fixtures.
struct Item {
    payload: Vec<u8>,
    av1c: Vec<u8>,
    bits: u8,
    nclx: Option<enc::NclxColorInformation>,
    alpha: bool,
}
fn test_item(bytes: &[u8], o: &enc::NativeEncodeOptions, alpha: bool) -> Item {
    let i = container::parse_avif(bytes).unwrap();
    Item {
        payload: i.primary_item_payload,
        av1c: i.av1_config.unwrap(),
        bits: o.bit_depth,
        nclx: if alpha {
            None
        } else {
            o.color_information.nclx
        },
        alpha,
    }
}
fn make_items(items: &[Item], extra_meta: &[u8], extra_props: &[Vec<u8>]) -> Vec<u8> {
    let mut props = Vec::new();
    let mut ipma = (items.len() as u32).to_be_bytes().to_vec();
    let mut infe = Vec::new();
    let mut refs = Vec::new();
    for (index, it) in items.iter().enumerate() {
        let id = (index + 1) as u16;
        let start = props.len() + 1;
        props.push(full(
            b"ispe",
            &[5u32.to_be_bytes(), 3u32.to_be_bytes()].concat(),
        ));
        let mut pixi = vec![if it.alpha { 1 } else { 3 }];
        pixi.extend(std::iter::repeat_n(it.bits, usize::from(pixi[0])));
        props.push(full(b"pixi", &pixi));
        props.push(boxed(b"av1C", &it.av1c));
        if let Some(c) = it.nclx {
            props.push(boxed(
                b"colr",
                &[
                    b"nclx".as_slice(),
                    &c.color_primaries.to_be_bytes(),
                    &c.transfer_characteristics.to_be_bytes(),
                    &c.matrix_coefficients.to_be_bytes(),
                    &[if c.full_range_flag { 128 } else { 0 }],
                ]
                .concat(),
            ));
        }
        if it.alpha {
            props.push(full(
                b"auxC",
                b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha\0",
            ));
            refs.extend(boxed(
                b"auxl",
                &[id.to_be_bytes(), 1u16.to_be_bytes(), 1u16.to_be_bytes()].concat(),
            ));
        }
        if index == 0 {
            props.extend(extra_props.iter().cloned());
        }
        ipma.extend(id.to_be_bytes());
        ipma.push((props.len() + 1 - start) as u8);
        for p in start..=props.len() {
            ipma.push(p as u8);
        }
        let parts: &[&[u8]] = &[
            &[2, 0, 0, 0],
            &id.to_be_bytes(),
            &[0, 0],
            b"av01",
            if it.alpha { b"Alpha\0" } else { b"Color\0" },
        ];
        infe.extend(boxed(b"infe", &parts.concat()));
    }
    let iprp = boxed(
        b"iprp",
        &[boxed(b"ipco", &props.concat()), full(b"ipma", &ipma)].concat(),
    );
    let iinf = full(
        b"iinf",
        &[&(items.len() as u16).to_be_bytes()[..], &infe].concat(),
    );
    let mut meta = vec![0; 4];
    meta.extend(full(
        b"hdlr",
        &[
            0, 0, 0, 0, b'p', b'i', b'c', b't', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b'P', 0,
        ],
    ));
    meta.extend(full(b"pitm", &1u16.to_be_bytes()));
    meta.extend(iinf);
    meta.extend(iprp);
    if !refs.is_empty() {
        meta.extend(full(b"iref", &refs));
    }
    meta.extend(extra_meta);
    let ftyp = boxed(b"ftyp", b"avif\0\0\0\0avifmif1miaf");
    let iloc_len = 16 + 14 * items.len();
    let mut offset = (ftyp.len() + 8 + meta.len() + iloc_len + 8) as u32;
    let mut locations = vec![0x44, 0];
    locations.extend((items.len() as u16).to_be_bytes());
    let mut mdat = Vec::new();
    for (index, it) in items.iter().enumerate() {
        locations.extend(((index + 1) as u16).to_be_bytes());
        locations.extend([0, 0, 0, 1]);
        locations.extend(offset.to_be_bytes());
        locations.extend((it.payload.len() as u32).to_be_bytes());
        offset += it.payload.len() as u32;
        mdat.extend(&it.payload);
    }
    let iloc = full(b"iloc", &locations);
    assert_eq!(iloc.len(), iloc_len);
    meta.extend(iloc);
    [ftyp, boxed(b"meta", &meta), boxed(b"mdat", &mdat)].concat()
}

fn with_metadata(item: &mut Item, config: bool, obu: &[u8]) {
    if config {
        item.av1c.extend_from_slice(obu);
    } else {
        let sequence = obu::find_obu_payload(&item.payload, obu::ObuType::SequenceHeader)
            .unwrap()
            .unwrap();
        let end = sequence.as_ptr() as usize - item.payload.as_ptr() as usize + sequence.len();
        item.payload.splice(end..end, obu.iter().copied());
    }
}

#[test]
fn metadata_obus_require_explicit_drop_in_primary_config_and_alpha() {
    let mut hdr = options(10, NativeModel::Yuv420);
    hdr.color_information.nclx = Some(enc::NclxColorInformation {
        color_primaries: 9,
        transfer_characteristics: 16,
        matrix_coefficients: 9,
        full_range_flag: true,
    });
    let ordinary = options(8, NativeModel::IdentityGbr);
    let alpha_options = options(8, NativeModel::Monochrome);
    for (config, alpha, metadata) in [
        (false, false, &[0x2a, 6, 1, 0x03, 0xe8, 0, 0xc8, 0x80][..]), // HDR_CLL 1000/200
        (true, false, &[0x2a, 2, 127, 0x80][..]),                     // unknown metadata type
        (false, true, &[0x2a, 2, 127, 0x80][..]),
        (true, true, &[0x2a, 2, 127, 0x80][..]),
    ] {
        let opts = if alpha { &ordinary } else { &hdr };
        let model = if alpha {
            NativeModel::IdentityGbr
        } else {
            NativeModel::Yuv420
        };
        let (bytes, _) = source(opts.bit_depth, model, false, opts);
        let mut items = vec![test_item(&bytes, opts, false)];
        if alpha {
            let (bytes, _) = source(8, NativeModel::Monochrome, false, &alpha_options);
            items.push(test_item(&bytes, &alpha_options, true));
        }
        with_metadata(items.last_mut().unwrap(), config, metadata);
        let input = make_items(&items, &[], &[]);
        let image = decode_native_ex(&input, &limits()).unwrap();
        assert_eq!(
            image
                .context()
                .information()
                .source_inventory()
                .av1_metadata_obus,
            1
        );
        for run in [false, true] {
            let error = if run {
                transcode_native(&image, opts, &PreservationPolicy::default(), &limits())
                    .unwrap_err()
            } else {
                preflight_native_transcode(&image, opts, &PreservationPolicy::default())
                    .unwrap_err()
            };
            assert!(matches!(
                error,
                PreservationError::PreservationUnsupported(PreservationFeature::Av1Metadata)
            ));
        }
        let policy = PreservationPolicy {
            allow_drop: vec![PreservationFeature::Av1Metadata],
            ..Default::default()
        };
        let preview = preflight_native_transcode(&image, opts, &policy).unwrap();
        let output = transcode_native(&image, opts, &policy, &limits()).unwrap();
        assert_eq!(preview.dropped, [PreservationFeature::Av1Metadata]);
        assert_eq!(preview.dropped_av1_metadata_obus, 1);
        assert_eq!(output.report.dropped_av1_metadata_obus, 1);
        assert_eq!(output.report.dropped, preview.dropped);
        assert_eq!(
            parse_native_info(&output.bytes, &limits())
                .unwrap()
                .source_inventory()
                .av1_metadata_obus,
            0
        );
        let decoded = decode_native_ex(&output.bytes, &limits()).unwrap();
        assert_eq!(decoded.view().unwrap(), image.view().unwrap());
    }
}

#[test]
fn malformed_metadata_obu_boundaries_are_rejected_even_with_drop_permission() {
    let opts = options(8, NativeModel::IdentityGbr);
    let (bytes, _) = source(8, NativeModel::IdentityGbr, false, &opts);
    for config in [false, true] {
        for metadata in [&[0x2a, 0x80][..], &[0x2a, 10, 1, 0x80][..]] {
            let mut item = test_item(&bytes, &opts, false);
            with_metadata(&mut item, config, metadata);
            // Put the broken item-data OBU at the end to prevent following frame
            // bytes from accidentally completing a truncated LEB128/payload.
            if !config {
                item.payload = vec![0x2a, 0x80];
            }
            let input = make_items(&[item], &[], &[]);
            assert!(parse_native_info(&input, &limits()).is_err());
            assert!(decode_native_ex(&input, &limits()).is_err());
        }
    }
}

#[test]
fn additional_alpha_drop_reports_source_ids_and_selected_samples_in_both_orders() {
    let opts = options(8, NativeModel::IdentityGbr);
    let (primary, _) = source(8, NativeModel::IdentityGbr, false, &opts);
    let ao = options(8, NativeModel::Monochrome);
    let (_, mut a) = source(8, NativeModel::Monochrome, false, &ao);
    a.planes[0].samples.fill(21);
    let first = enc::encode_native_bytes(&a, &ao).unwrap();
    a.planes[0].samples.fill(97);
    let second = enc::encode_native_bytes(&a, &ao).unwrap();
    for (a, b, selected) in [(&first, &second, 21), (&second, &first, 97)] {
        let input = make_items(
            &[
                test_item(&primary, &opts, false),
                test_item(a, &ao, true),
                test_item(b, &ao, true),
            ],
            &[],
            &[],
        );
        let image = decode_native_ex(&input, &limits()).unwrap();
        assert_eq!(
            image
                .context()
                .information()
                .info()
                .alpha_auxiliary_items
                .len(),
            2
        );
        for allow_drop in [vec![], vec![PreservationFeature::ExtraItems]] {
            let policy = PreservationPolicy {
                allow_drop,
                ..Default::default()
            };
            assert!(matches!(
                preflight_native_transcode(&image, &opts, &policy),
                Err(PreservationError::PreservationUnsupported(
                    PreservationFeature::AdditionalAlphaItems
                ))
            ));
            assert!(transcode_native(&image, &opts, &policy, &limits()).is_err());
        }
        let policy = PreservationPolicy {
            allow_drop: vec![PreservationFeature::AdditionalAlphaItems],
            ..Default::default()
        };
        let preview = preflight_native_transcode(&image, &opts, &policy).unwrap();
        assert_eq!(preview.retained_alpha_item_id, Some(2));
        assert_eq!(preview.dropped_alpha_item_ids, [3]);
        assert_eq!(preview.dropped, [PreservationFeature::AdditionalAlphaItems]);
        let output = transcode_native(&image, &opts, &policy, &limits()).unwrap();
        assert_eq!(output.report.dropped_alpha_item_ids, [3]);
        let decoded = decode_native_ex(&output.bytes, &limits()).unwrap();
        let info = decoded.context().information();
        assert_eq!(info.item_count(), 2);
        assert_eq!(info.info().alpha_auxiliary_items.len(), 1);
        let ImageViewEx::NativePlanar(view) = decoded.view().unwrap() else {
            panic!()
        };
        assert_eq!(
            view.planes[3].unwrap().sample(0, 0, 0, 1).unwrap(),
            selected
        );
        assert_eq!(decoded.view().unwrap(), image.view().unwrap());
    }
}

#[test]
fn dropping_three_source_alphas_records_every_omitted_id() {
    let opts = options(8, NativeModel::IdentityGbr);
    let (primary, _) = source(8, NativeModel::IdentityGbr, false, &opts);
    let ao = options(8, NativeModel::Monochrome);
    let (alpha, _) = source(8, NativeModel::Monochrome, false, &ao);
    let input = make_items(
        &[
            test_item(&primary, &opts, false),
            test_item(&alpha, &ao, true),
            test_item(&alpha, &ao, true),
            test_item(&alpha, &ao, true),
        ],
        &[],
        &[],
    );
    let image = decode_native_ex(&input, &limits()).unwrap();
    let policy = PreservationPolicy {
        allow_drop: vec![PreservationFeature::AdditionalAlphaItems],
        ..Default::default()
    };
    let output = transcode_native(&image, &opts, &policy, &limits()).unwrap();
    assert_eq!(output.report.retained_alpha_item_id, Some(2));
    assert_eq!(output.report.dropped_alpha_item_ids, [3, 4]);
    assert_eq!(
        output.report.dropped,
        [PreservationFeature::AdditionalAlphaItems]
    );
    assert_eq!(
        parse_native_info(&output.bytes, &limits())
            .unwrap()
            .item_count(),
        2
    );
}

#[test]
fn all_nonempty_entity_group_lists_require_drop_and_bad_bounds_fail() {
    let opts = options(8, NativeModel::IdentityGbr);
    let (bytes, _) = source(8, NativeModel::IdentityGbr, false, &opts);
    let entity = [1u32.to_be_bytes(), 1u32.to_be_bytes(), 1u32.to_be_bytes()].concat();
    let unknown = full(b"zzzz", &entity);
    let known = full(b"altr", &entity);
    for groups in [
        unknown.clone(),
        known.clone(),
        [unknown.clone(), known].concat(),
    ] {
        let input = make_items(
            &[test_item(&bytes, &opts, false)],
            &boxed(b"grpl", &groups),
            &[],
        );
        // Legacy decoding continues to ignore unknown groups, as AVIF recommends.
        assert!(decode_frame_bytes(&input).is_ok());
        let image = decode_native_ex(&input, &limits()).unwrap();
        assert!(
            image
                .context()
                .information()
                .source_inventory()
                .entity_groups
        );
        assert!(matches!(
            preflight_native_transcode(&image, &opts, &PreservationPolicy::default()),
            Err(PreservationError::PreservationUnsupported(
                PreservationFeature::EntityGroups
            ))
        ));
        let policy = PreservationPolicy {
            allow_drop: vec![PreservationFeature::EntityGroups],
            ..Default::default()
        };
        let output = transcode_native(&image, &opts, &policy, &limits()).unwrap();
        assert_eq!(output.report.dropped, [PreservationFeature::EntityGroups]);
        assert!(
            !parse_native_info(&output.bytes, &limits())
                .unwrap()
                .source_inventory()
                .entity_groups
        );
    }
    let input = make_items(
        &[test_item(&bytes, &opts, false)],
        &boxed(b"grpl", &[]),
        &[],
    );
    let image = decode_native_ex(&input, &limits()).unwrap();
    assert!(
        preflight_native_transcode(&image, &opts, &PreservationPolicy::default())
            .unwrap()
            .dropped
            .is_empty()
    );
    for broken in [&[0, 0, 0][..], &[0, 0, 0, 40, b'z', b'z', b'z', b'z'][..]] {
        let input = make_items(
            &[test_item(&bytes, &opts, false)],
            &boxed(b"grpl", broken),
            &[],
        );
        assert!(parse_native_info(&input, &limits()).is_err());
    }
}
