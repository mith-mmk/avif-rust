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
