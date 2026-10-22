use avif_rust::{decode_frame_bytes, decode_frame_bytes_strict};

#[test]
fn synthetic_native_planes_match_independent_libaom_and_dav1d_oracles() {
    for (name, avif, oracle) in [
        (
            "420-8",
            include_bytes!("native-pixel/yuv420p-lossy.avif").as_slice(),
            include_bytes!("native-pixel/yuv420p-lossy.yuv").as_slice(),
        ),
        (
            "422-8",
            include_bytes!("native-pixel/yuv422p-lossy.avif").as_slice(),
            include_bytes!("native-pixel/yuv422p-lossy.yuv").as_slice(),
        ),
        (
            "444-8",
            include_bytes!("native-pixel/yuv444p-lossy.avif").as_slice(),
            include_bytes!("native-pixel/yuv444p-lossy.yuv").as_slice(),
        ),
        (
            "mono-10",
            include_bytes!("native-pixel/gray10le-lossy.avif").as_slice(),
            include_bytes!("native-pixel/gray10le-lossy.yuv").as_slice(),
        ),
        (
            "444-12-lossless",
            include_bytes!("native-pixel/yuv444p12le-lossless.avif").as_slice(),
            include_bytes!("native-pixel/yuv444p12le-lossless.yuv").as_slice(),
        ),
        (
            "422-12-edge",
            include_bytes!("native-pixel/yuv422p12le-lossy.avif").as_slice(),
            include_bytes!("native-pixel/yuv422p12le-lossy.yuv").as_slice(),
        ),
        (
            "420-10-cdef",
            include_bytes!("native-pixel/yuv420p10le-cdef.avif").as_slice(),
            include_bytes!("native-pixel/yuv420p10le-cdef.yuv").as_slice(),
        ),
        (
            "mono-12-sgr",
            include_bytes!("native-pixel/gray12le-sgr.avif").as_slice(),
            include_bytes!("native-pixel/gray12le-sgr.yuv").as_slice(),
        ),
    ] {
        let legacy = decode_frame_bytes(avif).unwrap();
        let strict = decode_frame_bytes_strict(avif).unwrap();
        assert_eq!(legacy, strict, "{name}: strict and legacy");
        assert_eq!(
            (legacy.width, legacy.height),
            if name == "420-10-cdef" || name == "mono-12-sgr" {
                (93, 77)
            } else {
                (85, 73)
            }
        );
        let mut actual = Vec::new();
        for plane in &legacy.buffers.planes {
            for row in plane
                .samples
                .chunks_exact(plane.layout.stride())
                .take(plane.layout.height)
            {
                for &sample in &row[..plane.layout.width] {
                    if legacy.bit_depth > 8 {
                        actual.extend_from_slice(&sample.to_le_bytes());
                    } else {
                        actual.push(sample as u8);
                    }
                }
            }
        }
        assert_eq!(actual.len(), oracle.len(), "{name}: sample count");
        assert!(
            actual == oracle,
            "{name}: native planes differ from independent reference"
        );
    }
}
