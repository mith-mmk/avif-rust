use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};

mod support;

use avif_rust::av1::{
    alloc_frame_buffers, build_still_decode_plan, decode_luma_root_block_prefix,
    parse_frame_header, parse_sequence_header, parse_tile_group,
};
use avif_rust::container::parse_avif;
use avif_rust::obu::{ObuType, find_obu_payload};
use support::{
    assert_exact_samples, assert_rgba8_max_error, assert_rgba16_max_error, read_u16le_samples,
};

const ORACLE_MANIFEST: &str = "oracles.csv";
const SOURCE_MANIFEST: &str = "oracles.sources.csv";
const SOURCE_MANIFEST_HEADER: &str = "id,source,sha256,plane_format,generated_by";
const REQUIRED_STRICT_FIXTURE_IDS: [&str; 7] = [
    "BlackLossless",
    "filter-disabled-gbr",
    "filter-disabled-residual",
    "filter-disabled-partition",
    "filter-disabled-directional",
    "filter-disabled-palette",
    "WML2Viewer",
];
const ORACLE_HEADER: &str =
    "id,avif,width,height,bit_depth,plane_paths,plane_widths,plane_heights,rgba8,rgba16";

fn oracle_requirement_enabled(value: Option<&str>) -> bool {
    matches!(value, Some("1") | Some("true") | Some("TRUE"))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct OracleEntry {
    id: String,
    avif: String,
    width: usize,
    height: usize,
    bit_depth: u8,
    plane_paths: Vec<String>,
    plane_widths: Vec<usize>,
    plane_heights: Vec<usize>,
    rgba8: String,
    rgba16: String,
}

fn test_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data")
}

fn assert_palette_fixture_exercises_palette(avif_data: &[u8], fixture_id: &str) {
    let info = parse_avif(avif_data).expect("palette fixture container should parse");
    let sequence_payload = find_obu_payload(&info.primary_item_payload, ObuType::SequenceHeader)
        .expect("palette fixture sequence OBU lookup should succeed")
        .expect("palette fixture sequence OBU should exist");
    let sequence =
        parse_sequence_header(sequence_payload).expect("palette fixture sequence should parse");
    let frame_payload = find_obu_payload(&info.primary_item_payload, ObuType::Frame)
        .expect("palette fixture frame OBU lookup should succeed")
        .expect("palette fixture frame OBU should exist");
    let frame =
        parse_frame_header(frame_payload, &sequence).expect("palette fixture frame should parse");
    let tile_group = parse_tile_group(
        frame_payload,
        frame.uncompressed_header_bits,
        &frame.tile_info,
    )
    .expect("palette fixture tile group should parse");
    let plan = build_still_decode_plan(&sequence, &frame, &tile_group)
        .expect("palette fixture decode plan should build");
    let mut buffers = alloc_frame_buffers(&plan).expect("palette fixture buffers should allocate");
    let prefix = decode_luma_root_block_prefix(
        frame_payload,
        &tile_group,
        &sequence,
        &frame,
        &plan,
        &mut buffers,
        plan.width.saturating_mul(plan.height),
    )
    .expect("palette fixture block diagnostics should decode");

    assert!(
        prefix
            .blocks
            .iter()
            .any(|block| block.palette.has_palette()),
        "{fixture_id} must contain at least one palette block"
    );
    assert!(
        prefix
            .blocks
            .iter()
            .any(|block| block.palette.has_non_empty_color_map()),
        "{fixture_id} must contain a decoded palette color map"
    );
}

fn parse_oracle_manifest(input: &str) -> Result<Vec<OracleEntry>, String> {
    let mut lines = input.lines().filter(|line| {
        let trimmed = line.trim();
        !trimmed.is_empty() && !trimmed.starts_with('#')
    });
    let header = lines
        .next()
        .ok_or_else(|| "oracle manifest is empty".to_string())?;
    if header != ORACLE_HEADER {
        return Err(format!("unexpected oracle manifest header: {header}"));
    }

    let entries = lines
        .enumerate()
        .map(|(line_index, line)| parse_oracle_manifest_line(line_index + 2, line))
        .collect::<Result<Vec<_>, _>>()?;
    validate_oracle_entries(&entries)?;
    Ok(entries)
}

fn parse_oracle_manifest_line(line_number: usize, line: &str) -> Result<OracleEntry, String> {
    let columns: Vec<_> = line.split(',').collect();
    if columns.len() != 10 {
        return Err(format!(
            "oracle manifest line {line_number} has {} columns, expected 10",
            columns.len()
        ));
    }

    let id = required_column(line_number, "id", columns[0])?;
    validate_id(line_number, &id)?;
    let avif = required_path_column(line_number, "avif", columns[1])?;
    let width = parse_usize_column(line_number, "width", columns[2])?;
    let height = parse_usize_column(line_number, "height", columns[3])?;
    let bit_depth = parse_u8_column(line_number, "bit_depth", columns[4])?;
    let plane_paths = parse_path_list(line_number, "plane_paths", columns[5])?;
    let plane_widths = parse_usize_list(line_number, "plane_widths", columns[6])?;
    let plane_heights = parse_usize_list(line_number, "plane_heights", columns[7])?;
    let rgba8 = required_path_column(line_number, "rgba8", columns[8])?;
    let rgba16 = required_path_column(line_number, "rgba16", columns[9])?;

    if plane_paths.len() != plane_widths.len() || plane_paths.len() != plane_heights.len() {
        return Err(format!(
            "oracle manifest line {line_number} plane path/size counts differ"
        ));
    }

    Ok(OracleEntry {
        id,
        avif,
        width,
        height,
        bit_depth,
        plane_paths,
        plane_widths,
        plane_heights,
        rgba8,
        rgba16,
    })
}

fn validate_oracle_entries(entries: &[OracleEntry]) -> Result<(), String> {
    let mut ids = HashSet::new();
    for entry in entries {
        if !ids.insert(entry.id.as_str()) {
            return Err(format!("duplicate oracle fixture id: {}", entry.id));
        }
        if !matches!(entry.bit_depth, 8 | 10 | 12) {
            return Err(format!(
                "oracle fixture {} has unsupported bit depth {}",
                entry.id, entry.bit_depth
            ));
        }
        if !matches!(entry.plane_paths.len(), 1 | 3 | 4) {
            return Err(format!(
                "oracle fixture {} has unsupported plane count {}",
                entry.id,
                entry.plane_paths.len()
            ));
        }
        validate_oracle_dimensions(entry)?;
    }
    Ok(())
}

fn validate_required_strict_fixture_ids(entries: &[OracleEntry]) -> Result<(), String> {
    if entries.is_empty() {
        return Err("strict oracle manifest must contain at least one fixture".to_string());
    }

    let ids: HashSet<_> = entries.iter().map(|entry| entry.id.as_str()).collect();
    let missing: Vec<_> = REQUIRED_STRICT_FIXTURE_IDS
        .iter()
        .copied()
        .filter(|id| !ids.contains(id))
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "strict oracle manifest is missing required fixtures: {}",
            missing.join(", ")
        ))
    }
}

fn validate_source_manifest_text(input: &str) -> Result<(), String> {
    let mut lines = input.lines().filter(|line| {
        let trimmed = line.trim();
        !trimmed.is_empty() && !trimmed.starts_with('#')
    });
    let header = lines
        .next()
        .ok_or_else(|| "source manifest is empty".to_string())?;
    if header != SOURCE_MANIFEST_HEADER {
        return Err(format!("unexpected source manifest header: {header}"));
    }

    let mut ids = HashSet::new();
    for (line_index, line) in lines.enumerate() {
        let line_number = line_index + 2;
        let columns: Vec<_> = line.split(',').collect();
        if columns.len() != 5 {
            return Err(format!(
                "source manifest line {line_number} has {} columns, expected 5",
                columns.len()
            ));
        }
        let id = required_column(line_number, "id", columns[0])?;
        if !ids.insert(id.clone()) {
            return Err(format!("duplicate source manifest id: {id}"));
        }
        required_column(line_number, "source", columns[1])?;
        let hash = required_column(line_number, "sha256", columns[2])?;
        if hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(format!(
                "source manifest line {line_number} has invalid sha256"
            ));
        }
        required_column(line_number, "plane_format", columns[3])?;
        required_column(line_number, "generated_by", columns[4])?;
    }

    let missing: Vec<_> = REQUIRED_STRICT_FIXTURE_IDS
        .iter()
        .copied()
        .filter(|id| !ids.contains(*id))
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "source manifest is missing required fixtures: {}",
            missing.join(", ")
        ))
    }
}

fn validate_oracle_dimensions(entry: &OracleEntry) -> Result<(), String> {
    entry
        .width
        .checked_mul(entry.height)
        .ok_or_else(|| format!("oracle fixture {} frame sample count overflows", entry.id))?;
    entry
        .width
        .checked_mul(entry.height)
        .and_then(|sample_count| sample_count.checked_mul(4))
        .ok_or_else(|| format!("oracle fixture {} RGBA sample count overflows", entry.id))?;

    for (index, (&width, &height)) in entry
        .plane_widths
        .iter()
        .zip(entry.plane_heights.iter())
        .enumerate()
    {
        if width > entry.width || height > entry.height {
            return Err(format!(
                "oracle fixture {} plane {index} dimensions exceed frame dimensions",
                entry.id
            ));
        }
        width.checked_mul(height).ok_or_else(|| {
            format!(
                "oracle fixture {} plane {index} sample count overflows",
                entry.id
            )
        })?;
    }
    Ok(())
}

fn required_column(line_number: usize, name: &str, value: &str) -> Result<String, String> {
    if value.trim().is_empty() {
        return Err(format!("oracle manifest line {line_number} missing {name}"));
    }
    Ok(value.trim().to_string())
}

fn required_path_column(line_number: usize, name: &str, value: &str) -> Result<String, String> {
    let path = required_column(line_number, name, value)?;
    validate_relative_path(line_number, name, &path)?;
    Ok(path)
}

fn validate_id(line_number: usize, id: &str) -> Result<(), String> {
    if id
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        Ok(())
    } else {
        Err(format!("oracle manifest line {line_number} has invalid id"))
    }
}

fn validate_relative_path(line_number: usize, name: &str, value: &str) -> Result<(), String> {
    let path = Path::new(value);
    if path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::Prefix(_)))
    {
        return Err(format!(
            "oracle manifest line {line_number} {name} must be relative and stay inside test_data"
        ));
    }
    Ok(())
}

fn parse_path_list(line_number: usize, name: &str, value: &str) -> Result<Vec<String>, String> {
    let values: Vec<_> = value.split(';').filter(|value| !value.is_empty()).collect();
    if values.is_empty() {
        return Err(format!("oracle manifest line {line_number} missing {name}"));
    }
    values
        .into_iter()
        .map(|value| required_path_column(line_number, name, value))
        .collect()
}

fn parse_usize_list(line_number: usize, name: &str, value: &str) -> Result<Vec<usize>, String> {
    let values: Vec<_> = value.split(';').filter(|value| !value.is_empty()).collect();
    if values.is_empty() {
        return Err(format!("oracle manifest line {line_number} missing {name}"));
    }
    values
        .into_iter()
        .map(|value| parse_usize_column(line_number, name, value))
        .collect()
}

fn parse_usize_column(line_number: usize, name: &str, value: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| format!("oracle manifest line {line_number} has invalid {name}"))
        .and_then(|value| {
            if value == 0 {
                Err(format!(
                    "oracle manifest line {line_number} {name} must be non-zero"
                ))
            } else {
                Ok(value)
            }
        })
}

fn parse_u8_column(line_number: usize, name: &str, value: &str) -> Result<u8, String> {
    value
        .parse::<u8>()
        .map_err(|_| format!("oracle manifest line {line_number} has invalid {name}"))
}

fn test_data_path(root: &Path, relative: &str) -> PathBuf {
    root.join(relative)
}

#[test]
fn oracle_manifest_parser_accepts_schema_template() {
    let manifest = format!(
        "{ORACLE_HEADER}\nfixture,images/sample.avif,2,1,8,planes/y.u16le;planes/u.u16le;planes/v.u16le,2;2;2,1;1;1,rgba/sample.rgba,rgba/sample.rgba16le\n"
    );

    let entries = parse_oracle_manifest(&manifest).unwrap();

    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].id, "fixture");
    assert_eq!(entries[0].plane_paths.len(), 3);
    assert_eq!(entries[0].plane_widths, vec![2, 2, 2]);
    assert_eq!(entries[0].plane_heights, vec![1, 1, 1]);
}

#[test]
fn oracle_manifest_parser_rejects_unsafe_paths() {
    let manifest = format!(
        "{ORACLE_HEADER}\nfixture,../sample.avif,2,1,8,planes/y.u16le,2,1,rgba/sample.rgba,rgba/sample.rgba16le\n"
    );

    let err = parse_oracle_manifest(&manifest).unwrap_err();

    assert!(err.contains("relative"));
}

#[test]
fn oracle_manifest_parser_rejects_duplicate_ids() {
    let manifest = format!(
        "{ORACLE_HEADER}\nfixture,images/a.avif,2,1,8,planes/y.u16le,2,1,rgba/a.rgba,rgba/a.rgba16le\nfixture,images/b.avif,2,1,8,planes/y.u16le,2,1,rgba/b.rgba,rgba/b.rgba16le\n"
    );

    let err = parse_oracle_manifest(&manifest).unwrap_err();

    assert!(err.contains("duplicate"));
}

#[test]
fn oracle_manifest_parser_rejects_unsupported_bit_depth_and_plane_count() {
    let bad_depth = format!(
        "{ORACLE_HEADER}\nfixture,images/a.avif,2,1,16,planes/y.u16le,2,1,rgba/a.rgba,rgba/a.rgba16le\n"
    );
    let err = parse_oracle_manifest(&bad_depth).unwrap_err();
    assert!(err.contains("bit depth"));

    let bad_plane_count = format!(
        "{ORACLE_HEADER}\nfixture,images/a.avif,2,1,8,planes/y.u16le;planes/u.u16le,2;1,1;1,rgba/a.rgba,rgba/a.rgba16le\n"
    );
    let err = parse_oracle_manifest(&bad_plane_count).unwrap_err();
    assert!(err.contains("plane count"));
}

#[test]
fn oracle_manifest_parser_rejects_plane_dimensions_exceeding_frame() {
    let manifest = format!(
        "{ORACLE_HEADER}\nfixture,images/a.avif,2,1,8,planes/y.u16le,3,1,rgba/a.rgba,rgba/a.rgba16le\n"
    );

    let err = parse_oracle_manifest(&manifest).unwrap_err();

    assert!(err.contains("dimensions"));
}

#[test]
fn oracle_manifest_parser_rejects_sample_count_overflow() {
    let manifest = format!(
        "{ORACLE_HEADER}\nfixture,images/a.avif,{max},{max},8,planes/y.u16le,{max},{max},rgba/a.rgba,rgba/a.rgba16le\n",
        max = usize::MAX
    );

    let err = parse_oracle_manifest(&manifest).unwrap_err();

    assert!(err.contains("overflows"));
}

#[test]
fn strict_oracle_validation_rejects_empty_manifest() {
    let entries = Vec::new();

    let err = validate_required_strict_fixture_ids(&entries).unwrap_err();

    assert!(err.contains("at least one fixture"));
}

#[test]
fn strict_oracle_validation_rejects_header_only_manifest() {
    let entries = parse_oracle_manifest(ORACLE_HEADER).unwrap();

    let err = validate_required_strict_fixture_ids(&entries).unwrap_err();

    assert!(err.contains("at least one fixture"));
}

#[test]
fn strict_oracle_validation_rejects_missing_required_fixture() {
    let manifest = format!(
        "{ORACLE_HEADER}\nBlackLossless,images/BlackLossless.avif,64,64,8,planes/y.u16le;planes/u.u16le;planes/v.u16le,64;64;64,64;64;64,rgba/a.rgba,rgba/a.rgba16le\n"
    );
    let entries = parse_oracle_manifest(&manifest).unwrap();

    let err = validate_required_strict_fixture_ids(&entries).unwrap_err();

    assert!(err.contains("filter-disabled-gbr"));
}

#[test]
fn strict_source_manifest_rejects_invalid_hash() {
    let manifest = format!(
        "{SOURCE_MANIFEST_HEADER}\nBlackLossless,BlackLossless.avif,not-a-hash,gbrp,generate_oracles.ps1\n"
    );

    let err = validate_source_manifest_text(&manifest).unwrap_err();

    assert!(err.contains("invalid sha256"));
}

#[test]
fn external_supported_stream_oracles_match_when_present() {
    let root = test_data_dir();
    let manifest_path = root.join(ORACLE_MANIFEST);
    let require_oracles =
        oracle_requirement_enabled(std::env::var("AVIF_REQUIRE_ORACLES").ok().as_deref());
    if !manifest_path.exists() {
        assert!(
            !require_oracles,
            "AVIF_REQUIRE_ORACLES is enabled but {} is missing",
            manifest_path.display()
        );
        return;
    }

    let manifest = std::fs::read_to_string(&manifest_path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", manifest_path.display()));
    let entries = parse_oracle_manifest(&manifest).expect("oracle manifest should be valid");
    if require_oracles {
        validate_required_strict_fixture_ids(&entries)
            .unwrap_or_else(|err| panic!("strict oracle validation failed: {err}"));
        let source_manifest_path = root.join(SOURCE_MANIFEST);
        let source_manifest =
            std::fs::read_to_string(&source_manifest_path).unwrap_or_else(|err| {
                panic!(
                    "strict oracle source manifest {} is unavailable: {err}",
                    source_manifest_path.display()
                )
            });
        validate_source_manifest_text(&source_manifest)
            .unwrap_or_else(|err| panic!("strict source manifest validation failed: {err}"));
    }

    for entry in entries {
        let avif_data = std::fs::read(test_data_path(&root, &entry.avif))
            .unwrap_or_else(|err| panic!("failed to read AVIF fixture {}: {err}", entry.id));
        let decoded =
            avif_rust::decode_frame_bytes(&avif_data).expect("AVIF fixture should decode");

        if entry.id == "filter-disabled-palette" {
            assert_palette_fixture_exercises_palette(&avif_data, &entry.id);
        }

        assert_eq!(decoded.width, entry.width, "{} width", entry.id);
        assert_eq!(decoded.height, entry.height, "{} height", entry.id);
        assert_eq!(decoded.bit_depth, entry.bit_depth, "{} bit depth", entry.id);
        assert_eq!(
            decoded.buffers.planes.len(),
            entry.plane_paths.len(),
            "{} plane count",
            entry.id
        );

        for (plane_index, plane) in decoded.buffers.planes.iter().enumerate() {
            let width = entry.plane_widths[plane_index];
            let height = entry.plane_heights[plane_index];
            let expected_samples = read_u16le_samples(
                &test_data_path(&root, &entry.plane_paths[plane_index]),
                width * height,
            );

            assert_eq!(plane.layout.width, width, "{} plane width", entry.id);
            assert_eq!(plane.layout.height, height, "{} plane height", entry.id);
            assert_exact_samples(
                &plane.samples,
                &expected_samples,
                &format!("{} plane {plane_index} samples", entry.id),
            );
        }

        let rgba8 = decoded.to_rgba8().expect("fixture should convert to RGBA8");
        let expected_rgba8 = std::fs::read(test_data_path(&root, &entry.rgba8))
            .unwrap_or_else(|err| panic!("failed to read RGBA8 fixture {}: {err}", entry.id));
        assert_rgba8_max_error(&rgba8.rgba, &expected_rgba8, 1, &entry.id);

        let rgba16 = decoded
            .to_rgba16()
            .expect("fixture should convert to RGBA16");
        let expected_rgba16 = read_u16le_samples(
            &test_data_path(&root, &entry.rgba16),
            entry.width * entry.height * 4,
        );
        assert_rgba16_max_error(&rgba16.rgba, &expected_rgba16, 1, &entry.id);
    }
}

#[test]
fn oracle_requirement_flag_accepts_only_explicit_true_values() {
    assert!(oracle_requirement_enabled(Some("1")));
    assert!(oracle_requirement_enabled(Some("true")));
    assert!(oracle_requirement_enabled(Some("TRUE")));
    assert!(!oracle_requirement_enabled(Some("0")));
    assert!(!oracle_requirement_enabled(None));
}
