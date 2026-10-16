"""Build local archive chain without publishing or changing source worktrees.

Registry path patches apply only to subprocess commands in temporary snapshots.
This is NOT a verification of dependencies published on crates.io.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tomllib

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--encoder', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--offline', action='store_true')
args = parser.parse_args()
repo = Path(__file__).resolve().parents[1]
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=False)
env = os.environ.copy()
env['CARGO_TARGET_DIR'] = str(output / '.test-target')
flags = ['--offline'] if args.offline else []
commands = []


def run(command, cwd, name):
    result = subprocess.run(command, cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (output / (name + '.log')).write_text(result.stdout)
    commands.append({'name': name, 'command': command, 'exit_code': result.returncode})
    if result.returncode:
        raise RuntimeError(f'{name} failed; see {output / (name + ".log")}')
    return result.stdout


def snapshot(source, name):
    directory = output / name
    directory.mkdir()
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    archive = output / (name + '.tar')
    with archive.open('wb') as stream:
        subprocess.run(['git', 'archive', '--format=tar', sha], cwd=source,
                       stdout=stream, check=True)
    with tarfile.open(archive) as bundle:
        bundle.extractall(directory, filter='data')
    return directory, sha


avif, avif_sha = snapshot(repo, '.test-avif')
encoder, encoder_sha = snapshot(args.encoder.resolve(), '.test-encoder')
avif_manifest = tomllib.loads((avif / 'Cargo.toml').read_text())
encoder_manifest = tomllib.loads((encoder / 'Cargo.toml').read_text())
shared_source = avif_manifest['dependencies']['avif-image-buffer']
assert shared_source == encoder_manifest['dependencies']['avif-image-buffer']
assert avif_manifest['dependencies']['avifenc_codec']['rev'] == encoder_sha
for source in (avif / 'crates/avif-image-buffer/src').glob('*.rs'):
    pinned = subprocess.check_output(['git', 'show',
        shared_source['rev'] + ':crates/avif-image-buffer/src/' + source.name], cwd=repo)
    assert pinned == source.read_bytes(), 'local core differs from canonical Git source'


def package(manifest, name, version, patches):
    command = ['cargo', 'package', '--manifest-path', str(manifest), *flags]
    for key, path in patches.items():
        command += ['--config', 'patch.crates-io.' + key + '.path=' + json.dumps(str(path))]
    run(command, manifest.parent, name + '-package')
    directory = output / '.test-target/package' / (name + '-' + version)
    archive = directory.parent / (directory.name + '.crate')
    assert archive.is_file() and directory.is_dir()
    normalized = tomllib.loads((directory / 'Cargo.toml').read_text())
    assert normalized['package']['version'] == version
    for dependency in normalized.get('dependencies', {}).values():
        assert 'version' in dependency and 'git' not in dependency and 'path' not in dependency
    with tarfile.open(archive) as bundle:
        files = bundle.getnames()
    return directory, {'name': name, 'version': version,
        'archive': str(archive), 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'files': files, 'normalized_dependencies': normalized.get('dependencies', {})}


core, core_report = package(avif / 'crates/avif-image-buffer/Cargo.toml',
                            'avif-image-buffer', '0.1.0', {})
patches = {'avif-image-buffer': core}
encoded, encoder_report = package(encoder / 'Cargo.toml', 'avifenc-rust', '0.0.8', patches)
patches['avifenc-rust'] = encoded
decoded, decoder_report = package(avif / 'Cargo.toml', 'avif-rust', '0.0.9', patches)
config = []
for key, path in patches.items():
    config += ['--config', 'patch.crates-io.' + key + '.path=' + json.dumps(str(path))]
consumer = output / '.test-consumer'
consumer.mkdir()
(consumer / 'Cargo.toml').write_text(f'''[package]
name="packaged-avif-consumer"
version="0.0.0"
edition="2024"
[workspace]
[[bin]]
name="check"
path="main.rs"
[dependencies]
avif-rust={{path={json.dumps(str(decoded))},features=["encode"]}}
avifenc-rust={{path={json.dumps(str(encoded))},default-features=false}}
avif-image-buffer={{path={json.dumps(str(core))}}}
''')
(consumer / 'main.rs').write_text('''use avif_image_buffer::*;
fn same(v: avif_rust::buffer::ImageViewEx<'_>) -> avifenc_rust::buffer::ImageViewEx<'_> { v }
fn main() {
    let samples=[17,29,113,0];
    let v=ImageViewEx::Packed(PackedImageViewEx {
        plane:PlaneViewEx{samples:SamplesRef::U8(&samples),layout:PlaneLayoutEx{width:1,height:1,valid_bits:8,offset_bytes:0,row_stride_bytes:4}},
        channels:PackedChannels::Rgba,alpha:AlphaMode::Straight,
        color:ColorMetadataRef{icc:None,cicp:Some(Cicp{primaries:1,transfer:13,matrix:0})},
    });
    let options=avif_rust::EncodeOptionsEx {
        native:avifenc_rust::NativeEncodeOptions {
            bit_depth:8,subsampling:avifenc_rust::NativeSubsampling::Cs444,
            encoder:avifenc_rust::EncoderOptions{lossless:true,color_information:None,..Default::default()},
            color_information:avifenc_rust::NativeColorInformation::from_nclx(avifenc_rust::NclxColorInformation {
                color_primaries:1,transfer_characteristics:13,matrix_coefficients:0,full_range_flag:true}),
        },precision:PrecisionPolicy::Exact,color:avifenc_rust::ColorPolicyEx::Preserve,
    };
    let encoded=avif_rust::encode_ex(same(v),&options).unwrap();
    let limits=avif_rust::NativeDecodeLimits::new(1<<20,16,16,256,1<<20,1<<20,1<<20,8,32,8,4,1);
    let image=avif_rust::decode_native_ex(&encoded.bytes,&limits).unwrap();
    let ImageViewEx::NativePlanar(view)=image.view().unwrap() else {panic!()};
    for (role,value) in [29,113,17,0].into_iter().enumerate() {
        assert_eq!(view.planes[role].unwrap().sample(0,0,0,1).unwrap(),value);
    }
    println!("packaged_type_identity_and_transparent_pixel_roundtrip_ok");
}
''')
run(['cargo', 'run', '--manifest-path', str(consumer / 'Cargo.toml'), *flags, *config],
    consumer, 'packaged-consumer')
metadata = json.loads(run(['cargo', 'metadata', '--format-version', '1', '--filter-platform',
    'x86_64-unknown-linux-gnu', '--manifest-path', str(consumer / 'Cargo.toml'), *flags, *config],
    consumer, 'packaged-metadata'))
nodes = {node['id']: node for node in metadata['resolve']['nodes']}
packages = {package['id']: package for package in metadata['packages']}
shared = [key for key in nodes if packages[key]['name'] == 'avif-image-buffer']
assert len(shared) == 1 and not nodes[shared[0]]['dependencies']
encoder_id = next(key for key in nodes if packages[key]['name'] == 'avifenc-rust')
seen = set()
pending = [encoder_id]
while pending:
    for dependency in nodes[pending.pop()]['dependencies']:
        if dependency not in seen:
            seen.add(dependency)
            pending.append(dependency)
assert not any(packages[key]['name'] == 'avif-rust' for key in seen)
off_tree = run(['cargo', 'tree', '--manifest-path', str(decoded / 'Cargo.toml'),
    '--no-default-features', *flags, *config], decoded, 'packaged-encode-off')
assert all(name not in off_tree for name in ['avifenc-rust', 'rav1e', 'gamut-'])
report = {'kind': 'LOCAL ARCHIVE SUBSTITUTION ONLY; not a crates.io publication check',
    'source_heads': {'avif': avif_sha, 'encoder': encoder_sha},
    'canonical_shared_git_revision': shared_source['rev'],
    'archives': [core_report, encoder_report, decoder_report],
    'one_shared_type_source': True, 'encoder_has_no_decoder_dependency': True,
    'encode_off_excludes_encoder': True, 'packaged_consumer_roundtrip': True,
    'registry_packages_published': False, 'commands': commands}
(output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(output / 'report.json')
