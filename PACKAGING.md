# Distribution checks

## Decoder 0.1.1

The 0.1.0 versions of `avif-image-buffer`, `avifenc-rust` and `avif-rust` were
published on 2026-10-17. Decoder 0.1.1 includes the later AVIS decoding fixes and
bundled regressions. Shared and encoder versions and immutable dependency pins
remain unchanged. The known non-lossless AVIS pixel differences and strict
sequence layout limitation remain; see [CHANGELOG.md](CHANGELOG.md).

For this decoder-only release, use a clean, committed checkout and ordinary
registry resolution without patches or local source substitutions:

```sh
cargo fmt --all -- --check
cargo test --locked --release
cargo test --locked --release --features encode --test shared_buffer --test shared_transcode --test avis_regression
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo rustdoc --locked --all-features --lib -- -D warnings
cargo package --locked
cargo publish --dry-run --locked
```

Before the separately authorized real upload, confirm the exact version is absent
from crates.io and the clean source commit is pushed and matches its remote SHA.
Test the extracted archive against registry dependencies, including the embedded
AVIS regressions and optional encode/transcode integration. Check a registry-only
consumer with encode off and on, shared type identity and a lossless round trip.
After upload, verify the crates.io version, archive checksum and clean
`.cargo_vcs_info.json` commit SHA. Do not overwrite a published version.

## Historical shared-buffer 0.1.0 preparation

The remaining sections record the prepublication checkpoint on 2026-10-16.
Their unpublished-version statements and uniform-0.1.0 local-chain script describe
that historical checkpoint, not current registry availability or the decoder-only
0.1.1 release procedure above.

No crates were uploaded, credentials created, tags/releases published or main
branches merged by this work. Versions below are **unpublished candidates**.
The existing published releases remain decoder 0.0.8 and encoder 0.0.7.

## Registry check and publication order

Read-only crates.io API checks on 2026-10-16 returned:

| Package | Registry result | Prepared candidate | Publication prerequisite |
| --- | --- | --- | --- |
| avif-image-buffer | HTTP404: crate does not exist | 0.1.0 | None; std-only |
| avifenc-rust | Latest published 0.0.7 | 0.1.0 | avif-image-buffer 0.1.0 |
| avif-rust | Latest published 0.0.8 | 0.1.0 | shared 0.1.0 and encoder 0.1.0 |

The new encoder API is not in the already published 0.0.7, so a version fallback
to that release is incorrect. A registry version cannot be overwritten.
The shared name was unregistered at lookup time; it was not reserved. A dry run
does not check every server-side publication/ownership rule or create a package.
Actual publication requires separate authorization and server acceptance.

All three candidate versions are 0.1.0. The intended order is **shared -> encoder -> decoder**. The final decoder
package includes an optional encoder dependency; registry package resolution
still needs that candidate to exist even when encode is disabled for the build.

The shared archive contains Cargo manifests/lock, the two model source files,
contract tests, README and MIT license. It contains no codec, AVIF context,
fixture image, backend or native build dependency. Encoder and decoder archives
contain their existing Rust code plus the shared API integration and the
explicit corrections described in SHARED_BUFFER.md. The encoder repository
metadata now names the actual `mith-mmk/avifenc` repository.

## Development versus packaged dependencies

Both development manifests select the same immutable model source:

```toml
avif-image-buffer = { version = "0.1.0", git = "https://github.com/mith-mmk/avif-rust.git", rev = "5c73c7f26c276098ddd761fb20b956302a462001" }
```

The optional decoder-to-encoder dependency is:

```toml
avifenc_codec = { package = "avifenc-rust", version = "0.1.0", git = "https://github.com/mith-mmk/avifenc.git", rev = "e039e990cc5285f7219cedd827493c48a5c22b7d", optional = true, default-features = false }
```

Cargo uses the Git source during development and checks its package version.
When packaging, Cargo removes Git coordinates from the normalized `Cargo.toml`
and keeps the registry requirements. This is Cargo's documented
[multiple locations](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html#multiple-locations)
behavior. `Cargo.toml.orig` intentionally retains the original declarations.

The local model directory remains excluded from the root workspace. Changing
only the decoder's dependency to `path` while the encoder still uses Git would
create distinct Rust types, even at the same version. Both pinned Git consumers
use one model package. After publication, both normalized registry dependencies
use the same compatible 0.1 version family. No encoder dependency points back
to the decoder.

Prefer a complete Git graph or a complete registry graph. If deliberately
mixing sources, a top-level consumer must align the shared source and, when
passing encoder options across crates, the encoder source too. For example,
patch both crates.io packages to the same Git revisions shown above. Confirm
with Cargo metadata and a type assignment; matching version strings alone do
not establish type identity. No patch is installed in these production manifests.

## Checks that can pass before publication

The common crate has no registry dependencies. Its real Cargo archive build
and verification, plus `cargo publish --dry-run`, can complete before it is
published. The dry-run log ends with `aborting upload due to dry run`.

```sh
cargo test --locked --manifest-path crates/avif-image-buffer/Cargo.toml
cargo package --locked --manifest-path crates/avif-image-buffer/Cargo.toml
cargo publish --dry-run --locked --manifest-path crates/avif-image-buffer/Cargo.toml
```

A dry run may need read access to the registry index; offline failure alone is
not a publication restriction. This work performs no real publish.

For a local artifact chain, run from a clean, committed decoder checkout:

```sh
python3 scripts/verify-packages.py --encoder /absolute/encoder/checkout --output /absolute/new/.test-package-check --offline
```

Omit `--offline` if registry dependencies are not cached. The script snapshots
Git HEAD from each checkout; it does not test uncommitted files. It verifies
that both manifests use the same canonical shared Git revision and that local
shared Rust source matches that revision. It packages shared, encoder and
decoder in order, builds the unpacked archives, and compiles/runs an external
consumer using all three packages. That consumer checks shared Rust type
identity and an actual delegated lossless encode/decode with hidden RGB under
alpha0. Cargo metadata also checks one shared package, no reverse encoder
dependency, and no encoder/backend in the decoder's encode-off graph.

Command-scoped path patches supply the **unpacked local archives** in place of
the unpublished crates.io dependencies for these subprocesses only. Source
snapshots and lockfile edits stay under the output directory; production
manifests, user Cargo configuration and credentials are untouched. The normalized
archive manifests are checked to have version requirements and no Git/path
coordinates. This is a **local archive verification**, not a real registry
resolution or publication check. It cannot establish that the names are owned
or that crates.io accepted these archives.

## Checks still requiring publication

Unmodified `cargo package` / `cargo publish --dry-run` for the encoder needs
shared 0.1.0 in the registry. The decoder needs shared 0.1.0 and encoder 0.1.0.
These checks remain blocked before publication; the earlier missing-version
manifest error is fixed. CI retains the real decoder registry package gate,
and separately checks the dependency-free shared archive before it. The gate
is not replaced by a successful staging check or marked allowed-to-fail.

After separately authorized publication, rerun ordinary registry packaging and
check a fresh registry-only consumer with encode both off and on. Do not infer
registry readiness from Git tests, local patches, `--no-verify`, or package lists.

## Evidence and compatibility

See the repository [packaging-validation.json](https://github.com/mith-mmk/avif-rust/blob/codex/avif-native-context-20261016/packaging-validation.json)
for local gate results, source pins and evidence hashes. Shared implementation and codec behavior were not changed in this
packaging follow-up. Tests from the earlier integration checkpoint remain
historical evidence; new candidate-version and archive checks are recorded
separately.

[CI run38971277889](https://github.com/mith-mmk/avif-rust/actions/runs/38971277889/job/117209969190)
verified the preceding 2dc2d43 checkpoint: 10 of 11 jobs succeeded, including
three OSes, Rust1.88, WASM, strict Clippy, rustdoc and shared package/compile.
The sole failing step was downstream registry packaging because shared 0.1.0
was unpublished. That known publication prerequisite is not a new regression,
and those checks are not claimed as a new CI run for the uniform candidates. Local full decoder regression can still encounter
the previously reproduced baseline FFmpeg-generated AVIS entropy failure.

WML2 PR13 and JPEG2000 source worktrees remain unchanged.
