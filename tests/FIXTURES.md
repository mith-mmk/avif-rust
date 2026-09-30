# Required CI fixtures

These files are tracked in this repository so that `cargo test` works in a
standalone checkout. Tests must not depend on a sibling WML2 checkout or skip
the native allocation checks when these fixtures are missing. The fixtures
are also included in the crate package for the library's unit tests.

## WML2Viewer.avif

- Unmodified copy of the WML2 project sample, `samples/WML2Viewer.avif`.
- Source: https://github.com/mith-mmk/wml2-on-rust/blob/7328fcde2f0161694f63d8bec90d22be0135f69e/samples/WML2Viewer.avif
- SHA-256: `8b6846a35bc335a12cde9fa656b4add2c6a654f829bd996d228238b932897644`
- Used for palette, native header ownership, and allocation limit tests.

## plum-blossom-small.profile1.8bpc.yuv444.alpha-full.avif

- Author: Ryo Hirafuji (https://github.com/ledyba-z).
- License: Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/.
- Source: https://github.com/link-u/avif-sample-images/blob/c666a368b73006246694919b5dbcc078317af6cc/plum-blossom-small.profile1.8bpc.yuv444.alpha-full.avif
- Attribution: https://github.com/link-u/avif-sample-images/blob/c666a368b73006246694919b5dbcc078317af6cc/README.md
- Changes: none; the upstream AVIF file is copied byte for byte.
- SHA-256: `245a3dad6371dc702f29eb7e9735f843b63c525da871859728bedbe5bb274985`
- Used for native alpha header, split OBU, and allocation limit tests.
