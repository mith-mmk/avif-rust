# Native AV1 pixel investigation — 2026-10-22

調査のみ。製品 `src/` は変更していない。正式修正・新 PR・merge・release は行わない。
添付画像、OBU、raw planes、全画素トレースは private な `/workspace/avif-investigation-input/` に限定し、git に追加していない。

## 結論

libaom と dav1d は対象 7 枚で完全一致した。native PR #3 の差分数・最大差・MAE は Mac 計測と完全に再現した。native 側に複数の独立した問題がある。色変換、ICC、HDR 表示変換、film grain、byte stride、plane 順序の比較条件差では説明できない。

初回調査では6/7枚が一致し、fox8のdeblock後に158 samples/max2が残った。astra追補で、右端chroma ownerのmetadata lookupが次行へ折り返す原因を特定した。coded MI gridを保持する隔離probeを加え、**全7画像・全4段階・plum alphaが完全一致**した。製品修正は未実施。詳細・対照実験・限界は [FOLLOWUP.md](FOLLOWUP.md) と `residual-evidence.json` を参照。

## 固定した対象と参照

- main: `070ad6509bad18f9ab76732d543659ef6d7f9129`
- decoder PR #3: `d3ea4e602763951626d678291cc9c325e2c475b2`。main の 9 commit 先。
- x86_64 Linux、Rust 1.99.1 (`58cf8f9c3`)、release build。
- FFmpeg `7.1.5-0+deb13u1`。
- libaom runtime `v3.12.1`、Debian `3.12.1-1+deb13u1`。
- dav1d runtime `1.5.1`、Debian `1.5.1-1`。
- dav1d tag 1.5.1 の `42b2b24fb8819f1ed3643aa9cf2a62f03868e3aa` を C-only/ASM 無効で別 build。フィルター前の fox8/fox10 はインストール済み dav1d と byte-identical。ブロック予測・dequant トレースに使用。
- libaom tag v3.12.1 の `10aece4157eb79315da205f39e19bf6ab3ee30d0` をソース照合に使用。これは Debian 実行バイナリの exact build commit の主張ではない。
- encoder `avifenc` PR #3 と Mac の速度計測は今回の対象外。

checkout とその親に適用対象の AGENTS.md は存在せず、`/workspace/.agents` は空、checkout に `.agents/skills` もなかった。`.github/workflows/ci.yml`、`tests/FIXTURES.md`、既存 oracle scripts と native frame API を確認した。

## 入力と比較条件

Library skill により `libfile_e2b4999786848191b8936f59033c58a5` を消費環境自身に materialize。ファイル名 `avif.zip`、3,427,052 bytes、SHA-256 `26960271e1bd58a169b7653ec9b92ccf4beb1e169b833f60193abfcda0a8dba8`。ZIP 全 entry を展開前に一覧・サイズ・パス・リンク検査し、総展開 100 MB / 個別 30 MB 上限、絶対パス・`..`・backslash・symlink を拒否した。含有コードは実行していない。必要 7 枚だけ隔離ディレクトリに保持した。各画像 SHA-256 は `evidence.json` に記録。

native は `decode_frame_bytes` の source-plane 出力を Y/U/V/A 順に可視領域だけ取り出す。各 plane の `layout.stride()` を使い、8-bit は u8、それ以上は little-endian u16。比較は整数 sample 値で、range の rescale や色変換を行わない。plum alpha は別 OBU として両参照で decode し、16,384 samples 全て native と一致した。

native が抽出した同一 primary-item OBU を libaom と dav1d に渡した。FFmpeg OBU demux のためだけに Temporal Delimiter `12 00` を前置した。FFmpeg は `-map 0:v:0 -frames:v 1 -c:v rawvideo -threads 1 -f rawvideo`、入力 decoder を `libaom-av1` / `libdav1d` に固定。pixel format は入力の native format を維持。dav1d は filmgrain=0、全 7 枚で bitstream の grain present=0 を確認。全画像で AVIF を FFmpeg が直接 demux した dav1d 出力も同一だった。したがって Rust container parser の共通入力だけに依存した判定でもない。

dav1d public settings の inloop_filters mask 0 / 1 / 3 / 7 を用いて raw / deblock / CDEF / restoration の対応境界を得た。native に同じ境界の dump hook を一時追加した。plum の alpha decode が color dump を上書きしないよう monochrome auxiliary は stage dump から除いた。比較 hash、plane 寸法、各段階の差分数・bbox・最初の実値/期待値は `evidence.json` に保存。座標は各 plane の0始まりで、first samples は plane順・raster順の最初を意味する。

## 再現結果

count は不一致 sample 数。MAE は YUV 全 sample を分母とする非正規化平均絶対誤差。plum alpha は別比較でゼロ。

|画像|PR count|max|MAE|raw count|最初の相違処理|
|---|---:|---:|---:|---:|---|
|fox8 420|39,149|7|0.027452243|38,603|chroma 方向予測。その後 deblock に別残差|
|fox10 444|172,557|12|0.075373408|37|右端 V CfL 予測。その後 high-bit-depth deblock|
|fox12 444|396,432|46|0.359710341|0|deblock|
|fox odd 8-bit 444|731|5|0.000290610|0|右端 deblock|
|plum alpha|0|0|0|0|相違なし、alpha も一致|
|HDR Rec.2020|11,011|2|0.091883333|0|不要な deblock 実行|
|Paris ICC/Exif/XMP|218|6|0.000805219|0|右端 deblock|

main は fox8 のみ count=38,881、PR と 270 samples 異なる。他 6 枚は main/PR native 出力が全画素同一。提供情報の「5 枚同じ」より広い 6 枚一致を今回確認した。

全画像で native の deblock 後、CDEF 後、restoration 後の差分数は同じ。今回の差を CDEF / restoration 起点とする証拠はない。

## 原因の証拠と確度

1. **high-bit-depth flat threshold — 高確度。** `src/av1/tile_decode/post_filter_state.rs::deblock_filter_edge_with_visible_bounds` の flat8 / flat2 / flat6 は固定 `<= 1`。libaom `aom_dsp/loopfilter.c` の highbd flat mask は threshold を `bd-8` shift する。隔離 probe で `1 << (bit_depth-8)` にすると fox12 は **396,432 → 0**、fox10 は **172,557 → 35**。最初の fox12 差は Y(230,0) native=490、両参照=492。逆変換前後の推測ではなく、raw は完全一致し deblock で発生。

2. **可視端 guard — 高確度。** 同関数は radius 全体が visible bounds 内にない edge を丸ごと省略する。可視領域内の edge であることと、読み出しが coded padding 内にあることを分ける probe で odd **731 → 0**、Paris **218 → 0**。odd の bbox は Y x=1194..1202 / y=48..781、最初は Y(1200,48)=36、期待37。Paris は Y x=397..402 / y=1..298、最初は Y(398,1)=165、期待164。単に全 guard を緩めると可視幅と同位置の frame 外 edge まで処理して別の差を作るため、それは正しい修正案ではない。

3. **loopfilter disabled gate — 高確度。** HDR の frame header は loop_filter.levels=[0,0,0,0]。native `deblock_has_active_strengths` は既定 ref_delta=1 だけでも true にし、level=1 を導出してフィルターする。libaom `av1/decoder/decodeframe.c` は luma level[0] || level[1] を gate にする。隔離 gate probe で HDR **11,011 → 0**。最初は Y(15,0)=460、期待459。raw は完全一致。この画像では単独 flat-threshold probe は差分を増やすので、閾値問題とゼロ強度問題は分ける必要がある。

4. **右端 CfL luma backing — 高確度、正式な最小実装は未設計。** fox10 V block (1184,224), 32x32、alpha=-1。native と C dav1d で dequant 配列は一致するが、予測は full transform 内 47 samples 不一致、可視領域で raw 37 samples 不一致。例 V(1186,224) の予測503/502、再構成505/504。`alloc_coded_frame_buffers` は MI padding までしか luma を保持せず、CfL は layout 端を clamp する。luma transform の完全な再構成領域を保持する allocation probe と flat-threshold probe で fox10 **172,557 → 0**。全 plane を一律64-alignする probe は因果検証用で、正式なメモリ設計の提案ではない。

5. **subsampled chroma smooth context ownership — raw 差の原因として高確度。** fox8 chroma block (164,4), 4x4 D135。入力 top/left edge は先行復元と整合するが native smooth=false、dav1d smooth flag=512 (true)。native は UV を持たない luma sub8 block でも UV smooth grid を false で更新し、隣接判定も chroma ownership と異なる座標を使う。UV 構文がある時だけ対応 chroma footprint を更新し、chroma owner 座標で隣を読む probe で **raw 38,603 → 0**。最初の U(167,4) は130、期待131。初回5 probe併用時には **158 / max2 / MAE0.000110742** が残った。最初の残差は U(599,2)=139、期待138。初回で残差の最初の処理をdeblockと確定した。追補では右端metadata gridの行折り返しによる6-tap/4-tap選択差と特定し、coded grid probeで158差分もゼロになった。

これらは画像ごとの単なる最終 diff 減少だけでなく、独立参照の段階出力、予測/係数比較、一要因 probe を組み合わせた判定。ただし AV1 全 bitstream への一般適合性や strict allocation contract の適合を証明するものではない。

## 次の最小修正案

実装を承認・実施した状態ではない。次の修正作業では以下を独立した回帰単位に分ける。

1. highbd flat masks の threshold scaling と disabled loopfilter gate を修正。
2. visible edge eligibility と coded-padding tap readability を分離。frame 外の edge は禁止。
3. UV smooth context の更新・参照を chroma owner に揃え、UV のない luma block で破壊しない。
4. CfL に必要な luma reconstruction を保存。全面64-alignをそのまま採用せず strict/native budget、crop、allocation tests を設計する。
5. deblock metadata gridをcoded MI寸法へ統一し、2次元lookupに範囲チェックを追加する。strict grid planとbudgetも併せて検証する。追補で、この所有ブロック情報の不足がfox8の残る158差分の原因と確定した。

## 再現用ファイル

- `examples/investigate_dump.rs`: 公開 API から primary/alpha OBU と native planes を出す独立計測例。
- `examples/investigate_dav1d.c`: 1.5.1 API で filter mask を指定する参照 harness。
- `probes.patch`: disposable PR checkout 専用の stage hooks、局所トレース、環境変数付き仮説 probe。製品コードには適用していない。
- `dav1d-trace.patch`: 上記 dav1d tag の C recon への局所予測・係数トレース。
- `measure.py`: 入力7枚の native/reference比較を実行する runner。
- `summarize.py`, `evidence.json`: local results の集計と本調査の集計結果。画像そのものは含まない。

例: PR SHA の別 checkout に `git apply probes.patch` し、`cargo build --locked --release --example investigate_dump`。dav1d 1.5.1 の matching headers と library に C harness を link する。`python3 measure.py INPUT PRIVATE_OUT NATIVE_DUMP DAV1D_DUMP` を probe env なしで実行。private output は results/*.native, *.obu, *.meta, *.davstage{0,1,3,7}, *.libaom-av1, *.libdav1d と stages/IMAGE/{raw,deblock,cdef,restoration}.yuv。main は別 checkout/binary で同じ example を実行する。

probe は `AVIF_PROBE_FLAT`, `AVIF_PROBE_EDGE`, `AVIF_PROBE_ZERO`, `AVIF_PROBE_CFL_PAD`, `AVIF_PROBE_UV_SMOOTH`、追補の `AVIF_PROBE_GRID_PAD` を必要な実験だけ1に設定する。`AVIF_RECON_TRACE` / `AVIF_PRED_TRACE` は private file path、`AVIF_STAGE_DIR` は private directory path。本 probe はエラー処理・効率・allocation保証を持つ正式パッチではない。

既存 CI の全マトリクス、Wasm、fuzz、encode/package は調査のためには実行していない。変更対象は計測・記録のみで、製品ソースは指定 PR SHA のまま。

検証結果: `cargo test --locked` は全 suite 成功（903 passed / 0 failed / 9 ignored）。CI 必須 `cargo test --locked --test avis_regression` も7/7成功。`cargo fmt --all -- --check`、probe patch apply check、dav1d trace reverse check、Python構文検査、再現runnerの7枚native byte一致・4参照経路一致を確認。`validation.json` に集計。
