# astra follow-up: fox8 deblock residual

開始点は `70dc474759ac85649e35987200eefe05ea1d3695`。既存の画像 SHA、4参照経路の output SHA、調査ログを再利用・検証した。Library の再取得・再展開はしていない。製品 checkout の `src/`, Cargo.toml, Cargo.lock は開始点と同一。実験は private な detached worktree `/workspace/avif-investigation-input/residual` で行った。

**残る158 samples/max2の原因は、deblock metadata grid の右端での行折り返し誤参照。coded MI padding を保持する probe により全7画像が完全一致した。**

## 最初の相違処理

fox8 は 1204×800、4:2:0。U plane は602×400。先行5 probe適用時の raw は参照と完全一致する。plane順・raster順の最初の残差は **U(599,2): native139、参照138**。

影響する最初の処理は U の垂直境界 `(600,0)` の lane2。native と C dav1d に同じ6入力を確認した:

|項目|native（以前の5 probe）|dav1d / GRID_PAD probe|
|---|---|---|
|p2,p1,p0,q0,q1,q2|141,140,140,136,136,136|同一|
|level / E / I / H|7 / 25 / 7 / 0|同一|
|選択filter length|6|4|
|処理後の6値|141,140,**139,138,137**,136|141,140,**138,137,136**,136|

4-tapでは HEV=false、`f=3*(136-140)=-12`, `f1=(-12+4)>>3=-1`, `f2=(-12+3)>>3=-2`、従って p0=138、q0=137。誤って6-tapを選ぶとこの入力はflat判定を通り、p0=139になる。入力誤差や整数丸めの差ではなく、kernel選択前のmetadata相違である。

## 誤ったmetadataが選ばれる経路

`deblock_luma_owner_coordinate(600,1)` は、chroma共有ブロックを所有する右側luma MIを選ぶため `1204` を返す。y=0側も所有座標は4になる。つまり参照する MI は `(col=301,row=1)`。

ところが `apply_deblock_stage` と `apply_deblock_plane_with_filter_grid` の grid幅は visible `ceil(1204/4)=301`。`block_filter_state_at` は2次元範囲を確認せず、一次元の `get(row*width+col)` だけを行う。`1*301+301=602` は配列全体の範囲内で、**次行 `(col=0,row=2)`** に折り返す。

- 誤参照: `(x=0,y=0), Block16x16` → chroma width8 → previous width8とのmin8 → filter length6。
- 正しい参照: `(x=1200,y=0), Block8x16` → chroma width4 → min4 → filter length4。
- 正しいcoded幅は `ceil(1204/8)*8=1208`、MI幅302。この最後のMIをgridに保持すると正しいBlock8x16を取得できる。

局所トレースは `first-edge.txt`、構造化された数値は `residual-evidence.json`。C-only dav1dのトレース版は、全7画像の最終出力がインストール済みdav1d/libaom参照と同一であることを確認した。

## 反証実験と全画像確認

追加probeだけを変え、他の先行5 probeを固定した:

|追加条件|fox8の最終差分|max|
|---|---:|---:|
|追加なし|158|2|
|2次元bounds checkだけ、範囲外はNoneにして既存fallbackを使用|10|1|
|gridをcoded MI幅・高さへpaddingし、metadataを保持|0|0|

したがって、範囲チェックだけの修正は不十分。fallbackで失われる所有ブロックの情報も必要である。

6 probe併用時は **全7画像、raw / deblock / CDEF / restorationの全段階で不一致0**。plum alphaの16,384 samplesも不一致0。各画像のnative色plane SHAと独立参照SHAが一致する。

先行仮説を一つずつ外した対照実験も実施した。他のprobeは全て有効。これは一要因を戻した値なので、未変更PRのbaseline値と一致するとは限らない。

|無効にしたprobe|画像|raw差分|最終差分|max|
|---|---|---:|---:|---:|
|FLAT|fox10|0|172,519|12|
|FLAT|fox12|0|396,432|46|
|EDGE|odd|0|731|5|
|EDGE|Paris|0|218|6|
|ZERO|HDR|0|14,757|3|
|CFL_PAD|fox10|37|35|1|
|UV_SMOOTH|fox8|38,603|38,871|7|
|GRID_PAD|fox8|0|158|2|

probeを全て無効にした新しい計測バイナリは、7枚とも元のPR出力とbyte-identicalだった。これにより追加したトレース自体が差分を消していないことも確認した。

## 最小修正の方針と限界

将来の正式修正では、deblock metadata gridの全利用箇所でcoded MI寸法を統一し、2次元lookupの範囲を明示的に検証する。`StrictDeblockFilterGridPlan::for_frame` のchecked arithmeticとbudget計上も同じ寸法へ変更し、strict/legacyの両経路をテストする必要がある。今回の `GRID_PAD` はlegacy計測経路だけの隔離probeで、strict allocation contractを実装・検証したものではない。

前回特定した5原因も対照実験で支持された。CfLの全面64-align probeは保持領域不足の因果検証であり、そのまま正式修正として採用する設計ではない。各問題の合成効果をこの7枚で解消したという結果であり、AV1全ストリーム、AVIS/inter、全tile構成・subsamplingへの適合保証ではない。

## 再現と検証

更新済み `probes.patch` は未変更PR `d3ea4e6` または調査開始点 `70dc474` の製品ソースに適用できる。元の5環境変数に **`AVIF_PROBE_GRID_PAD=1`** を追加する。`AVIF_PROBE_GRID_BOUNDS=1` はbounds-only反証用。`AVIF_EDGE_TRACE=1` で対象U境界のmetadataとtapsをstderrへ出力する。

```sh
# disposable checkoutでのみ適用・buildする
git apply investigation/native-pixel-20261022/probes.patch
cargo build --locked --release --example investigate_dump

AVIF_PROBE_FLAT=1 AVIF_PROBE_EDGE=1 AVIF_PROBE_ZERO=1 \
AVIF_PROBE_CFL_PAD=1 AVIF_PROBE_UV_SMOOTH=1 AVIF_PROBE_GRID_PAD=1 \
python3 investigation/native-pixel-20261022/measure.py \
  PRIVATE_INPUT_DIR PRIVATE_OUTPUT_DIR \
  ABSOLUTE_NATIVE_DUMP ABSOLUTE_DAV1D_DUMP
```

`dav1d-trace.patch` は既存recon traceにloop-filter traceを追加。対象tagは引き続きdav1d 1.5.1 `42b2b24`。private出力は `residual-final/`, `residual-ablation/`, `residual-control/` に残した。成果JSONには画素配列や画像を含めず、hash・集計値・最初の小さなtap列のみ保存した。

検証済み: Rust release build、C dav1d build、7枚の4段階完全一致、alpha一致、8 ablation、bounds-only反証、7枚のnative baseline不変、7枚のC dav1d最終出力不変、patch forward/reverse apply check。製品ソースは不変のため、前回の既存テスト903 passed/0 failed/9 ignoredを再実行していない。新規の製品修正・PR・merge・releaseは行っていない。
