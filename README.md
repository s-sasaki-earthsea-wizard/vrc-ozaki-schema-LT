# vrc-ozaki-schema-LT

コンシューマ GPU (GeForce RTX) で、CUDA 13.4 の cuBLAS に入った FP64 エミュレーション
(Ozaki scheme I / II) がどれだけ DGEMM と PDE 計算を速くするか、また精度が FP64 相当に
保たれるかを、Docker で分離した CUDA 12 環境との対照実験で定量評価するプロジェクト。
成果は VRChat 物理学集会の LT で発表する (スライドは [slides-jp/](slides-jp/))。

## 実験デザイン

### 条件

「CUDA 12 vs 13.4」の 2 条件だけだと、cuBLAS のバージョン差と Ozaki の効果が分離できない。
そこで CUDA 13.4 イメージを環境変数で native / emulated に切り替え、計 4 条件で測る。

| 条件 | イメージ | `CUBLAS_EMULATE_DOUBLE_PRECISION` | `CUBLAS_EMULATION_STRATEGY` | 役割 |
|---|---|---|---|---|
| `cuda12` | CUDA 12.9.1 | (なし) | (なし) | 対照 (旧環境) |
| `cuda13-native` | CUDA 13.4.2 | `0` | - | 同じ toolkit で Ozaki なし。バージョン差の切り分け |
| `cuda13-emu` | CUDA 13.4.2 | `1` | `performant` | cuBLAS が「得なら」エミュレーション |
| `cuda13-emu-eager` | CUDA 13.4.2 | `1` | `eager` | 可能な限りエミュレーション |

- **Ozaki の正味の効果** = `cuda13-emu*` vs `cuda13-native`
- **「CUDA を上げるだけで何が変わるか」** = `cuda13-emu*` vs `cuda12`

### CPU ベースライン (C++ / Eigen)

「GPU のエミュレーション FP64 は、ちゃんと書いた CPU の FP64 と比べてどうか」を見るため、CPU 側に 2 条件を置く。

| 条件 | 実装 | 役割 |
|---|---|---|
| `cpu-eigen` | C++17 + Eigen 3.4 のネイティブ GEMM (BLAS 委譲なし) + OpenMP、`-O3 -march=native` (AVX2 + FMA) | CPU 代表 (`--cpu-ref`) |
| `cpu-numpy` | NumPy → OpenBLAS 0.3.29 | 定番ライブラリの参照 |

- 入力は GPU 版と同じ関数・同じシードで Python が生成し、`.npy` で C++ (`cpp/cpu_bench`) に渡す。
  結果も `.npy` で戻し、精度評価は GPU と同じロジックで行うので、全条件が同じデータ・同じ物差しで比べられる
- CPU は P コア 6 + E コア 8 + LP-E コア 2 のハイブリッド構成。`OMP_PLACES=cores OMP_PROC_BIND=close` で
  コア ID 順に割り当て、**6 (P のみ) / 14 (P+E) / 16 (全コア)** スレッドを掃引し、最速の構成を代表値にする
  (全構成は report.md の「CPU thread sweep」に残る)
- PDE は Eigen 版のみ (ADI は Eigen GEMM、FTCS は OpenMP + SIMD のループ)。NumPy のスライス版ステンシルは
  CPU の実力を表さないので速度比較には使わない

### ワークロード

| ベンチ | 内容 | cuBLAS | 期待 |
|---|---|---|---|
| DGEMM | `C = A @ B`, N ∈ {2000, 4000, 8000, 16000} | `cublasDgemm` | エミュレーションで大幅高速化 |
| PDE `adi-gemm` | 2D 熱方程式, Peaceman–Rachford ADI (= 因子分解された Crank–Nicolson)。1 ステップ = `U ← M U M` の DGEMM 2 回 | 使う | 高速化 |
| PDE `ftcs-stencil` | 2D 熱方程式, 陽解法 5 点ステンシル (自前 CUDA カーネル) | 使わない | **変化なし** (負の対照) |

`ftcs-stencil` は「Ozaki は FP64 全般ではなく GEMM を速くする技術」であることを示す負の対照。
ステンシルや三重対角ソルバのような GEMM でない FP64 計算は、エミュレーションの恩恵を受けない。

### 精度評価

- **DGEMM**
  - NumPy (CPU) との比較: `max|C − C_np| / max|C_np|`, Frobenius 相対誤差
  - **厳密丸め参照との比較**: ランダムに選んだ要素について、誤差なし積 (Dekker TwoProduct) +
    `math.fsum` で正しく丸めた内積を計算し、`|c_ij − exact_ij| / (|A||B|)_ij` を単位丸め
    u = 2⁻⁵³ 単位で報告。NumPy 自身の誤差も同じ物差しで出るので「GPU エミュレーションが CPU の
    FP64 と同等か」を直接比べられる
  - 入力は Ozaki 論文の慣習に従い `A = (U(0,1) − 0.5)·exp(φ·N(0,1))`。φ を大きくすると
    指数のばらつきが増え、エミュレーションに必要なスライス数が増える (`--phi 0.5 2.0` などで掃引可)
- **PDE**: 初期条件を離散正弦固有モードの重ね合わせにしているので、差分スキームの離散厳密解が
  閉形式で得られる。これとの差は純粋に丸め誤差。参考として連続解との差 (離散化誤差) と NumPy 解との差も出す

### 評価指標

実行時間 (CUDA event、median/min/max/std)、TFLOPS、Speedup (median 時間比)、上記の誤差。

## 前提環境

開発機での確認状況 (2026-10-06):

- GPU: GeForce RTX 5080 16 GB (Blackwell, sm_120)
- Driver: 590.48.01 (CUDA 13.1 世代)
- Docker 29.2 / Compose v5 / NVIDIA Container Toolkit 1.18.2
- CPU: Core Ultra 9 285H (NumPy 参照計算用)

### 注意点

1. **CUDA 12.4.1 ではなく 12.9.1 を対照にしている。** CUDA 12.4 は Blackwell (sm_120) 登場前で、
   cuBLAS/NVRTC がネイティブコードを持たない (PTX JIT 頼みか動かない)。RTX 50 系で
   12.4 を使うと「古い CUDA が遅い理由」に JIT が混ざり、比較が汚れる。
   Blackwell 以前の GPU で 12.4.1 を使いたい場合は `CUDA12_BASE_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04`。
2. **ドライバが 13.4 の要求 (R615+) より古い。** CUDA 13.x のマイナーバージョン互換で R580+ でも
   13.4 のユーザ空間ライブラリは動く想定だが、イメージの `NVIDIA_REQUIRE_CUDA=cuda>=13.4` で
   コンテナ起動が拒否されるため、`docker-compose.yml` で `NVIDIA_DISABLE_REQUIRE=true` にしている。
   ドライバを R615+ に上げたら `NVIDIA_DISABLE_REQUIRE=false` に戻すのが望ましい。
   さらに Container Toolkit の `cuda-compat-mode = "ldconfig"` によって、13.4 イメージ同梱の
   forward-compat libcuda (R615) がホストのドライバより優先されてしまう (GeForce では非サポート構成)。
   そこで `Dockerfile.cuda13` で `/usr/local/cuda-*/compat` を削除し、両条件ともホストの libcuda を使う
   (`env.json` の `libcuda_loaded` で確認できる)。
3. **エミュレーションは opt-in。** CUDA 13.4 にしただけでは DGEMM はネイティブ FP64 のまま。
   `CUBLAS_EMULATE_DOUBLE_PRECISION=1` が必要 (run_benchmarks.sh が条件ごとに設定)。
4. **CuPy は wheel を使い、コンテナ内 toolkit の cuBLAS をロードしていることを実行時に検証する。**
   `/proc/self/maps` から実際にロードされた `libcublas` のパスと `cublasGetVersion` を JSON に記録する。
   pip 由来の `nvidia-cublas-*` が入っていたらビルドを失敗させる。
   ソースビルドしたい場合は `CUPY_FROM_SOURCE=1 make build` (`pip install --no-binary cupy`、20 分以上かかる)。
5. N=16000 の DGEMM は入出力だけで 6 GB。エミュレーションのワークスペースと合わせて 16 GB を
   超える可能性があり、OOM になったケースは `status: failed` として記録され、スイープは続行する。

## 結果サマリ (RTX 5080, 2026-10-06)

詳細は [results/report.md](results/report.md)、[results-phi/report.md](results-phi/report.md)、
[results/nsys/kernels.md](results/nsys/kernels.md)。

| 項目 | 結果 |
|---|---|
| DGEMM (φ=0.5) | native ~0.82 TFLOPS → emu 5.5 / 9.0 / 13.4 / 15.6 TFLOPS (N=2000/4000/8000/16000)、**最大 19.2x** |
| cuda12 vs cuda13-native | 同等 (CUDA を上げるだけでは速くならない) |
| ADI-GEMM (PDE) | 4.2x / 7.5x / 11.5x / **15.9x** (N=1024..8192) |
| FTCS ステンシル (PDE) | 全条件 1.00x (GEMM でない FP64 計算は速くならない) |
| 精度 | emu の scaled err は 0.05–1.9u。native (1.6–86u)、NumPy (0.2–27u) より小さい |
| φ 依存性 (N=8000) | φ=0 → 4 で 17.1x → 12.4x。指数レンジが広いほど遅くなるが、精度は保たれる |
| 実行カーネル | エミュレーション時は常に **Ozaki-II** (`oz2_int8_dgemm` + INT8 Tensor Core `nvjet_sm120_biu_mma`)。native は `cutlass_80_tensorop_d884gemm` |
| performant vs eager | N=512 では performant が native を選ぶが、eager (Ozaki-II) の方が 2.4x 速い |
| CPU (C++/Eigen, 14 スレッド) | DGEMM 0.30–0.36 TFLOPS、OpenBLAS は 0.21–0.38 TFLOPS。Eigen 比で GPU native は **2.2–2.6x**、GPU Ozaki は **16–48x** (φ 掃引を含む) |
| CPU の PDE | ADI: GPU Ozaki が Eigen 比 11.7–39x。FTCS: GPU は Ozaki と無関係に 6.5–18x (メモリ帯域の差) |
| CPU の精度 | Eigen / OpenBLAS の scaled err は 0.2–0.9u (φ=0.5)、N=8000・φ=4 で 25–27u。Ozaki (0.05–1.9u) はどちらより小さい |
| CPU のスレッド | 14 (P+E) が DGEMM / ADI で最速。16 (LP-E コアを含む) は Eigen で約 3 倍遅くなる。FTCS (メモリ律速) は 6 (P のみ) が最速 |

### カーネル確認 (Nsight Systems)

`make profile` (`profile_kernels.sh`) で、cuda13 イメージ同梱の nsys を使って DGEMM を NVTX 範囲内で
プロファイルし、カーネル名から実際の経路 (Ozaki-II / Ozaki-I / native FP64) を判定する。
cuda12 イメージには nsys が無いため対象外 (cuda13-native が同じ native 経路の代表)。

```bash
SIZES="512 2048 8000" PHIS="0.5 4" ./profile_kernels.sh   # -> results/nsys/kernels.md
```

`.nsys-rep` も残すので、Nsight Systems GUI でタイムラインを確認できる。

### φ 掃引

入力の指数レンジ φ を変えた DGEMM 比較は `make phi-sweep` (`results-phi/`)。
φ=0 は一様乱数 `U(-0.5, 0.5)`、φ=4 では要素の大きさが `exp(±4σ)` 程度に広がる。

### Fortran の既存バイナリ + NVBLAS (drop-in)

「既存の Fortran コードを書き換えずに Ozaki の恩恵を受けられるか」を確かめる追加実験 (`make fortran-bench`、
[results-fortran/dgemm.txt](results-fortran/dgemm.txt))。`external dgemm` を呼ぶだけの Fortran
(`fortran/dgemm_bench.f90`) を OpenBLAS にリンクして 1 回だけビルドし、同じバイナリを次の 3 条件で動かす。

| 条件 | 実行方法 |
|---|---|
| `fortran-openblas` | そのまま (OpenBLAS、14 スレッド) |
| `fortran-nvblas-native` | `LD_PRELOAD=libnvblas.so.13` + `CUBLAS_EMULATE_DOUBLE_PRECISION=0` |
| `fortran-nvblas-emu` | `LD_PRELOAD=libnvblas.so.13` + `CUBLAS_EMULATE_DOUBLE_PRECISION=1` |

NVBLAS は呼び出しのたびに A, B を GPU に送り、C を戻すので、時間は PCIe 転送込みの end-to-end になる。
精度は real128 の内積を参照にしたサンプル比較 (`benchmark_dgemm.py` と同じ scaled err)。

| N (φ=0.5) | OpenBLAS | NVBLAS native | NVBLAS emu | emu / OpenBLAS | emu / native |
|---:|---:|---:|---:|---:|---:|
| 4000 | 0.391 s | 0.238 s | 0.099 s | 4.0x | 2.4x |
| 8000 | 2.537 s | 1.530 s | 0.361 s | 7.0x | 4.2x |
| 16000 | 23.40 s | 11.75 s | 2.258 s | **10.4x** | 5.2x |

- NVBLAS 経由でもエミュレーションの環境変数が効き、nsys で Ozaki-II (`oz2_int8_dgemm`) を確認した。
  N=8000 のカーネル時間は 71.4 ms で、CuPy 経路 (71.7 ms) と同じ
- 誤差は emu が 0.026–0.080u、native が 0.47–1.70u、OpenBLAS が 0.21–0.34u
- このマシンの GPU は PCIe x4 接続 (実効 5–6 GB/s) で、N=8000 では end-to-end の約 7 割が転送。
  GPU 常駐の CuPy (OpenBLAS 比 44x) には届かない
- `NVBLAS_AUTOPIN_MEM_ENABLED` は emu では逆効果 (N=8000 で 0.53 s → なしで 0.36 s)。既定ではオフ
- **NVBLAS は無言で失敗する**: `NVBLAS_TILE_DIM 16384` (16 GB の GPU) では `cublasXtDgemm` が
  `CUBLAS_STATUS_ALLOC_FAILED` で失敗するが、`dgemm_` は C に触れずに戻り、エラーは NVBLAS のログにしか出ない。
  `run_fortran_benchmarks.sh` は実行ごとにログを確認し、失敗を `FAILED` 行として記録する

## 使い方

```bash
make build        # イメージのビルド
make env-check    # GPU / CUDA / cuBLAS / エミュレーション設定の確認
make quick        # 小サイズでの動作確認 (results-quick/)
make bench        # 本番 (results/)。DGEMM + PDE、全 4 条件、最後にレポート生成
make phi-sweep    # φ = 0, 0.5, 1, 2, 4 の DGEMM 比較 (results-phi/)
make profile      # nsys で実行カーネルを確認 (results/nsys/)
make cpu-quick    # C++/Eigen + NumPy の CPU ベースラインを小サイズで動作確認 (results-quick/)
make cpu-bench    # CPU ベースライン本番 (results/)。DGEMM + PDE、スレッド 6/14/16
make cpu-phi-sweep  # CPU の φ 掃引 (results-phi/)。THREADS=14 のようにスレッド数を指定
make fortran-quick  # Fortran + NVBLAS の drop-in を小サイズで動作確認 (results-quick/fortran/)
make fortran-bench  # Fortran + NVBLAS の drop-in 本番 (results-fortran/)。先に make build が必要
make analyze      # results/ と results-phi/ から図と report.md を再生成
```

GPU と CPU のベンチは**同時に走らせない** (どちらもホスト CPU を使うため)。

`run_benchmarks.sh` は環境変数で調整できる:

```bash
CONDITIONS="cuda13-native cuda13-emu" ./run_benchmarks.sh        # 条件を絞る
RESULTS_DIR=results-foo ./run_benchmarks.sh                        # 出力先を変える
DGEMM_ARGS="--phi 0.5 2.0 --repeats 10" SKIP_PDE=1 ./run_benchmarks.sh
PDE_ARGS="--sizes 2048 4096 --adi-steps 50" SKIP_DGEMM=1 ./run_benchmarks.sh
```

各スクリプトは単体でも実行できる (`python src/benchmark_dgemm.py --help`)。

## 出力

```text
results/
├── <condition>/
│   ├── env.json      # 環境情報 (CUDA / cuBLAS バージョン、ロードされた libcublas、エミュレーション設定)
│   ├── dgemm.json    # DGEMM の生データ (各 repeat の時間、誤差)
│   ├── pde.json      # PDE の生データ
│   └── *.log
├── figures/
│   ├── dgemm_tflops.png
│   ├── dgemm_speedup.png
│   ├── dgemm_error.png
│   ├── pde_time_per_step.png
│   ├── pde_speedup.png
│   ├── dgemm_speedup_vs_cpu.png   # cpu-eigen 基準の speedup (CPU 結果がある場合)
│   └── pde_speedup_vs_cpu.png
├── nsys/
│   ├── <condition>/  # *.nsys-rep と NVTX 範囲内のカーネル集計 CSV
│   ├── kernels.json
│   └── kernels.md    # 条件・N・φ ごとの実行経路と上位カーネル
└── report.md         # 表と図をまとめたレポート

results-phi/          # φ 掃引 (図に dgemm_phi_tflops.png / dgemm_phi_error.png が加わる)

results-fortran/
├── dgemm.txt         # Fortran drop-in の RESULT 行 (条件・タイルごと) と nsys の判定 (NSYS 行)
└── nsys/fortran-nvblas-emu/  # *.nsys-rep とカーネル / memcpy の集計 CSV
```

NumPy の参照解は `cache/` にキャッシュされ、全条件で同じものを使う (NumPy は全イメージで同一バージョンに固定)。

## ディレクトリ構成

```text
.
├── docker/
│   ├── Dockerfile.cuda12      # CUDA 12.9.1 + CuPy (cupy-cuda12x)
│   ├── Dockerfile.cuda13      # CUDA 13.4.2 + CuPy (cupy-cuda13x)
│   ├── Dockerfile.cpu         # Ubuntu 24.04 + g++ 13 + Eigen 3.4 + CMake
│   ├── Dockerfile.fortran     # cuda13 イメージ + gfortran 13 + OpenBLAS (NVBLAS は CUDA 同梱)
│   ├── requirements.txt       # NumPy / matplotlib (全イメージ共通で固定)
│   └── requirements-cpu.txt   # + threadpoolctl (OpenBLAS のスレッド数制御)
├── docker-compose.yml
├── cpp/
│   ├── cpu_bench.cpp          # C++/Eigen の DGEMM / ADI / FTCS (サブコマンド)
│   ├── npy.hpp                # 2D float64 .npy の読み書き
│   └── CMakeLists.txt
├── fortran/
│   └── dgemm_bench.f90        # BLAS の dgemm を呼ぶだけの Fortran ドライバ (real128 で精度確認)
├── src/
│   ├── common.py              # 環境情報の収集、CUDA event タイマー
│   ├── benchmark_dgemm.py     # DGEMM ベンチマーク
│   ├── benchmark_pde.py       # 2D 熱方程式ベンチマーク (ADI-GEMM / FTCS ステンシル)
│   ├── analyze_results.py     # 集計、PNG 出力、Markdown レポート生成
│   ├── profile_dgemm.py       # nsys 用の DGEMM ドライバ (NVTX 範囲付き)
│   ├── summarize_nsys.py      # nsys のカーネル集計から実行経路を判定
│   └── benchmark_cpu.py       # CPU ベースライン (C++/Eigen と NumPy) のオーケストレーション
├── scripts/
│   └── conditions.sh          # 実験条件の定義 (run_benchmarks.sh / profile_kernels.sh が共有)
├── run_benchmarks.sh          # 全条件・全ベンチを一括実行
├── profile_kernels.sh         # nsys で実行カーネルを確認
├── run_cpu_benchmarks.sh      # CPU ベースラインを一括実行
├── run_fortran_benchmarks.sh  # Fortran + NVBLAS の drop-in 実験を一括実行
├── Makefile
└── slides-jp/                 # 発表スライド
```

## 参考

- [cuBLAS documentation — Floating Point Emulation](https://docs.nvidia.com/cuda/cublas/)
- [CUDA Toolkit 13.4 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
- Ozaki, Uchino, Imamura, "Ozaki Scheme II: A GEMM-oriented emulation of floating-point matrix multiplication using an integer modular technique" (2025)
