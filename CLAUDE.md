# vrc-ozaki-schema-LT

## プロジェクト概要

GeForce RTX (RTX 5080, Blackwell sm_120) で、CUDA 13.4 の cuBLAS FP64 エミュレーション
(Ozaki scheme I / II) による DGEMM・PDE 計算の高速化と精度を、Docker で分離した
CUDA 12 環境との対照実験で定量評価する。成果は VRChat 物理学集会の LT で発表する (slides-jp/)。

### 技術仕様

- 条件: `cuda12` (12.9.1) / `cuda13-native` / `cuda13-emu` (performant) / `cuda13-emu-eager`。
  エミュレーションは `CUBLAS_EMULATE_DOUBLE_PRECISION` で実行時に切り替える
- CPU ベースライン: `cpu-eigen` (C++17 + Eigen 3.4 ネイティブ GEMM + OpenMP, `cpp/cpu_bench`) と
  `cpu-numpy` (OpenBLAS)。入力は Python が `.npy` で渡し、精度評価も Python 側の同じロジックで行う。
  スレッド 6 / 14 / 16 (P / P+E / 全コア) を掃引し、最速を代表値にする
- 両イメージともホストドライバの libcuda を使う (cuda13 イメージから forward-compat libcuda を削除済み)
- ワークロード: DGEMM、2D 熱方程式 (ADI-GEMM / FTCS ステンシル = 負の対照)
- 精度: NumPy 比較、厳密丸め参照 (TwoProduct + fsum) とのサンプル比較、PDE の離散厳密解との比較
- カーネル確認: `profile_kernels.sh` (nsys) で実際に動いたカーネルから Ozaki-I / II / native を判定

### 進捗

- [x] ベンチマーク環境 (Docker / スクリプト / レポート生成)
- [x] 本番測定 (phi=0.5): DGEMM 最大 19.2x、ADI 最大 15.9x、ステンシル 1.00x
- [x] nsys によるカーネル確認、phi 掃引
- [x] C++/Eigen の CPU ベースライン: Eigen 比で GPU native 2.5x、GPU Ozaki 48x (DGEMM N=16000)、ADI 39x
- [ ] スライド作成

## 言語設定

このプロジェクトでは**日本語**での応答を行ってください。ただし、コード内のコメント、ログメッセージ、エラーメッセージ、ドキュメンテーション文字列などは**英語**で記述してください。

## 開発ルール

### コーディング規約

- Python: PEP 8準拠
- 関数名: snake_case
- クラス名: PascalCase
- 定数: UPPER_SNAKE_CASE
- Docstring: Google Style

## Git運用

- ブランチ戦略: feature/*, fix/*, refactor/*
- コミットメッセージ: 英文を使用、動詞から始める
- PRはmainブランチへ

## 開発ガイドライン

### ドキュメント更新プロセス

機能追加やPhase完了時には、以下のドキュメントを同期更新する：

1. **CLAUDE.md**: プロジェクト全体状況、Phase完了記録、技術仕様
2. **README.md**: ユーザー向け機能概要、実装状況、使用方法
3. **Makefile**: コマンドヘルプテキスト（## コメント）の更新
4. **makefiles/**: コマンドヘルプテキスト（## コメント）の更新

### コミットメッセージ規約

#### コミット粒度

- **1コミット = 1つの主要な変更**: 複数の独立した機能や修正を1つのコミットにまとめない
- **論理的な単位でコミット**: 関連する変更は1つのコミットにまとめる
- **段階的コミット**: 大きな変更は段階的に分割してコミット

#### プレフィックスと絵文字

- ✨ feat: 新機能
- 🐞 fix: バグ修正
- 📚 docs: ドキュメント
- 🎨 style: コードスタイル修正
- 🛠️ refactor: リファクタリング
- ⚡ perf: パフォーマンス改善
- ✅ test: テスト追加・修正
- 🏗️ chore: ビルド・補助ツール
- 🚀 deploy: デプロイ
- 🔒 security: セキュリティ修正
- 📝 update: 更新・改善
- 🗑️ remove: 削除

**重要**: Claude Codeを使用してコミットする場合は、必ず以下の署名を含める：

```text
🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
```
