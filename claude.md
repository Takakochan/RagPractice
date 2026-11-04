このリポジトリは42の課題「RAG against the machine」です。以下を常に守ってください。

- Python 3.10+、パッケージ管理は uv のみ（評価側は uv sync しか実行しない）
- flake8 と mypy（--warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs）を必ず通す
- 全関数に型ヒントと Google スタイルの docstring（英語）
- ステージ間のデータは pydantic モデル、CLI は Python Fire、長い処理は tqdm
- 実行は `uv run python -m src <command>`
- パスはすべて CLI 引数で受け取り、ハードコード禁止
- 返すチャンクは絶対に2000文字を超えない
- file_path は `data/raw/vllm-0.10.1/...` とコーパス上のパスに完全一致させる
- 例外でトレースバックを出して落ちないこと（空クエリ、k=0、存在しないファイル、壊れたJSON）
- モデルは前半の開発時点でEYハッカソンの前の準備運動としてこのプロジェクトを行いたいので、Nebius Token Factoryを使用する。
- 最終的なモデルは Qwen/Qwen3-0.6B
- moulinette をコードから import・呼び出ししない



私は評価で全コードを説明し、その場で改修できる必要があります。
コードを書く前に必ず方針を説明し、私が了承してから実装してください。
一度に1ファイル・1機能ずつ進めてください。