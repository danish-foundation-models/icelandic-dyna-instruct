# Icelandic reverse instruction generation

Generate instruction-response data from Icelandic DynaWord. Gemma writes the instruction; the original passage is the response.

```text
DynaWord -> filter passage -> generate instruction -> validate -> Parquet
```

## Setup

```bash
uv sync --extra dev
uv run pytest
```

Install vLLM separately in the GPU environment. The configured model is `google/gemma-4-31B-it`.

Sampling follows [Google's Gemma 4 recommendation](https://ai.google.dev/gemma/docs/core/model_card_4): `temperature=1.0`, `top_p=0.95`, and `top_k=64`. This uses sampling rather than greedy decoding.

## Run

Start the inference server:

```bash
uv run reverse-instruct serve configs/is.yaml
```

Inspect rendered prompts without inference:

```bash
uv run reverse-instruct run configs/is.yaml --dry-run --limit 3
```

Generate a small sample:

```bash
uv run reverse-instruct run configs/is.yaml --limit 100
```

Generate everything allowed by a config:

```bash
uv run reverse-instruct run configs/is.yaml
```

Resume an interrupted run with `--resume`. Existing output is never overwritten automatically.

## Current capacity

| Language | Sources | Maximum candidates |
|---|---|---:|
| Icelandic | Wikipedia, blogs, court decisions, laws, Wikibooks, Wikisource | 10,000 |

One candidate produces at most one instruction. The accepted total will be lower because unsuitable passages and invalid or duplicate instructions are rejected.

Change source caps in `configs/is.yaml` to generate more.

## Output

The Hugging Face-ready dataset is written to `outputs/<run>/dataset/` as Parquet shards:

```text
dataset/
  train-00000.parquet
  train-00001.parquet
```

Each shard contains exactly these columns:

```text
id         string
messages   list<struct<role: string, content: string>>
model      string
```

`id` is the original DynaWord row ID. Rejections are stored as Parquet under `rejected/`; `run.json` contains run metadata. The prompt is in `prompts/is.md`.
