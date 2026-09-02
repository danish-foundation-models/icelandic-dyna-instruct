# Faroese and Icelandic reverse instruction generation

Generate instruction-response data from Faroese and Icelandic DynaWord. Gemma writes the instruction; the original passage is the response.

```text
DynaWord -> filter passage -> generate instruction -> validate -> Parquet
```

## Setup

```bash
uv sync --extra dev
uv run pytest
```

Install vLLM separately in the GPU environment. The configured model is `google/gemma-4-31B-it`.

## Run

Start the inference server:

```bash
uv run reverse-instruct serve configs/fo.yaml
```

Inspect rendered prompts without inference:

```bash
uv run reverse-instruct run configs/fo.yaml --dry-run --limit 3
uv run reverse-instruct run configs/is.yaml --dry-run --limit 3
```

Generate a small sample:

```bash
uv run reverse-instruct run configs/fo.yaml --limit 100
```

Generate everything allowed by a config:

```bash
uv run reverse-instruct run configs/fo.yaml
uv run reverse-instruct run configs/is.yaml
```

Resume an interrupted run with `--resume`. Existing output is never overwritten automatically.

## Current capacity

| Language | Sources | Maximum candidates |
|---|---|---:|
| Faroese | 3,000 Wikipedia + 7,000 BLARK | 10,000 |
| Icelandic | Wikipedia, blogs, court decisions, laws, Wikibooks, Wikisource | 10,000 |

One candidate produces at most one instruction. The accepted total will be lower because unsuitable passages and invalid or duplicate instructions are rejected.

Change source caps in `configs/fo.yaml` and `configs/is.yaml` to generate more.

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

`id` is the original DynaWord row ID. Rejections are stored as Parquet under `rejected/`; `run.json` contains run metadata. Prompts are in `prompts/fo.md` and `prompts/is.md`.
