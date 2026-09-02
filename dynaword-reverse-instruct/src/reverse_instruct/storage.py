import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from reverse_instruct.config import AppConfig
from reverse_instruct.models import AcceptedRecord, RejectedRecord
from reverse_instruct.validation import normalize_instruction

DATASET_DIR = "dataset"
REJECTED_DIR = "rejected"
RUN_FILE = "run.json"

MESSAGE_TYPE = pa.struct([("role", pa.string()), ("content", pa.string())])
ACCEPTED_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("messages", pa.list_(MESSAGE_TYPE)),
        ("model", pa.string()),
    ]
)
REJECTED_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("source", pa.string()),
        ("domain", pa.string()),
        ("instruction", pa.string()),
        ("response", pa.string()),
        ("reason", pa.string()),
        ("model", pa.string()),
        ("dataset_revision", pa.string()),
    ]
)


def parquet_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.parquet")) if directory.exists() else []


def read_records(directory: Path) -> Iterator[dict]:
    for path in parquet_files(directory):
        yield from pq.read_table(path).to_pylist()


def row_count(directory: Path) -> int:
    return sum(pq.ParquetFile(path).metadata.num_rows for path in parquet_files(directory))


def next_shard_index(output_dir: Path) -> int:
    files = parquet_files(output_dir / DATASET_DIR) + parquet_files(output_dir / REJECTED_DIR)
    indexes = [int(path.stem.rsplit("-", 1)[1]) for path in files]
    return max(indexes, default=-1) + 1


def prepare_output(config: AppConfig, resume: bool) -> tuple[set[str], set[str], int]:
    dataset_dir = config.output_dir / DATASET_DIR
    rejected_dir = config.output_dir / REJECTED_DIR
    has_output = bool(
        parquet_files(dataset_dir)
        or parquet_files(rejected_dir)
        or (config.output_dir / RUN_FILE).exists()
    )
    if has_output and not resume:
        raise FileExistsError(f"Output already exists in {config.output_dir}; use --resume")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    accepted = list(read_records(dataset_dir))
    rejected = list(read_records(rejected_dir))
    processed_ids = {str(row["id"]) for row in accepted + rejected}
    seen_instructions = {
        normalize_instruction(str(row["messages"][0]["content"])) for row in accepted
    }
    return processed_ids, seen_instructions, next_shard_index(config.output_dir)


def write_shard(
    output_dir: Path,
    index: int,
    accepted: list[AcceptedRecord],
    rejected: list[RejectedRecord],
) -> Path | None:
    (output_dir / DATASET_DIR).mkdir(parents=True, exist_ok=True)
    (output_dir / REJECTED_DIR).mkdir(parents=True, exist_ok=True)

    accepted_path = None
    if accepted:
        rows = [record.model_dump(mode="json") for record in accepted]
        table = pa.Table.from_pylist(rows, schema=ACCEPTED_SCHEMA)
        accepted_path = output_dir / DATASET_DIR / f"train-{index:05d}.parquet"
        pq.write_table(table, accepted_path, compression="zstd")

    if rejected:
        rows = [record.model_dump(mode="json") for record in rejected]
        table = pa.Table.from_pylist(rows, schema=REJECTED_SCHEMA)
        path = output_dir / REJECTED_DIR / f"rejected-{index:05d}.parquet"
        pq.write_table(table, path, compression="zstd")

    return accepted_path


def write_run_file(config: AppConfig, status: str, started_at: str) -> None:
    payload = {
        "run_name": config.run_name,
        "status": status,
        "started_at": started_at,
        "updated_at": datetime.now(UTC).isoformat(),
        "accepted": row_count(config.output_dir / DATASET_DIR),
        "rejected": row_count(config.output_dir / REJECTED_DIR),
        "config": config.model_dump(mode="json"),
    }
    (config.output_dir / RUN_FILE).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
