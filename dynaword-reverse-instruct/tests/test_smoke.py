from pathlib import Path

import pyarrow.parquet as pq
from datasets import load_dataset

from reverse_instruct.config import load_config
from reverse_instruct.data import make_passage, source_limits
from reverse_instruct.models import AcceptedRecord, Candidate, GenerationDecision, Message
from reverse_instruct.prompting import PromptTemplate
from reverse_instruct.runner import flush_full_shards
from reverse_instruct.storage import prepare_output, write_shard
from reverse_instruct.validation import validation_errors

ROOT = Path(__file__).parents[1]


def test_configs_and_prompts_load() -> None:
    for language in ("fo", "is"):
        config = load_config(ROOT / "configs" / f"{language}.yaml")
        candidate = Candidate(
            id="example_1",
            source="example",
            language=language,
            domain_hint="encyclopedic",
            source_context="Reference text.",
            passage="A complete example passage.",
        )
        rendered = PromptTemplate(config.prompt_path).render(candidate)
        assert "A complete example passage." in rendered
        assert "$passage" not in rendered


def test_make_passage_filters_and_truncates() -> None:
    assert make_passage("too short", min_chars=20, max_chars=100) is None
    text = "First complete sentence. " * 20
    passage = make_passage(text, min_chars=20, max_chars=100)
    assert passage is not None
    assert len(passage) <= 100
    assert passage.endswith(".")


def test_limit_is_distributed_across_sources() -> None:
    config = load_config(ROOT / "configs" / "is.yaml")
    limits = source_limits(config, total_limit=12)
    assert set(limits.values()) == {2}


def test_validation_accepts_valid_instruction() -> None:
    decision = GenerationDecision(
        accept=True,
        domain="encyclopedic",
        instruction="Explain the history and significance of this subject.",
        reason="Complete explanatory passage.",
    )
    errors = validation_errors(decision, {"encyclopedic"}, set())
    assert errors == []


def test_validation_rejects_meta_reference() -> None:
    decision = GenerationDecision(
        accept=True,
        domain="encyclopedic",
        instruction="Summarize the provided text in a clear and neutral style.",
        reason="Complete passage.",
    )
    errors = validation_errors(decision, {"encyclopedic"}, set())
    assert errors == ["instruction refers to the supplied passage"]


def test_generation_schema_requires_all_fields() -> None:
    schema = GenerationDecision.model_json_schema()
    assert set(schema["required"]) == {"accept", "domain", "instruction", "reason"}
    assert schema["additionalProperties"] is False
    assert schema["properties"]["reason"]["maxLength"] == 300


def test_accepted_dataset_has_three_parquet_columns(tmp_path: Path) -> None:
    record = AcceptedRecord(
        id="wiki_123",
        messages=[
            Message(role="user", content="Explain the subject."),
            Message(role="assistant", content="The original passage."),
        ],
        model="google/gemma-4-31B-it",
    )
    path = write_shard(tmp_path, 0, [record], [])
    assert path == tmp_path / "dataset" / "train-00000.parquet"

    table = pq.read_table(tmp_path / "dataset" / "train-00000.parquet")
    assert table.column_names == ["id", "messages", "model"]
    assert table.to_pylist()[0]["messages"] == [
        {"role": "user", "content": "Explain the subject."},
        {"role": "assistant", "content": "The original passage."},
    ]

    dataset = load_dataset(
        "parquet",
        data_files=str(tmp_path / "dataset" / "*.parquet"),
        split="train",
    )
    assert dataset.column_names == ["id", "messages", "model"]
    assert dataset[0]["messages"][0] == {
        "role": "user",
        "content": "Explain the subject.",
    }

    config = load_config(ROOT / "configs" / "fo.yaml").model_copy(update={"output_dir": tmp_path})
    processed_ids, seen_instructions, next_index = prepare_output(config, resume=True)
    assert processed_ids == {"wiki_123"}
    assert seen_instructions == {"explain the subject."}
    assert next_index == 1


def test_full_shards_contain_the_configured_number_of_rows(tmp_path: Path) -> None:
    record = AcceptedRecord(
        id="example",
        messages=[
            Message(role="user", content="Explain the subject."),
            Message(role="assistant", content="The original passage."),
        ],
        model="google/gemma-4-31B-it",
    )
    accepted = [record, record, record]
    config = load_config(ROOT / "configs" / "fo.yaml").model_copy(
        update={"output_dir": tmp_path, "rows_per_shard": 2}
    )

    next_index = flush_full_shards(config, 0, accepted, [], uploader=None)

    assert next_index == 1
    assert len(accepted) == 1
    assert pq.ParquetFile(tmp_path / "dataset" / "train-00000.parquet").metadata.num_rows == 2
