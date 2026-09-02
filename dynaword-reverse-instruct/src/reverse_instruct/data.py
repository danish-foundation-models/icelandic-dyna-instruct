import math
import re
from collections.abc import Iterator

from datasets import load_dataset

from reverse_instruct.config import AppConfig
from reverse_instruct.models import Candidate


def clean_text(value: str) -> str:
    paragraphs = re.split(r"\n\s*\n", value.strip())
    cleaned = [" ".join(paragraph.split()) for paragraph in paragraphs]
    return "\n\n".join(paragraph for paragraph in cleaned if paragraph)


def make_passage(text: str, min_chars: int, max_chars: int) -> str | None:
    text = clean_text(text)
    if len(text) < min_chars:
        return None
    if len(text) <= max_chars:
        return text

    window = text[:max_chars]
    sentence_end = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
    if sentence_end + 1 >= min_chars:
        return window[: sentence_end + 1]

    word_end = window.rfind(" ")
    return window[:word_end] if word_end >= min_chars else None


def source_limits(config: AppConfig, total_limit: int | None) -> dict[str, int]:
    if total_limit is None:
        return {name: source.max_documents for name, source in config.sources.items()}

    per_source = math.ceil(total_limit / len(config.sources))
    return {name: min(source.max_documents, per_source) for name, source in config.sources.items()}


def iter_candidates(config: AppConfig, total_limit: int | None = None) -> Iterator[Candidate]:
    limits = source_limits(config, total_limit)
    total_yielded = 0

    for source_name, source_config in config.sources.items():
        dataset = load_dataset(
            config.dataset.name,
            source_name,
            split=config.dataset.split,
            revision=config.dataset.revision,
            streaming=True,
        )
        dataset = dataset.shuffle(
            seed=config.seed,
            buffer_size=config.dataset.shuffle_buffer,
        )

        source_yielded = 0
        for row in dataset:
            passage = make_passage(
                str(row["text"]),
                min_chars=config.text.min_chars,
                max_chars=config.text.max_chars,
            )
            if passage is None:
                continue

            source_id = str(row["id"])
            yield Candidate(
                id=source_id,
                source=source_name,
                language=config.language.code,
                domain_hint=source_config.domain_hint,
                source_context=source_config.context,
                passage=passage,
            )
            source_yielded += 1
            total_yielded += 1

            if total_limit is not None and total_yielded >= total_limit:
                return
            if source_yielded >= limits[source_name]:
                break
