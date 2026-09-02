import asyncio
from datetime import UTC, datetime

from reverse_instruct.config import AppConfig
from reverse_instruct.data import iter_candidates
from reverse_instruct.hub import HubUploader
from reverse_instruct.inference import InferenceClient
from reverse_instruct.models import (
    AcceptedRecord,
    Candidate,
    GenerationDecision,
    Message,
    RejectedRecord,
)
from reverse_instruct.prompting import PromptTemplate
from reverse_instruct.storage import (
    DATASET_DIR,
    parquet_files,
    prepare_output,
    write_run_file,
    write_shard,
)
from reverse_instruct.validation import normalize_instruction, validation_errors


def make_accepted_record(
    candidate: Candidate,
    decision: GenerationDecision,
    config: AppConfig,
) -> AcceptedRecord:
    return AcceptedRecord(
        id=candidate.id,
        messages=[
            Message(role="user", content=decision.instruction or ""),
            Message(role="assistant", content=candidate.passage),
        ],
        model=config.model.name,
    )


def make_rejected_record(
    candidate: Candidate,
    decision: GenerationDecision,
    config: AppConfig,
    errors: list[str],
) -> RejectedRecord:
    reason = "; ".join(errors) if errors else decision.reason
    return RejectedRecord(
        id=candidate.id,
        source=candidate.source,
        domain=decision.domain,
        instruction=decision.instruction,
        response=candidate.passage,
        reason=reason,
        model=config.model.name,
        dataset_revision=config.dataset.revision,
    )


async def infer_batch(
    candidates: list[Candidate],
    client: InferenceClient,
    prompt: PromptTemplate,
) -> list[GenerationDecision]:
    requests = [client.generate(prompt.render(candidate)) for candidate in candidates]
    return list(await asyncio.gather(*requests))


async def process_batch(
    candidates: list[Candidate],
    client: InferenceClient,
    prompt: PromptTemplate,
    config: AppConfig,
    seen_instructions: set[str],
) -> tuple[list[AcceptedRecord], list[RejectedRecord]]:
    decisions = await infer_batch(candidates, client, prompt)
    accepted: list[AcceptedRecord] = []
    rejected: list[RejectedRecord] = []
    allowed_domains = set(config.language.allowed_domains)

    for candidate, decision in zip(candidates, decisions, strict=True):
        errors = validation_errors(decision, allowed_domains, seen_instructions)
        if decision.accept and not errors:
            record = make_accepted_record(candidate, decision, config)
            accepted.append(record)
            seen_instructions.add(normalize_instruction(record.messages[0].content))
        else:
            rejected.append(make_rejected_record(candidate, decision, config, errors))

    return accepted, rejected


def write_buffers(
    config: AppConfig,
    shard_index: int,
    accepted: list[AcceptedRecord],
    rejected: list[RejectedRecord],
    uploader: HubUploader | None,
) -> int:
    if not accepted and not rejected:
        return shard_index
    path = write_shard(config.output_dir, shard_index, accepted, rejected)
    if path is not None and uploader is not None:
        uploader.upload(path)
    return shard_index + 1


def flush_full_shards(
    config: AppConfig,
    shard_index: int,
    accepted: list[AcceptedRecord],
    rejected: list[RejectedRecord],
    uploader: HubUploader | None,
) -> int:
    while len(accepted) >= config.rows_per_shard:
        shard = accepted[: config.rows_per_shard]
        del accepted[: config.rows_per_shard]
        shard_index = write_buffers(config, shard_index, shard, rejected, uploader)
        rejected.clear()
    return shard_index


async def run(config: AppConfig, limit: int | None, resume: bool) -> None:
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be greater than zero")

    processed_ids, seen_instructions, shard_index = prepare_output(config, resume)
    prompt = PromptTemplate(config.prompt_path)
    client = InferenceClient(config.model)
    uploader = HubUploader(config.hub_repo) if config.hub_repo else None
    if uploader is not None:
        uploader.upload_missing(parquet_files(config.output_dir / DATASET_DIR))
    started_at = datetime.now(UTC).isoformat()
    write_run_file(config, "running", started_at)

    batch: list[Candidate] = []
    accepted_buffer: list[AcceptedRecord] = []
    rejected_buffer: list[RejectedRecord] = []
    processed_now = 0
    accepted_now = 0

    try:
        for candidate in iter_candidates(config, total_limit=limit):
            if candidate.id in processed_ids:
                continue
            batch.append(candidate)
            if len(batch) < config.model.concurrency:
                continue

            accepted, rejected = await process_batch(
                batch, client, prompt, config, seen_instructions
            )
            accepted_buffer.extend(accepted)
            rejected_buffer.extend(rejected)
            processed_now += len(batch)
            accepted_now += len(accepted)
            print(f"processed={processed_now} accepted={accepted_now}", flush=True)
            batch = []

            shard_index = flush_full_shards(
                config,
                shard_index,
                accepted_buffer,
                rejected_buffer,
                uploader,
            )

        if batch:
            accepted, rejected = await process_batch(
                batch, client, prompt, config, seen_instructions
            )
            accepted_buffer.extend(accepted)
            rejected_buffer.extend(rejected)
            processed_now += len(batch)
            accepted_now += len(accepted)
            print(f"processed={processed_now} accepted={accepted_now}", flush=True)

        write_buffers(
            config,
            shard_index,
            accepted_buffer,
            rejected_buffer,
            uploader,
        )
    finally:
        await client.close()

    write_run_file(config, "complete", started_at)


def dry_run(config: AppConfig, limit: int) -> None:
    prompt = PromptTemplate(config.prompt_path)
    for candidate in iter_candidates(config, total_limit=limit):
        print(f"\n===== {candidate.id} =====\n")
        print(prompt.render(candidate))
