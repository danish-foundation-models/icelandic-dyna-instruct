import re

from reverse_instruct.models import GenerationDecision

META_REFERENCES = (
    "text above",
    "passage above",
    "provided text",
    "teksturin omanfyri",
    "brotið omanfyri",
    "tekstinn hér að ofan",
    "ofangreindur texti",
)


def normalize_instruction(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def validation_errors(
    decision: GenerationDecision,
    allowed_domains: set[str],
    seen_instructions: set[str],
) -> list[str]:
    if not decision.accept:
        return []

    errors: list[str] = []
    instruction = decision.instruction or ""

    if decision.domain not in allowed_domains:
        errors.append(f"unknown domain: {decision.domain}")
    if not 20 <= len(instruction) <= 600:
        errors.append("instruction must contain 20 to 600 characters")

    normalized = normalize_instruction(instruction)
    if any(reference in normalized for reference in META_REFERENCES):
        errors.append("instruction refers to the supplied passage")
    if normalized in seen_instructions:
        errors.append("duplicate instruction")

    return errors
