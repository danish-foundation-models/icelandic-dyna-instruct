from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Candidate(BaseModel):
    id: str
    source: str
    language: str
    domain_hint: str
    source_context: str
    passage: str


class GenerationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accept: bool
    domain: str | None
    instruction: str | None = Field(max_length=300)
    reason: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def accepted_rows_have_an_instruction(self) -> "GenerationDecision":
        if self.accept and (self.domain is None or self.instruction is None):
            raise ValueError("Accepted output requires domain and instruction")
        return self


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AcceptedRecord(BaseModel):
    id: str
    messages: list[Message]
    model: str


class RejectedRecord(BaseModel):
    id: str
    source: str
    domain: str | None
    instruction: str | None
    response: str
    reason: str
    model: str
    dataset_revision: str
