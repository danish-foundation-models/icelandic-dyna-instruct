from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class LanguageConfig(BaseModel):
    code: str
    name: str
    allowed_domains: list[str]


class DatasetConfig(BaseModel):
    name: str
    revision: str
    split: str = "train"
    shuffle_buffer: int = Field(default=10_000, gt=0)


class SourceConfig(BaseModel):
    max_documents: int = Field(gt=0)
    domain_hint: str
    context: str


class TextConfig(BaseModel):
    min_chars: int = Field(gt=0)
    max_chars: int = Field(gt=0)


class ModelConfig(BaseModel):
    name: str
    base_url: str
    api_key_env: str
    temperature: float = Field(ge=0)
    max_tokens: int = Field(gt=0)
    concurrency: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0)


class ServerConfig(BaseModel):
    host: str
    port: int = Field(gt=0, le=65_535)
    tensor_parallel_size: int = Field(gt=0)
    gpu_memory_utilization: float = Field(gt=0, le=1)
    max_model_len: int = Field(gt=0)
    dtype: str


class AppConfig(BaseModel):
    run_name: str
    seed: int
    prompt_path: Path
    output_dir: Path
    rows_per_shard: int = Field(gt=0)
    hub_repo: str | None = None
    language: LanguageConfig
    dataset: DatasetConfig
    sources: dict[str, SourceConfig]
    text: TextConfig
    model: ModelConfig
    server: ServerConfig


def load_config(path: Path) -> AppConfig:
    config_path = path.resolve()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    for key in ("prompt_path", "output_dir"):
        value = Path(data[key])
        if not value.is_absolute():
            data[key] = config_path.parent / value

    config = AppConfig.model_validate(data)
    if config.text.min_chars >= config.text.max_chars:
        raise ValueError("text.min_chars must be smaller than text.max_chars")
    if not config.sources:
        raise ValueError("At least one source must be configured")
    return config
