import json
import os

from openai import AsyncOpenAI

from reverse_instruct.config import ModelConfig
from reverse_instruct.models import GenerationDecision


class InferenceClient:
    def __init__(self, config: ModelConfig) -> None:
        api_key = os.getenv(config.api_key_env) or "unused"
        self.config = config
        self.client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=api_key,
            timeout=config.timeout_seconds,
            max_retries=0,
        )

    async def generate(self, prompt: str) -> GenerationDecision:
        schema = GenerationDecision.model_json_schema()
        response = await self.client.chat.completions.create(
            model=self.config.name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "generation_decision",
                    "strict": True,
                    "schema": schema,
                },
            },
        )
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("Inference server returned no content")
        return GenerationDecision.model_validate(json.loads(content))

    async def close(self) -> None:
        await self.client.close()
