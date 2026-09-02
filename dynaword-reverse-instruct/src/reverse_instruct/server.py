import os
import shlex

from reverse_instruct.config import AppConfig


def serve(config: AppConfig) -> None:
    server = config.server
    command = [
        "vllm",
        "serve",
        config.model.name,
        "--host",
        server.host,
        "--port",
        str(server.port),
        "--tensor-parallel-size",
        str(server.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(server.gpu_memory_utilization),
        "--max-model-len",
        str(server.max_model_len),
        "--dtype",
        server.dtype,
        "--limit-mm-per-prompt",
        '{"image": 0, "audio": 0}',
        "--async-scheduling",
        "--structured-outputs-config",
        '{"backend": "xgrammar", "disable_any_whitespace": true}',
    ]
    print(shlex.join(command), flush=True)
    os.execvp(command[0], command)
