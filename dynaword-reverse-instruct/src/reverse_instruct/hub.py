from pathlib import Path

from huggingface_hub import HfApi


class HubUploader:
    def __init__(self, repo_id: str) -> None:
        self.repo_id = repo_id
        self.api = HfApi()
        self.api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    def upload(self, path: Path) -> None:
        self.api.upload_file(
            path_or_fileobj=path,
            path_in_repo=f"data/{path.name}",
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message=f"Add {path.name}",
        )

    def upload_missing(self, paths: list[Path]) -> None:
        remote_files = set(self.api.list_repo_files(self.repo_id, repo_type="dataset"))
        for path in paths:
            if f"data/{path.name}" not in remote_files:
                self.upload(path)
