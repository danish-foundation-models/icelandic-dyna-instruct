from pathlib import Path
from string import Template

from reverse_instruct.models import Candidate


class PromptTemplate:
    def __init__(self, path: Path) -> None:
        self.template = Template(path.read_text(encoding="utf-8"))

    def render(self, candidate: Candidate) -> str:
        return self.template.substitute(
            source=candidate.source,
            domain_hint=candidate.domain_hint,
            source_context=candidate.source_context,
            passage=candidate.passage,
        )
