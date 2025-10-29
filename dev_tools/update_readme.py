import inspect
import re
import sys
from pathlib import Path

from polars_pipe.core import transform as tf
from polars_pipe.core import validation as vl


def find_pipe_funcs(
    pipeline_path: str = "./src/polars_pipe/services/basic_pipeline.py",
) -> list[str]:
    src_code = Path(pipeline_path).read_text()
    matches = re.findall(r"(?<=pipe\()\s*([^,)\s]+)", src_code)
    return [m.split(".")[-1] for m in matches]


def extract_stages_docs(matches: list[str]) -> str:
    fns = {
        **dict(inspect.getmembers(tf, inspect.isroutine)),
        **dict(inspect.getmembers(vl, inspect.isroutine)),
    }

    lines = []
    for m in matches:
        lines.append(f"\n## `{m}`")
        if (doc := fns[m].__doc__) is not None:
            lines.append("\n".join([line.removeprefix("    ") for line in doc.splitlines()]))
    return "\n".join(lines)


def update_readme(pipeline_docs: str, readme_path: str = "./README.md") -> None:
    readme_path = Path(readme_path)
    readme_txt = readme_path.read_text(encoding="utf-8")
    pattern = r"(# pipeline stages)(.*?)(::)"
    updated_readme = re.sub(pattern, rf"\1\n\n{pipeline_docs}\n\3", readme_txt, flags=re.DOTALL)
    if readme_txt != updated_readme:
        readme_path.write_text(updated_readme)
        return 1
    return 0


if __name__ == "__main__":
    matches = find_pipe_funcs()
    pipeline_docs = extract_stages_docs(matches)
    sys.exit(update_readme(pipeline_docs))
