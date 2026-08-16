import sys
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_REPO = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


destination = Path(sys.argv[1])
revision_path = destination / "REVISION"
if revision_path.exists():
    assert revision_path.read_text().strip() == MODEL_REVISION
    assert (destination / "config.json").is_file()
    raise SystemExit(0)

snapshot_download(
    repo_id=MODEL_REPO,
    revision=MODEL_REVISION,
    local_dir=destination,
)
temporary_revision = destination / "REVISION.tmp"
temporary_revision.write_text(MODEL_REVISION + "\n")
temporary_revision.replace(revision_path)
