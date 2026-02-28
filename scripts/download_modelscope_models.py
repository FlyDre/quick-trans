import argparse
import os
from typing import Iterable


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="download_modelscope_models")
    p.add_argument(
        "--dir",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "llm")),
        help="模型下载目录（默认：项目根目录下的 llm）",
    )
    p.add_argument(
        "--models",
        nargs="*",
        default=[
            "gpustack/faster-whisper-small",
            "gpustack/faster-whisper-medium",
            "facebook/nllb-200-distilled-600M",
        ],
        help="要下载的 ModelScope 模型ID 列表",
    )
    p.add_argument("--revision", default="master")
    return p.parse_args()


def _download(model_id: str, base_dir: str, revision: str) -> str:
    try:
        from modelscope import snapshot_download  # type: ignore
    except Exception as e:
        raise RuntimeError("缺少 modelscope。请先安装：pip install -U modelscope") from e

    os.makedirs(base_dir, exist_ok=True)
    return snapshot_download(model_id, cache_dir=base_dir, revision=revision)


def main() -> None:
    args = _parse_args()
    base_dir = os.path.abspath(args.dir)
    revision = str(args.revision).strip()

    print(f"目标目录: {base_dir}")
    for m in args.models:
        model_id = str(m).strip().strip("`'\"").rstrip("/")
        if not model_id:
            continue
        print(f"下载: {model_id} (revision={revision})")
        local_path = _download(model_id, base_dir=base_dir, revision=revision)
        print(f"完成: {model_id} -> {local_path}")


if __name__ == "__main__":
    main()
