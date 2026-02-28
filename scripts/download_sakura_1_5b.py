import argparse
import os
from typing import Iterable


DEFAULT_HF_REPO = "SakuraLLM/Sakura-1.5B-Qwen2.5-v1.0-GGUF"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="download_sakura_1_5b")
    p.add_argument(
        "--dir",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "llm")),
        help="模型下载目录（默认：项目根目录下的 llm）",
    )
    p.add_argument("--model", default=DEFAULT_HF_REPO, help="HuggingFace 模型ID")
    p.add_argument("--revision", default="main")
    p.add_argument("--hf-endpoint", default=None, help="例如 https://hf-mirror.com")
    p.add_argument(
        "--include",
        nargs="*",
        default=["*.gguf", "*.json", "README*", "LICENSE*"],
        help="仅 HuggingFace：下载文件匹配（glob），默认只拉 gguf/元数据文件",
    )
    p.add_argument("--exclude", nargs="*", default=[], help="仅 HuggingFace：排除文件匹配（glob）")
    return p.parse_args()


def _hf_download(repo_id: str, dst_dir: str, revision: str, hf_endpoint: str | None, include: Iterable[str], exclude: Iterable[str]) -> str:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as e:
        raise RuntimeError("缺少 huggingface_hub。请先安装：pip install -U huggingface_hub") from e

    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = str(hf_endpoint).strip().rstrip("/")

    local_dir = os.path.join(dst_dir, repo_id.replace("/", os.sep))
    os.makedirs(local_dir, exist_ok=True)
    return snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=list(include) if include else None,
        ignore_patterns=list(exclude) if exclude else None,
    )


def main() -> None:
    args = _parse_args()
    dst_dir = os.path.abspath(args.dir)
    model_id = str(args.model).strip().strip("`'\"").rstrip("/")
    revision = str(args.revision).strip()
    hf_endpoint = str(args.hf_endpoint).strip() if args.hf_endpoint else None

    if not model_id:
        raise SystemExit("model 不能为空")

    print(f"目标目录: {dst_dir}")
    print(f"模型: {model_id}")
    print("来源: HuggingFace")

    p = _hf_download(
        repo_id=model_id,
        dst_dir=dst_dir,
        revision=revision if revision else "main",
        hf_endpoint=hf_endpoint,
        include=args.include or [],
        exclude=args.exclude or [],
    )
    print(f"完成(HuggingFace): {p}")


if __name__ == "__main__":
    main()
