import argparse
import os
import subprocess
import tempfile


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="ollama_create_sakura")
    p.add_argument("--name", default="sakura-1.5b", help="Ollama 模型名（后续用 --mt-model 指向它）")
    p.add_argument("--gguf", required=True, help="本地 gguf 文件路径")
    p.add_argument(
        "--system",
        default=(
            "你是一个轻小说/歌词风格的日中翻译模型。"
            "你要把日文翻译成简体中文。"
            "要求：忠实原意；尽量不要增删信息；不要擅自添加原文没有的主语/人称代词；不要胡编剧情；仅输出译文。"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    name = str(args.name).strip()
    gguf = os.path.abspath(str(args.gguf).strip().strip('"'))
    if not os.path.isfile(gguf):
        raise SystemExit(f"gguf 文件不存在：{gguf}")

    modelfile = f'FROM "{gguf}"\nSYSTEM """{args.system}"""\nPARAMETER temperature 0.1\nPARAMETER top_p 0.3\nPARAMETER repeat_penalty 1.0\n'
    with tempfile.TemporaryDirectory(prefix="ollama_sakura_") as td:
        path = os.path.join(td, "Modelfile")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(modelfile)
        subprocess.check_call(["ollama", "create", name, "-f", path])
    print(f"完成：ollama create {name}")


if __name__ == "__main__":
    main()
