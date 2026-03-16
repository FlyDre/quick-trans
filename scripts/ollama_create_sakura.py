import argparse
import os
import subprocess
import tempfile


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="ollama_create_sakura")
    p.add_argument("--name", default="sakura-1.5b", help="Ollama model name")
    p.add_argument("--gguf", required=True, help="Local gguf file path")
    p.add_argument(
        "--system",
        default=(
            "You are a helpful assistant."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    name = str(args.name).strip()
    gguf = os.path.abspath(str(args.gguf).strip().strip('"'))
    if not os.path.isfile(gguf):
        raise SystemExit(f"gguf file not found: {gguf}")

    # Qwen2.5 chat template for Ollama
    template = (
        "{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}"
        "{{ range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{ end }}"
        "<|im_start|>assistant\n"
    )
    modelfile = (
        f'FROM "{gguf}"\n'
        f'SYSTEM """{args.system}"""\n'
        f'TEMPLATE """{template}"""\n'
        "PARAMETER temperature 0.7\n"
        "PARAMETER top_p 0.9\n"
        "PARAMETER repeat_penalty 1.1\n"
        'PARAMETER stop "<|im_end|>"\n'
    )
    with tempfile.TemporaryDirectory(prefix="ollama_sakura_") as td:
        path = os.path.join(td, "Modelfile")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(modelfile)
        subprocess.check_call(["ollama", "create", name, "-f", path])
    print(f"done: ollama create {name}")


if __name__ == "__main__":
    main()
