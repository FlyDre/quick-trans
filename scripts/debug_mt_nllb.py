import argparse
import os


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="debug_mt_nllb")
    p.add_argument(
        "--model",
        default=os.path.join(_repo_root(), "llm", "facebook", "nllb-200-distilled-600M"),
    )
    p.add_argument("--src", default="jpn_Jpan")
    p.add_argument("--tgt", default="zho_Hans")
    p.add_argument(
        "--text",
        default="強くなれる理由を知った 僕を連れて",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    model_dir = os.path.abspath(args.model)
    src = str(args.src).strip()
    tgt = str(args.tgt).strip()
    text = str(args.text)

    import transformers
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print("transformers:", transformers.__version__)
    print("model_dir:", model_dir)

    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True)
    print("model_type:", getattr(getattr(model, "config", None), "model_type", None))
    print("tokenizer:", tok.__class__.__name__)

    src_id = tok.convert_tokens_to_ids(src)
    tgt_id = tok.convert_tokens_to_ids(tgt)
    unk = getattr(tok, "unk_token_id", None)
    if src_id is None or tgt_id is None or (unk is not None and (src_id == unk or tgt_id == unk)):
        raise RuntimeError(f"tokenizer 不支持 src/tgt 语言token：src={src} tgt={tgt}")

    eos_id = int(getattr(getattr(model, "config", None), "eos_token_id", 2))
    ids = tok.encode(text, add_special_tokens=False)
    input_ids = torch.tensor([[int(src_id), *ids, eos_id]], dtype=torch.long)
    out = model.generate(input_ids=input_ids, forced_bos_token_id=int(tgt_id), num_beams=1, do_sample=False, max_new_tokens=64)
    first_token = int(out[0][0].item()) if out.numel() > 0 else None
    print("src_lang_id:", int(src_id))
    print("forced_bos_token_id:", int(tgt_id))
    print("first_generated_token_id:", first_token)
    print("decoded:", tok.batch_decode(out, skip_special_tokens=True)[0])


if __name__ == "__main__":
    main()
