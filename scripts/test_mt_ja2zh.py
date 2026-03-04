import argparse
import difflib
import os
import sys


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


_ROOT = _repo_root()
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


CASES = [
    {
        "id": "dialog-1",
        "ja": "おはよう。昨日はごめんね。急に帰らなきゃいけなくなって、何も言えなかった。",
        "zh": "早上好。昨天对不起。我不得不突然回去，什么也没来得及说。",
    },
    {
        "id": "dialog-2",
        "ja": "大丈夫？無理しないで。痛かったら、すぐに言って。",
        "zh": "没事吧？别勉强。要是疼就马上说。",
    },
    {
        "id": "narr-1",
        "ja": "窓の外では雨が静かに降っていた。彼は冷めたコーヒーを一口飲み、ため息をついた。",
        "zh": "窗外静静地下着雨。他抿了一口凉掉的咖啡，叹了口气。",
    },
    {
        "id": "narr-2",
        "ja": "約束は約束だ。どんな理由があっても、破っていいものじゃない。",
        "zh": "约定就是约定。无论有什么理由，都不该违背。",
    },
    {
        "id": "instr-1",
        "ja": "このファイルを保存したら、アプリを再起動してください。設定が反映されます。",
        "zh": "保存这个文件后，请重启应用，设置会生效。",
    },
    {
        "id": "tech-1",
        "ja": "ログを見ると、ネットワークが不安定なときだけタイムアウトが発生している。",
        "zh": "从日志来看，只有在网络不稳定时才会发生超时。",
    },
    {
        "id": "song-1",
        "ja": "深い夢から覚めて隣を見ても、君はやっぱりいなくて。",
        "zh": "从深梦中醒来，转头看向身旁，你果然还是不在。",
    },
    {
        "id": "song-2",
        "ja": "空から見る世界は何色に見えますか？",
        "zh": "从天空俯瞰的世界，会是什么颜色？",
    },
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="test_mt_ja2zh")
    p.add_argument("--only", default=None, help="仅运行指定 case id（例如 dialog-1）")
    p.add_argument("--ollama-host", default=None, help="默认 http://127.0.0.1:11434")
    p.add_argument("--sakura-model", default="sakura-1.5b", help="Ollama 模型名")
    p.add_argument(
        "--nllb-model",
        default=os.path.join(_repo_root(), "llm", "facebook", "nllb-200-distilled-600M"),
        help="NLLB 模型目录或模型名",
    )
    p.add_argument("--nllb-device", default="cuda")
    p.add_argument("--nllb-beam-size", type=int, default=1)
    p.add_argument("--hf-token", default=None)
    return p.parse_args()


def _ratio(a: str, b: str) -> float:
    return float(difflib.SequenceMatcher(a=a or "", b=b or "").ratio())


def main() -> None:
    args = _parse_args()

    from quick_trans.mt import Translator
    from quick_trans.mt_sakura import SakuraOllamaTranslator

    nllb_device = str(args.nllb_device).strip()
    if nllb_device.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                nllb_device = "cpu"
        except Exception:
            nllb_device = "cpu"

    nllb = Translator(
        model_name=str(args.nllb_model).strip(),
        device=nllb_device,
        beam_size=int(args.nllb_beam_size),
        hf_token=(str(args.hf_token).strip() if args.hf_token else None),
    )
    sakura = SakuraOllamaTranslator(
        model_name=str(args.sakura_model).strip(),
        host=(str(args.ollama_host).strip() if args.ollama_host else None),
    )

    cases = CASES
    if args.only:
        cases = [c for c in CASES if c["id"] == args.only]
        if not cases:
            raise SystemExit(f"未找到 case id：{args.only}")

    for c in cases:
        ja = c["ja"]
        gold = c["zh"]
        out_nllb = nllb.translate_ja_to_zh(ja)
        out_sakura = sakura.translate_ja_to_zh(ja)

        r_nllb = _ratio(out_nllb, gold)
        r_sakura = _ratio(out_sakura, gold)

        print("=" * 80)
        print(f"CASE: {c['id']}")
        print(f"JA: {ja}")
        print(f"GOLD: {gold}")
        print(f"NLLB : {out_nllb}")
        print(f"SAK  : {out_sakura}")
        print(f"SCORE (char ratio)  NLLB={r_nllb:.3f}  SAKURA={r_sakura:.3f}")


if __name__ == "__main__":
    main()
