from quick_trans.ollama_client import generate


_DEFAULT_SYSTEM = (
    "你是一个轻小说/歌词风格的日中翻译模型。"
    "你要把日文翻译成简体中文。"
    "要求：忠实原意；尽量不要增删信息；不要擅自添加原文没有的主语/人称代词；不要胡编剧情；仅输出译文。"
)


def _build_sakura_prompt(system: str, japanese: str) -> str:
    user_prompt = "将下面的日文文本翻译成中文：" + japanese
    return (
        "<|im_start|>system\n"
        + system
        + "<|im_end|>\n"
        + "<|im_start|>user\n"
        + user_prompt
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )


class SakuraOllamaTranslator:
    def __init__(
        self,
        model_name: str,
        host: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._model = model_name
        self._host = host
        self._system = system_prompt or _DEFAULT_SYSTEM

    def translate_ja_to_zh(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        prompt = _build_sakura_prompt(self._system, text)
        out = generate(
            model=self._model,
            prompt=prompt,
            host=self._host,
            options={
                "temperature": 0.1,
                "top_p": 0.3,
                "repeat_penalty": 1.15,
                "num_predict": 128,
                "stop": ["<|im_end|>"],
            },
            raw=True,
            timeout_s=180.0,
        )
        return (out or "").strip()
