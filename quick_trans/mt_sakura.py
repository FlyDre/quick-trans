from quick_trans.ollama_client import generate


_DEFAULT_SYSTEM = (
    "你是严谨的日中字幕翻译引擎。"
    "你只做日文到简体中文翻译，不解释，不改写，不扩写，不总结。"
    "必须忠实原句语义与语气，不得凭空补充主语、剧情和背景。"
    "仅输出简体中文译文，不要输出任何标签或额外文本。"
)


def _build_sakura_prompt(system: str, japanese: str, prev_text: str | None = None, next_text: str | None = None) -> str:
    sys = (system or "").strip() or _DEFAULT_SYSTEM
    _ = prev_text
    _ = next_text
    return (
        "<|im_start|>system\n"
        + sys
        + "<|im_end|>\n"
        + "<|im_start|>user\n"
        + japanese
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )


def _clean_translation_output(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    for prefix in (
        "任务：",
        "请把【当前句】翻译成简体中文",
        "上文：",
        "当前句：",
        "下文：",
    ):
        if s.startswith(prefix):
            lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
            filtered = [ln for ln in lines if not (ln.startswith("任务：") or ln.startswith("上文：") or ln.startswith("当前句：") or ln.startswith("下文："))]
            s = "\n".join(filtered).strip()
            break
    return s


def _looks_untranslated(src: str, out: str) -> bool:
    s = (src or "").strip()
    t = (out or "").strip()
    if not t:
        return True
    if "<|im_start|>" in t or "<|im_end|>" in t:
        return True
    if "严谨的日中字幕翻译引擎" in t or "仅输出简体中文译文" in t:
        return True
    if t == s:
        return True
    if all(ch in "。！？,.!?;；:：、」「『』（）()[]{}" for ch in t):
        return True
    return False


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
        return self.translate_ja_to_zh_with_context(text=text, prev_text=None, next_text=None)

    def translate_ja_to_zh_with_context(self, text: str, prev_text: str | None = None, next_text: str | None = None) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        prompt = _build_sakura_prompt(self._system, text, prev_text=prev_text, next_text=next_text)
        out = generate(
            model=self._model,
            prompt=prompt,
            host=self._host,
            options={
                "temperature": 0.0,
                "top_p": 1.0,
                "repeat_penalty": 1.08,
                "num_predict": 256,
                "stop": ["<|im_end|>"],
            },
            raw=True,
            timeout_s=180.0,
        )
        cleaned = _clean_translation_output(out)
        if _looks_untranslated(text, cleaned):
            retry_prompt = _build_sakura_prompt(self._system, text, prev_text=None, next_text=None)
            retry_out = generate(
                model=self._model,
                prompt=retry_prompt,
                host=self._host,
                options={
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "repeat_penalty": 1.08,
                    "num_predict": 256,
                    "stop": ["<|im_end|>"],
                },
                raw=True,
                timeout_s=180.0,
            )
            cleaned = _clean_translation_output(retry_out)
        if _looks_untranslated(text, cleaned):
            return ""
        return cleaned
