import os

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Translator:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        beam_size: int = 1,
        hf_token: str | None = None,
    ) -> None:
        is_local = os.path.isdir(model_name)
        extra_kwargs = {"token": hf_token}
        if is_local:
            extra_kwargs["local_files_only"] = True
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name, **extra_kwargs)
        except Exception as e:
            raise RuntimeError(
                "翻译模型加载失败。常见原因：1）无法访问 Hugging Face；2）HF_TOKEN 无效；3）模型名拼写错误；4）公司/校园网拦截。"
                "可尝试：a) 不设置 HF_TOKEN 重试；b) 传 --hf-endpoint https://hf-mirror.com；"
                "c) 手动下载模型到本地目录后用 --mt-model 指向该目录。"
            ) from e
        torch_device = torch.device(device)
        dtype = torch.float16 if torch_device.type == "cuda" else torch.float32
        try:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype, **extra_kwargs)
        except Exception as e:
            raise RuntimeError(
                "翻译模型权重加载失败。可尝试：a) 传 --hf-token；b) 传 --hf-endpoint；c) 先手动下载到本地再加载。"
            ) from e
        self._model.to(torch_device)
        self._model.eval()
        self._device = torch_device
        self._beam_size = beam_size
        model_type = getattr(getattr(self._model, "config", None), "model_type", None)
        self._src_lang_id: int | None = None
        self._tgt_lang_id: int | None = None
        if model_type == "m2m_100":
            self._src_lang_id = self._maybe_lang_id("jpn_Jpan")
            self._tgt_lang_id = self._maybe_lang_id("zho_Hans")
        self._use_m2m_lang_tokens = self._src_lang_id is not None and self._tgt_lang_id is not None

    def _maybe_lang_id(self, lang_token: str) -> int | None:
        tid = self._tokenizer.convert_tokens_to_ids(lang_token)
        unk = getattr(self._tokenizer, "unk_token_id", None)
        if tid is None:
            return None
        if unk is not None and tid == unk:
            return None
        if isinstance(tid, int) and tid >= 0:
            return tid
        return None

    def translate_ja_to_zh(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        with torch.inference_mode():
            if self._use_m2m_lang_tokens:
                src_id = int(self._src_lang_id)
                tgt_id = int(self._tgt_lang_id)
                eos_id = int(getattr(getattr(self._model, "config", None), "eos_token_id", 2))
                ids = self._tokenizer.encode(text, add_special_tokens=False)
                input_ids = torch.tensor([[src_id, *ids, eos_id]], dtype=torch.long, device=self._device)
                out = self._model.generate(
                    input_ids=input_ids,
                    forced_bos_token_id=tgt_id,
                    num_beams=self._beam_size,
                    do_sample=False,
                    max_new_tokens=128,
                )
                s = self._tokenizer.batch_decode(out, skip_special_tokens=True)[0]
                return (s or "").strip()

            enc = self._tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(self._device)
            out = self._model.generate(
                **enc,
                num_beams=self._beam_size,
                do_sample=False,
                max_new_tokens=128,
            )
            s = self._tokenizer.batch_decode(out, skip_special_tokens=True)[0]
            return (s or "").strip()
