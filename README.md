# quick-trans
A project for generate Chinese subtitles for Japanese videos in real-time. 实时为日语视频生成中文字幕。

## quick-trans（主干流程：日语音频→中文字幕）

### 目标
- 输入：本地音频文件（后续可扩展到 Windows 系统输出设备回环）
- 输出：实时生成 WebVTT 字幕（中文字幕）
- 默认档：faster-whisper（small）+ 日语VAD分段 + NLLB（日→中）

### 依赖
- Python 3.10+
- 内置解码依赖 miniaudio（可解码 wav/flac/mp3/ogg 等）
- 支持自动解密并解码网易云 .ncm（容器内通常为 mp3 或 flac）
- Python 依赖见 requirements.txt

### 运行
```bash
python -m quick_trans.cli --input .\demo.wav --output .\out.vtt
```

### Web 实时字幕（本地）
启动服务：
```bash
python -m quick_trans.webui --host 127.0.0.1 --port 8000
```
浏览器打开：
- http://127.0.0.1:8000/
在页面里选择音频/视频文件并点击开始，字幕会随着播放时间显示（ASR+中文）。
### 下载模型
python .\scripts\download_modelscope_models.py --dir f:\proj-audio\quick-trans\llm

###
python -m quick_trans.cli --input "1.ncm" --output out-s.vtt --asr-model ".\llm\gpustack\faster-whisper-small" --mt-model ".\llm\facebook\nllb-200-distilled-600M" --asr-compute-type int8 --asr-text
###
python -m quick_trans.cli --input "2.mp3" --output out-m.vtt --asr-model ".\llm\gpustack\faster-whisper-medium" --mt-model ".\llm\facebook\nllb-200-distilled-600M" --asr-compute-type int8 --asr-text
###

### small
python -m quick_trans.cli --input "1.ncm" --output out-s.vtt --asr-model ".\llm\gpustack\faster-whisper-small"

### meduim
python -m quick_trans.cli --input "1.ncm" --output out-m.vtt --asr-model ".\llm\gpustack\faster-whisper-medium"



离线/国内网络推荐：先用 ModelScope 预下载模型到项目的 llm 目录：
```bash
python .\scripts\download_modelscope_models.py --dir .\llm
```
如提示缺少 modelscope，请先安装：
```bash
pip install -U modelscope
```

输入格式：
- 直接支持：wav / flac / mp3 / ogg（由 miniaudio 解码并自动转为 16k 单声道）
- 支持：ncm（自动解密并解码，无需手动转换）
- 其他格式（如 m4a/aac）：需要额外解码后再输入

常用参数：
- --asr-model：可用 .\llm\gpustack\faster-whisper-small 或 .\llm\gpustack\faster-whisper-medium；默认优先 small
- --asr-device：cuda / cpu
- --cpu：强制 ASR 与翻译都在 CPU 上跑
- --asr-text：把 ASR 原文落盘为 txt（不给值则自动生成为 output 同名的 .asr.txt）
- --mt-model：默认优先使用 .\llm\facebook\nllb-200-distilled-600M，若不存在则用 facebook/nllb-200-distilled-600M
- --mt-device：cuda / cpu
- --realtime：按音频时长节奏实时推进（默认会尽快处理）

### 输出
- out.vtt：WebVTT 格式，可直接被播放器/浏览器/OBS 浏览器源使用
