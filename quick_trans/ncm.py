import base64
import json
import struct
from dataclasses import dataclass
from typing import Optional


try:
    from Crypto.Cipher import AES  # type: ignore
except Exception:
    AES = None


_CORE_KEY = bytes.fromhex("687A4852416D736F356B496E62617857")
_META_KEY = bytes.fromhex("2331346C6A6B5F215C5D2630553C2728")


def _unpad_pkcs7(data: bytes) -> bytes:
    if not data:
        return data
    pad = data[-1]
    if pad <= 0 or pad > 16:
        return data
    return data[:-pad]


@dataclass(frozen=True)
class NcmDecrypted:
    audio_bytes: bytes
    audio_format: Optional[str]


def decrypt_ncm(path: str) -> NcmDecrypted:
    if AES is None:
        raise RuntimeError("缺少 pycryptodome（Crypto.Cipher.AES），无法解密 .ncm")

    with open(path, "rb") as f:
        header = f.read(8)
        if header != b"CTENFDAM":
            raise ValueError("不是标准 NCM 文件（文件头不匹配）")

        f.seek(2, 1)

        key_length = struct.unpack("<I", f.read(4))[0]
        key_data = bytearray(f.read(key_length))
        for i in range(len(key_data)):
            key_data[i] ^= 0x64
        key_data = bytes(key_data)

        cryptor = AES.new(_CORE_KEY, AES.MODE_ECB)
        key_data = _unpad_pkcs7(cryptor.decrypt(key_data))[17:]

        key_data_ba = bytearray(key_data)
        key_len = len(key_data_ba)
        if key_len == 0:
            raise ValueError("NCM 密钥解析失败")

        key_box = bytearray(range(256))
        c = 0
        last_byte = 0
        key_offset = 0
        for i in range(256):
            swap = key_box[i]
            c = (swap + last_byte + key_data_ba[key_offset]) & 0xFF
            key_offset += 1
            if key_offset >= key_len:
                key_offset = 0
            key_box[i] = key_box[c]
            key_box[c] = swap
            last_byte = c

        meta_length = struct.unpack("<I", f.read(4))[0]
        meta_data = f.read(meta_length)
        audio_format: Optional[str] = None
        try:
            meta_ba = bytearray(meta_data)
            for i in range(len(meta_ba)):
                meta_ba[i] ^= 0x63
            meta_raw = base64.b64decode(bytes(meta_ba)[22:])
            meta_plain = _unpad_pkcs7(AES.new(_META_KEY, AES.MODE_ECB).decrypt(meta_raw))
            meta_json = meta_plain.decode("utf-8", errors="ignore")[6:]
            meta_obj = json.loads(meta_json)
            if isinstance(meta_obj, dict):
                v = meta_obj.get("format")
                if isinstance(v, str) and v:
                    audio_format = v.lower()
        except Exception:
            audio_format = None

        f.read(4)
        f.seek(5, 1)
        image_size = struct.unpack("<I", f.read(4))[0]
        if image_size > 0:
            f.seek(image_size, 1)

        out = bytearray()
        while True:
            chunk = bytearray(f.read(0x8000))
            if not chunk:
                break
            chunk_len = len(chunk)
            for i in range(1, chunk_len + 1):
                j = i & 0xFF
                chunk[i - 1] ^= key_box[(key_box[j] + key_box[(key_box[j] + j) & 0xFF]) & 0xFF]
            out.extend(chunk)

        return NcmDecrypted(audio_bytes=bytes(out), audio_format=audio_format)
