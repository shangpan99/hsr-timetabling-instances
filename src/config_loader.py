# -*- coding: utf-8 -*-
"""统一实验配置读取工具。

支持 UTF-8 / UTF-8-BOM / UTF-16 / GB18030 / GBK 等常见编码。
即使配置文件的注释因 Windows 编辑器被保存成混合编码，只要参数键和值仍可识别，
也会尽量继续读取并给出明确警告，而不会把最后一次解码失败误报成唯一原因。
"""
from __future__ import annotations

import csv
import io
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

_STRICT_ENCODINGS = (
    "utf-8-sig",
    "utf-8",
    "utf-16",
    "utf-16-le",
    "utf-16-be",
    "gb18030",
    "gbk",
    "cp1252",
)
_FALLBACK_ENCODINGS = ("utf-8-sig", "gb18030", "gbk", "cp1252", "latin-1")
_SENTINELS = {"", "none", "null", "nil", "nan"}
_TRUE = {"true", "yes", "y", "on", "是"}
_FALSE = {"false", "no", "n", "off", "否"}

# 这些键均为 ASCII。它们用于避免把 UTF-16 数据误当成含大量 NUL 的 UTF-8 文本，
# 也用于确认一次“宽松解码”确实找到了配置内容，而非偶然解出了乱码。
_SIGNATURE_KEYS = {"dep_window_fast", "dep_window_normal", "sampling_allocation_mode"}


def cast_value(value: Any) -> Any:
    """将配置值转换为 None / bool / int / float / str。

    数字 0/1 保留为整数，使用 ``as_bool`` 时仍可按布尔值读取；
    这样可以避免 alpha=1、min_obs=1 被误读为 True。
    """
    if value is None:
        return None
    text = str(value).strip()
    low = text.lower()
    if low in _SENTINELS:
        return None
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    try:
        if low.startswith("0x"):
            return int(low, 16)
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _extract_config_keys(text: str) -> set[str]:
    """在不转换值的情况下提取有效配置键，用于验证解码结果。"""
    if not text:
        return set()
    # 正常解码后的配置不应包含 NUL；含 NUL 通常说明 UTF-16 被误按单字节编码读取。
    if "\x00" in text:
        return set()

    keys: set[str] = set()
    try:
        reader = csv.reader(io.StringIO(text, newline=""))
        for row in reader:
            if not row:
                continue
            key = str(row[0]).lstrip("\ufeff").strip()
            if not key or key.startswith("#") or key.lower() == "key":
                continue
            keys.add(key)
    except (csv.Error, UnicodeError):
        return set()
    return keys


def _looks_like_expected_config(text: str) -> bool:
    keys = _extract_config_keys(text)
    return _SIGNATURE_KEYS.issubset(keys)


def _encoding_candidates(raw: bytes) -> Tuple[str, ...]:
    """根据 BOM/NUL 特征调整尝试顺序，同时保留完整候选集合。"""
    preferred: list[str] = []
    if raw.startswith(b"\xef\xbb\xbf"):
        preferred.append("utf-8-sig")
    elif raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        preferred.append("utf-16")
    elif raw[:200].count(b"\x00") >= 5:
        # 无 BOM 的 UTF-16 配置文件在 Windows/Excel 中并不少见。
        preferred.extend(("utf-16-le", "utf-16-be"))

    ordered: list[str] = []
    for enc in (*preferred, *_STRICT_ENCODINGS):
        if enc not in ordered:
            ordered.append(enc)
    return tuple(ordered)


def _decode_config_bytes(path: Path) -> Tuple[str, str, bool, Dict[str, str]]:
    """读取并解码配置文件。

    Returns
    -------
    text, encoding_label, used_replacement, diagnostics
    """
    raw = path.read_bytes()
    if not raw:
        raise ValueError(f"配置文件为空：{path}")

    diagnostics: Dict[str, str] = {}

    # 第一轮：严格解码。只接受确实包含关键配置项的结果。
    for encoding in _encoding_candidates(raw):
        try:
            text = raw.decode(encoding, errors="strict")
        except (UnicodeDecodeError, LookupError) as exc:
            diagnostics[encoding] = f"{type(exc).__name__}: {exc}"
            continue
        if _looks_like_expected_config(text):
            return text, encoding, False, diagnostics
        diagnostics[encoding] = "decoded, but required configuration keys were not found"

    # 第二轮：容错解码。主要处理“ASCII 参数行正常、中文注释有个别坏字节”的混合文件。
    # 参数键和值在本项目默认配置中均为 ASCII，因此注释里的替换字符不会改变实验参数。
    for encoding in _FALLBACK_ENCODINGS:
        try:
            text = raw.decode(encoding, errors="replace")
        except LookupError as exc:
            diagnostics[f"{encoding}+replace"] = f"{type(exc).__name__}: {exc}"
            continue
        if _looks_like_expected_config(text):
            return text, f"{encoding}+replace", True, diagnostics
        diagnostics[f"{encoding}+replace"] = (
            "decoded with replacement, but required configuration keys were not found"
        )

    details = "\n".join(f"  - {enc}: {msg}" for enc, msg in diagnostics.items())
    raise UnicodeError(
        "无法识别 config.csv 的编码，或文件内容已损坏。\n"
        f"文件：{path}\n"
        "请用 Windows 记事本打开该文件，选择“另存为”，编码选 UTF-8；"
        "也可用随包提供的原始 config.csv 覆盖。\n"
        f"尝试记录：\n{details}"
    )


def load_config(
    path: str | Path | None = None,
    defaults: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    path = project_root() / "settings/config.csv" if path is None else Path(path)
    if not path.exists():
        if defaults is None:
            raise FileNotFoundError(f"未找到配置文件：{path}")
        return dict(defaults)

    text, encoding, used_replacement, diagnostics = _decode_config_bytes(path)

    cfg: Dict[str, Any] = {}
    reader = csv.reader(io.StringIO(text, newline=""))
    for line_no, row in enumerate(reader, start=1):
        if not row:
            continue
        key = str(row[0]).lstrip("\ufeff").strip()
        if not key or key.startswith("#") or key.lower() == "key":
            continue
        if len(row) < 2:
            raise ValueError(
                f"config.csv 第 {line_no} 行缺少 value 列：{row!r}；"
                "每个参数应写成 key,value"
            )
        cfg[key] = cast_value(row[1])

    missing = sorted(_SIGNATURE_KEYS - set(cfg))
    if missing:
        raise ValueError(f"config.csv 缺少必要参数：{missing}")

    if defaults:
        for key, value in defaults.items():
            cfg.setdefault(key, value)

    cfg["_config_encoding"] = encoding
    cfg["_config_decode_replaced"] = int(used_replacement)
    cfg["_config_path"] = str(path.resolve())

    if used_replacement:
        warnings.warn(
            "config.csv 含有混合编码或无效字节；程序已在不影响可识别参数行的前提下"
            "进行容错读取。建议稍后用 Windows 记事本将 config.csv 另存为 UTF-8。"
            f" 当前读取方式：{encoding}",
            RuntimeWarning,
            stacklevel=2,
        )

    return cfg


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    low = str(value).strip().lower()
    if low in _TRUE or low == "1":
        return True
    if low in _FALSE or low == "0":
        return False
    return default


def as_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    return int(float(value))


def as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    return float(value)


def parse_list(value: Any, *, sep: str = ";") -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip().replace("，", sep).replace(",", sep)
    return [x.strip() for x in text.split(sep) if x.strip()]


def parse_int_list(value: Any) -> List[int]:
    return [int(float(x)) for x in parse_list(value)]


def parse_float_list(value: Any) -> List[float]:
    return [float(x) for x in parse_list(value)]


def parse_named_designs(value: Any) -> Dict[str, List[int]]:
    """解析 ``D2:80+100;D4:60+80+100+120``。"""
    result: Dict[str, List[int]] = {}
    for item in parse_list(value):
        if ":" not in item:
            raise ValueError(f"multi-R design 格式错误：{item!r}；应为 D2:80+100")
        name, sizes_text = item.split(":", 1)
        sizes = [int(float(x.strip())) for x in sizes_text.replace(",", "+").split("+") if x.strip()]
        if not name.strip() or not sizes:
            raise ValueError(f"multi-R design 格式错误：{item!r}")
        result[name.strip()] = sizes
    return result


def get_k_list(cfg: Mapping[str, Any], prefix: str, k: int) -> List[int]:
    return parse_int_list(cfg.get(f"{prefix}_K{k}"))


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_project_path(value: Any, default: str | None = None) -> Path:
    raw = value if value not in {None, ""} else default
    if raw is None:
        raise ValueError("路径配置为空")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = project_root() / path
    return path.resolve()
