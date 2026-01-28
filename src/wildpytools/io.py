from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Optional, Union, Sequence, Literal, Tuple, Set, List
import logging

logger = logging.getLogger(__name__)





from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict

import re
import soundfile as sf
from mutagen import File as MutagenFile
from dateutil import parser as dateparser


DateResult = Tuple[Optional[datetime], Optional[str]]


def extract_recording_datetime(
    path: str | Path,
    *,
    allow_filename_fallback: bool = True,
    allow_filesystem_fallback: bool = True,
) -> DateResult:
    """
    Extract a recording datetime from an audio file using a prioritized strategy.

    Returns
    -------
    (datetime | None, source | None)

    source is one of:
        - 'bwf'
        - 'ixml'
        - 'container_tag'
        - 'filename'
        - 'filesystem'
    """

    def _to_seconds(dt):
        return dt.replace(microsecond=0) if dt is not None else None

    path = Path(path)

    # -----------------------------
    # 1. BWF (Broadcast Wave) fields
    # -----------------------------
    if path.suffix.lower() == ".wav":
        try:
            with sf.SoundFile(path) as f:
                info = f.extra_info or ""

            date_match = re.search(r"OriginationDate\s*:\s*(\d{4}-\d{2}-\d{2})", info)
            time_match = re.search(r"OriginationTime\s*:\s*(\d{2}:\d{2}:\d{2})", info)

            if date_match:
                dt_str = date_match.group(1)
                if time_match:
                    dt_str += " " + time_match.group(1)
                return _to_seconds(dateparser.parse(dt_str)), "bwf"

        except Exception:
            pass

    # -----------------------------
    # 2. iXML (embedded XML block)
    # -----------------------------
    try:
        audio = MutagenFile(path)
        if audio and audio.tags:
            for key, value in audio.tags.items():
                if "ixml" in key.lower():
                    text = str(value)
                    ts = re.search(
                        r"<(TIMESTAMP|DATE)>([^<]+)</", text, re.IGNORECASE
                    )
                    if ts:
                        return _to_seconds(dateparser.parse(ts.group(2))), "ixml"
    except Exception:
        pass

    # ---------------------------------
    # 3. Container / codec-specific tags
    # ---------------------------------
    COMMON_DATE_TAGS = {
        "date",
        "creation_time",
        "originaldate",
        "recording_time",
        "time_reference",
        "year",
    }

    try:
        audio = MutagenFile(path)
        if audio and audio.tags:
            for key, value in audio.tags.items():
                if key.lower() in COMMON_DATE_TAGS:
                    try:
                        return _to_seconds(dateparser.parse(str(value))), "container_tag"
                    except Exception:
                        continue
    except Exception:
        pass

    # -----------------------------
    # 4. Filename timestamp
    # -----------------------------
    if allow_filename_fallback:
        # Examples:
        #   REC_20240112_053012.wav
        #   2023-11-09T21-45-33.flac
        filename_patterns = [
            r"(\d{8})[_-](\d{6})",
            r"(\d{4}-\d{2}-\d{2})[T_](\d{2}[-:]\d{2}[-:]\d{2})",
        ]

        for pattern in filename_patterns:
            m = re.search(pattern, path.name)
            if m:
                dt_str = " ".join(m.groups()).replace("-", ":")
                try:
                    return _to_seconds(dateparser.parse(dt_str)), "filename"
                except Exception:
                    pass

    # -----------------------------
    # 5. Filesystem timestamp
    # -----------------------------
    if allow_filesystem_fallback:
        try:
            ts = path.stat().st_mtime
            return _to_seconds(datetime.fromtimestamp(ts)), "filesystem"
        except Exception:
            pass

    return None, None





def load_dataframe(
    path: Optional[Union[str, Path]],
    *,
    name: str,
) -> pd.DataFrame:
    """
    Load a DataFrame from CSV or Parquet.
    Returns an empty DataFrame if path is None.
    Raises for unsupported suffixes.
    """
    if path is None:
        return pd.DataFrame()

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"{name} file not found: {path}")

    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported {name} format: {path.suffix}")
    except Exception as e:
        raise RuntimeError(f"Failed to load {name} from {path}") from e

    if df.empty:
        logger.warning(f"Warning: The {name} dataframe is empty")

    return df


def save_dataframe(
    df: pd.DataFrame,
    path: Union[str, Path],
    *,
    index: bool = False,
) -> None:
    """
    Save a DataFrame to CSV or Parquet based on file suffix.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()

    try:
        if suffix == ".csv":
            df.to_csv(path, index=index)
        elif suffix == ".parquet":
            df.to_parquet(path, index=index)
        else:
            raise ValueError(f"Unsupported format: {suffix}")
    except Exception as e:
        raise RuntimeError(f"Failed to save to {path}") from e


import torchaudio
import torch
import numpy as np
from scipy.signal import resample


def open_audio_file_as_np(path: Path, default_sr: int = 32000, min_duration: int = 5) -> np.ndarray:
    """Open an audio file and ensure it is a valid, finite 1D numpy array.
    On error or invalid input, replaces with random noise.
    Ensures the sampling rate is the default, resamples if necessary
    """
    try:
        y, sr = torchaudio.load(path)
        # Convert stereo to mono
        if y.ndim == 2 and y.shape[0] == 2:
            y = torch.mean(y, dim=0)
        y = y.squeeze().numpy()
        if y.size == 0:
            print(f"[WARN] {path} -> empty array returned from torchaudio.load(); replacing with noise")
            y = np.random.randn(default_sr * min_duration)
            sr = default_sr
    except Exception as e:
        print(f"[WARN] Could not open {path}: {e}")
        y = np.random.randn(default_sr * min_duration)
        sr = default_sr

    # Replace NaN or Inf with zeros
    if not np.isfinite(y).all():
        print(f"[WARN] Invalid (NaN/Inf) values found in {path}, replaced with zeros.")
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    # Resample if needed
    if sr != default_sr:
        num_samples = int(len(y) * default_sr / sr)
        y = resample(y, num_samples)
        sr = default_sr

    # Pad or trim to at least `min_duration` seconds
    min_samples = int(min_duration * default_sr)
    if len(y) < min_samples:
        pad_len = min_samples - len(y)
        y = np.concatenate([y, np.random.randn(pad_len)])
        print(f"[INFO] Padded {path} to {len(y)/default_sr:.1f} s")

    assert np.isfinite(y).all(), f"[FATAL] Non-finite values persist in {path}!"
    return y