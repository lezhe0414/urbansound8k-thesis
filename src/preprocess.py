from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def _load_audio(path: Path, sample_rate: int, duration: float) -> np.ndarray:
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - exercised only without deps
        raise RuntimeError("librosa is required. Install dependencies with `pip install -r requirements.txt`.") from exc

    target_samples = int(sample_rate * duration)
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))
    elif len(audio) > target_samples:
        audio = audio[:target_samples]
    return audio.astype(np.float32)


def _mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    import librosa

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mean = float(mel_db.mean())
    std = float(mel_db.std())
    if std < 1e-8:
        std = 1.0
    return ((mel_db - mean) / std).astype(np.float32)


def preprocess_urbansound8k(
    raw_dir: Path,
    out_dir: Path,
    sample_rate: int,
    duration: float,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    limit: int | None = None,
) -> None:
    metadata_path = raw_dir / "metadata" / "UrbanSound8K.csv"
    audio_dir = raw_dir / "audio"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing UrbanSound8K metadata file: {metadata_path}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"Missing UrbanSound8K audio directory: {audio_dir}")

    metadata = pd.read_csv(metadata_path)
    required = {"slice_file_name", "fold", "classID", "class"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"UrbanSound8K metadata is missing columns: {sorted(missing)}")
    if limit is not None:
        metadata = metadata.head(limit)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for row in tqdm(metadata.to_dict("records"), desc="Preprocessing UrbanSound8K"):
        fold = int(row["fold"])
        file_name = str(row["slice_file_name"])
        source_path = audio_dir / f"fold{fold}" / file_name
        if not source_path.exists():
            raise FileNotFoundError(f"Missing audio file referenced by metadata: {source_path}")

        audio = _load_audio(source_path, sample_rate=sample_rate, duration=duration)
        mel = _mel_spectrogram(audio, sample_rate=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)

        fold_dir = out_dir / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        output_path = fold_dir / f"{Path(file_name).stem}.npz"
        np.savez_compressed(
            output_path,
            mel=mel,
            classID=int(row["classID"]),
            class_name=str(row["class"]),
            fold=fold,
            slice_file_name=file_name,
        )
        rows.append(
            {
                "slice_file_name": file_name,
                "fold": fold,
                "classID": int(row["classID"]),
                "class": str(row["class"]),
                "path": str(output_path.relative_to(out_dir)),
            }
        )

    processed_metadata = pd.DataFrame(rows)
    processed_metadata.to_csv(out_dir / "metadata.csv", index=False)
    params = {
        "sample_rate": sample_rate,
        "duration": duration,
        "n_mels": n_mels,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "items": len(processed_metadata),
    }
    (out_dir / "preprocess_params.json").write_text(_json_dumps(params), encoding="utf-8")


def _json_dumps(payload: dict) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess UrbanSound8K audio into cached Mel-spectrograms.")
    parser.add_argument("--raw-dir", default="data/raw/UrbanSound8K", help="Path to the UrbanSound8K root directory.")
    parser.add_argument("--out-dir", default="data/processed/urbansound8k_mels", help="Output directory for .npz files.")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="Optional small-sample limit for smoke testing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    preprocess_urbansound8k(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        sample_rate=args.sample_rate,
        duration=args.duration,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
