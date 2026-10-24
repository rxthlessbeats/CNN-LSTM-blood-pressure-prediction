"""MIMIC record fetch, 5-second windows, and subject splits."""

import pickle
import re
import sys
from functools import partial
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np


def _repo_root():
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (
            parent / "src" / "cnn_lstm_bp"
        ).is_dir():
            return parent
    return Path.cwd()


ROOT = _repo_root()
DATA = ROOT / "data"
MODEL_DIR = ROOT / "model"
PARQUET = DATA / "final_records_mimic3_all_with_id.parquet"
MATCHING = DATA / "matching_records_mimic3.csv"
INDEX = DATA / "mimic3_records.pkl"
DATABASE = "mimic3wdb/1.0"
REQUIRED = ("II", "ABP", "PLETH")
SAMPLES = 625  # 125 Hz x 5 s; identity when the record is already 125 Hz


def resample_to(signal, n):
    signal = np.asarray(signal, dtype=np.float32)
    if len(signal) == n:
        return signal.copy()
    if len(signal) < 2:
        raise ValueError("segment shorter than 2 samples")
    grid = np.linspace(0.0, 1.0, n)
    src = np.linspace(0.0, 1.0, len(signal))
    return np.interp(grid, src, signal.astype(np.float64)).astype(np.float32)


def is_numeric_segment(name):
    return bool(re.fullmatch(r"[\d_]+", str(name)))


def _split_record(record):
    text = str(record).replace("\\", "/").rstrip("/")
    parent, _, name = text.rpartition("/")
    return parent, name


def _sid(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _load_pickle(path):
    # WindowsPath cannot be constructed on Linux; keep the parts as a pure path.
    class _Paths(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "pathlib" and name == "WindowsPath":
                return PureWindowsPath
            if module == "pathlib" and name == "PosixPath":
                return PurePosixPath
            return super().find_class(module, name)

    with open(path, "rb") as handle:
        return _Paths(handle).load()


def _parallel(fn, items, jobs):
    if jobs == 1:
        return [fn(item) for item in items]
    from joblib import Parallel, delayed

    return Parallel(n_jobs=jobs)(delayed(fn)(item) for item in items)


def _match_record(record, database_name=DATABASE, required=REQUIRED, min_seconds=5):
    import wfdb

    parent, name = _split_record(record)
    if not re.fullmatch(r"\d+", name):
        return []
    record_dir = f"{database_name}/{parent}".replace("\\", "/")
    try:
        header = wfdb.rdheader(record_name=name, pn_dir=record_dir, rd_segments=True)
    except Exception as exc:  # noqa: BLE001  # a bad header should skip the record
        print(f"skip {record}: {exc}", file=sys.stderr)
        return []
    present = list(header.sig_name or [])
    if not all(sig in present for sig in required):
        return []
    hits = []
    for segment_name in header.seg_name or []:
        if not is_numeric_segment(segment_name):
            continue
        try:
            segment = wfdb.rdheader(record_name=str(segment_name), pn_dir=record_dir)
        except Exception as exc:  # noqa: BLE001  # a bad segment should be skipped
            print(f"skip {segment_name}: {exc}", file=sys.stderr)
            continue
        if not segment.fs:
            continue
        duration = segment.sig_len / segment.fs
        if duration < min_seconds:
            continue
        names = list(segment.sig_name or [])
        if all(sig in names for sig in required):
            hits.append(
                {
                    "subject_id": name,
                    "seg_name": str(segment_name),
                    "path": f"{parent}/{segment_name}",
                    "length": duration,
                }
            )
    return hits


def _windows_from_row(row, database_name=DATABASE, split_sec=5, samples=SAMPLES):
    from cnn_lstm_bp import processing

    processing.database_name = database_name
    try:
        segments, fs = processing.split_segment(str(row["path"]), split_sec=split_sec)
        kept = processing.signal_selection_algorithm(segments, fs)
    except Exception as exc:  # noqa: BLE001  # a bad window should be skipped
        print(f"skip {row.get('seg_name')}: {exc}", file=sys.stderr)
        return []
    subject = _sid(row["subject_id"])
    windows = []
    for i, rec in enumerate(kept):
        windows.append(
            {
                "subject_id": subject,
                "seg_name": f"{row['seg_name']}_{i}",
                "ecg": resample_to(rec["ECG"], samples).tolist(),
                "ppg": resample_to(rec["PPG"], samples).tolist(),
                "true_sbp": float(rec["true_sbp"]),
                "true_dbp": float(rec["true_dbp"]),
            }
        )
    return windows


def fetch_records(database_name, limit=0):
    import wfdb

    try:
        subjects = wfdb.get_record_list(database_name)
    except Exception as exc:
        raise SystemExit(
            f"{exc}\n\nMIMIC-III Waveform is credentialed. Run `wfdb login`, then "
            "`cnn-lstm-bp prepare --fetch`."
        ) from exc
    records = []
    for subject in subjects:
        studies = wfdb.get_record_list(f"{database_name}/{subject}")
        for study in studies:
            records.append(f"{subject}{study}".replace("\\", "/"))
            if limit and len(records) >= limit:
                return records
    return records


def prepare(
    index=INDEX,
    out=MATCHING,
    database_name=DATABASE,
    fetch=False,
    limit=0,
    jobs=1,
    min_seconds=5,
):
    import pandas as pd

    index, out = Path(index), Path(out)
    if fetch or not index.exists():
        if not fetch and not index.exists():
            raise SystemExit(f"missing {index}. Run: cnn-lstm-bp prepare --fetch")
        records = fetch_records(database_name, limit=0)
        index.parent.mkdir(parents=True, exist_ok=True)
        with open(index, "wb") as handle:
            pickle.dump(records, handle)
        print(f"wrote {index} ({len(records)} records)")
    else:
        raw = _load_pickle(index)
        records = [str(r).replace("\\", "/") for r in raw]
    if limit:
        records = records[:limit]
    print(f"scanning {len(records)} records")
    grouped = _parallel(
        partial(
            _match_record,
            database_name=database_name,
            required=REQUIRED,
            min_seconds=min_seconds,
        ),
        records,
        jobs,
    )
    rows = [row for group in grouped for row in group]
    if not rows:
        raise SystemExit("no records had II, ABP, and PLETH long enough to keep")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} segments)")
    return out


def build(
    matching=MATCHING,
    out=PARQUET,
    database_name=DATABASE,
    limit=0,
    jobs=1,
    samples=SAMPLES,
    flush_every=2000,
):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    matching, out = Path(matching), Path(out)
    if not matching.exists():
        raise SystemExit(f"missing {matching}. Run prepare first.")
    frame = pd.read_csv(matching)
    if limit:
        frame = frame.head(limit)
    rows = frame.to_dict(orient="records")
    print(f"windowing {len(rows)} segments")
    grouped = _parallel(
        partial(
            _windows_from_row, database_name=database_name, split_sec=5, samples=samples
        ),
        rows,
        jobs,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    writer = None
    buf = []
    kept = 0

    def flush():
        nonlocal writer, buf
        if not buf:
            return
        table = pa.Table.from_pandas(pd.DataFrame(buf), preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out, table.schema)
        writer.write_table(table)
        buf = []

    try:
        for group in grouped:
            buf.extend(group)
            kept += len(group)
            if len(buf) >= flush_every:
                flush()
        flush()
    finally:
        if writer is not None:
            writer.close()
    if kept == 0:
        raise SystemExit("quality filter kept zero windows")
    print(f"wrote {out} ({kept} windows, {samples} samples)")
    return out


def load_parquet(path):
    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise SystemExit(f"missing {path}. Run build first, or pass --data.")
    frame = pd.read_parquet(path)
    missing = {"ecg", "ppg", "true_sbp", "true_dbp"} - set(frame.columns)
    if missing:
        raise SystemExit(f"{path} missing columns: {sorted(missing)}")
    if "subject_id" not in frame.columns:
        if "seg_name" not in frame.columns:
            raise SystemExit(f"{path} has no subject_id or seg_name")
        frame["subject_id"] = frame["seg_name"].map(lambda s: str(s).split("_")[0])
    try:
        ecg = np.stack([np.asarray(v, dtype=np.float32) for v in frame["ecg"]])
        ppg = np.stack([np.asarray(v, dtype=np.float32) for v in frame["ppg"]])
    except ValueError as exc:
        raise SystemExit(
            f"windows differ in length; rebuild with cnn-lstm-bp build ({exc})"
        ) from exc
    signals = np.stack((ecg, ppg), axis=-1)
    subjects = np.array([_sid(v) for v in frame["subject_id"]])
    return (
        signals,
        frame["true_sbp"].to_numpy(np.float32),
        frame["true_dbp"].to_numpy(np.float32),
        subjects,
    )


def subject_split(subject_ids, seed):
    from sklearn.model_selection import train_test_split

    ids = np.unique(np.asarray(subject_ids).astype(str))
    if len(ids) < 5:
        raise SystemExit(
            f"need at least 5 subjects for an 80/10/10 split, found {len(ids)}"
        )
    train_ids, temp_ids = train_test_split(
        ids, test_size=0.2, random_state=seed, shuffle=True
    )
    if len(temp_ids) < 2:
        raise SystemExit(
            f"{len(ids)} subjects is not enough to hold out both val and test"
        )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.5, random_state=seed, shuffle=True
    )
    return (
        [str(x) for x in train_ids],
        [str(x) for x in val_ids],
        [str(x) for x in test_ids],
    )
