import argparse
import json
import sys
from pathlib import Path

import numpy as np

from cnn_lstm_bp.data import (
    DATABASE,
    INDEX,
    MATCHING,
    MODEL_DIR,
    PARQUET,
    SAMPLES,
    build,
    is_numeric_segment,
    load_parquet,
    prepare,
    resample_to,
    subject_split,
)

MODEL_FOR = {"sbp": "BP_Model_DO", "dbp": "BP_Model"}


def _write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def _take(mask, signals, y_sbp, y_dbp):
    return signals[mask], y_sbp[mask], y_dbp[mask]


def _loader(x, y, batch_size, shuffle):
    import torch
    from torch.utils.data import DataLoader, Dataset

    class _Set(Dataset):
        def __init__(self, x, y):
            self.x = torch.as_tensor(x, dtype=torch.float32)
            self.y = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)

        def __len__(self):
            return self.x.shape[0]

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

    return DataLoader(_Set(x, y), batch_size=batch_size, shuffle=shuffle)


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _models():
    from cnn_lstm_bp.models import BP_Model, BP_Model_DO

    return {"BP_Model": BP_Model, "BP_Model_DO": BP_Model_DO}


def _set_seed(seed):
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metrics(row):
    keys = ("mae", "rmse", "std", "r2", "bhs5", "bhs10", "bhs15", "grade")
    return {k: v if isinstance(v, str) else float(v) for k, v in zip(keys, row)}


def train(
    data=PARQUET,
    model_dir=MODEL_DIR,
    epochs=1000,
    batch_size=128,
    patience=50,
    lr=0.01,
    seed=38,
):
    import torch

    from cnn_lstm_bp.evaluation import evaluate_model_performance
    from cnn_lstm_bp.models import (
        EarlyStopping,
        evaluate_torch_model,
        train_torch_model,
    )

    signals, y_sbp, y_dbp, subjects = load_parquet(data)
    train_ids, val_ids, test_ids = subject_split(subjects, seed)
    parts = []
    for chosen in (train_ids, val_ids, test_ids):
        mask = np.isin(subjects, chosen)
        parts.append(_take(mask, signals, y_sbp, y_dbp))
    if any(len(part[0]) == 0 for part in parts):
        raise SystemExit("subject split produced an empty set")
    print(
        f"windows train/val/test: {len(parts[0][0])}/{len(parts[1][0])}/{len(parts[2][0])}"
    )
    _set_seed(seed)
    device = _device()
    print(f"device: {device}")
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    classes = _models()
    run = {
        "data": str(data),
        "seed": seed,
        "samples": int(signals.shape[1]),
        "subjects": {"train": train_ids, "val": val_ids, "test": test_ids},
        "models": {},
    }
    targets = {
        "sbp": [part[1] for part in parts],
        "dbp": [part[2] for part in parts],
    }
    xs = [part[0] for part in parts]
    for name, labels in targets.items():
        cls_name = MODEL_FOR[name]
        model = classes[cls_name]().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.8, patience=patience
        )
        stopper = EarlyStopping(patience=patience * 3, min_delta=0)
        # ponytail: shuffle train batches; the notebook left every loader unshuffled
        train_loader = _loader(xs[0], labels[0], batch_size, True)
        val_loader = _loader(xs[1], labels[1], batch_size, False)
        test_loader = _loader(xs[2], labels[2], batch_size, False)
        ckpt = model_dir / f"torch_cnn_lstm_{name}.pth"
        train_loss, val_loss, _, _, epochs_ran = train_torch_model(
            model,
            train_loader,
            val_loader,
            epochs,
            torch.nn.L1Loss(),
            optimizer,
            device,
            name,
            scheduler,
            stopper,
            ckpt_path=str(ckpt),
        )
        pred, true = evaluate_torch_model(model, test_loader, device)
        scores = _metrics(evaluate_model_performance(true, pred, label=name.upper()))
        run["models"][name] = {
            "class": cls_name,
            "checkpoint": str(ckpt),
            "epochs": int(epochs_ran),
            "train_loss": [float(v) for v in train_loss],
            "val_loss": [float(v) for v in val_loss],
            "test": scores,
        }
    _write_json(model_dir / "run.json", run)
    print(f"wrote {model_dir / 'run.json'}")
    return run


def evaluate(data=PARQUET, model_dir=MODEL_DIR, batch_size=128):
    from cnn_lstm_bp.evaluation import evaluate_model_performance
    from cnn_lstm_bp.models import evaluate_torch_model, load_checkpoint

    model_dir = Path(model_dir)
    run_path = model_dir / "run.json"
    if not run_path.exists():
        raise SystemExit(f"missing {run_path}. Run train first.")
    run = json.loads(run_path.read_text())
    signals, y_sbp, y_dbp, subjects = load_parquet(data)
    test_ids = run["subjects"]["test"]
    mask = np.isin(subjects, test_ids)
    if not mask.any():
        raise SystemExit("none of the saved test subjects are in this data file")
    x = signals[mask]
    labels = {"sbp": y_sbp[mask], "dbp": y_dbp[mask]}
    device = _device()
    classes = _models()
    results = {}
    for name, y in labels.items():
        spec = run["models"][name]
        model = classes[spec["class"]]().to(device)
        model.load_state_dict(load_checkpoint(spec["checkpoint"], device))
        pred, true = evaluate_torch_model(
            model, _loader(x, y, batch_size, False), device
        )
        results[name] = _metrics(
            evaluate_model_performance(true, pred, label=name.upper())
        )
    return results


def predict(window, target, checkpoint):
    import torch

    from cnn_lstm_bp.models import load_checkpoint

    window = np.asarray(np.load(window), dtype=np.float32)
    if window.ndim == 2:
        window = window[None, ...]
    if window.ndim != 3 or window.shape[-1] != 2 or window.shape[1] < 32:
        raise SystemExit(
            "window must be (T, 2) or (N, T, 2) with T>=32 and columns [ECG, PPG]"
        )
    if not Path(checkpoint).exists():
        raise SystemExit(f"missing {checkpoint}. Run train first.")
    device = _device()
    model = _models()[MODEL_FOR[target]]().to(device)
    model.load_state_dict(load_checkpoint(checkpoint, device))
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(window).to(device)).cpu().numpy().ravel()
    for value in pred:
        print(f"{value:.4f}")
    return pred


def _synthetic(path, n_subjects=10, windows=2, samples=64, seed=0):
    import pandas as pd

    rng = np.random.default_rng(seed)
    rows = []
    t = np.linspace(0, 1, samples, dtype=np.float32)
    for s in range(n_subjects):
        for k in range(windows):
            phase = 0.15 * k
            tone = np.sin(2 * np.pi * (2 + s) * t + phase).astype(np.float32)
            rows.append(
                {
                    "subject_id": f"s{s}",
                    "seg_name": f"s{s}_{k}",
                    "ecg": (tone + 0.01 * rng.normal(size=samples))
                    .astype(np.float32)
                    .tolist(),
                    "ppg": (np.roll(tone, 4) + 0.01 * rng.normal(size=samples))
                    .astype(np.float32)
                    .tolist(),
                    "true_sbp": float(110 + 3 * s),
                    "true_dbp": float(70 + s),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def demo(epochs=1):
    import tempfile

    from cnn_lstm_bp.evaluation import assign_bhs_grade

    assert is_numeric_segment("3000063_0006")
    assert not is_numeric_segment("3000063_layout")
    assert not is_numeric_segment("~")
    original = np.arange(SAMPLES, dtype=np.float32)
    assert np.array_equal(resample_to(original, SAMPLES), original)
    assert (
        assign_bhs_grade({"within_5mmHg": 60, "within_10mmHg": 85, "within_15mmHg": 95})
        == "A"
    )
    assert (
        assign_bhs_grade({"within_5mmHg": 0, "within_10mmHg": 0, "within_15mmHg": 0})
        == "D"
    )
    train_ids, val_ids, test_ids = subject_split([f"s{i}" for i in range(10)], 38)
    assert set(train_ids).isdisjoint(val_ids) and set(train_ids).isdisjoint(test_ids)
    assert set(val_ids).isdisjoint(test_ids)
    assert len(train_ids) + len(val_ids) + len(test_ids) == 10

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data = tmp / "windows.parquet"
        _synthetic(data)
        run = train(
            data,
            tmp / "model",
            epochs=epochs,
            batch_size=8,
            patience=50,
            lr=1e-3,
            seed=38,
        )
        scores = evaluate(data, tmp / "model", batch_size=8)
        for name in ("sbp", "dbp"):
            assert np.isfinite(scores[name]["mae"])
            assert scores[name]["grade"] in "ABCD"
            assert Path(run["models"][name]["checkpoint"]).exists()
        signals, _, _, _ = load_parquet(data)
        window = tmp / "window.npy"
        np.save(window, signals[0])
        pred = predict(window, "sbp", run["models"]["sbp"]["checkpoint"])
        assert pred.shape == (1,) and np.isfinite(pred).all()
    print("demo ok")


def _add_train_args(parser):
    parser.add_argument("--data", type=Path, default=PARQUET)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=38)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="MIMIC ECG/PPG blood-pressure pipeline"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare", help="find II+ABP+PLETH records")
    prep.add_argument("--index", type=Path, default=INDEX)
    prep.add_argument("--out", type=Path, default=MATCHING)
    prep.add_argument("--database", default=DATABASE)
    prep.add_argument("--fetch", action="store_true")
    prep.add_argument("--limit", type=int, default=0)
    prep.add_argument("--jobs", type=int, default=1)
    prep.add_argument("--min-seconds", type=int, default=5)

    bld = sub.add_parser("build", help="window and quality-filter into parquet")
    bld.add_argument("--matching", type=Path, default=MATCHING)
    bld.add_argument("--out", type=Path, default=PARQUET)
    bld.add_argument("--database", default=DATABASE)
    bld.add_argument("--limit", type=int, default=0)
    bld.add_argument("--jobs", type=int, default=1)
    bld.add_argument("--samples", type=int, default=SAMPLES)

    tr = sub.add_parser("train", help="fit SBP and DBP models")
    _add_train_args(tr)

    ev = sub.add_parser("evaluate", help="score the saved test subjects")
    ev.add_argument("--data", type=Path, default=PARQUET)
    ev.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    ev.add_argument("--batch-size", type=int, default=128)

    pr = sub.add_parser("predict", help="run one checkpoint on a window")
    pr.add_argument("--target", choices=("sbp", "dbp"), required=True)
    pr.add_argument("--window", type=Path, required=True)
    pr.add_argument("--checkpoint", type=Path)

    run = sub.add_parser("run", help="train, then evaluate")
    _add_train_args(run)

    shown = sub.add_parser("demo", help="synthetic smoke test, no PhysioNet")
    shown.add_argument("--epochs", type=int, default=1)

    args = parser.parse_args(argv)
    if args.cmd == "prepare":
        prepare(
            args.index,
            args.out,
            args.database,
            args.fetch,
            args.limit,
            args.jobs,
            args.min_seconds,
        )
    elif args.cmd == "build":
        build(
            args.matching, args.out, args.database, args.limit, args.jobs, args.samples
        )
    elif args.cmd == "train":
        train(
            args.data,
            args.model_dir,
            args.epochs,
            args.batch_size,
            args.patience,
            args.lr,
            args.seed,
        )
    elif args.cmd == "evaluate":
        evaluate(args.data, args.model_dir, args.batch_size)
    elif args.cmd == "predict":
        ckpt = args.checkpoint or (MODEL_DIR / f"torch_cnn_lstm_{args.target}.pth")
        predict(args.window, args.target, ckpt)
    elif args.cmd == "run":
        train(
            args.data,
            args.model_dir,
            args.epochs,
            args.batch_size,
            args.patience,
            args.lr,
            args.seed,
        )
        evaluate(args.data, args.model_dir, args.batch_size)
    elif args.cmd == "demo":
        demo(args.epochs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
