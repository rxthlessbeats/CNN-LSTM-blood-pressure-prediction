# CNN-LSTM blood pressure prediction

Subject-wise SBP and DBP regression from 5-second MIMIC-III ECG (lead II) and PPG (PLETH). ABP is used only as the label. SBP uses `BP_Model_DO`. DBP uses `BP_Model`.

## Setup

```bash
uv sync
uv run pre-commit install
```

`torch` is the CPU wheel, so `uv sync` stays small. For a CUDA GPU, change the `pytorch-cpu` index URL in `pyproject.toml` to the matching `cuXXX` index and run `uv lock`. The Keras notebook needs the extra: `uv sync --extra keras`.

MIMIC-III Waveform (`mimic3wdb/1.0`) needs a credentialed PhysioNet account. Run `wfdb login` once before `prepare`. Waveforms are not in git (`data/` and `model/` are ignored).

## Layout

```
src/cnn_lstm_bp/     import package
  cli.py             cnn-lstm-bp command
  data.py            prepare, build, load
  processing.py      window quality filter
  models.py          BP_Model and BP_Model_DO
  evaluation.py      MAE and BHS grade
notebooks/
tests/
```

## Pipeline

```bash
uv run cnn-lstm-bp prepare --fetch
uv run cnn-lstm-bp build
uv run cnn-lstm-bp train
uv run cnn-lstm-bp evaluate
uv run cnn-lstm-bp predict --target sbp --window window.npy
uv run cnn-lstm-bp demo
uv run pytest
```

`prepare` writes `data/matching_records_mimic3.csv`. `build` writes `data/final_records_mimic3_all_with_id.parquet`: 5-second windows that pass the heart-rate, peak-ratio, and pulse-pressure checks, resampled to 625 samples (125 Hz). A record that is already 125 Hz is unchanged.

`train` holds out subjects 80/10/10 (`--seed 38`), then fits with Adam (`--lr 0.01`), L1 loss, batch size 128, and up to 1000 epochs. Training stops after 150 epochs without a better validation loss. Checkpoints are that best epoch:

- `model/torch_cnn_lstm_sbp.pth`
- `model/torch_cnn_lstm_dbp.pth`
- `model/run.json` (the subject split and test MAE / BHS grade)

`predict` reads a `.npy` array of shape `(T, 2)` or `(N, T, 2)`, columns `[ECG, PPG]`.

`demo` trains on synthetic windows and exits. It does not download MIMIC.

`--jobs` on `prepare` and `build` defaults to 1. Raise it when the header scan is the slow part.

## Notebooks

The original notebooks are in `notebooks/`. The first cell switches the working directory to the repo root so `data/` still resolves. Imports are `cnn_lstm_bp` (`from cnn_lstm_bp.models import ...`). `notebooks/train.ipynb` is the earlier Keras run. The pipeline follows `notebooks/train_torch.ipynb`.

`notebooks/cv.ipynb` scores saved weights on random folds. It does not retrain. Use `evaluate` for the subject-wise test set.
