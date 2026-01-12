# Med-VQA on VQA-RAD

Compare:
- Baseline: CNN–BiLSTM (late fusion)
- Advanced: Multimodal Transformer fusion (cross-modal attention)

## Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Data
Put your parquet files in `data/`:
- data/train-*.parquet
- data/test-*.parquet

## Run
python3 -m src.train_baseline
python3 -m src.train_transformer
python3 -m src.eval --model baseline
python3 -m src.eval --model transformer
