# AI-Based Rockfall Prediction and Alert System

Production-style SIH machine-learning pipeline for predicting binary `Rockfall_Risk` (`0` safe, `1` rockfall) from historical environmental and geological data.

## Installation

Use Python 3.11+ and install dependencies:

```bash
cd rockfall_ai
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Train and evaluate

```bash
python -m src.train
```

Training discovers CSV/XLSX files in `config.json:data_dir`, cleans and merges them, creates EDA, compares Random Forest/XGBoost/CatBoost with 5-fold stratified CV and randomized tuning, and stores the best model by ROC-AUC then F1.

## Inference

```bash
python -m src.predict path\\to\\new_records.csv --output outputs\\predictions.csv
```

## FastAPI server

After training, start the prediction service from `rockfall_ai/`:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation. Check readiness with `GET /health`; submit one or more records to `POST /predict`:

```json
{
  "alert_threshold": 0.5,
  "records": [{"latitude": 32.56, "longitude": 107.45, "landslide_trigger": "rain"}]
}
```

## Dataset assumptions

- `Rockfall_Risk` is used automatically when supplied. For NASA's catalog, `landslide_category == rock_fall` maps to `1`; all other categories map to `0`.
- Datasets merge on available Date, Latitude, Longitude, and Mine ID keys. If no key overlap exists, weather is not merged to avoid invented links. Geology without keys is represented only by profile summary placeholders.
- DEM and earthquake inputs are optional; columns from future datasets are discovered and included without changing the core pipeline.

## Structure

```text
rockfall_ai/
├── data/                 # optional project-local datasets
├── notebooks/            # experimentation only
├── src/                  # reusable pipeline modules
├── models/               # serialized model artifacts
├── outputs/plots/        # tables and visual outputs
├── reports/              # generated training report
├── config.json
├── requirements.txt
├── README.md
└── WORK.md
```

## Future work

Add FastAPI endpoints, alert thresholds, a React dashboard, live weather feeds, tests, monitoring, and deployment automation.
