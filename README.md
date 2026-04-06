# Cavitator ML Design Tool

ML-based drag prediction and optimization for supercavitating torpedo cavitators.

## Fixed Body Geometry
```
Total length : 115 mm
Cylinder     : ∅30 mm × 85 mm
Hemisphere   : ∅15 mm (nose)
Cavitator    : ∅12 mm (varies in SHAPE only, not size)
```

## Cavitator Shapes
| Shape | Parameter 1 | Parameter 2 |
|---|---|---|
| Disc | — | — |
| Cone | Half-angle θ [20–80°] | — |
| Truncated cone | Half-angle θ [30–75°] | Truncation ratio [0.1–0.7] |
| Spherical | R/dc ratio [0.6–2.5] | — |
| Elliptical | Axis ratio a/b [0.5–3.0] | — |
| Ogive polynomial | Exponent n [1–4] | — |

## Project Structure
```
cavitator_app/
├── data/
│   ├── generate_dataset.py       # physics dataset generator
│   └── cavitator_dataset_1000.csv
├── models/
│   ├── train_models.py           # model training
│   ├── *.pkl                     # saved models
│   └── metadata.json
├── api/
│   └── main.py                   # FastAPI backend
├── app/
│   └── streamlit_app.py          # Streamlit frontend
└── README.md
```

## Setup & Run

### 1. Install dependencies
```bash
pip install xgboost scikit-learn shap scipy numpy pandas \
            fastapi uvicorn streamlit joblib matplotlib seaborn
```

### 2. Generate dataset
```bash
cd cavitator_app
python data/generate_dataset.py
```

### 3. Train models
```bash
python models/train_models.py
```

### 4. Run FastAPI backend
```bash
uvicorn api.main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

### 5. Run Streamlit frontend
```bash
streamlit run app/streamlit_app.py
# Opens: http://localhost:8501
```

## API Endpoints
| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/shapes` | Available shapes + parameters |
| GET | `/model-info` | Model metrics + feature importance |
| POST | `/predict` | Predict Cd, Csf, Cdp, L̃, D̃ |
| POST | `/optimize` | Find optimal cavitator shape |
| GET | `/batch-compare` | Compare all shapes at given conditions |

## Example API call
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"shape_name":"cone","param1":45,"param2":0,"sigma_c":0.10,"Re":1e6,"depth_m":10}'
```

## Physics Basis
- **Cdp** = Cdp0(θ) × (1 + σ_c)  — Reichardt (1946)
- **Cdp0** anchored at disc=0.827 — Rouse & McNown (1948)
- **L̃** = Cdp0/σ_c — Logvinovich (1980)
- **D̃** = √(Cdp0/σ_c) — Garabedian (1956)
- **Csf** = Prandtl turbulent BL × wetted fraction

## Model Performance
| Target | RF R² | XGB R² |
|---|---|---|
| Cdp | 0.997 | 0.997 |
| Csf | 0.976 | 0.976 |
| Cd  | 0.988 | 0.990 |
| L̃   | 0.988 | 0.989 |
| D̃   | 0.988 | 0.987 |
