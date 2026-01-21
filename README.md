# Churn ML API (FastAPI + Docker)

End-to-end ML project:
- train a churn model
- save model artifact
- serve predictions via FastAPI
- tests + Docker
- deploy to Render or AWS EC2

## Quickstart (local)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# train model (creates models/model.joblib)
python -m training.train

# run API
uvicorn app.main:app --reload
