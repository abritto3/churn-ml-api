# Churn Prediction ML API (FastAPI + Docker)

This is an end-to-end machine learning project that:
- trains a churn prediction model
- packages it into a REST API using FastAPI
- exposes live prediction endpoints
- is containerised with Docker
- deployed publicly on Render

---

## 🌐 Live Demo
- **Health check:** https://churn-ml-api-65bt.onrender.com/health
- **Interactive API docs:** https://churn-ml-api-65bt.onrender.com/docs

---

## 🧠 Problem Overview
Customer churn is a common business problem where companies want to identify users likely to stop using a service.

**Goal:**  
Predict the probability that a customer will churn based on account and billing features.

## Architecture
![Architecture Diagram](architecture.png)

**Inputs include:**
- tenure (months)
- monthly charges
- total charges
- contract type
- internet service
- payment method
- paperless billing

**Outputs:**
- churn probability (0–1)
- binary churn decision

---

## 🛠 Tech Stack
- **Language:** Python
- **API:** FastAPI, Uvicorn
- **ML:** scikit-learn (pipeline + model)
- **Data:** pandas, numpy
- **Validation:** Pydantic
- **Testing:** pytest
- **Containerisation:** Docker
- **Deployment:** Render

---

## 🔌 API Endpoints

### `GET /health`
Simple health check to confirm the service is running.

**Response**
```json
{"status": "ok"}
