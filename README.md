# StockPulse

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-Frontend-black.svg)](https://nextjs.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

StockPulse is a stock anomaly detection platform with a **FastAPI backend** and a **Next.js 15 frontend**.  
It ingests stock data, detects anomalies using machine learning, and provides a clean UI to explore results.

---

## 🚀 Project Structure

```

StockPulse/
│── backend/               # FastAPI backend
│── data/                  # CSV datasets (e.g., microsoft_stocks.csv)
│── docs/                  # Documentation, SRS, Demo & Database Schema
│── frontend/              # Next.js frontend
│── models/                # Trained anomaly detection models
│── UI/                    # Screenshots of the interface

````

---

## 🖥️ Backend (FastAPI)

### Prerequisites
- Python **3.10+**
- Virtual environment (`venv` or similar)

### Setup
```bash
cd backend

# Create & activate virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
````

### Running the Backend

```bash
uvicorn app:app --reload --port 8000
```

* API will be available at: [http://localhost:8000](http://localhost:8000)
* Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🌐 Frontend (Next.js 15 + TypeScript)

### Prerequisites

* Node.js **20+**
* npm or yarn

### Setup

```bash
cd frontend

# Install dependencies
npm install
```

### Running the Frontend

```bash
npm run dev
```

* Frontend will be available at: [http://localhost:3000](http://localhost:3000)

---

## 🔗 Connecting Frontend & Backend

* The frontend fetches data from the FastAPI backend (`http://localhost:8000`).
* Ensure both servers are running before testing.
* If deploying, update the API base URL in frontend config (`.env.local`).

---

## 📊 Features

### Backend

* Data ingestion & preprocessing
* Anomaly detection custom models
* REST API for anomalies, reports, and stats

### Frontend

* Dashboard with charts and tables
* Interactive anomaly exploration
* Reports Management
* Filters, severity levels, and pagination

---

## 🧪 Development Notes

* Use **`.env` files** in both backend & frontend for environment variables.
* Backend requires clean CSVs (see `data/`).
* Frontend uses **shadcn/ui** for components and **Recharts** for charts.

---

## 📦 Deployment

* **Backend**: Deploy with Docker, Uvicorn, or on platforms like Render/Heroku.
* **Frontend**: Deploy on Netlify, Vercel, or similar.

---

## 📚 Documentation

* 📄 [StockPulse SRS](./docs/StockPulse%20SRS.pdf) — Software Requirements Specification
* 📄 [StockPulse Plan](./docs/StockPulse.pdf) — Implementation roadmap

---
