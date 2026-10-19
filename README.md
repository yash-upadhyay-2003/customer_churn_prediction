# Customer Churn Prediction & Churn Observatory

A modular, production-ready Machine Learning system comprising a FastAPI backend and a bespoke, editorial Streamlit frontend (**Churn Observatory**) powered by a trained scikit-learn Decision Tree Pipeline.

---

## 1. Architecture & Design

The project is structured with strict separation of concerns, providing both a high-throughput REST API and a distinctive B2B SaaS intelligence frontend:

```
customer_churn_prediction/
│
├── app/
│   ├── __init__.py               # Package marker and version
│   ├── main.py                   # FastAPI application initialization & route definitions
│   ├── schemas.py                # Pydantic schemas for strict validation & OpenAPI documentation
│   ├── config.py                 # Centralized configuration and path management
│   └── services/
│       ├── __init__.py           # Service exports
│       └── prediction_service.py # Singleton ML lifecycle manager and inference logic
│
├── saved_model/
│   └── churn_model.pkl           # Trained scikit-learn Pipeline (18 KB)
│
├── tests/
│   ├── __init__.py
│   └── test_api.py               # Automated pytest suite (11 test cases)
│
├── streamlit_app.py              # Churn Observatory frontend (editorial B2B SaaS UI)
├── .streamlit/
│   ├── config.toml               # Streamlit theme configuration (white & burgundy palette)
│   ├── secrets.toml              # Local development secrets (ignored by git)
│   └── secrets.toml.example      # Example secrets template
│
├── requirements.txt              # Production, API, and UI dependencies
├── .gitignore                    # Git rules (preserves model file, ignores secrets)
└── README.md                     # Documentation and usage guide
```

---

## 2. Machine Learning Pipeline Details

The trained pipeline (`saved_model/churn_model.pkl`) was inspected directly:

- **Pipeline Steps**:
  1. `preprocessor`: `ColumnTransformer`
     - **Numerical features (`num`)**: Median `SimpleImputer` followed by `StandardScaler` on:
       - `BaseFee`
       - `AdditionalFeatureSpend`
       - `NumberOfPenalties`
       - `CurrentPeriodUsage`
       - `PriorPeriodUsage`
       - `ServiceRating`
       - `ServiceTickets`
       - `CustomerTenure`
     - **Categorical features (`cat`)**: Most frequent `SimpleImputer` followed by `OneHotEncoder(handle_unknown='ignore')` on:
       - `Territory` (50 US States + DC)
       - `DeviceType` (`'-'`, `'Galaxy'`, `'iPhone'`)
       - `Promotion` (`'yes'`, `'no'`)
       - `HasRenewed` (`'yes'`, `'no'`)
       - `PlanType` (`'Blue Plan'`, `'Green Plan'`, `'Purple Plan'`, `'Red Plan'`)
       - `StartMonth` (`'Apr'`, `'Aug'`, `'Dec'`, `'Feb'`, `'Jan'`, `'Jul'`, `'Jun'`, `'Mar'`, `'May'`, `'Nov'`, `'Oct'`, `'Sep'`)
       - `StartWeek` (`'W01'` through `'W53'`)
  2. `classifier`: `DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=42)`
- **Target Classes**: `0` = No Churn, `1` = Churn
- **Excluded Features**: `AccountID` and `Country` are excluded as per model training.
- **Preprocessing**: Imputing, standard scaling, and one-hot encoding are handled entirely within the saved scikit-learn pipeline. The backend and frontend accept raw customer values.

---

## 3. Local Setup (Windows / PowerShell)

### Step 1: Create a Virtual Environment

```powershell
python -m venv .venv
```

### Step 2: Activate the Virtual Environment in PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

*(If PowerShell script execution is restricted, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

### Step 3: Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

---

## 4. Running the Applications

### Option A: Start the FastAPI Backend

Run from the project root:

```powershell
python -m uvicorn app.main:app --port 8081 --reload
```

- **API Root**: [http://127.0.0.1:8081/](http://127.0.0.1:8081/)
- **Health Check**: [http://127.0.0.1:8081/health](http://127.0.0.1:8081/health)
- **Model Metadata**: [http://127.0.0.1:8081/model-info](http://127.0.0.1:8081/model-info)
- **Interactive Swagger UI**: [http://127.0.0.1:8081/docs](http://127.0.0.1:8081/docs)

### Option B: Start the Streamlit Frontend ("Churn Observatory")

In a separate terminal (with virtual environment activated):

```powershell
python -m streamlit run streamlit_app.py
```

- **Frontend URL**: [http://localhost:8501](http://localhost:8501)

### Configuring `API_BASE_URL` for Streamlit

The Streamlit app checks for `API_BASE_URL` in the following priority order:
1. **Streamlit Secrets**: `.streamlit/secrets.toml`
   ```toml
   API_BASE_URL = "http://127.0.0.1:8081"
   ```
2. **Environment Variable**: `API_BASE_URL`
   ```powershell
   $env:API_BASE_URL="http://127.0.0.1:8081"
   ```
3. **Default Fallback**: `http://127.0.0.1:8081`

---

## 5. Churn Observatory Design Identity

- **Design Philosophy**: Editorial minimalism, quiet confidence, precise typography, and generous whitespace.
- **Strict Colour System**: 90–95% white (`#FFFFFF`) and near-white (`#FAF9F7`) with fine separators (`#EAE6E3`) and restrained burgundy accents (`#762F43`).
- **Features Grouping**:
  - `01 — Account & Plan`: Territory, DeviceType, PlanType, Promotion, HasRenewed
  - `02 — Usage & Spend`: BaseFee, AdditionalFeatureSpend, CurrentPeriodUsage, PriorPeriodUsage
  - `03 — Customer Experience`: NumberOfPenalties, ServiceRating, ServiceTickets, CustomerTenure, StartMonth, StartWeek
- **Prediction Workspace**:
  - Asymmetric desktop layout (~60% input form, ~40% prediction workspace).
  - Standby state when awaiting input.
  - Active prediction result with probability percentage, horizontal indicator bar, contextual interpretation, and model disclaimer.

---

## 6. How to Test via Swagger UI (Step-by-Step)

1. Open your browser to: **[http://127.0.0.1:8081/docs](http://127.0.0.1:8081/docs)**
2. In the **Prediction** section, click on **`POST /predict`**.
3. Click the **"Try it out"** button.
4. Paste the sample payload:

```json
{
  "Territory": "CA",
  "DeviceType": "iPhone",
  "Promotion": "no",
  "HasRenewed": "yes",
  "PlanType": "Purple Plan",
  "BaseFee": 50.0,
  "AdditionalFeatureSpend": 15.5,
  "NumberOfPenalties": 0,
  "CurrentPeriodUsage": 120.0,
  "PriorPeriodUsage": 115.0,
  "ServiceRating": 4.0,
  "ServiceTickets": 1,
  "StartMonth": "Jan",
  "StartWeek": "W02",
  "CustomerTenure": 24
}
```

5. Click **"Execute"** to view the response:
```json
{
  "prediction": "No Churn",
  "predicted_class": 0,
  "churn_probability": 0.0
}
```

---

## 7. Running the Automated Tests

Execute the automated test suite using `pytest`:

```powershell
python -m pytest tests/ -v
```

All 11 unit and integration tests validate:
- Root endpoint response
- Health check status
- Model information and schema
- Valid single prediction (No Churn)
- Valid single prediction (Churn)
- Missing required features (HTTP 422)
- Invalid types (HTTP 422)
- Negative value boundary checks (HTTP 422)
- Graceful handling of unknown categorical values
- Batch predictions
- Empty batch prevention (HTTP 422)
