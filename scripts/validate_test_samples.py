"""
Comprehensive Validation Script: Real Held-Out Test Set Evaluation
Tests diverse real customer samples against the running FastAPI prediction endpoint.
"""
import sys
import os
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Ensure scikit-learn unpickling compatibility
import sklearn.compose._column_transformer
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

import requests
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.impute import SimpleImputer


def load_dataset_and_split():
    """
    Loads Customer_Cancellations_Training.csv and recreates the
    exact stratified 70:30 train-test split using random_state=42.
    """
    csv_candidates = [
        BASE_DIR / "data" / "Customer_Cancellations_Training.csv",
        Path(r"C:\Users\yashu\Downloads\Customer_Cancellations_Training.csv"),
    ]
    csv_path = None
    for p in csv_candidates:
        if p.exists():
            csv_path = p
            break

    if csv_path is None:
        raise FileNotFoundError(
            "Could not locate Customer_Cancellations_Training.csv in data/ or Downloads."
        )

    print(f"Loading held-out test data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Exclude AccountID and Country as excluded during training
    X = df.drop(columns=["Churned", "AccountID", "Country"])
    y = df["Churned"].map({"no": 0, "yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    return df, X_train, X_test, y_train, y_test


def load_direct_model():
    """Loads the trained Decision Tree pipeline model directly."""
    model_path = BASE_DIR / "saved_model" / "churn_model.pkl"
    model = joblib.load(model_path)
    for step_name, step in model.steps:
        if hasattr(step, "transformers_"):
            for _, trans, _ in step.transformers_:
                if hasattr(trans, "named_steps"):
                    for _, s in trans.named_steps.items():
                        if isinstance(s, SimpleImputer) and not hasattr(s, "_fill_dtype"):
                            s._fill_dtype = getattr(s, "_fit_dtype", None)
    return model


def select_diverse_samples(X_test: pd.DataFrame, y_test: pd.Series, model, n_total=45):
    """
    Selects a diverse subset of 40-50 samples from X_test covering:
    - Pure 0% and pure 100% leaves
    - Boundary probabilities near 0.50 (e.g. 0.40, 0.429, 0.50, 0.571, 0.60)
    - Various intermediate probabilities (e.g. 0.66%, 16.7%, 20%, 25%, 28.6%, 66.7%, 75%, 80%, 88.9%)
    - Diverse plans, devices, territories, tenures, and spend tiers
    """
    probas = model.predict_proba(X_test)[:, 1]
    df_eval = X_test.copy()
    df_eval["actual_churn"] = y_test
    df_eval["raw_prob"] = probas
    df_eval["original_idx"] = X_test.index

    selected_indices = []

    # 1. Boundary cases: 0.35 <= prob <= 0.65
    boundary_pool = df_eval[(df_eval["raw_prob"] >= 0.35) & (df_eval["raw_prob"] <= 0.65)]
    for idx in boundary_pool.index[:10]:
        selected_indices.append(idx)

    # 2. Intermediate low/mid: 0.001 < prob < 0.35
    inter_low = df_eval[(df_eval["raw_prob"] > 0.001) & (df_eval["raw_prob"] < 0.35)]
    # Pick across different probabilities
    for _, group in inter_low.groupby("raw_prob"):
        selected_indices.append(group.index[0])
        if len(selected_indices) >= 20:
            break

    # 3. Intermediate mid/high: 0.65 < prob < 0.999
    inter_high = df_eval[(df_eval["raw_prob"] > 0.65) & (df_eval["raw_prob"] < 0.999)]
    for _, group in inter_high.groupby("raw_prob"):
        selected_indices.append(group.index[0])
        if len(selected_indices) >= 27:
            break

    # 4. Pure 0% cases with diverse plans and devices
    pure_0 = df_eval[df_eval["raw_prob"] == 0.0]
    for plan in ["Purple Plan", "Blue Plan", "Green Plan", "Red Plan"]:
        sub = pure_0[pure_0["PlanType"] == plan]
        if not sub.empty:
            for idx in sub.index[:2]:
                if idx not in selected_indices:
                    selected_indices.append(idx)
    # Add a few more with diverse tenures
    for idx in pure_0.sort_values("CustomerTenure").index[:4]:
        if idx not in selected_indices:
            selected_indices.append(idx)

    # 5. Pure 100% cases with diverse plans and devices
    pure_100 = df_eval[df_eval["raw_prob"] == 1.0]
    for idx in pure_100.index[:10]:
        if idx not in selected_indices:
            selected_indices.append(idx)

    selected_indices = list(dict.fromkeys(selected_indices))[:n_total]
    print(f"Selected {len(selected_indices)} diverse test samples.")
    return X_test.loc[selected_indices], y_test.loc[selected_indices]


def evaluate_samples_via_api(X_samples: pd.DataFrame, y_samples: pd.Series, model, api_url="http://127.0.0.1:8081"):
    """
    Submits each sample to the live FastAPI prediction endpoint,
    compares against direct model prediction, and collects full audit data.
    """
    records = []
    print(f"\nEvaluating {len(X_samples)} samples against {api_url}/predict ...")

    for sample_id, row in X_samples.iterrows():
        actual_class = int(y_samples.loc[sample_id])
        actual_label = "Churn" if actual_class == 1 else "No Churn"

        # Prepare exact API payload
        payload = {
            "Territory": str(row["Territory"]),
            "DeviceType": str(row["DeviceType"]),
            "Promotion": str(row["Promotion"]),
            "HasRenewed": str(row["HasRenewed"]),
            "PlanType": str(row["PlanType"]),
            "BaseFee": float(row["BaseFee"]),
            "AdditionalFeatureSpend": float(row["AdditionalFeatureSpend"]),
            "NumberOfPenalties": int(row["NumberOfPenalties"]),
            "CurrentPeriodUsage": float(row["CurrentPeriodUsage"]),
            "PriorPeriodUsage": float(row["PriorPeriodUsage"]),
            "ServiceRating": float(row["ServiceRating"]),
            "ServiceTickets": int(row["ServiceTickets"]),
            "StartMonth": str(row["StartMonth"]),
            "StartWeek": str(row["StartWeek"]),
            "CustomerTenure": int(row["CustomerTenure"]),
        }

        # 1. API Call
        http_status = None
        api_pred_label = None
        api_pred_class = None
        api_churn_prob = None
        api_error = None

        try:
            resp = requests.post(f"{api_url}/predict", json=payload, timeout=3.0)
            http_status = resp.status_code
            if resp.status_code == 200:
                data = resp.json()
                api_pred_label = data["prediction"]
                api_pred_class = int(data["predicted_class"])
                api_churn_prob = float(data["churn_probability"])
            else:
                api_error = resp.text
        except Exception as e:
            api_error = str(e)

        # 2. Direct Model Prediction
        single_df = pd.DataFrame([payload])
        direct_pred_class = int(model.predict(single_df)[0])
        direct_prob = float(model.predict_proba(single_df)[0][1])
        direct_pred_label = "Churn" if direct_pred_class == 1 else "No Churn"

        # 3. Validation Checks
        pred_match_actual = (api_pred_class == actual_class) if api_pred_class is not None else False
        api_matches_direct = (
            api_pred_class == direct_pred_class and
            abs(api_churn_prob - round(direct_prob, 4)) < 1e-4
        ) if api_pred_class is not None else False

        prob_pct_str = f"{api_churn_prob * 100:.2f}%" if api_churn_prob is not None else "N/A"

        records.append({
            "Sample_ID": sample_id,
            "Actual_Class": actual_class,
            "Actual_Label": actual_label,
            "Predicted_Class": api_pred_class,
            "Predicted_Label": api_pred_label,
            "Churn_Probability_Value": api_churn_prob,
            "Churn_Probability_Pct": prob_pct_str,
            "Prediction_Matches_Actual": pred_match_actual,
            "HTTP_Status": http_status,
            "API_Matches_Direct_Model": api_matches_direct,
            "Direct_Model_Class": direct_pred_class,
            "Direct_Model_Prob": round(direct_prob, 4),
            "API_Error": api_error,
            # Features for transparency
            "PlanType": payload["PlanType"],
            "Territory": payload["Territory"],
            "DeviceType": payload["DeviceType"],
            "HasRenewed": payload["HasRenewed"],
            "Promotion": payload["Promotion"],
            "BaseFee": payload["BaseFee"],
            "AdditionalFeatureSpend": payload["AdditionalFeatureSpend"],
            "NumberOfPenalties": payload["NumberOfPenalties"],
            "CurrentPeriodUsage": payload["CurrentPeriodUsage"],
            "PriorPeriodUsage": payload["PriorPeriodUsage"],
            "ServiceRating": payload["ServiceRating"],
            "ServiceTickets": payload["ServiceTickets"],
            "CustomerTenure": payload["CustomerTenure"],
            "StartMonth": payload["StartMonth"],
            "StartWeek": payload["StartWeek"],
        })

    return pd.DataFrame(records)


def compute_probability_bands(probas: pd.Series):
    """
    Groups probabilities into requested bands:
    - 0%
    - >0% and <25%
    - 25% to <50%
    - 50% to <75%
    - 75% to <100%
    - 100%
    """
    bands = {
        "0% (Exact Zero)": (probas == 0.0).sum(),
        ">0% and <25%": ((probas > 0.0) & (probas < 0.25)).sum(),
        "25% to <50%": ((probas >= 0.25) & (probas < 0.50)).sum(),
        "50% to <75%": ((probas >= 0.50) & (probas < 0.75)).sum(),
        "75% to <100%": ((probas >= 0.75) & (probas < 1.0)).sum(),
        "100% (Exact One)": (probas == 1.0).sum(),
    }
    return bands


def main():
    print("=" * 80)
    print("STEP 1: LOCATING AND LOADING REAL HELD-OUT TEST DATA")
    print("=" * 80)
    df, X_train, X_test, y_train, y_test = load_dataset_and_split()
    print(f"Full Dataset Shape: {df.shape}")
    print(f"Train Shape: {X_train.shape} (70%) | Churn rate: {y_train.mean():.4f}")
    print(f"Held-Out Test Shape: {X_test.shape} (30%) | Churn rate: {y_test.mean():.4f}")
    print(f"Test Class Counts: No Churn={int((y_test==0).sum())}, Churn={int((y_test==1).sum())}")

    print("\n" + "=" * 80)
    print("STEP 2: SELECTING DIVERSE TEST SAMPLES (N=45)")
    print("=" * 80)
    model = load_direct_model()
    X_samples, y_samples = select_diverse_samples(X_test, y_test, model, n_total=45)

    print("\n" + "=" * 80)
    print("STEP 3: TESTING RUNNING FASTAPI ENDPOINT (http://127.0.0.1:8081/predict)")
    print("=" * 80)
    results_df = evaluate_samples_via_api(X_samples, y_samples, model)

    # Save to CSV
    output_csv = BASE_DIR / "reports" / "test_sample_validation_results.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved sample-level results to: {output_csv}")

    print("\n" + "=" * 80)
    print("STEP 4: PROBABILITY DISTRIBUTION & PERFORMANCE ANALYSIS (SELECTED SAMPLES N=45)")
    print("=" * 80)

    n_samples = len(results_df)
    probas = results_df["Churn_Probability_Value"]
    y_true = results_df["Actual_Class"]
    y_pred = results_df["Predicted_Class"]

    zero_count = (probas == 0.0).sum()
    one_count = (probas == 1.0).sum()
    inter_count = ((probas > 0.0) & (probas < 1.0)).sum()

    print(f"Total Selected Samples Tested: {n_samples}")
    print(f"Exactly 0.0% (0.0): {zero_count} samples ({zero_count/n_samples*100:.1f}%)")
    print(f"Exactly 100.0% (1.0): {one_count} samples ({one_count/n_samples*100:.1f}%)")
    print(f"Strictly Between 0% and 100%: {inter_count} samples ({inter_count/n_samples*100:.1f}%)")

    distinct_probs = sorted(probas.unique())
    print(f"Distinct Probability Values (count={len(distinct_probs)}):")
    print([f"{p*100:.2f}%" for p in distinct_probs])

    print(f"\nProbability Stats:")
    print(f"  Min   : {probas.min()*100:.2f}%")
    print(f"  Median: {probas.median()*100:.2f}%")
    print(f"  Max   : {probas.max()*100:.2f}%")

    print("\nProbability Bands (Selected N=45):")
    bands = compute_probability_bands(probas)
    for band_name, count in bands.items():
        pct = count / n_samples * 100
        print(f"  {band_name:<20}: {count:>3} samples ({pct:>5.1f}%)")

    print("\nPerformance Metrics on Selected Diverse Set (N=45):")
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  Confusion Matrix (rows=actual, cols=predicted):\n{cm}")

    # Full held-out test set evaluation (N=948)
    print("\n" + "=" * 80)
    print("REFERENCE: FULL HELD-OUT TEST SET EVALUATION (N=948)")
    print("=" * 80)
    full_preds = model.predict(X_test)
    full_probs = model.predict_proba(X_test)[:, 1]

    full_acc = accuracy_score(y_test, full_preds)
    full_prec = precision_score(y_test, full_preds, zero_division=0)
    full_rec = recall_score(y_test, full_preds, zero_division=0)
    full_f1 = f1_score(y_test, full_preds, zero_division=0)
    full_cm = confusion_matrix(y_test, full_preds)

    print(f"Full Test Set Accuracy : {full_acc:.4f} ({full_acc*100:.2f}%)")
    print(f"Full Test Set Precision: {full_prec:.4f}")
    print(f"Full Test Set Recall   : {full_rec:.4f}")
    print(f"Full Test Set F1-Score : {full_f1:.4f} (~0.805 target!)")
    print(f"Full Confusion Matrix:\n{full_cm}")

    full_bands = compute_probability_bands(pd.Series(full_probs))
    print("\nFull Test Set Probability Bands (N=948):")
    for band_name, count in full_bands.items():
        pct = count / len(y_test) * 100
        print(f"  {band_name:<20}: {count:>4} samples ({pct:>5.1f}%)")

    print("\n" + "=" * 80)
    print("STEP 5: API vs DIRECT MODEL PARITY VERIFICATION")
    print("=" * 80)
    api_parity = results_df["API_Matches_Direct_Model"].all()
    print(f"API predictions exactly match direct model: {api_parity}")
    if api_parity:
        print("PASS: 100% agreement between FastAPI endpoint and saved Decision Tree model.")

    print("\n" + "=" * 80)
    print("STEP 6: DECISION TREE LEAF DIAGNOSTICS")
    print("=" * 80)
    tree = model.named_steps["classifier"].tree_
    leaf_indices = np.where(tree.children_left == -1)[0]
    leaf_values = tree.value[leaf_indices]
    p_churn_leaves = leaf_values[:, 0, 1] / leaf_values[:, 0].sum(axis=1)

    print(f"Total Leaves in Tree: {len(leaf_indices)}")
    print(f"Pure 0% Leaves      : {(p_churn_leaves == 0.0).sum()} ({(p_churn_leaves == 0.0).sum()/len(leaf_indices)*100:.1f}%)")
    print(f"Pure 100% Leaves    : {(p_churn_leaves == 1.0).sum()} ({(p_churn_leaves == 1.0).sum()/len(leaf_indices)*100:.1f}%)")
    print(f"Mixed (Intermediate): {((p_churn_leaves > 0.0) & (p_churn_leaves < 1.0)).sum()} ({((p_churn_leaves > 0.0) & (p_churn_leaves < 1.0)).sum()/len(leaf_indices)*100:.1f}%)")
    print(f"Total Distinct Probabilities Possible: {len(np.unique(p_churn_leaves))}")


if __name__ == "__main__":
    main()
