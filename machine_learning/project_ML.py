# automl_dashboard.py
# AutoML Web Dashboard - Classification & Regression

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
import joblib

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, KFold, cross_val_score
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score
)

# ===============================
# Base model imports
# ===============================
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, SGDClassifier,
    LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    AdaBoostClassifier,
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Optional libs (if installed)
optional_models = []
try:
    from xgboost import XGBClassifier, XGBRegressor
    optional_models.append(("XGB", XGBClassifier, XGBRegressor))
except:
    pass

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    optional_models.append(("LGBM", LGBMClassifier, LGBMRegressor))
except:
    pass

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    optional_models.append(("CAT", CatBoostClassifier, CatBoostRegressor))
except:
    pass


# ===============================
# Build model zoo
# ===============================
def build_models(task: str):
    models = []

    if task == "classification":
        # Logistic Regression
        for C in [0.01, 0.1, 1, 5, 10]:
            for solver in ["liblinear", "lbfgs"]:
                models.append(
                    (f"LogReg_C{C}_{solver}",
                     LogisticRegression(C=C, solver=solver, max_iter=2000))
                )

        # Linear models (SGD + RidgeClassifier)
        for alpha in [0.0001, 0.001, 0.01, 0.1]:
            models.append(
                (f"SGD_alpha{alpha}",
                 SGDClassifier(alpha=alpha, max_iter=2000, tol=1e-3))
            )
        for alpha in [0.5, 1.0, 2.0]:
            models.append(
                (f"RidgeClf_alpha{alpha}", RidgeClassifier(alpha=alpha))
            )

        # KNN
        for k in [3, 5, 7, 9, 11, 15]:
            for w in ["uniform", "distance"]:
                models.append(
                    (f"KNN_k{k}_{w}",
                     KNeighborsClassifier(n_neighbors=k, weights=w))
                )

        # Decision Trees
        for depth in [None, 3, 5, 8, 12]:
            for crit in ["gini", "entropy"]:
                models.append(
                    (f"DT_depth{depth}_{crit}",
                     DecisionTreeClassifier(max_depth=depth, criterion=crit))
                )

        # SVM (بدون poly لتسريع)
        for C in [0.1, 1, 5, 10]:
            for kernel in ["linear", "rbf"]:
                models.append(
                    (f"SVC_{kernel}_C{C}",
                     SVC(C=C, kernel=kernel, probability=True))
                )
        for C in [0.1, 1, 5, 10]:
            models.append(
                (f"LinearSVC_C{C}", LinearSVC(C=C))
            )

        # Ensembles
        for n in [100, 200, 400]:
            for depth in [None, 8, 15]:
                models.append(
                    (f"RF_n{n}_d{depth}",
                     RandomForestClassifier(n_estimators=n, max_depth=depth))
                )
                models.append(
                    (f"ET_n{n}_d{depth}",
                     ExtraTreesClassifier(n_estimators=n, max_depth=depth))
                )

        for lr in [0.01, 0.05, 0.1]:
            for n in [100, 200]:
                models.append(
                    (f"GB_n{n}_lr{lr}",
                     GradientBoostingClassifier(n_estimators=n,
                                                learning_rate=lr))
                )

        for n in [50, 100, 200]:
            models.append(
                (f"AdaBoost_n{n}", AdaBoostClassifier(n_estimators=n))
            )

        # Naive Bayes
        models.append(("GaussianNB", GaussianNB()))

        # MLP
        for h in [(50,), (100,), (50, 50), (100, 50)]:
            for alpha in [0.0001, 0.001, 0.01]:
                models.append(
                    (f"MLP_{h}_a{alpha}",
                     MLPClassifier(hidden_layer_sizes=h,
                                   alpha=alpha,
                                   max_iter=800))
                )

        # Optional boosted libs (silent)
        for name, Cls, _ in optional_models:
            for n in [200, 500]:
                for depth in [3, 6, 10]:
                    try:
                        if name == "CAT":
                            models.append(
                                (f"{name}_n{n}_d{depth}",
                                 Cls(iterations=n,
                                     depth=depth,
                                     verbose=False))
                            )
                        elif name == "LGBM":
                            models.append(
                                (f"{name}_n{n}_d{depth}",
                                 Cls(n_estimators=n,
                                     max_depth=depth,
                                     verbosity=-1))
                            )
                        elif name == "XGB":
                            models.append(
                                (f"{name}_n{n}_d{depth}",
                                 Cls(n_estimators=n,
                                     max_depth=depth,
                                     eval_metric="logloss",
                                     verbosity=0))
                            )
                    except:
                        pass

    else:  # regression
        models.append(("LinearRegression", LinearRegression()))
        for a in [0.1, 1, 5, 10, 50]:
            models.append((f"Ridge_a{a}", Ridge(alpha=a)))
        for a in [0.001, 0.01, 0.1, 1]:
            models.append((f"Lasso_a{a}", Lasso(alpha=a)))
        for a in [0.001, 0.01, 0.1]:
            for l1 in [0.2, 0.5, 0.8]:
                models.append(
                    (f"Elastic_a{a}_l1{l1}",
                     ElasticNet(alpha=a, l1_ratio=l1))
                )

        for alpha in [0.0001, 0.001, 0.01, 0.1]:
            models.append(
                (f"SGDReg_alpha{alpha}",
                 SGDRegressor(alpha=alpha, max_iter=5000))
            )

        for k in [3, 5, 7, 9, 11, 15]:
            for w in ["uniform", "distance"]:
                models.append(
                    (f"KNNReg_k{k}_{w}",
                     KNeighborsRegressor(n_neighbors=k, weights=w))
                )

        for depth in [None, 3, 5, 8, 12]:
            models.append(
                (f"DTReg_depth{depth}",
                 DecisionTreeRegressor(max_depth=depth))
            )

        for C in [0.1, 1, 5, 10]:
            for kernel in ["linear", "rbf"]:
                models.append(
                    (f"SVR_{kernel}_C{C}",
                     SVR(C=C, kernel=kernel))
                )
        for C in [0.1, 1, 5, 10]:
            models.append(
                (f"LinearSVR_C{C}", LinearSVR(C=C))
            )

        for n in [100, 200, 400]:
            for depth in [None, 8, 15]:
                models.append(
                    (f"RFReg_n{n}_d{depth}",
                     RandomForestRegressor(n_estimators=n, max_depth=depth))
                )
                models.append(
                    (f"ETReg_n{n}_d{depth}",
                     ExtraTreesRegressor(n_estimators=n, max_depth=depth))
                )

        for lr in [0.01, 0.05, 0.1]:
            for n in [100, 200]:
                models.append(
                    (f"GBReg_n{n}_lr{lr}",
                     GradientBoostingRegressor(n_estimators=n,
                                               learning_rate=lr))
                )

        for n in [50, 100, 200]:
            models.append(
                (f"AdaReg_n{n}", AdaBoostRegressor(n_estimators=n))
            )

        for h in [(50,), (100,), (50, 50), (100, 50)]:
            for alpha in [0.0001, 0.001, 0.01]:
                models.append(
                    (f"MLPReg_{h}_a{alpha}",
                     MLPRegressor(hidden_layer_sizes=h,
                                  alpha=alpha,
                                  max_iter=1200))
                )

        for name, _, Reg in optional_models:
            for n in [200, 500]:
                for depth in [3, 6, 10]:
                    try:
                        if name == "CAT":
                            models.append(
                                (f"{name}Reg_n{n}_d{depth}",
                                 Reg(iterations=n,
                                     depth=depth,
                                     verbose=False))
                            )
                        elif name == "LGBM":
                            models.append(
                                (f"{name}Reg_n{n}_d{depth}",
                                 Reg(n_estimators=n,
                                     max_depth=depth,
                                     verbosity=-1))
                            )
                        elif name == "XGB":
                            models.append(
                                (f"{name}Reg_n{n}_d{depth}",
                                 Reg(n_estimators=n,
                                     max_depth=depth,
                                     eval_metric="rmse",
                                     verbosity=0))
                            )
                    except:
                        pass

    return models


# ===============================
# AutoML main function
# ===============================
def run_automl(df, target_col, task, test_size=0.2, random_state=42, n_splits=5):
    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col])
    y = df[target_col]

    num_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop"
    )

    if task == "classification":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    models = build_models(task)
    st.write(f"✅ Total models to evaluate: **{len(models)}**")

    results = []

    def evaluate_model(model):
        pipe = Pipeline(steps=[
            ("preprocess", preprocess),
            ("model", model)
        ])
        if task == "classification":
            scores = cross_val_score(
                pipe, X_train, y_train,
                cv=cv, scoring="f1_weighted", n_jobs=-1
            )
        else:
            scores = cross_val_score(
                pipe, X_train, y_train,
                cv=cv, scoring="r2", n_jobs=-1
            )
        return scores.mean(), scores.std()

    progress = st.progress(0)
    status = st.empty()

    for i, (name, model) in enumerate(models, start=1):
        try:
            t0 = time.time()
            mean_score, std_score = evaluate_model(model)
            elapsed = time.time() - t0

            results.append([name, mean_score, std_score, elapsed])
            status.text(f"[{i}/{len(models)}] {name}  score={mean_score:.4f}  time={elapsed:.2f}s")
            progress.progress(i / len(models))
        except Exception as e:
            results.append([name, np.nan, np.nan, np.nan])
            print(f"{name} FAILED: {e}")

    progress.empty()
    status.empty()

    results_df = pd.DataFrame(
        results,
        columns=["Model", "CV_Score_Mean", "CV_Score_Std", "Train_Time_sec"]
    )
    results_df = results_df.sort_values("CV_Score_Mean", ascending=False).reset_index(drop=True)

    best_model_name = results_df.loc[0, "Model"]
    best_model_obj = dict(models)[best_model_name]

    best_pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", best_model_obj)
    ])
    best_pipe.fit(X_train, y_train)
    y_pred = best_pipe.predict(X_test)

    if task == "classification":
        acc = accuracy_score(y_test, y_pred)
        f1w = f1_score(y_test, y_pred, average="weighted")
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        metrics = {
            "Accuracy": acc,
            "F1_weighted": f1w,
            "Precision_weighted": prec,
            "Recall_weighted": rec
        }
    else:
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics = {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }

    return {
        "results_df": results_df,
        "best_model_name": best_model_name,
        "best_pipe": best_pipe,
        "metrics": metrics,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "task": task,
        "target_col": target_col
    }


# ===============================
# Plot helpers
# ===============================
def plot_comparison(results_df):
    topN = min(15, len(results_df))
    plot_df = results_df.head(topN).copy()

    # Top-N bar
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_df["Model"][::-1], plot_df["CV_Score_Mean"][::-1])
    ax.set_title(f"Top {topN} Models by CV Score")
    ax.set_xlabel("CV Score (Mean)")
    ax.set_ylabel("Model")
    st.pyplot(fig)

    # Error bars
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(plot_df["CV_Score_Mean"], plot_df["Model"],
                xerr=plot_df["CV_Score_Std"], fmt="o")
    ax.set_title(f"Top {topN} Models (Mean ± Std)")
    ax.set_xlabel("CV Score")
    ax.set_ylabel("Model")
    st.pyplot(fig)

    # Histogram
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(results_df["CV_Score_Mean"].dropna(), bins=25)
    ax.set_title("Distribution of CV Scores (All Models)")
    ax.set_xlabel("CV Score Mean")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Mean vs Std
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(results_df["CV_Score_Mean"], results_df["CV_Score_Std"])
    ax.set_title("CV Mean vs CV Std (Stability)")
    ax.set_xlabel("CV Score Mean")
    ax.set_ylabel("CV Score Std")
    st.pyplot(fig)

    # Fastest models
    time_df = results_df.nsmallest(topN, "Train_Time_sec").copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(time_df["Model"][::-1], time_df["Train_Time_sec"][::-1])
    ax.set_title(f"Top {topN} Fastest Models (Train Time)")
    ax.set_xlabel("Train Time (sec)")
    ax.set_ylabel("Model")
    st.pyplot(fig)


def plot_best_model_details(result_dict):
    task = result_dict["task"]
    y_test = result_dict["y_test"]
    y_pred = result_dict["y_pred"]
    best_pipe = result_dict["best_pipe"]

    if task == "classification":
        model = best_pipe.named_steps["model"]
        unique_classes = np.unique(y_test)

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(values_format="d", ax=ax)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        cmn = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cmn)
        disp.plot(values_format=".2f", ax=ax)
        ax.set_title("Confusion Matrix (Normalized)")
        st.pyplot(fig)

        if hasattr(model, "predict_proba") and len(unique_classes) == 2:
            y_proba = best_pipe.predict_proba(result_dict["X_test"])[:, 1]

            fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=unique_classes[1])
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.set_title("ROC Curve")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()
            st.pyplot(fig)

            prec, rec, _ = precision_recall_curve(y_test, y_proba, pos_label=unique_classes[1])
            ap = average_precision_score(y_test, y_proba)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(rec, prec, label=f"AP = {ap:.3f}")
            ax.set_title("Precision-Recall Curve")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.legend()
            st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist(y_proba, bins=25)
            ax.set_title("Predicted Probability Histogram")
            ax.set_xlabel("P(class=1)")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        recalls = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(range(len(recalls)), recalls)
        ax.set_title("Per-Class Recall")
        ax.set_xlabel("Class index")
        ax.set_ylabel("Recall")
        st.pyplot(fig)

    else:
        errors = y_test - y_pred

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test, y_pred)
        ax.set_title("Actual vs Predicted")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_pred, errors)
        ax.axhline(0, linestyle="--")
        ax.set_title("Residuals vs Predicted")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual")
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(errors, bins=25)
        ax.set_title("Residual Distribution")
        ax.set_xlabel("Residual")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(np.abs(errors), bins=25)
        ax.set_title("Absolute Error Distribution")
        ax.set_xlabel("|Residual|")
        ax.set_ylabel("Count")
        st.pyplot(fig)


def plot_feature_importance(result_dict):
    best_pipe = result_dict["best_pipe"]
    model = best_pipe.named_steps["model"]

    try:
        feature_names = best_pipe.named_steps["preprocess"].get_feature_names_out()
    except Exception:
        return  # إذا ما قدر يجيب الأسماء نطلع بهدوء

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1])
        ax.set_title("Top 20 Feature Importances")
        ax.set_xlabel("Importance")
        st.pyplot(fig)

    if hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim == 2:
            coef = np.mean(np.abs(coef), axis=0)
        else:
            coef = np.abs(coef)
        coef_df = pd.DataFrame({
            "Feature": feature_names,
            "AbsCoef": coef
        }).sort_values("AbsCoef", ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(coef_df["Feature"][::-1], coef_df["AbsCoef"][::-1])
        ax.set_title("Top 20 Coefficients (Absolute)")
        ax.set_xlabel("|Coefficient|")
        st.pyplot(fig)


# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="AutoML Dashboard", layout="wide")

st.markdown(
    "<h1 style='text-align:center;'>🤖 AutoML Benchmark Dashboard</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;'>Upload your dataset, choose the target, and let the system try 100+ models automatically.</p>",
    unsafe_allow_html=True,
)

st.sidebar.header("⚙️ Settings")

uploaded_csv = st.sidebar.file_uploader("⬆️ Upload CSV data", type=["csv"])
task = st.sidebar.selectbox("Task Type", ["classification", "regression"])
test_size = st.sidebar.slider("Test size (test split)", 0.1, 0.4, 0.2, 0.05)
n_splits = st.sidebar.slider("CV folds (K-Fold)", 3, 7, 5, 1)

if "automl_result" not in st.session_state:
    st.session_state.automl_result = None

if uploaded_csv is not None:
    # ✅ محاولة القراءة بـ UTF-8، وإذا فشل نستخدم windows-1256 (مناسب للملفات العربية من الويندوز)
    try:
        df = pd.read_csv(uploaded_csv)
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_csv, encoding="windows-1256", engine="python")

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    target_col = st.selectbox("🎯 Choose target column", df.columns.tolist())

    if st.button("🚀 Run AutoML"):
        with st.spinner("Running AutoML... this may take a while depending on dataset size and models..."):
            res = run_automl(
                df, target_col, task,
                test_size=test_size,
                random_state=42,
                n_splits=n_splits,
            )
            st.session_state.automl_result = res
            joblib.dump(res["best_pipe"], "best_model_pipeline.joblib")


if st.session_state.automl_result is not None:
    res = st.session_state.automl_result

    tab_overview, tab_models, tab_charts, tab_best, tab_explain, tab_predict, tab_download = st.tabs(
        ["Overview", "Models & Scores", "Charts", "Best Model Details", "Explainability", "Predict on New Data", "Download"]
    )

    # -------- Overview --------
    with tab_overview:
        st.subheader("🏆 Best Model Overview")
        st.write(f"**Best model:** `{res['best_model_name']}`")
        st.write("**Task:**", res["task"])
        st.write("**Target column:**", res["target_col"])

        st.subheader("📌 Test Metrics")
        st.json(res["metrics"])

    # -------- Models & Scores --------
    with tab_models:
        st.subheader("📋 All Models (sorted by CV score)")
        st.dataframe(res["results_df"])

    # -------- Charts --------
    with tab_charts:
        st.subheader("📈 Model Comparison Charts")
        plot_comparison(res["results_df"])

    # -------- Best Model Details --------
    with tab_best:
        st.subheader("🔍 Best Model Performance")
        plot_best_model_details(res)

    # -------- Explainability --------
    with tab_explain:
        st.subheader("🧠 Feature Importance / Coefficients")
        st.info("Only available for tree / linear models that expose feature_importances_ or coef_.")
        plot_feature_importance(res)

    # -------- Predict on New Data --------
    with tab_predict:
        st.subheader("🤖 Predict using Best Model")
        new_file = st.file_uploader("Upload new CSV for prediction", type=["csv"], key="predict_csv")

        if new_file is not None:
            new_df = pd.read_csv(new_file)
            st.write("New data preview:")
            st.dataframe(new_df.head())

            # Remove target if exists
            if res["target_col"] in new_df.columns:
                X_new = new_df.drop(columns=[res["target_col"]])
            else:
                X_new = new_df.copy()

            preds = res["best_pipe"].predict(X_new)
            new_df["Prediction"] = preds

            st.success("Predictions generated successfully ✅")
            st.dataframe(new_df.head())

            csv_bytes = new_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download predictions CSV",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv"
            )

    # -------- Download --------
    with tab_download:
        st.subheader("💾 Download Artifacts")

        # comparison CSV
        comp_csv = res["results_df"].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download model comparison (CSV)",
            data=comp_csv,
            file_name="model_comparison_results.csv",
            mime="text/csv"
        )

        # best model joblib
        try:
            with open("best_model_pipeline.joblib", "rb") as f:
                st.download_button(
                    "⬇️ Download best model pipeline (.joblib)",
                    data=f,
                    file_name="best_model_pipeline.joblib"
                )
        except FileNotFoundError:
            st.warning("best_model_pipeline.joblib file not found on disk yet.")
else:
    st.info("Upload a CSV, choose the target and task, then click **Run AutoML** to start.")