"""
=============================================================
CHURN PREDICTION — PHASE 2: MODELING & OPTIMIZATION
=============================================================
Pipeline:
  1. Train/Test Split (time-based + leakage check)
  2. Baseline Models  (Logistic Regression, Decision Tree)
  3. Main Models      (Random Forest, XGBoost, LightGBM)
  4. Evaluation       (Accuracy, Precision, Recall, F1, ROC-AUC)
  5. Visualization    (Confusion Matrix, ROC Curve)
  6. Feature Importance & Top Churn Drivers
  7. Feature Selection
  8. Hyperparameter Tuning
  9. Imbalance Handling (class_weight + SMOTE)
 10. Cross-Validation & Model Comparison
 11. Export Best Model + Preprocessing Pipeline
=============================================================
"""

# ─────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────
import warnings, pickle, os, joblib
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from xgboost                 import XGBClassifier
from lightgbm                import LGBMClassifier

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, classification_report
)
from sklearn.inspection      import permutation_importance
from sklearn.pipeline        import Pipeline
from sklearn.feature_selection import SelectFromModel

from imblearn.over_sampling  import SMOTE

# ─────────────────────────────────────────────
# HELPER: bảng kết quả đẹp
# ─────────────────────────────────────────────
def print_section(title: str):
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")

def metrics_row(name, y_true, y_pred, y_prob=None) -> dict:
    row = dict(
        Model     = name,
        Accuracy  = round(accuracy_score(y_true, y_pred),  4),
        Precision = round(precision_score(y_true, y_pred, zero_division=0), 4),
        Recall    = round(recall_score(y_true, y_pred,    zero_division=0), 4),
        F1        = round(f1_score(y_true, y_pred,        zero_division=0), 4),
        ROC_AUC   = round(roc_auc_score(y_true, y_prob), 4) if y_prob is not None else None,
    )
    return row

# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# PHẦN 1 — DATA LOADING & TRAIN/TEST SPLIT
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
print_section("1. TRAIN/TEST SPLIT (TIME-BASED)")

# ------------------------------------------------------------------
# THAY ĐỔI: đọc dữ liệu thực tế của bạn ở đây
#   df = pd.read_csv("data/churn_dataset.csv", parse_dates=["order_date"])
# ------------------------------------------------------------------
# DEMO: tạo dữ liệu mẫu với cột thời gian để minh hoạ time-split
np.random.seed(42)
N = 5000

df = pd.DataFrame({
    "order_date"             : pd.date_range("2023-01-01", periods=N, freq="6h"),
    "days_since_last_order"  : np.random.randint(1, 180, N),
    "total_orders"           : np.random.randint(1, 50,  N),
    "orders_last_30d"        : np.random.randint(0, 10,  N),
    "avg_days_between_orders": np.random.uniform(1, 60,  N),
    "avg_order_value"        : np.random.uniform(50, 500, N),
    "spending_last_30d"      : np.random.uniform(0, 1000, N),
    "voucher_usage_rate"     : np.random.uniform(0, 1,   N),
    "recent_activity_drop"   : np.random.uniform(0, 1,   N),
    "unique_products"        : np.random.randint(1, 20,  N),
    "weekend_order_ratio"    : np.random.uniform(0, 1,   N),
    "max_days_between_orders": np.random.randint(1, 200, N),
    "favorite_channel"       : np.random.choice(["app", "web", "store"], N),
    "churn"                  : np.random.choice([0, 1], N, p=[0.75, 0.25]),
})

TARGET = "churn"
DATE_COL = "order_date"

# ── Time-based split ────────────────────────────────────────────
df_sorted = df.sort_values(DATE_COL).reset_index(drop=True)

split_idx   = int(len(df_sorted) * 0.80)          # 80% train / 20% test
split_date  = df_sorted[DATE_COL].iloc[split_idx]

df_train    = df_sorted.iloc[:split_idx].copy()
df_test     = df_sorted.iloc[split_idx:].copy()

print(f"  Split date    : {split_date.date()}")
print(f"  Train rows    : {len(df_train):,}  ({df_train[DATE_COL].min().date()} → {df_train[DATE_COL].max().date()})")
print(f"  Test  rows    : {len(df_test):,}  ({df_test[DATE_COL].min().date()}  → {df_test[DATE_COL].max().date()})")
print(f"  Train churn % : {df_train[TARGET].mean():.2%}")
print(f"  Test  churn % : {df_test[TARGET].mean():.2%}")

# ── Leakage check ───────────────────────────────────────────────
print("\n  [Leakage Check]")
train_max = df_train[DATE_COL].max()
test_min  = df_test[DATE_COL].min()
if train_max < test_min:
    print(f"  ✅  Không có overlap: train_max={train_max.date()} < test_min={test_min.date()}")
else:
    print(f"  ❌  CẢNH BÁO: train_max={train_max.date()} >= test_min={test_min.date()} — kiểm tra lại!")

FEATURE_COLS = [c for c in df.columns if c not in [TARGET, DATE_COL]]

X_train = df_train[FEATURE_COLS]
y_train = df_train[TARGET]
X_test  = df_test[FEATURE_COLS]
y_test  = df_test[TARGET]

# ─────────────────────────────────────────────
# LOAD PREPROCESSOR & transform
# ─────────────────────────────────────────────
print_section("1b. APPLY PREPROCESSOR")

# Load preprocessor đã fit ở notebook trước
try:
    with open("/mnt/user-data/uploads/preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    print("  ✅  Đã load preprocessor.pkl")
except Exception as e:
    print(f"  ⚠️  Không load được pkl ({e}) — tạo preprocessor mới")
    preprocessor = None

# Fit trên train, transform cả hai — KHÔNG fit lại trên test!
if preprocessor is None:
    from sklearn.compose       import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    num_cols = X_train[FEATURE_COLS].select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in FEATURE_COLS if c not in num_cols]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(),                       num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)
    print("  fit_transform() train + transform() test (fallback)")
else:
    # Preprocessor đã fit — chỉ cần transform
    from sklearn.utils.validation import check_is_fitted
    try:
        check_is_fitted(preprocessor)
        X_train_proc = preprocessor.transform(X_train)
        print("  transform() — preprocessor đã fit từ trước")
    except Exception:
        X_train_proc = preprocessor.fit_transform(X_train)
        print("  fit_transform() train")
    X_test_proc = preprocessor.transform(X_test)

print(f"  X_train_proc shape: {X_train_proc.shape}")
print(f"  X_test_proc  shape: {X_test_proc.shape}")

# Lấy tên feature sau transform để dùng sau
try:
    feature_names = preprocessor.get_feature_names_out()
except Exception:
    feature_names = [f"f{i}" for i in range(X_train_proc.shape[1])]

# Lưu results
all_results = []

# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# PHẦN 2 — BASELINE MODELS
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
print_section("2. BASELINE MODELS")

# ── 2a. Logistic Regression ────────────────
print("\n  [2a] Logistic Regression")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_proc, y_train)
lr_pred = lr.predict(X_test_proc)
lr_prob = lr.predict_proba(X_test_proc)[:, 1]
all_results.append(metrics_row("Logistic Regression", y_test, lr_pred, lr_prob))
print(classification_report(y_test, lr_pred, target_names=["Không churn", "Churn"]))

# ── 2b. Decision Tree ──────────────────────
print("\n  [2b] Decision Tree")
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train_proc, y_train)
dt_pred = dt.predict(X_test_proc)
dt_prob = dt.predict_proba(X_test_proc)[:, 1]
all_results.append(metrics_row("Decision Tree", y_test, dt_pred, dt_prob))
print(classification_report(y_test, dt_pred, target_names=["Không churn", "Churn"]))

# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# PHẦN 3 — MAIN MODELS
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
print_section("3. MAIN MODELS")

# ── 3a. Random Forest ──────────────────────
print("\n  [3a] Random Forest")
rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                             n_jobs=-1, random_state=42)
rf.fit(X_train_proc, y_train)
rf_pred = rf.predict(X_test_proc)
rf_prob = rf.predict_proba(X_test_proc)[:, 1]
all_results.append(metrics_row("Random Forest", y_test, rf_pred, rf_prob))
print(classification_report(y_test, rf_pred, target_names=["Không churn", "Churn"]))

# ── 3b. XGBoost ────────────────────────────
print("\n  [3b] XGBoost")
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(n_estimators=200, max_depth=6,
                     learning_rate=0.05, scale_pos_weight=scale_pos,
                     eval_metric="logloss", random_state=42,
                     verbosity=0)
xgb.fit(X_train_proc, y_train)
xgb_pred = xgb.predict(X_test_proc)
xgb_prob = xgb.predict_proba(X_test_proc)[:, 1]
all_results.append(metrics_row("XGBoost", y_test, xgb_pred, xgb_prob))
print(classification_report(y_test, xgb_pred, target_names=["Không churn", "Churn"]))

# ── 3c. LightGBM ───────────────────────────
print("\n  [3c] LightGBM")
lgbm = LGBMClassifier(n_estimators=200, max_depth=6,
                       learning_rate=0.05, class_weight="balanced",
                       random_state=42, verbose=-1)
lgbm.fit(X_train_proc, y_train)
lgbm_pred = lgbm.predict(X_test_proc)
lgbm_prob = lgbm.predict_proba(X_test_proc)[:, 1]
all_results.append(metrics_row("LightGBM", y_test, lgbm_pred, lgbm_prob))
print(classification_report(y_test, lgbm_pred, target_names=["Không churn", "Churn"]))

# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# PHẦN 4 — EVALUATION TABLE
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
print_section("4. EVALUATION — SO SÁNH CÁC MODEL")

results_df = pd.DataFrame(all_results).set_index("Model")
print(results_df.to_string())

# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# PHẦN 5 — VISUALIZATION
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
print_section("5. VISUALIZATION")

MODELS = {
    "Logistic Regression": (lr_pred, lr_prob),
    "Decision Tree"      : (dt_pred, dt_prob),
    "Random Forest"      : (rf_pred, rf_prob),
    "XGBoost"            : (xgb_pred, xgb_prob),
    "LightGBM"           : (lgbm_pred, lgbm_prob),
}
PALETTE = ["#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f"]

# ── 5a. Confusion Matrices ─────────────────
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
fig.suptitle("Confusion Matrices — Tất cả Model", fontsize=14, fontweight="bold")

for ax, (name, (pred, _)), color in zip(axes, MODELS.items(), PALETTE):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                cmap="Blues", cbar=False,
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["Actual 0", "Actual 1"])
    ax.set_title(name, fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/01_confusion_matrices.png", dpi=130, bbox_inches="tight")
plt.close()
print("  Saved: 01_confusion_matrices.png")

# ── 5b. ROC Curves ─────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.50)")

for (name, (_, prob)), color in zip(MODELS.items(), PALETTE):
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    ax.plot(fpr, tpr, lw=2, color=color, label=f"{name}  (AUC={auc:.3f})")

ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate",  fontsize=12)
ax.set_title("ROC Curves — Tất cả Model", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/02_roc_curves.png", dpi=130, bbox_inches="tight")
plt.close()
print("  Saved: 02_roc_curves.png")

# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# PHẦN 6 — FEATURE IMPORTANCE
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
print_section("6. FEATURE IMPORTANCE & TOP CHURN DRIVERS")

def get_importance_df(model, names):
    """Lấy feature importance từ tree-based models."""
    imp = model.feature_importances_
    return (pd.DataFrame({"feature": names, "importance": imp})
              .sort_values("importance", ascending=False)
              .reset_index(drop=True))

imp_rf   = get_importance_df(rf,   feature_names)
imp_xgb  = get_importance_df(xgb,  feature_names)
imp_lgbm = get_importance_df(lgbm, feature_names)

TOP_N = 15
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Feature Importance — Top 15 Features", fontsize=14, fontweight="bold")

for ax, imp_df, name, color in zip(
    axes,
    [imp_rf, imp_xgb, imp_lgbm],
    ["Random Forest", "XGBoost", "LightGBM"],
    ["#4e79a7", "#e15759", "#59a14f"]
):
    top = imp_df.head(TOP_N)
    ax.barh(top["feature"][::-1], top["importance"][::-1], color=color)
    ax.set_title(name, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.tick_params(axis="y", labelsize=8)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/03_feature_importance.png", dpi=130, bbox_inches="tight")
plt.close()
print("  Saved: 03_feature_importance.png")

# ── Top Churn Drivers (trung bình 3 model) ──
print("\n  Top 10 Churn Drivers (avg importance — RF, XGB, LGBM):")
merged = (imp_rf.set_index("feature")[["importance"]]
          .rename(columns={"importance": "rf"})
          .join(imp_xgb.set_index("feature")[["importance"]].rename(columns={"importance": "xgb"}))
          .join(imp_lgbm.set_index("feature")[["importance"]].rename(columns={"importance": "lgbm"})))
merged["avg"] = merged.mean(axis=1)
top_drivers = merged.sort_values("avg", ascending=False).head(10)
print(top_drivers[["rf","xgb","lgbm","avg"]].round(4).to_string())

# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# PHẦN 7 — OPTIMIZATION: FEATURE SELECTION
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
print_section("7. FEATURE SELECTION (Remove low-importance features)")

# Giữ những feature có importance trung bình > threshold
THRESHOLD = merged["avg"].quantile(0.25)   # loại 25% dưới cùng
selected_mask    = merged["avg"] >= THRESHOLD
selected_features = merged[selected_mask].index.tolist()
feature_idx       = [list(feature_names).index(f) for f in selected_features
                     if f in list(feature_names)]

X_train_sel = X_train_proc[:, feature_idx]
X_test_sel  = X_test_proc[:,  feature_idx]

print(f"  Tổng features ban đầu : {X_train_proc.shape[1]}")
print(f"  Features giữ lại      : {len(feature_idx)}  (threshold avg={THRESHOLD:.4f})")
print(f"  Features loại bỏ      : {X_train_proc.shape[1] - len(feature_idx)}")

# Train lại XGBoost với features đã chọn để so sánh
xgb_sel = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                          scale_pos_weight=scale_pos, eval_metric="logloss",
                          random_state=42, verbosity=0)
xgb_sel.fit(X_train_sel, y_train)
xgb_sel_pred = xgb_sel.predict(X_test_sel)
xgb_sel_prob = xgb_sel.predict_proba(X_test_sel)[:, 1]
row_sel = metrics_row("XGBoost (selected)", y_test, xgb_sel_pred, xgb_sel_prob)
print(f"\n  XGBoost full     ROC-AUC: {roc_auc_score(y_test, xgb_prob):.4f}")
print(f"  XGBoost selected ROC-AUC: {row_sel['ROC_AUC']:.4f}")

# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# PHẦN 8 — HYPERPARAMETER TUNING (XGBoost)
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
print_section("8. HYPERPARAMETER TUNING — XGBoost (RandomizedSearchCV)")

param_dist = {
    "max_depth"     : [3, 4, 5, 6, 7, 8],
    "learning_rate" : [0.01, 0.03, 0.05, 0.1, 0.15],
    "n_estimators"  : [100, 150, 200, 300],
    "subsample"     : [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
}

cv_inner = StratifiedKFold(n_splits=3, shuffle=False)   # shuffle=False giữ thứ tự thời gian

xgb_base = XGBClassifier(scale_pos_weight=scale_pos, eval_metric="logloss",
                           random_state=42, verbosity=0)

search = RandomizedSearchCV(
    xgb_base, param_dist,
    n_iter=30,
    scoring="roc_auc",
    cv=cv_inner,
    n_jobs=-1,
    random_state=42,
    verbose=0,
)
search.fit(X_train_proc, y_train)

best_xgb  = search.best_estimator_
best_pred = best_xgb.predict(X_test_proc)
best_prob = best_xgb.predict_proba(X_test_proc)[:, 1]

print(f"\n  Best params   : {search.best_params_}")
print(f"  CV ROC-AUC    : {search.best_score_:.4f}")
print(f"  Test ROC-AUC  : {roc_auc_score(y_test, best_prob):.4f}")

all_results.append(metrics_row("XGBoost (tuned)", y_test, best_pred, best_prob))

# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# PHẦN 9 — IMBALANCE HANDLING
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
print_section("9. IMBALANCE HANDLING")

# ── 9a. class_weight="balanced" ────────────
print("\n  [9a] class_weight='balanced' — LightGBM")
lgbm_cw = LGBMClassifier(n_estimators=200, learning_rate=0.05,
                           class_weight="balanced", random_state=42, verbose=-1)
lgbm_cw.fit(X_train_proc, y_train)
lgbm_cw_pred = lgbm_cw.predict(X_test_proc)
lgbm_cw_prob = lgbm_cw.predict_proba(X_test_proc)[:, 1]
row_cw = metrics_row("LightGBM (class_weight)", y_test, lgbm_cw_pred, lgbm_cw_prob)
all_results.append(row_cw)
print(f"  F1={row_cw['F1']:.4f}  ROC-AUC={row_cw['ROC_AUC']:.4f}")

# ── 9b. SMOTE ──────────────────────────────
print("\n  [9b] SMOTE oversampling")
print(f"  Trước SMOTE: {pd.Series(y_train).value_counts().to_dict()}")

smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X_train_proc, y_train)
print(f"  Sau  SMOTE: {pd.Series(y_sm).value_counts().to_dict()}")

lgbm_sm = LGBMClassifier(n_estimators=200, learning_rate=0.05,
                           random_state=42, verbose=-1)
lgbm_sm.fit(X_sm, y_sm)
lgbm_sm_pred = lgbm_sm.predict(X_test_proc)
lgbm_sm_prob = lgbm_sm.predict_proba(X_test_proc)[:, 1]
row_sm = metrics_row("LightGBM (SMOTE)", y_test, lgbm_sm_pred, lgbm_sm_prob)
all_results.append(row_sm)
print(f"  F1={row_sm['F1']:.4f}  ROC-AUC={row_sm['ROC_AUC']:.4f}")

# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# PHẦN 10 — CROSS-VALIDATION & MODEL COMPARISON
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
print_section("10. CROSS-VALIDATION & MODEL COMPARISON")

cv_outer = StratifiedKFold(n_splits=5, shuffle=False)

CV_MODELS = {
    "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree"       : DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    "XGBoost (tuned)"     : best_xgb,
    "LightGBM (SMOTE)"    : lgbm_sm,
}

cv_results = []
for name, model in CV_MODELS.items():
    X_fit = X_sm if name == "LightGBM (SMOTE)" else X_train_proc
    y_fit = y_sm if name == "LightGBM (SMOTE)" else y_train

    scores = cross_validate(
        model, X_fit, y_fit,
        cv=cv_outer,
        scoring=["accuracy", "f1", "roc_auc"],
        n_jobs=-1,
    )
    cv_results.append({
        "Model"        : name,
        "CV Accuracy"  : round(scores["test_accuracy"].mean(), 4),
        "CV F1"        : round(scores["test_f1"].mean(), 4),
        "CV ROC-AUC"   : round(scores["test_roc_auc"].mean(), 4),
        "CV ROC-AUC std": round(scores["test_roc_auc"].std(), 4),
    })

cv_df = pd.DataFrame(cv_results).set_index("Model")
print("\n  Cross-Validation Results (5-fold):")
print(cv_df.to_string())

# ── Final Model Comparison Plot ────────────
fig, ax = plt.subplots(figsize=(10, 5))
cv_df["CV ROC-AUC"].sort_values().plot(kind="barh", ax=ax, color="#4e79a7", xerr=cv_df["CV ROC-AUC std"])
ax.set_title("CV ROC-AUC — So sánh các Model", fontsize=13, fontweight="bold")
ax.set_xlabel("ROC-AUC (mean ± std)")
ax.axvline(0.5, ls="--", color="red", alpha=0.5, label="Random baseline")
ax.legend()
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/04_cv_model_comparison.png", dpi=130, bbox_inches="tight")
plt.close()
print("\n  Saved: 04_cv_model_comparison.png")

# ── Bảng tổng hợp toàn bộ experiment ──────
print("\n  Bảng tổng hợp toàn bộ models:")
final_df = pd.DataFrame(all_results).set_index("Model")
print(final_df.to_string())

# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# PHẦN 11 — EXPORT BEST MODEL + PIPELINE
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
print_section("11. EXPORT BEST MODEL & PREPROCESSING PIPELINE")

# Chọn model tốt nhất theo CV ROC-AUC
best_model_name = cv_df["CV ROC-AUC"].idxmax()
print(f"\n  🏆  Best model: {best_model_name}  (CV ROC-AUC = {cv_df.loc[best_model_name, 'CV ROC-AUC']:.4f})")

best_model_obj = CV_MODELS[best_model_name]

# ── Export model ───────────────────────────
model_path = "/mnt/user-data/outputs/best_model.pkl"
joblib.dump(best_model_obj, model_path)
print(f"  ✅  best_model.pkl saved → {model_path}")

# ── Export preprocessing pipeline ─────────
pipeline_path = "/mnt/user-data/outputs/preprocessing_pipeline.pkl"
joblib.dump(preprocessor, pipeline_path)
print(f"  ✅  preprocessing_pipeline.pkl saved → {pipeline_path}")

# ── Export full inference pipeline ─────────
# Nếu model tốt nhất KHÔNG dùng SMOTE, ta có thể đóng gói thành 1 Pipeline
if "SMOTE" not in best_model_name:
    inference_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model",        best_model_obj),
    ])
    infer_path = "/mnt/user-data/outputs/inference_pipeline.pkl"
    joblib.dump(inference_pipeline, infer_path)
    print(f"  ✅  inference_pipeline.pkl (preprocessor + model) saved → {infer_path}")
else:
    print("  ℹ️  SMOTE model không thể đóng gói chung với preprocessor")
    print("       Dùng preprocessing_pipeline.pkl + best_model.pkl riêng biệt khi inference")

# ── Export results summary ─────────────────
summary_path = "/mnt/user-data/outputs/model_results_summary.csv"
final_df.to_csv(summary_path)
print(f"  ✅  model_results_summary.csv saved → {summary_path}")

# ── Optimal Threshold (Youden's J) ────────
print("\n  [Threshold Optimization — Youden's J = max(Sensitivity + Specificity - 1)]")

# Lấy prob của best model trên test set
best_prob_test = best_model_obj.predict_proba(
    X_sm if "SMOTE" in best_model_name else X_test_proc
)[:, 1]

# Với SMOTE model, cần predict trên X_test_proc (không phải X_sm)
best_prob_test = best_model_obj.predict_proba(X_test_proc)[:, 1]

fpr_t, tpr_t, thresholds_t = roc_curve(y_test, best_prob_test)

# Youden's J statistic: tối đa hoá (TPR - FPR) = (Sensitivity + Specificity - 1)
youdens_j      = tpr_t - fpr_t
optimal_idx    = np.argmax(youdens_j)
optimal_threshold = float(thresholds_t[optimal_idx])

print(f"  Threshold tối ưu (Youden's J) : {optimal_threshold:.4f}")
print(f"    → TPR (Recall) tại ngưỡng này : {tpr_t[optimal_idx]:.4f}")
print(f"    → FPR (1-Specificity)          : {fpr_t[optimal_idx]:.4f}")
print(f"    → Youden's J                   : {youdens_j[optimal_idx]:.4f}")

# Áp dụng threshold tối ưu và tính lại metrics
best_pred_opt = (best_prob_test >= optimal_threshold).astype(int)
print(f"\n  Metrics với default threshold=0.5:")
print(f"    F1={f1_score(y_test, best_model_obj.predict(X_test_proc), zero_division=0):.4f}  "
      f"Recall={recall_score(y_test, best_model_obj.predict(X_test_proc), zero_division=0):.4f}  "
      f"Precision={precision_score(y_test, best_model_obj.predict(X_test_proc), zero_division=0):.4f}")
print(f"  Metrics với optimal threshold={optimal_threshold:.4f}:")
print(f"    F1={f1_score(y_test, best_pred_opt, zero_division=0):.4f}  "
      f"Recall={recall_score(y_test, best_pred_opt, zero_division=0):.4f}  "
      f"Precision={precision_score(y_test, best_pred_opt, zero_division=0):.4f}")

# Plot Youden's J curve
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- subplot 1: ROC + optimal point ---
ax = axes[0]
ax.plot(fpr_t, tpr_t, color="#4e79a7", lw=2,
        label=f"ROC (AUC={roc_auc_score(y_test, best_prob_test):.3f})")
ax.scatter(fpr_t[optimal_idx], tpr_t[optimal_idx],
           color="crimson", s=120, zorder=5,
           label=f"Optimal threshold={optimal_threshold:.3f}")
ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.5)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC Curve — {best_model_name}", fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# --- subplot 2: Youden's J vs threshold ---
ax2 = axes[1]
ax2.plot(thresholds_t, youdens_j, color="#59a14f", lw=2, label="Youden's J (TPR - FPR)")
ax2.axvline(optimal_threshold, color="crimson", ls="--", lw=1.5,
            label=f"Optimal = {optimal_threshold:.3f}")
ax2.set_xlabel("Threshold"); ax2.set_ylabel("Youden's J")
ax2.set_title("Threshold Optimization", fontweight="bold")
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

plt.suptitle("Optimal Classification Threshold", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/05_optimal_threshold.png", dpi=130, bbox_inches="tight")
plt.close()
print("  Saved: 05_optimal_threshold.png")

# Export threshold
threshold_data = {
    "optimal_threshold": optimal_threshold,
    "method"           : "Youden's J (max TPR - FPR)",
    "best_model"       : best_model_name,
    "tpr_at_threshold" : float(tpr_t[optimal_idx]),
    "fpr_at_threshold" : float(fpr_t[optimal_idx]),
    "youdens_j"        : float(youdens_j[optimal_idx]),
}
threshold_path = "/mnt/user-data/outputs/optimal_threshold.pkl"
joblib.dump(threshold_data, threshold_path)
print(f"  ✅  optimal_threshold.pkl saved → {threshold_path}")
print(f"      Content: {threshold_data}")

# ── Hướng dẫn load & dự đoán ──────────────
print("""
  ═══════════════════════════════════════════════════════════
  CÁCH LOAD & DỰ ĐOÁN VỚI OPTIMAL THRESHOLD:

  import joblib, pandas as pd

  preproc   = joblib.load("preprocessing_pipeline.pkl")
  model     = joblib.load("best_model.pkl")
  threshold = joblib.load("optimal_threshold.pkl")["optimal_threshold"]

  X_new = preproc.transform(new_df)
  prob  = model.predict_proba(X_new)[:, 1]
  pred  = (prob >= threshold).astype(int)   # dùng optimal threshold
  ═══════════════════════════════════════════════════════════
""")

print_section("HOÀN THÀNH")
print("  Output files:")
for f in ["01_confusion_matrices.png","02_roc_curves.png","03_feature_importance.png",
          "04_cv_model_comparison.png","05_optimal_threshold.png",
          "best_model.pkl","preprocessing_pipeline.pkl",
          "optimal_threshold.pkl","model_results_summary.csv"]:
    full = f"/mnt/user-data/outputs/{f}"
    size = os.path.getsize(full) if os.path.exists(full) else 0
    print(f"    {'✅' if size else '❌'}  {f}  ({size:,} bytes)")
