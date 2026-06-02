import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import timedelta
import joblib
import plotly.express as px

# Ứng dụng Streamlit: Dự đoán khách quay lại trong 30 ngày (return_30d)
# - Input: CSV upload với cột `customer_id`, `order_datetime`
# - Lịch sử: Data/processed_data.csv (gồm các giao dịch/chi tiết đơn)
# - Model: Model/best_model.pkl và Model/preprocessing_pipeline.pkl

PROCESSED_PATH = os.path.join("Data", "processed_data.csv")
MODEL_PATH = os.path.join("Model", "best_model.pkl")
PREPROC_PATH = os.path.join("Model", "preprocessing_pipeline.pkl")
THRESHOLD_PATH = os.path.join("Model", "optimal_threshold.pkl")

# Các feature cố định theo thứ tự model đã train
MODEL_FEATURES = [
    "days_since_last_order",
    "total_orders",
    "orders_last_30d",
    "avg_days_between_orders",
    "avg_order_value",
    "spending_last_30d",
    "voucher_usage_rate",
    "recent_activity_drop",
    "unique_products",
    "weekend_order_ratio",
    "max_days_between_orders",
    "favorite_channel",
]


def load_processed_data(path=PROCESSED_PATH):
    """Tải dữ liệu lịch sử và chuyển cột thời gian sang datetime nếu có"""
    if not os.path.exists(path):
        st.error(f"Không tìm thấy file lịch sử: {path}")
        return None
    df = pd.read_csv(path)
    if "customer_id" not in df.columns:
        st.error("File processed_data.csv phải có cột 'customer_id'.")
        return None
    if "order_datetime" in df.columns:
        df["order_datetime"] = pd.to_datetime(df["order_datetime"], errors="coerce")
    else:
        df["order_datetime"] = pd.NaT
    return df


def first_existing(col_candidates, df):
    for c in col_candidates:
        if c in df.columns:
            return c
    return None


def compute_features_for_customer(customer_id, ref_date, hist_df):
    """Tạo feature theo mô tả cho 1 khách dựa trên history trước ref_date."""
    cust = hist_df[hist_df["customer_id"] == customer_id].copy()
    # chỉ dùng lịch sử trước ref_date
    if "order_datetime" in cust.columns:
        cust = cust[cust["order_datetime"] < ref_date]
    else:
        cust = cust.iloc[0:0]

    # columns candidates
    order_id_col = first_existing(["transaction_id", "order_id", "order_number", "invoice_no"], hist_df)
    qty_col = first_existing(["quantity", "qty", "quantity_ordered"], hist_df)
    price_col = first_existing(["unit_price", "price", "order_value", "amount"], hist_df)
    product_col = first_existing(["product_id", "sku", "item_id"], hist_df)
    voucher_col = first_existing(["voucher_used", "voucher", "coupon_used"], hist_df)
    channel_col = first_existing(["channel", "favorite_channel", "sales_channel"], hist_df)

    # nếu không có order_id, coi mỗi hàng là 1 order
    if order_id_col is None:
        cust["_order_id_tmp"] = np.arange(len(cust))
        order_id_col = "_order_id_tmp"

    total_orders = int(cust[order_id_col].nunique()) if not cust.empty else 0

    # days_since_last_order
    if cust.shape[0] == 0:
        days_since_last = np.nan
    else:
        last_dt = cust["order_datetime"].max() if "order_datetime" in cust.columns else pd.NaT
        days_since_last = (ref_date - last_dt).days if pd.notna(last_dt) else np.nan

    # orders_last_30d
    window_start = ref_date - timedelta(days=30)
    if "order_datetime" in cust.columns:
        orders_last_30d = int(cust[(cust["order_datetime"] >= window_start) & (cust["order_datetime"] < ref_date)][order_id_col].nunique())
    else:
        orders_last_30d = 0

    # avg_days_between_orders & max_days_between_orders
    if "order_datetime" in cust.columns and cust[order_id_col].nunique() > 1:
        order_dates = cust.groupby(order_id_col)["order_datetime"].min().sort_values()
        diffs = order_dates.diff().dt.days.dropna()
        avg_days_between = float(diffs.mean())
        max_days_between = float(diffs.max())
    else:
        avg_days_between = np.nan
        max_days_between = np.nan

    # avg_order_value & spending_last_30d
    if qty_col and price_col and not cust.empty:
        cust["_line_value"] = cust[qty_col].fillna(0) * cust[price_col].fillna(0)
        order_vals = cust.groupby(order_id_col)["_line_value"].sum().rename("order_total").reset_index()
        avg_order_value = float(order_vals["order_total"].mean()) if not order_vals.empty else np.nan
        if "order_datetime" in cust.columns:
            orders_in_30 = cust[(cust["order_datetime"] >= window_start) & (cust["order_datetime"] < ref_date)]
            spending_last_30d = float((orders_in_30[qty_col].fillna(0) * orders_in_30[price_col].fillna(0)).sum())
        else:
            spending_last_30d = 0.0
    else:
        avg_order_value = np.nan
        spending_last_30d = 0.0

    # voucher_usage_rate
    if voucher_col is not None and total_orders > 0:
        try:
            voucher_cnt = cust.drop_duplicates(order_id_col).set_index(order_id_col)[voucher_col].astype(float).fillna(0).sum()
            voucher_usage_rate = float(voucher_cnt / total_orders)
        except Exception:
            voucher_usage_rate = np.nan
    else:
        voucher_usage_rate = np.nan

    # recent_activity_drop (orders_last_30d / avg_monthly_orders_previous_3_months)
    prev_start = ref_date - timedelta(days=120)
    prev_end = ref_date - timedelta(days=30)
    if "order_datetime" in cust.columns:
        prev_orders = cust[(cust["order_datetime"] >= prev_start) & (cust["order_datetime"] < prev_end)][order_id_col].nunique()
        avg_monthly_orders_previous_3_months = prev_orders / 3.0
        if avg_monthly_orders_previous_3_months > 0:
            recent_activity_drop = float(orders_last_30d / avg_monthly_orders_previous_3_months)
        else:
            recent_activity_drop = 0.0
    else:
        recent_activity_drop = 0.0

    # unique_products
    unique_products = int(cust[product_col].nunique()) if (product_col is not None and not cust.empty) else np.nan

    # weekend_order_ratio
    if "order_datetime" in cust.columns and not cust.empty:
        weekend_ratio = float(cust.drop_duplicates(order_id_col)["order_datetime"].dt.weekday.isin([5, 6]).mean())
    else:
        weekend_ratio = np.nan

    # favorite_channel
    favorite_channel = cust[channel_col].mode().iloc[0] if (channel_col in cust.columns and not cust[channel_col].mode().empty) else np.nan

    return {
        "days_since_last_order": days_since_last,
        "total_orders": total_orders,
        "orders_last_30d": orders_last_30d,
        "avg_days_between_orders": avg_days_between,
        "avg_order_value": avg_order_value,
        "spending_last_30d": spending_last_30d,
        "voucher_usage_rate": voucher_usage_rate,
        "recent_activity_drop": recent_activity_drop,
        "unique_products": unique_products,
        "weekend_order_ratio": weekend_ratio,
        "max_days_between_orders": max_days_between,
        "favorite_channel": favorite_channel,
    }


def compute_features(upload_df: pd.DataFrame, hist_df: pd.DataFrame):
    rows = []
    for _, r in upload_df.iterrows():
        cid = r.get("customer_id")
        ref = r.get("order_datetime")
        if pd.isna(cid):
            continue
        try:
            ref_date = pd.to_datetime(ref)
        except Exception:
            ref_date = pd.Timestamp.now()
        feats = compute_features_for_customer(cid, ref_date, hist_df)
        row = {"customer_id": cid, "order_datetime": ref_date}
        row.update(feats)
        rows.append(row)
    return pd.DataFrame(rows)


def load_model_and_preprocessor():
    """Load model và preprocessor; raise error nếu thiếu để hiển thị rõ ràng"""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROC_PATH):
        raise FileNotFoundError(f"Thiếu file model/preprocessor trong Model/: {MODEL_PATH} hoặc {PREPROC_PATH}")
    model = joblib.load(MODEL_PATH)
    preproc = joblib.load(PREPROC_PATH)
    return model, preproc


def load_threshold(path=THRESHOLD_PATH, default=0.3):
    """Load optimal threshold nếu có; hỗ trợ int/float, dict, list/tuple. Fallback về default với warning."""
    default_threshold = float(default)

    if not os.path.exists(path):
        st.warning(f"Không tìm thấy optimal_threshold.pkl tại {path}. Sử dụng ngưỡng mặc định {default_threshold}.")
        return default_threshold

    try:
        threshold_obj = joblib.load(path)

        # trực tiếp nếu là số
        if isinstance(threshold_obj, (int, float)):
            return float(threshold_obj)

        # nếu là dict, ưu tiên một số key rồi fallback qua giá trị số đầu tiên
        if isinstance(threshold_obj, dict):
            for key in ["threshold", "optimal_threshold", "best_threshold"]:
                if key in threshold_obj:
                    try:
                        return float(threshold_obj[key])
                    except Exception:
                        continue
            for value in threshold_obj.values():
                try:
                    return float(value)
                except Exception:
                    continue

        # nếu là list/tuple, thử lấy phần tử đầu
        if isinstance(threshold_obj, (list, tuple)) and len(threshold_obj) > 0:
            try:
                return float(threshold_obj[0])
            except Exception:
                pass

        st.warning(f"Không đọc được giá trị threshold từ {path}. Sử dụng ngưỡng mặc định {default_threshold}.")
        return default_threshold

    except Exception as e:
        st.warning(f"Lỗi khi load optimal_threshold.pkl: {e}. Sử dụng ngưỡng mặc định {default_threshold}.")
        return default_threshold


def predict_return_probability(feature_df: pd.DataFrame, model, preproc):
    """Dự đoán return_probability bằng model.

    Lưu ý: risk_group được gán cố định theo BR-03 (không dùng optimal_threshold.pkl).
    """
    df = feature_df.copy()
    # Sử dụng bộ feature cố định cùng thứ tự với mô hình đã train
    X = preproc.transform(df[MODEL_FEATURES])
    probs = model.predict_proba(X)[:, 1]
    df["return_probability"] = probs

    # Risk group BR-03 (cố định, không phụ thuộc optimal threshold)
    def rg(p):
        if p < 0.3:
            return "High Risk"
        elif p < 0.7:
            return "Medium"
        else:
            return "Loyal"

    df["risk_group"] = df["return_probability"].apply(rg)

    # recommendation ngắn
    def rec(row):
        if row["risk_group"] == "High Risk":
            return "Gửi ưu đãi lớn, gọi CSKH"
        if row["risk_group"] == "Medium":
            return "Gửi email nhắc nhở + coupon nhỏ"
        return "Duy trì chương trình loyalty"

    df["recommendation"] = df.apply(rec, axis=1)
    return df


def main():
    st.set_page_config(page_title="Return30d Prediction", layout="wide")
    st.title("Hệ thống dự đoán khả năng khách hàng quay lại mua hàng ít nhất trong 30 ngày tới")

    st.markdown("""
    ### Quy trình dự đoán

    Người dùng chỉ cần tải lên file CSV chứa customer_id và order_datetime.

    Hệ thống sẽ tự động tra cứu lịch sử giao dịch, xây dựng bộ đặc trưng khách hàng, thực hiện tiền xử lý dữ liệu và sử dụng mô hình Machine Learning để dự đoán xác suất khách hàng quay lại trong 30 ngày tiếp theo.

    Kết quả bao gồm:
    - Return Probability
    - Risk Group
    - Dự đoán quay lại / không quay lại
    - Báo cáo chi tiết có thể tải xuống
    """)

    uploaded = st.file_uploader("Upload CSV (customer_id, order_datetime)", type=["csv"])
    show_raw = st.checkbox("Hiện raw upload")

    if uploaded is None:
        st.info("Vui lòng upload file CSV để bắt đầu.")
        return

    try:
        upload_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Không đọc được file upload: {e}")
        return

    if upload_df.shape[0] == 0:
        st.error("File upload rỗng.")
        return

    if not {"customer_id", "order_datetime"}.issubset(set(upload_df.columns)):
        st.error("File upload thiếu cột bắt buộc: customer_id hoặc order_datetime")
        return

    if show_raw:
        st.subheader("Raw uploaded data")
        st.dataframe(upload_df)

    hist_df = load_processed_data()
    if hist_df is None:
        return

    with st.spinner("Tạo feature..."):
        feat_df = compute_features(upload_df, hist_df)
    st.success("Tạo feature xong")

    # Xác định khách không có lịch sử (sẽ hiển thị warning sau bảng báo cáo)
    missing = feat_df[feat_df["total_orders"] == 0]

    # load model và preproc
    try:
        model, preproc = load_model_and_preprocessor()
    except Exception as e:
        st.error(f"Không load được model/preprocessor: {e}")
        return

    # load optimal threshold để tạo prediction_label (fallback bên trong load_threshold)
    threshold = load_threshold()

    try:
        result = predict_return_probability(feat_df, model, preproc)
    except Exception as e:
        st.error(f"Lỗi khi predict: {e}")
        return

    # tạo prediction_label dựa trên threshold (return_probability -> Quay lại / Không quay lại)
    result["prediction_label"] = result["return_probability"].apply(lambda p: "Quay lại" if p >= threshold else "Không quay lại")

    # KPI
    total = len(result)
    high = (result["risk_group"] == "High Risk").sum()

    k1, k2, k3 = st.columns(3)
    k1.metric("Tổng khách dự đoán", total)
    k2.metric("Số khách High Risk", int(high))
    k3.metric("% High Risk", f"{(high/total*100 if total else 0):.1f}%")

    st.subheader("Phân bố nhóm rủi ro")
    st.plotly_chart(px.pie(result, names="risk_group", title="Risk Group Distribution"), use_container_width=True)

    st.subheader("Top khách có return_probability thấp nhất")
    low = result.sort_values("return_probability", ascending=True).head(20)
    st.plotly_chart(px.bar(low, x="customer_id", y="return_probability", color="risk_group", title="Lowest return_probability"), use_container_width=True)

    st.subheader("Báo cáo chi tiết")
    disp = result.copy()
    disp["order_datetime"] = disp["order_datetime"].astype(str)
    st.dataframe(disp)

    csv = disp.to_csv(index=False).encode("utf-8")
    st.download_button("Tải báo cáo CSV", data=csv, file_name="return30d_report.csv", mime="text/csv")

    # Hiển thị cảnh báo missing sau khi người dùng đã thấy báo cáo và có thể tải xuống
    if not missing.empty:
        st.warning(f"Có {len(missing)} khách không tồn tại trong dữ liệu lịch sử hoặc chưa có giao dịch trước snapshot date.")


if __name__ == "__main__":
    main()
