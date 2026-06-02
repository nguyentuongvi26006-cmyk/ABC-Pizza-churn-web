import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import timedelta
import joblib
import plotly.express as px

# Ứng dụng Streamlit: Dự đoán khách quay lại trong 30 ngày (return_30d)
# - Input: CSV upload với cột `customer_id`, `order_datetime`
# - Lịch sử: Data/processed_data.csv
# - Model: Model/best_model.pkl và Model/preprocessing_pipeline.pkl
# - Output: báo cáo dự đoán `return_probability`, `risk_group`, bảng và biểu đồ

PROCESSED_PATH = os.path.join("Data", "processed_data.csv")
MODEL_PATH = os.path.join("Model", "best_model.pkl")
PREPROC_PATH = os.path.join("Model", "preprocessing_pipeline.pkl")
THRESHOLD_PATH = os.path.join("Model", "optimal_threshold.pkl")


def load_processed_data(path=PROCESSED_PATH):
    """Tải dữ liệu lịch sử và chuẩn hoá cột thời gian"""
    if not os.path.exists(path):
        st.error(f"Không tìm thấy file lịch sử: {path}")
        return None
    df = pd.read_csv(path)
    if "customer_id" not in df.columns:
        st.error("File processed_data.csv phải có cột 'customer_id'.")
        return None
    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    # chuẩn hoá tên cột thời gian nếu có
    if "order_datetime" in df.columns:
        df["order_datetime"] = pd.to_datetime(df["order_datetime"], errors="coerce")
    else:
        df["order_datetime"] = pd.NaT
    return df


def safe_first_existing(col_candidates, df):
    for c in col_candidates:
        if c in df.columns:
            return c
    return None


def compute_features_for_customer(customer_id, ref_date, hist_df):
    """Tạo các feature cho 1 khách dựa trên lịch sử trước ref_date.

    Logic được map theo Feature Engineering:
    - total_orders = COUNT(DISTINCT transaction_id)
    - avg_order_value = trung bình giá trị mỗi đơn, với order_value = SUM(quantity * unit_price) hoặc SUM(revenue)
    - spending_last_30d = tổng chi tiêu trong 30 ngày gần nhất
    """
    # chuẩn hoá customer_id để tránh lệch do khoảng trắng/kiểu dữ liệu
    customer_id = str(customer_id).strip()
    cust = hist_df[hist_df["customer_id"].astype(str).str.strip() == customer_id].copy()

    # chỉ lấy giao dịch trước thời điểm tham chiếu
    if "order_datetime" in cust.columns:
        cust["order_datetime"] = pd.to_datetime(cust["order_datetime"], errors="coerce")
        cust = cust[cust["order_datetime"] < ref_date]

    voucher_col = safe_first_existing(["voucher_used", "voucher", "used_voucher", "coupon_used"], hist_df)
    product_col = safe_first_existing(["product_id", "product_name", "sku", "item_id"], hist_df)
    channel_col = safe_first_existing(["channel", "favorite_channel", "sales_channel"], hist_df)

    # tính line_value theo dữ liệu thực tế
    if "revenue" in cust.columns:
        cust["line_value"] = pd.to_numeric(cust["revenue"], errors="coerce").fillna(0)
    elif "quantity" in cust.columns and "unit_price" in cust.columns:
        qty = pd.to_numeric(cust["quantity"], errors="coerce").fillna(0)
        price = pd.to_numeric(cust["unit_price"], errors="coerce").fillna(0)
        cust["line_value"] = qty * price
    else:
        money_col = safe_first_existing(["order_value", "amount", "total", "price", "order_amount"], hist_df)
        if money_col is not None:
            cust["line_value"] = pd.to_numeric(cust[money_col], errors="coerce").fillna(0)
        else:
            cust["line_value"] = 0

    # total_orders theo FE: COUNT(DISTINCT transaction_id)
    if "transaction_id" in cust.columns:
        total_orders = cust["transaction_id"].nunique()
    else:
        total_orders = len(cust)

    if total_orders == 0:
        return {
            "days_since_last_order": 999,
            "total_orders": 0,
            "orders_last_30d": 0,
            "avg_days_between_orders": 0,
            "avg_order_value": 0,
            "spending_last_30d": 0.0,
            "voucher_usage_rate": 0,
            "recent_activity_drop": 0,
            "unique_products": 0,
            "weekend_order_ratio": 0,
            "max_days_between_orders": 0,
            "favorite_channel": "Unknown",
        }

    # tạo bảng order-level để tính đúng theo đơn hàng, tránh 1 transaction nhiều dòng bị đếm lặp
    if "transaction_id" in cust.columns:
        agg_dict = {
            "order_datetime": "min",
            "line_value": "sum",
        }
        if voucher_col is not None:
            agg_dict[voucher_col] = "max"
        if channel_col is not None:
            agg_dict[channel_col] = lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"

        orders = cust.groupby("transaction_id", as_index=False).agg(agg_dict)
    else:
        orders = cust.copy()

    # recency
    last_dt = orders["order_datetime"].max() if "order_datetime" in orders.columns else pd.NaT
    days_since_last = (ref_date - last_dt).days if pd.notna(last_dt) else 999

    # orders in last 30 days
    window_start = ref_date - timedelta(days=30)
    if "order_datetime" in orders.columns:
        orders_last_30d = orders[
            (orders["order_datetime"] >= window_start) &
            (orders["order_datetime"] < ref_date)
        ].shape[0]
    else:
        orders_last_30d = 0

    # avg_days_between_orders và max_days_between_orders theo khoảng cách giữa các đơn
    if "order_datetime" in orders.columns and orders.shape[0] > 1:
        s = orders.sort_values("order_datetime")["order_datetime"]
        diffs = s.diff().dt.days.dropna()
        avg_days_between = diffs.mean() if not diffs.empty else 0
        max_days_between = diffs.max() if not diffs.empty else 0
    else:
        avg_days_between = 0
        max_days_between = 0

    # avg_order_value = trung bình giá trị từng đơn
    avg_order_value = orders["line_value"].mean() if "line_value" in orders.columns else 0

    # spending_last_30d
    if "order_datetime" in orders.columns and "line_value" in orders.columns:
        spending_last_30d = orders.loc[
            (orders["order_datetime"] >= window_start) &
            (orders["order_datetime"] < ref_date),
            "line_value"
        ].sum()
    else:
        spending_last_30d = 0.0

    # voucher usage rate = số đơn có voucher / total_orders
    if voucher_col is not None and voucher_col in orders.columns:
        try:
            voucher_rate = pd.to_numeric(orders[voucher_col], errors="coerce").fillna(0).mean()
        except Exception:
            voucher_rate = 0
    else:
        voucher_rate = 0

    # recent activity drop: so sánh orders_last_30d với 30 ngày trước đó
    prev_start = ref_date - timedelta(days=60)
    prev_end = ref_date - timedelta(days=30)
    if "order_datetime" in orders.columns:
        prev_count = orders[
            (orders["order_datetime"] >= prev_start) &
            (orders["order_datetime"] < prev_end)
        ].shape[0]
    else:
        prev_count = 0
    recent_drop = (prev_count - orders_last_30d) / prev_count if prev_count > 0 else 0

    # unique products
    if product_col is not None and product_col in cust.columns:
        unique_products = cust[product_col].nunique()
    else:
        unique_products = 0

    # weekend order ratio theo order-level
    if "order_datetime" in orders.columns:
        weekend_ratio = orders["order_datetime"].dt.weekday.isin([5, 6]).mean()
    else:
        weekend_ratio = 0

    # favorite channel
    if channel_col is not None and channel_col in cust.columns:
        mode_channel = cust[channel_col].dropna().mode()
        fav = mode_channel.iloc[0] if not mode_channel.empty else "Unknown"
    else:
        fav = "Unknown"

    return {
        "days_since_last_order": days_since_last,
        "total_orders": total_orders,
        "orders_last_30d": orders_last_30d,
        "avg_days_between_orders": avg_days_between,
        "avg_order_value": avg_order_value,
        "spending_last_30d": spending_last_30d,
        "voucher_usage_rate": voucher_rate,
        "recent_activity_drop": recent_drop,
        "unique_products": unique_products,
        "weekend_order_ratio": weekend_ratio,
        "max_days_between_orders": max_days_between,
        "favorite_channel": fav,
    }

def compute_features(upload_df: pd.DataFrame, hist_df: pd.DataFrame):
    """Tạo feature cho tất cả customer trong upload_df bằng cách lookup history"""
    rows = []
    for _, r in upload_df.iterrows():
        cid = r.get("customer_id")
        ref = r.get("order_datetime")
        if pd.isna(cid):
            continue
        if not pd.isna(ref):
            ref = pd.to_datetime(ref)
        else:
            # nếu user không cung cấp thời điểm, mặc định là now
            ref = pd.Timestamp.now()

        feats = compute_features_for_customer(cid, ref, hist_df)
        row = {"customer_id": cid, "order_datetime": ref}
        row.update(feats)
        rows.append(row)

    feat_df = pd.DataFrame(rows)
    return feat_df


def load_model_and_preprocessor(
    model_path=MODEL_PATH,
    preproc_path=PREPROC_PATH,
    threshold_path=THRESHOLD_PATH
):
    model = None
    preproc = None
    threshold = 0.5

    if os.path.exists(model_path) and os.path.exists(preproc_path):
        try:
            model = joblib.load(model_path)
            preproc = joblib.load(preproc_path)

            if os.path.exists(threshold_path):
                threshold_data = joblib.load(threshold_path)

                if isinstance(threshold_data, dict):
                    threshold = threshold_data.get("optimal_threshold", 0.5)
                else:
                    threshold = float(threshold_data)

        except Exception as e:
            st.warning(f"Lỗi khi load model/preprocessor/threshold: {e}")
            model = None
            preproc = None
            threshold = 0.5
    else:
        st.warning("Model hoặc preprocessing pipeline không tìm thấy trong thư mục Model/. Ứng dụng sẽ không dùng model thật.")

    return model, preproc, threshold


def predict_return_probability(feature_df: pd.DataFrame, model, preproc, threshold=0.5):
    """Dùng model + preproc để dự đoán return_probability. Nếu model không có -> fallback heuristic."""
    df = feature_df.copy()
    if model is not None and preproc is not None:
        try:
            X = preproc.transform(df)
            probs = model.predict_proba(X)[:, 1]
            df["return_probability"] = probs
            used_model = True
        except Exception as e:
            st.warning(f"Lỗi khi transform/predict: {e} — dùng heuristic fallback.")
            used_model = False
    else:
        used_model = False

    if not used_model:
        # heuristic fallback: simple score -> probability
        r = df["days_since_last_order"].fillna(999).astype(float)
        f = df["total_orders"].fillna(0).astype(float)
        m = df["spending_last_30d"].fillna(0).astype(float)
        score = -0.02 * r + 0.3 * np.log1p(f) + 0.0005 * m
        prob = 1 / (1 + np.exp(-score))
        df["return_probability"] = prob.clip(0, 1)

    # risk group theo BR-03
    def rg(p):
        if p < 0.3:
            return "High"
        elif p < 0.7:
            return "Medium"
        else:
            return "Loyal"

    df["risk_group"] = df["return_probability"].apply(rg)
    # prediction theo optimal_threshold
    df["prediction_by_threshold"] = df["return_probability"].apply(
    lambda p: "Quay lại" if p >= threshold else "Không quay lại")
    

    # recommendation ngắn
    def rec(row):
        if row["risk_group"] == "High":
            return "Gửi ưu đãi lớn, gọi CSKH"
        if row["risk_group"] == "Medium":
            return "Gửi email nhắc nhở + coupon nhỏ"
        return "Duy trì chương trình loyalty"

    df["recommendation"] = df.apply(rec, axis=1)
    df["model_source"] = "best_model.pkl" if used_model else "heuristic_fallback"
    return df


def main():
    st.set_page_config(page_title="Churn -> Return30d Dashboard", layout="wide")
    st.title("Dự đoán khả năng khách hàng không quay lại mua hàng ít nhất 30 ngày tới")

    st.markdown("""
    Upload CSV với 2 cột: `customer_id`, `order_datetime` (thời điểm muốn dự đoán).
    Ứng dụng sẽ lookup lịch sử trong `Data/processed_data.csv`, tạo feature, dùng model trong `Model/` để dự đoán xác suất quay lại trong 30 ngày.
    """)

    # Sidebar
    st.sidebar.header("Cấu hình")
    show_raw = st.sidebar.checkbox("Hiện raw upload", value=False)

    uploaded = st.file_uploader("Upload CSV (customer_id, order_datetime)", type=["csv"])

    # load lịch sử
    hist_df = load_processed_data()

    if uploaded is None:
        st.info("Vui lòng upload file CSV để bắt đầu.")
        return

    try:
        upload_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Không đọc được file upload: {e}")
        return

    # chuẩn hoá dữ liệu upload
    if "customer_id" in upload_df.columns:
        upload_df["customer_id"] = upload_df["customer_id"].astype(str).str.strip()
    if "order_datetime" in upload_df.columns:
        upload_df["order_datetime"] = pd.to_datetime(upload_df["order_datetime"], errors="coerce")

    # kiểm tra cột
    required = {"customer_id", "order_datetime"}
    if not required.issubset(set(upload_df.columns)):
        st.error(f"File upload thiếu cột bắt buộc: {', '.join(required - set(upload_df.columns))}")
        return

    if upload_df.shape[0] == 0:
        st.error("File upload rỗng.")
        return

    if show_raw:
        st.subheader("Raw uploaded data")
        st.dataframe(upload_df)

    if hist_df is None:
        st.error("Không có dữ liệu lịch sử để lookup. Kiểm tra Data/processed_data.csv")
        return

    # tạo feature
    with st.spinner("Tạo feature từ lịch sử..."):
        feat_df = compute_features(upload_df, hist_df)

    st.success("Tạo feature xong")

    # load model & preproc
    model, preproc, threshold = load_model_and_preprocessor()

    # dự đoán
    result = predict_return_probability(feat_df, model, preproc, threshold)

    # KPI
    total = len(result)
    high = (result["risk_group"] == "High").sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Tổng khách dự đoán", total)
    c2.metric("Số khách High Risk", int(high))
    c3.metric("% High Risk", f"{(high/total*100 if total else 0):.1f}%")

    # Pie chart
    st.subheader("Phân bố nhóm rủi ro")
    pie = px.pie(result, names="risk_group", title="Risk Group Distribution")
    st.plotly_chart(pie, use_container_width=True)

    # Bar chart: top khách có return_probability thấp nhất
    st.subheader("Top khách có return_probability thấp nhất")
    low = result.sort_values("return_probability", ascending=True).head(20)
    bar = px.bar(low, x="customer_id", y="return_probability", color="risk_group", title="Lowest return_probability")
    st.plotly_chart(bar, use_container_width=True)

    # Table
    st.subheader("Báo cáo chi tiết")
    df_display = result.copy()
    df_display["order_datetime"] = df_display["order_datetime"].astype(str)
    st.dataframe(df_display)

    csv = df_display.to_csv(index=False).encode("utf-8")
    st.download_button("Tải báo cáo CSV", data=csv, file_name="return30d_report.csv", mime="text/csv")

    # cảnh báo nếu có customer không có lịch sử
    missing_hist = feat_df[feat_df["total_orders"] == 0]
    if not missing_hist.empty:
        st.warning(f"Có {len(missing_hist)} khách không có lịch sử trong Data/processed_data.csv — các feature được gán mặc định.")


if __name__ == "__main__":
    main()
