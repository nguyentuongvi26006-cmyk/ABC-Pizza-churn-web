import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import plotly.express as px

# -----------------------------
# App: Churn prediction placeholder
# Mục tiêu: Dự đoán khả năng khách hàng KHÔNG quay lại mua hàng trong 30 ngày
# - User upload CSV gồm `customer_id` và `order_datetime`
# - Lookup lịch sử mua hàng trong `data/processed_data.csv`
# - Tạo feature tạm: recency, frequency, monetary
# - Placeholder `predict_churn()` để thay bằng model sau này
# File này chứa cả backend + frontend
# -----------------------------

PROCESSED_DATA_PATH = os.path.join("data", "processed_data.csv")


def load_processed_data(path=PROCESSED_DATA_PATH):
    """Tải file processed_data.csv từ thư mục Data.
    Kỳ vọng file có tối thiểu: customer_id, order_datetime. Có thể có 'order_value'/'amount'...
    """
    if not os.path.exists(path):
        st.error(f"File dữ liệu lịch sử không tìm thấy: {path}. Vui lòng đặt file vào thư mục Data/")
        return None
    df = pd.read_csv(path)
    # Chuẩn hoá tên cột
    if "customer_id" not in df.columns:
        st.error("File processed_data.csv cần cột 'customer_id'.")
        return None
    # chuyển order_datetime sang kiểu datetime nếu có
    if "order_datetime" in df.columns:
        df["order_datetime"] = pd.to_datetime(df["order_datetime"], errors="coerce")
    else:
        st.warning("File processed_data.csv không có cột 'order_datetime'. Tính toán recency/frequency sẽ giới hạn.")
        df["order_datetime"] = pd.NaT

    return df


def validate_upload(uploaded_df: pd.DataFrame):
    """Kiểm tra file upload có chứa customer_id và order_datetime"""
    if uploaded_df is None:
        return False, "File rỗng hoặc không đọc được"
    required = {"customer_id", "order_datetime"}
    missing = required - set(uploaded_df.columns)
    if missing:
        return False, f"File upload thiếu cột: {', '.join(missing)}"
    # parse datetimes
    uploaded_df["order_datetime"] = pd.to_datetime(uploaded_df["order_datetime"], errors="coerce")
    if uploaded_df["order_datetime"].isna().all():
        return False, "Không thể chuyển 'order_datetime' sang datetime. Kiểm tra định dạng."
    return True, ""


def compute_features_for_customer(customer_id, ref_date, processed_df):
    """Tạo các feature tạm cho 1 khách hàng, dựa trên lịch sử trước ref_date.
    Trả về dict có recency (ngày), frequency (số đơn), monetary (tổng hoặc trung bình)
    """
    # lọc lịch sử trước ref_date
    cust_hist = processed_df[processed_df["customer_id"] == customer_id].copy()
    if "order_datetime" in cust_hist.columns:
        cust_hist = cust_hist[cust_hist["order_datetime"] < ref_date]
    else:
        # nếu không có thời gian trong lịch sử, coi như không có lịch sử
        cust_hist = cust_hist.iloc[0:0]

    # frequency: số đơn trước ref_date
    frequency = len(cust_hist)

    # recency: số ngày kể từ đơn cuối đến ref_date
    if frequency == 0:
        recency = np.nan
    else:
        last_dt = cust_hist["order_datetime"].max()
        if pd.isna(last_dt):
            recency = np.nan
        else:
            recency = (ref_date - last_dt).days

    # monetary: tìm cột amount/order_value/total
    monetary_cols = [c for c in cust_hist.columns if c.lower() in ("order_value", "amount", "total", "price", "order_amount")]
    if monetary_cols:
        # lấy tổng tiền (hoặc trung bình nếu muốn)
        monetary = cust_hist[monetary_cols[0]].sum()
    else:
        monetary = np.nan

    return {"recency": recency, "frequency": frequency, "monetary": monetary}


def compute_features(upload_df: pd.DataFrame, processed_df: pd.DataFrame):
    """Tạo feature cho tất cả khách trong upload_df bằng cách lookup processed_df"""
    features = []
    for _, row in upload_df.iterrows():
        cid = row["customer_id"]
        ref = row["order_datetime"]
        if pd.isna(ref):
            features.append({"customer_id": cid, "order_datetime": ref, "recency": np.nan, "frequency": 0, "monetary": np.nan})
            continue
        feat = compute_features_for_customer(cid, ref, processed_df)
        features.append({"customer_id": cid, "order_datetime": ref, **feat})
    feat_df = pd.DataFrame(features)
    return feat_df


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict_churn(features_df: pd.DataFrame):
    """Placeholder predict_churn: trả về churn_probability.
    Công thức đơn giản (tạm): recency tăng -> churn tăng; frequency tăng -> churn giảm; monetary giảm -> churn tăng
    Khi có best_model.pkl và preprocessor.pkl, sẽ load model ở đây.
    """
    df = features_df.copy()
    # xử lý NaN: gán giá trị mặc định
    # recency: nếu NaN -> lớn (ví dụ 999 ngày)
    df["recency_filled"] = df["recency"].fillna(999)
    df["frequency_filled"] = df["frequency"].fillna(0)
    df["monetary_filled"] = df["monetary"].fillna(0)

    # scale đơn giản
    r = df["recency_filled"].astype(float)
    f = df["frequency_filled"].astype(float)
    m = df["monetary_filled"].astype(float)

    # công thức heuristic để tạo probability (chỉ là placeholder)
    score = 0.02 * r - 0.3 * np.log1p(f) - 0.0005 * m
    prob = sigmoid(score)
    df["churn_probability"] = prob.clip(0, 1)

    # nhóm rủi ro
    def risk_group(p):
        if p >= 0.66:
            return "High"
        elif p >= 0.33:
            return "Medium"
        else:
            return "Low"

    df["risk_group"] = df["churn_probability"].apply(risk_group)

    # recommendation
    def recommend(row):
        if row["risk_group"] == "High":
            return "Gửi ưu đãi lớn, gọi CSKH, chương trình khuyến mãi cá nhân hoá"
        if row["risk_group"] == "Medium":
            return "Gửi email nhắc nhở + coupon nhỏ"
        return "Giữ liên lạc định kỳ, ưu đãi nhẹ"

    df["recommendation"] = df.apply(recommend, axis=1)

    # giữ chỉ các cột cần thiết
    out = df[["customer_id", "order_datetime", "recency", "frequency", "monetary", "churn_probability", "risk_group", "recommendation"]]
    return out


def main():
    st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")
    st.title("Dự đoán khách không quay lại trong 30 ngày (placeholder)")

    st.markdown("""
    Hướng dẫn nhanh:
    - Upload file CSV chứa `customer_id` và `order_datetime` (mốc thời gian muốn dự đoán)
    - Ứng dụng sẽ lookup lịch sử từ `Data/processed_data.csv` để tạo feature tạm
    - Hiện tại dùng hàm `predict_churn()` placeholder; khi có `best_model.pkl` sẽ tích hợp sau
    """)

    # Sidebar
    st.sidebar.header("Cấu hình")
    show_raw = st.sidebar.checkbox("Hiển thị raw upload", value=False)

    uploaded_file = st.file_uploader("Upload CSV (customer_id, order_datetime)", type=["csv"])

    # load processed data once
    processed_df = None
    if os.path.exists(PROCESSED_DATA_PATH):
        processed_df = load_processed_data(PROCESSED_DATA_PATH)

    if uploaded_file is not None:
        try:
            upload_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Không đọc được file upload: {e}")
            return

        ok, msg = validate_upload(upload_df)
        if not ok:
            st.error(msg)
            return

        if show_raw:
            st.subheader("Raw uploaded data")
            st.dataframe(upload_df)

        if processed_df is None:
            st.error(f"Không tìm thấy hoặc không thể load {PROCESSED_DATA_PATH}. Vui lòng cung cấp file lịch sử.")
            return

        with st.spinner("Tạo feature từ lịch sử..."):
            feat_df = compute_features(upload_df, processed_df)

        st.success("Tạo feature xong")

        # predict
        result_df = predict_churn(feat_df)

        # KPI
        total_customers = len(result_df)
        high_risk = (result_df["risk_group"] == "High").sum()

        k1, k2, k3 = st.columns(3)
        k1.metric("Tổng khách dự đoán", total_customers)
        k2.metric("Số khách High Risk", int(high_risk))
        k3.metric("Tỷ lệ High Risk", f"{(high_risk/total_customers*100 if total_customers else 0):.1f}%")

        # Pie chart risk groups
        st.subheader("Tỷ lệ nhóm rủi ro")
        pie = px.pie(result_df, names="risk_group", title="Risk Group Distribution")
        st.plotly_chart(pie, use_container_width=True)

        # Bar chart top churn_probability
        st.subheader("Top khách có churn_probability cao nhất")
        topk = result_df.sort_values("churn_probability", ascending=False).head(20)
        bar = px.bar(topk, x="customer_id", y="churn_probability", color="risk_group", title="Top churn probability")
        st.plotly_chart(bar, use_container_width=True)

        # Detailed table
        st.subheader("Bảng báo cáo chi tiết")
        # hiển thị với format
        display_df = result_df.copy()
        display_df["order_datetime"] = display_df["order_datetime"].astype(str)
        st.dataframe(display_df)

        # download link
        csv = display_df.to_csv(index=False).encode("utf-8")
        st.download_button("Tải báo cáo CSV", data=csv, file_name="churn_report.csv", mime="text/csv")

    else:
        st.info("Vui lòng upload file CSV để chạy báo cáo.")

    # Footer: hướng dẫn deploy
    st.markdown("---")
    st.header("Deployment & Notes")
    st.markdown("""
    - Chạy ứng dụng: `streamlit run app.py`
    - `requirements.txt` đã được chuẩn bị.
    - Khi có `best_model.pkl`, `preprocessor.pkl`, `optimal_threshold.pkl`:
      + Sửa hàm `predict_churn()` để load model và preprocessor từ disk và trả về `churn_probability` thực tế
      + Thay phần phân ngưỡng bằng `optimal_threshold.pkl` nếu cần
    - Để deploy lên Streamlit Cloud: tạo repo GitHub chứa project, commit `app.py`, `requirements.txt`, đảm bảo thư mục `Data/processed_data.csv` hoặc tải data từ remote.
    """)


if __name__ == "__main__":
    main()
