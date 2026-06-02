# DỰ ÁN DỰ ĐOÁN KHẢ NĂNG KHÁCH HÀNG QUAY LẠI MUA HÀNG TRONG 30 NGÀY

## 1. Giới thiệu dự án

Dự án được thực hiện nhằm dự đoán khả năng khách hàng sẽ quay lại mua hàng trong vòng 30 ngày tiếp theo dựa trên lịch sử giao dịch.

Hệ thống hỗ trợ doanh nghiệp:

* Xác định khách hàng có nguy cơ không quay lại mua hàng.
* Xây dựng các chương trình chăm sóc khách hàng phù hợp.
* Tăng tỷ lệ giữ chân khách hàng.
* Hỗ trợ ra quyết định kinh doanh dựa trên dữ liệu.

---

## 2. Mục tiêu dự án

* Làm sạch và phân tích dữ liệu khách hàng.
* Xây dựng các đặc trưng phản ánh hành vi mua sắm.
* Huấn luyện mô hình Machine Learning dự đoán khả năng khách quay lại.
* Xây dựng ứng dụng Web hỗ trợ dự đoán và xuất báo cáo.

---

## 3. Thành viên thực hiện

| Thành viên      | Vai trò                             |
| --------------- | ----------------------------------- |
| Nguyễn Tường Vi | Data Cleaning, EDA, Web Application |
| Linh            | Feature Engineering                 |
| Trang           | Modeling                            |

---

## 4. Bộ dữ liệu sử dụng

Các trường dữ liệu chính:

* customer_id
* transaction_id
* order_datetime
* product_name
* quantity
* unit_price
* payment_method
* channel
* voucher_used

---

## 5. Các bước thực hiện

### 5.1 Data Cleaning

Thực hiện xử lý:

* Giá trị thiếu (Missing Value)
* Dữ liệu trùng lặp (Duplicate)
* Chuẩn hóa dữ liệu dạng văn bản
* Chuyển đổi định dạng thời gian
* Xử lý dữ liệu bất thường (Outlier) bằng phương pháp IQR

---

### 5.2 Exploratory Data Analysis (EDA)

Phân tích:

* Tổng quan doanh thu
* Hiệu quả sản phẩm
* Hành vi khách hàng
* Xu hướng theo thời gian
* Hiệu quả các kênh bán hàng

---

### 5.3 Feature Engineering

Các đặc trưng được sử dụng:

* days_since_last_order
* total_orders
* orders_last_30d
* avg_days_between_orders
* avg_order_value
* spending_last_30d
* voucher_usage_rate
* recent_activity_drop
* unique_products
* weekend_order_ratio
* max_days_between_orders
* favorite_channel

Biến mục tiêu:

**return_30d**

* 1: Khách hàng phát sinh giao dịch trong 30 ngày tiếp theo.
* 0: Khách hàng không phát sinh giao dịch trong 30 ngày tiếp theo.

---

### 5.4 Modeling

Mô hình Machine Learning được sử dụng để dự đoán:

**Return Probability** (xác suất khách hàng quay lại mua hàng).

Phân nhóm khách hàng:

| Return Probability | Nhóm khách hàng |
| ------------------ | --------------- |
| < 30%              | High Risk       |
| 30% - 70%          | Medium          |
| > 70%              | Loyal           |

---

### 5.5 Web Application

Ứng dụng được xây dựng bằng Streamlit.

#### Input

Người dùng tải lên file CSV gồm:

* customer_id
* order_datetime

#### Quy trình xử lý

1. Đọc dữ liệu lịch sử.
2. Tạo feature cho từng khách hàng.
3. Tiền xử lý dữ liệu.
4. Dự đoán bằng mô hình Machine Learning.
5. Trả về kết quả dự đoán.

#### Output

* Return Probability
* Risk Group
* Recommendation
* File báo cáo CSV

---

## 6. Cấu trúc thư mục

```text
Project/
│
├── Data/
│   └── processed_data.csv
│
├── Model/
│   ├── best_model.pkl
│   ├── preprocessing_pipeline.pkl
│   └── optimal_threshold.pkl
│
├── app_final.py
├── requirements.txt
└── README.md
```

---

## 7. Công nghệ sử dụng

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Plotly
* Joblib

---

## 8. Hướng dẫn chạy chương trình

Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

Chạy ứng dụng:

```bash
streamlit run app_final.py
```

---

## 9. Kết quả đạt được

* Hoàn thành quy trình Data Analytics từ làm sạch dữ liệu đến triển khai ứng dụng.
* Xây dựng mô hình dự đoán khả năng khách hàng quay lại mua hàng.
* Phát triển ứng dụng Web hỗ trợ doanh nghiệp theo dõi và phân loại khách hàng theo mức độ rủi ro.

---

## 10. Kết luận

Dự án giúp doanh nghiệp nhận diện sớm khách hàng có nguy cơ không quay lại mua hàng, từ đó đưa ra các chương trình chăm sóc phù hợp nhằm nâng cao tỷ lệ giữ chân khách hàng và gia tăng doanh thu.
