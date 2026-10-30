# Chạy ứng dụng Credit Risk

Ứng dụng hợp nhất hiện nằm trong `app.py`: gồm dashboard chỉ số và đánh giá, mô phỏng ngưỡng/chi phí, phân tích biến, chấm điểm hồ sơ mới, giải thích điểm, what-if và chấm hàng loạt.

Từ thư mục gốc project, cài thư viện và tạo dữ liệu hiển thị dashboard một lần:

    python -m pip install -r requirements.txt
    python export_dashboard_data.py
    streamlit run app.py

`export_dashboard_data.py` tạo các file kết quả trong `dashboard_data/` theo quy trình notebook; bước này chạy tìm kiếm/đánh giá mô hình và có thể mất thời gian. Khi các file đã có, lần chạy tiếp theo chỉ cần `streamlit run app.py`.

Trong trang **Chấm điểm hồ sơ mới**, các biến categorical hiện nhãn chữ; app chuyển lựa chọn đó về mã số đúng trước khi đưa vào model. Upload CSV cũng nhận mã số hoặc nhãn chữ. Với mục tiêu kỹ thuật, các biến vẫn phải dùng đúng nhãn/mã được liệt kê trong app.

`dashboard.py` là bản dashboard độc lập cũ; app hợp nhất chính để trình bày và chấm điểm là `app.py`.

Đây là demo học thuật German Credit. Nhãn Good/Bad và các gợi ý what-if không phải quyết định cho vay thực tế; P(Bad) cũng chưa được hiệu chuẩn thành xác suất vỡ nợ thực.
