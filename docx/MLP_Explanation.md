# Giải thích: Gated-ResMLP (MLP.py)

Mô hình **Gated Residual Multi-Layer Perceptron (Gated-ResMLP)** là trái tim của hệ thống CausalFlowNet, chịu trách nhiệm xấp xỉ các hàm cơ chế nhân quả phi tuyến giữa các biến.

## 1. Khái niệm (Concept)
Gated-ResMLP là một mạng nơ-ron sâu được thiết kế dựa trên sự kết hợp của hai kỹ thuật hiện đại: **Residual Connections** (Kết nối tắt) và **Gating Mechanisms** (Cơ chế cổng).
- **Residual**: Giúp dòng gradient chảy mượt mà qua các lớp sâu, tránh hiện tượng triệt tiêu gradient (vanishing gradient).
- **Gating**: Cho phép mạng "chọn lọc" thông tin nào quan trọng để truyền tiếp và thông tin nào nên bị loại bỏ, tương tự như cơ chế trong mạng LSTM hay GRU.

## 2. Nguyên lý hoạt động trong dự án (Working Principle)
Trong `modules/MLP.py`:
1.  **Shared Weights**: Một mạng MLP duy nhất được dùng chung cho tất cả các nút nhân quả để giảm số lượng tham số và tăng tính tổng quát hóa.
2.  **Cơ chế Cổng (Gating)**: Tín hiệu đầu vào được nhân với một vector cổng (gate) được tính qua hàm Sigmoid. Điều này giúp mô hình hóa các tương tác phức tạp (ví dụ: biến A chỉ tác động lên B trong một số điều kiện nhất định).
3.  **Lọc biến cha (Masking)**: MLP kết hợp với ma trận kề $W$ để chỉ nhận đầu vào từ các biến được coi là "cha" của biến hiện tại.

## 3. Vì sao chọn (Rationale)
- **Xử lý phi tuyến**: Khác với NOTEARS truyền thống dùng hồi quy tuyến tính, Gated-ResMLP có khả năng học bất kỳ hàm phi tuyến phức tạp nào.
- **Tính ổn định**: Kết nối Residual giúp huấn luyện các mạng sâu hơn mà không bị mất ổn định số học.
- **Hiệu quả tham số**: Việc chia sẻ trọng số (shared weights) giúp mô hình nhẹ hơn và hội tụ nhanh hơn trên các tập dữ liệu có số lượng biến lớn.
