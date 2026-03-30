# Giải thích Thực nghiệm: Tập dữ liệu Sachs (test_sachs.py)

Tệp này thực hiện việc huấn luyện và đánh giá mô hình CausalFlowNet trên tập dữ liệu Protein thực tế từ Sachs et al. (2005).

## 1. Tại sao chọn các tham số này?

Trong `test_sachs.py`, cấu hình được thiết lập như sau:

- **`n_clusters: 5`**: 
    - *Lý do*: Tập dữ liệu Sachs bao gồm các phép đo tế bào đơn dưới các điều kiện thí nghiệm khác nhau (can thiệp bằng các chất ức chế/kích thích). Do đó, nhiễu thực tế không đồng nhất mà bao gồm nhiều "ngữ cảnh" khác nhau. Việc chọn 5 cụm giúp GMM Prior của Flow bắt được các trạng thái sinh lý khác nhau này.
- **`flow_bins: 12`**: 
    - *Lý do*: Dữ liệu sinh học thực tế rất nhiễu và có phân phối phi chuẩn phức tạp xấp xỉ các đường cong hữu tỷ bậc hai. 12 thùng (bins) cung cấp đủ độ phân giải để Spline Flow "khớp" được chính xác hình dạng của nhiễu.
- **`lda_hsic: 0.03`**: 
    - *Lý do*: Một giá trị trung bình giúp cân bằng giữa việc tối ưu hóa khả năng hợp lý (Likelihood) và việc đảm bảo tính độc lập. Nếu đặt quá cao, mô hình sẽ khó hội tụ; nếu quá thấp, mô hình sẽ dễ nhầm lẫn hướng nhân quả.
- **`stage1_epochs: 30` / `stage2_epochs: 20`**: 
    - *Lý do*: Với 11 nút, đồ thị Sachs không quá lớn. Số lượng vòng lặp này đủ để mô hình hoàn thành quá trình "Khám phá" (Stage 1) và "Tinh chỉnh" (Stage 2) mà không bị Overfitting.
- **`l1_stage2: 0.012`**: 
    - *Lý do*: Ở giai đoạn 2, chúng ta tăng hình phạt L1 (từ 0.001 lên 0.012) để triệt tiêu các cạnh yếu và nhiễu, giúp đồ thị cuối cùng sạch và rõ ràng hơn.

## 2. Kết quả kỳ vọng
Với bộ tham số này, mô hình hướng tới việc đạt được chỉ số **SID thấp** (Structural Intervention Distance), vì SID phản ánh chính xác khả năng của mô hình trong việc trả lời các câu hỏi về can thiệp protein trong y sinh.
