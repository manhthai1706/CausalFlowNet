# Giải thích: Neural Spline Flow (Flow.py)

**Neural Spline Flow (NSF)** là thành phần chịu trách nhiệm mô hình hóa phân phối xác suất của nhiễu (noise) trong hệ thống nhân quả.

## 1. Khái niệm (Concept)
Normalizing Flows là một lớp các mô hình xác suất cho phép biến đổi một phân phối đơn giản (như Gauss) thành một phân phối phức tạp thông qua các hàm khả nghịch. **Neural Spline Flow** là một dạng Flow tiên tiến sử dụng các hàm Spline (đa thức bậc hai hữu tỷ) để thực hiện phép biến đổi này.

## 2. Nguyên lý hoạt động trong dự án (Working Principle)
Trong `modules/Flow.py`:
1.  **Mô hình hóa Nhiễu**: Thay vì giả định nhiễu là phân phối Gauss (như đa số các thuật toán cũ), NSF học trực tiếp hình dạng thực tế của nhiễu từ dữ liệu.
2.  **Rational-Quadratic Splines (RQS)**: Sử dụng các đoạn đa thức để khớp với dữ liệu nhiễu, cho phép mô hình hóa các đặc trưng phức tạp như đa đỉnh (multimodal) hoặc đuôi nặng (heavy-tailed).
3.  **GMM Prior**: Sử dụng hỗn hợp Gaussian (Gaussian Mixture Model) làm phân phối gốc để tăng khả năng biểu diễn các ngữ cảnh (context) khác nhau trong dữ liệu sinh học.

## 3. Vì sao chọn (Rationale)
- **Độ chính xác**: Khám phá nhân quả phụ thuộc rất nhiều vào việc ước lượng đúng mật độ nhiễu. NSF cung cấp khả năng ước lượng mật độ chính xác nhất hiện nay.
- **Tính linh hoạt**: Dữ liệu thực tế (như tập Sachs) thường có nhiễu rất phức tạp và không tuân theo phân phối chuẩn. NSF là công cụ duy nhất đủ mạnh để xử lý vấn đề này mà vẫn đảm bảo tính khả nghịch (invertible) để tính toán Likelihood.
