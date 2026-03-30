# Giải thích: Tối ưu hóa & Ràng buộc DAG (Optimization.py)

Tệp này chứa các thuật toán quan trọng nhất để đảm bảo cấu trúc học được là một đồ thị có hướng không chu trình (Directed Acyclic Graph - DAG).

## 1. Khái niệm (Concept)
- **Acyclicity Constraint**: Khám phá nhân quả yêu cầu đồ thị không được có vòng lặp (ví dụ: A -> B -> C -> A là sai). Chúng ta sử dụng hàm h(W) dựa trên lũy thừa ma trận để định lượng "độ vòng" của đồ thị.
- **Augmented Lagrangian (ALM)**: Là một khung tối ưu hóa dùng để giải quyết các bài toán có ràng buộc cứng (trong trường hợp này là ràng buộc $h(W)=0$).

## 2. Nguyên lý hoạt động trong dự án (Working Principle)
Trong `core/Optimization.py`:
1.  **Hàm h(W)**: Tính toán vết của hàm mũ ma trận ($\text{Tr}(\exp(W \circ W)) - d$). Giá trị này bằng 0 khi và chỉ khi đồ thị là DAG.
2.  **Vòng lặp Lagrangian**: Thay vì ép đồ thị là DAG ngay lập tức (rất khó), ALM tăng dần hình phạt cho các vòng lặp theo thời gian. Ban đầu mô hình tự do khám phá cấu trúc, sau đó dần bị "siết chặt" lại để loại bỏ các vòng lặp.
3.  **Cập nhật tham số ($\alpha, \rho$)**: Tự động điều chỉnh trọng số của ràng buộc dựa trên mức độ vi phạm thực tế của đồ thị.

## 3. Vì sao chọn (Rationale)
- **Tính khả vi (Differentiability)**: Cách tiếp cận này biến bài toán tìm kiếm đồ thị (vốn là bài toán tổ hợp khó) thành bài toán tối ưu hóa liên tục, cho phép sử dụng Gradient Descent để giải nhanh chóng.
- **Độ tin cậy**: ALM là thuật toán chuẩn xác nhất để đảm bảo mô hình cuối cùng hội tụ về một cấu trúc hơp lệ mà không làm hy sinh quá nhiều độ chính xác của Likelihood.
