# Giải thích: HSIC & Kernel (HSIC.py & Kernel.py)

Các tệp này triển khai tiêu chuẩn độc lập **HSIC (Hilbert-Schmidt Independence Criterion)** để đảm bảo tính đúng đắn của các mối quan hệ nhân quả tìm được.

## 1. Khái niệm (Concept)
- **HSIC**: Là một độ đo phi tuyến tính dùng để kiểm tra tính độc lập thống kê giữa hai tập biến. Nếu HSIC bằng 0, hai biến đó độc lập hoàn toàn.
- **Kernel/RFF**: Thay vì tính toán HSIC trên không gian gốc (rất chậm), chúng ta sử dụng **Random Fourier Features (RFF)** để xấp xỉ hàm nhân Gauss, chuyển bài toán về không gian Fourier để tính toán cực nhanh.

## 2. Nguyên lý hoạt động trong dự án (Working Principle)
Trong `core/HSIC.py`:
1.  **Kiểm định phần dư**: Trong mô hình nhân quả ANM, nhiễu (phần dư) phải độc lập với các biến cha. HSIC được dùng để đo lường sự phụ thuộc này.
2.  **Song song hóa (Parallelization)**: Thuật toán được thiết kế để tính toán sự độc lập trên tất cả các nút của đồ thị cùng một lúc bằng phép nhân ma trận lô (batch matrix multiplication) trên GPU.
3.  **Hạng phạt (Penalty)**: Giá trị HSIC được đưa trực tiếp vào hàm mất mát như một hình phạt. Nếu mô hình chọn sai hướng cạnh, HSIC sẽ tăng cao, buộc mô hình phải chọn lại hướng đúng.

## 3. Vì sao chọn (Rationale)
- **Xác định hướng cạnh**: HSIC là công cụ mạnh nhất để phân biệt hướng nhân quả trong các trường hợp phi tuyến (ví dụ: phân biệt $A \rightarrow B$ hay $B \rightarrow A$).
- **Hiệu năng**: Việc sử dụng RFF giúp HSIC có tốc độ xử lý nhanh gấp hàng chục lần so với cách tính kernel truyền thống, cho phép huấn luyện trên tập dữ liệu lớn.
