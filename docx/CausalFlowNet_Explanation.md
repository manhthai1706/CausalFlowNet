# Tổng quan: CausalFlowNet (CausalFlowNet.py)

Đây là file điều phối chính, kết hợp tất cả các module lại thành một mô hình hoàn chỉnh để khám phá cấu trúc nhân quả.

## 1. Khái niệm (Concept)
CausalFlowNet là một mạng thần kinh nhân quả tích hợp, kết hợp sức mạnh của **Deep Learning** (để học cơ chế), **Normalizing Flows** (để học phân phối nhiễu) và **Continuous Optimization** (để đảm bảo cấu trúc đồ thị).

## 2. Nguyên lý hoạt động trong dự án (Working Principle)
Trong `CausalFlowNet.py`:
1.  **Khởi tạo**: Thiết lập ma trận kề $W$ dưới dạng các tham số có thể học được.
2.  **Quy trình Forward**:
    - Dữ liệu đi qua **Gated-ResMLP** để dự báo giá trị các biến dựa trên "cha" của chúng.
    - Tính phần dư (residual) giữa giá trị thực và dự báo.
    - Phần dư đi qua **Neural Spline Flow** để tính Log-Likelihood.
    - Song song đó, tính hạng phạt **HSIC** để đảm bảo tính độc lập.
3.  **Huấn luyện**: Sử dụng vòng lặp **Augmented Lagrangian** để vừa tối ưu hóa độ khớp dữ liệu, vừa đảm bảo tính không chu trình (acyclicity).

## 3. Vì sao chọn (Rationale)
- **End-to-End**: Cho phép huấn luyện toàn bộ hệ thống từ đầu đến cuối chỉ bằng một hàm mất mát duy nhất.
- **Đa năng**: Không chỉ tìm ra các cạnh (edges), tệp này còn tích hợp các phương pháp suy luận sau huấn luyện như **ATE (Average Treatment Effect)** để đo lường mức độ tác động và **Clustering** để phân nhóm các mẫu.
- **Tính hiện đại**: Kết hợp những kỹ thuật mới nhất trong lĩnh vực Causal AI hiện nay.
