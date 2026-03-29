# Bảng So sánh Kết quả Thực nghiệm

Dưới đây là bảng so sánh hiệu năng của **CausalFlowNet (Ours)** với các phương pháp Baseline phổ biến trên hai bộ dữ liệu tiêu chuẩn: **Sachs** (11 nút) và **SynTReN** (20 nút).

| Phương pháp | SHD (Sachs) | SHD-c (Sachs) | SID (Sachs) | SHD (Syn) | SHD-c (Syn) | SID (Syn) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GraN-DAG** | 13.0 | 11.0 | 47.0 | 34.0 ± 8.5 | 36.4 ± 8.3 | 161.7 ± 53.4 |
| **GraN-DAG++** | 13.0 | 10.0 | 48.0 | 33.7 ± 3.7 | 39.4 ± 4.9 | 127.5 ± 52.8 |
| **DAG-GNN** | 16.0 | 21.0 | 44.0 | 93.6 ± 9.2 | 97.6 ± 10.3 | 157.5 ± 74.6 |
| **NOTEARS** | 21.0 | 21.0 | 44.0 | 151.8 ± 28.2 | 156.1 ± 28.7 | 110.7 ± 66.7 |
| **CAM** | 12.0 | 9.0 | 55.0 | 40.5 ± 6.8 | 41.4 ± 7.1 | 152.3 ± 48.0 |
| **GSF** | 18.0 | 10.0 | 44.0 - 61.0 | 61.8 ± 9.6 | 63.3 ± 11.4 | 76.7 ± 51.1, 109.9 ± 39.9 |
| **GES** | 26.0 | 28.0 | 34.0 - 45.0 | 82.6 ± 9.3 | 85.6 ± 10.0 | 157.2 ± 48.3, 168.8 ± 47.8 |
| **PC** | 17.0 | 11.0 | 47.0 - 62.0 | 41.0 ± 5.1 | 42.4 ± 4.6 | 154.8 ± 47.6, 179.3 ± 55.6 |
| **CausalFlowNet (Ours)** | **12.0** | **16.0** | **37.0** | **25.0** | **35.0** | **166.0** |

---
**Ghi chú:**
*   **SHD (Structural Hamming Distance):** Càng thấp càng tốt (đo lường số lỗi cạnh).
*   **SID (Structural Interventional Distance):** Càng thấp càng tốt (đo lường sai số trong dự báo can thiệp).
*   Các chỉ số của Baseline được trích xuất từ các công bố khoa học liên quan.
