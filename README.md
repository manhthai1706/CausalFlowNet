# CausalFlowNet V2: Enhanced Causal Discovery & Inference
## Nghiên cứu Khám phá và Suy luận Nhân quả

CausalFlowNet V2 is a research framework for **Causal Discovery** and **Causal Inference** that builds upon foundational work in Neural Additive Models and Normalizing Flows. It aims to provide an efficient and accurate approach for analyzing causal relationships in observational data.

CausalFlowNet V2 là một khung nghiên cứu về **Khám phá** và **Suy luận Nhân quả**, được xây dựng dựa trên những nghiên cứu nền tảng về Neural Additive Models và Normalizing Flows. Dự án hướng tới việc cung cấp một phương pháp hiệu quả và chính xác để phân tích các mối quan hệ nhân quả trong dữ liệu quan sát.

---

## 🙏 Scientific Context & Acknowledgement (Bối cảnh Khoa học & Ghi nhận)

This project is an evolution of existing causal discovery methodologies. We would like to acknowledge the pioneering researchers in the fields of:
- **Neural ANM & DAG-GNN**: For the fundamental concepts of differentiable causal discovery.
- **Normalizing Flows (NSF)**: For advanced density estimation techniques.
- **Causal Inference (Do-calculus)**: Inspired by the structural causal models (SCM) framework.

Dự án này là một bước tiến hóa dựa trên các phương pháp khám phá nhân quả hiện có. Chúng tôi xin ghi nhận công lao của các nhà nghiên cứu tiên phong trong các lĩnh vực:
- **Neural ANM & DAG-GNN**: Cho các khái niệm nền tảng về khám phá nhân quả có thể vi phân.
- **Normalizing Flows (NSF)**: Cho các kỹ thuật ước lượng mật độ tiên tiến.
- **Causal Inference (Do-calculus)**: Được truyền cảm hứng từ khung làm việc của các mô hình nhân quả cấu trúc (SCM).

---

## 🚀 Refined Features (Các tính năng cải tiến)

- **Vectorized Architecture**: Efficient training process optimized for modern computing resources.
  - *Kiến trúc vectơ hóa*: Quy trình huấn luyện hiệu quả, được tối ưu hóa cho các nguồn lực tính toán hiện đại.
- **Gated-ResMLP**: Incorporates Gated Linear Units (GLU) to better model context-dependent biological interactions.
  - *MLP Residual có cổng*: Sử dụng khối GLU để mô hình hóa tốt hơn các tương tác sinh học phụ thuộc vào ngữ cảnh.
- **Neural Spline Flow (GMM-Prior)**: Adaptively learns complex noise structures and multi-modal distributions.
  - *Neural Spline Flow*: Học một cách thích nghi các cấu trúc nhiễu phức tạp và phân phối đa chế độ.
- **Parallel Independence Testing**: Integrative HSIC implementation for simultaneous structural verification.
  - *Kiểm tra tính độc lập song song*: Triển khai HSIC tích hợp để xác thực cấu trúc đồng thời.
- **ATE Estimation**: Provides quantitative insights into causal strengths via do-calculus simulation.
  - *Ước lượng ATE*: Cung cấp các hiểu biết định lượng về sức mạnh nhân quả thông qua giả lập do-calculus.

---

## 🏗️ Architecture (Kiến trúc)

The model follows a structured discovery pipeline:
1. **Feature Learning**: Gated-ResMLP captures non-linear relationships.
2. **Noise Analysis**: Neural Spline Flow handles non-Gaussian residuals.
3. **Constrained Optimization**: Augmented Lagrangian enforces a directed acyclic graph (DAG) structure.
4. **Causal Quantification**: Simulation-based ATE estimation for impact analysis.

---

## 📊 Evaluation Results (Kết quả Đánh giá)

*These results reflect the performance of our enhanced implementation on standard benchmarks.*

| Dataset | TPR (Recall) | FPR | SHD (Errors) | Note |
| :--- | :---: | :---: | :---: | :--- |
| **Sachs** (Protein) | 0.44 | 0.06 | **12** | Competitive Accuracy |
| **SynTReN** (Gene) | 0.55 | 0.08 | **8** | Precise Discovery |

---

## 🛠️ Usage (Cách sử dụng)

### 1. Requirements (Yêu cầu)
```bash
pip install torch numpy pandas networkx matplotlib scikit-learn
```

### 2. Running Sachs Experiment
```bash
python test_sachs.py
```

---

## 📜 License (Giấy phép)

Distributed under the **MIT License**.

Copyright (c) 2026 **ManhThai**
