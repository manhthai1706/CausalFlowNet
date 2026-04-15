# Experiments / Thực nghiệm

This document details the datasets and experimental setup used to evaluate **CausalFlowNet**.

Tài liệu này trình bày chi tiết về các bộ dữ liệu và thiết lập thực nghiệm dùng để đánh giá **CausalFlowNet**.

---

## 1. Datasets / Dữ liệu

### 1.1 Sachs Dataset (Protein Signaling Network)
- **Description:** A classic biological dataset recording the expression levels of 11 proteins and phospholipids in human immune system cells under various interventional conditions.
- **Dimensionality:** 11 nodes, 853 samples.
- **Challenge:** Non-Gaussian noise, heavy tails, and context-dependent interactions across different interventions.
- **Execution:** Run `python test_sachs.py`

### 1.2 SynTReN Dataset (Synthetic Gene Regulatory Network)
- **Description:** Synthetic gene expression data generated using the **SynTReN** generator, simulating biological network topologies (based on E. coli and Yeast networks) while introducing biological noise.
- **Dimensionality:** 20 nodes, 1000 samples.
- **Challenge:** Higher dimensionality, sparse underlying ground-truth DAG.
- **Execution:** Run `python test_syntren.py`

---

## 2. Evaluation Metrics / Các chỉ số Đánh giá

The output graph $\hat{W}$ is compared against the Ground-Truth $W^*$ using the following metrics:

1.  **SHD (Structural Hamming Distance):** 
    Computes the number of structural modifications (add, delete, reverse) required to transform $\hat{W}$ into $W^*$. Lower is better.
2.  **SHD-c (Structural Hamming Distance for CPDAGs):**
    Evaluates the graph within its Markov equivalence class (CPDAG), avoiding penalization for edges whose direction cannot be statistically distinguished by observational data alone. Lower is better.
3.  **SID (Structural Interventional Distance):**
    A causal-specific metric that evaluates how well the estimated graph can be used for do-calculus interventions. It penalizes paths that are incorrectly stated, heavily favoring models that produce reliable causal impacts. Lower is better.
4.  **TPR (True Positive Rate) / FPR (False Positive Rate):**
    Measures the recall and the false alarm rate of the binary edge detection.
5.  **FDR (False Discovery Rate):**
    The proportion of false edges among all detected edges.

---

## 3. Training Details / Cấu hình Huấn luyện

Our default experiments use the following hyperparameter configurations:

*   **Architecture:** Gated-ResMLP (Hidden Dims: `[32, 32]`)
*   **NSF Configuration:** 2 layers, 8 bins per dimension.
*   **HSIC Penalization ($\lambda_{\text{HSIC}}$):** Automatically scaled per dataset (e.g., `0.01`).
*   **L1 Penalty ($\lambda_{L1}$):** Promotes sparsity effectively based on edge weight decay.
*   **Optimizer:** Adam optimizer with `learning_rate=0.01`.
*   **Constraint Thresholds:** The outer continuous ALM loop aggressively updates $\alpha$ and $\rho$ with a maximum penalty cap to avoid numerical explosion. Graph edges below an absolute weight of `0.1` are thresholded to $0$.

Check `benchmark.md` for a comprehensive comparison with existing algorithms (GraN-DAG, NOTEARS, CAM, GES, etc.).
