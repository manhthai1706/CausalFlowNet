# Methodology / Phương pháp luận

This document provides a technical overview of the components within **CausalFlowNet**.

Tài liệu này cung cấp cái nhìn tổng quan về kỹ thuật của các khối chức năng trong **CausalFlowNet**.

---

## 1. Gated-ResMLP (Học Cơ chế Nhân quả)

In the Structural Equation Model (SEM) framework, the relationship between a child node and its parent nodes is represented by a nonlinear function $f_i$:
$$ X_i = f_i(PA_i) + \epsilon_i $$

**CausalFlowNet** uses a shared **Gated Residual MLP** to approximate all structural mechanisms simultaneously. 
*   **Gating Mechanism:** By mapping the input to a doubled dimension and splitting it into `features` and `gate`, the network dynamically controls information flow using $h = \sigma_{\text{act}}(\text{features}) \circ \sigma(\text{gate})$.
*   **Residual Connections:** To propagate gradients deeper without vanishing issues.
*   **Orthogonal Initialization:** Weights are initialized orthogonally to keep the variance of the gradients stable.

---

## 2. Neural Spline Flows (Mô hình hóa Phân phối Nhiễu)

Instead of strictly assuming that the noise $\epsilon_i$ follows a Gaussian distribution, we use **Neural Spline Flows (NSF)** with a Gaussian Mixture Model (GMM) prior.
*   **Rational-Quadratic Splines (RQS):** Piecewise rational quadratic functions used in coupling layers to model highly complex, multi-modal noise distributions commonly found in biological data.
*   **Density Estimation:** The exact likelihood of the residuals is evaluated. The Negative Log-Likelihood (NLL) acts as a principal reconstruction loss:
    $$ \log p(x) = \log p(z) + \sum \log | \det J | $$
*   **GMM Prior:** Allows the latent embeddings to cluster naturally, providing a mechanism for causal context identification.

---

## 3. Parallel Fast HSIC (Kiểm định Độc lập Thống kê)

To ensure the causal discovery directs edges properly, the estimated residuals (noise) must be statistically independent of the predicted parent variables. We apply the **Hilbert-Schmidt Independence Criterion (HSIC)**.
*   To avoid the cubic complexity $O(n^2)$ of exact HSIC computation across a large mini-batch, we deploy **Random Fourier Features (RFF)**.
*   The parallelized implementation applies dot products over mapped cosine/sine features for all variables simultaneously across the batch, reducing the complexity and fully utilizing GPU vectors.

---

## 4. Augmented Lagrangian Optimization (Tối ưu hóa Bậc cao)

Structure learning requires enforcing acyclicity. We map the graph edges $W$ into a continuous constraint:
$$ h(W) = \text{Tr}(e^{W \circ W}) - d = 0 $$

We solve this constraint using the **Augmented Lagrangian Method (ALM)**:
$$ L_{\text{aug}} = \text{NLL} + \lambda_{\text{HSIC}} L_{\text{HSIC}} + \lambda_{\text{L1}} \|W\|_1 + \alpha h(W) + \frac{\rho}{2} h(W)^2 $$

Where $\alpha$ (the Lagrangian multiplier) and $\rho$ (the penalty parameter) are dynamically updated outside the inner optimization loop (Adam).
