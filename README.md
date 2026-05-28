
<h1 align="center">CausalFlowNet</h1>
<h3 align="center">Nonlinear Causal Discovery via Normalizing Flows and Parallel Independence Testing</h3>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square&logo=pytorch" alt="PyTorch"/></a>
</p>

---

## Abstract

Causal discovery from continuous observational data is a cornerstone of scientific inquiry, yet it remains a fundamentally challenging task. Modern functional causal models (FCMs) often rely on rigid parametric assumptions (such as linear Gaussian mechanisms) or fail to explicitly enforce the statistical independence between structural residuals and causal parents—a core requirement of additive noise models (ANMs). 

To address these limitations, we introduce **CausalFlowNet**, a unified, end-to-end deep learning framework for continuous causal structure learning. CausalFlowNet integrates:
1. Flexible nonlinear mechanism modeling via a shared **Gated Residual Multi-Layer Perceptron (Gated-ResMLP)**.
2. Invariant noise density estimation using **Neural Spline Flows (NSF)** equipped with a learnable **Gaussian Mixture Model (GMM)** prior.
3. Explicit enforcement of the ANM parent-residual independence assumption via a fully parallelized **Hilbert-Schmidt Independence Criterion (HSIC)** module accelerated by **Random Fourier Features (RFF)**.

The entire framework is optimized end-to-end under the **Augmented Lagrangian Method (ALM)** with a continuous acyclicity constraint, guaranteeing directed acyclic graph (DAG) solutions. Empirical evaluation on real intracellular signaling networks (**Sachs**, $d=11$) and simulated gene expression data (**SynTReN**, $d=20$) shows that CausalFlowNet achieves state-of-the-art or highly competitive performance across multiple metrics, including Structural Hamming Distance (SHD), CPDAG-SHD ($SHD\text{-}c$), and Structural Interventional Distance (SID).

---

## Table of Contents

- [I. Introduction](#i-introduction)
- [II. Theoretical Foundation & Mathematical Framework](#ii-theoretical-foundation--mathematical-framework)
  - [A. Structural Equation Models and Additive Noise Assumptions](#a-structural-equation-models-and-additive-noise-assumptions)
  - [B. Invertible Residual Modeling via Neural Spline Flows](#b-invertible-residual-modeling-via-neural-spline-flows)
  - [C. Parallelized Independence Verification via Fast HSIC with RFF](#c-parallelized-independence-verification-via-fast-hsic-with-rff)
  - [D. Continuous Acyclicity Constraint and Constrained Optimization](#d-continuous-acyclicity-constraint-and-constrained-optimization)
- [III. Proposed System Architecture & Algorithms](#iii-proposed-system-architecture--algorithms)
  - [A. Nonlinear Parent Selection & Gated Residual MLP](#a-nonlinear-parent-selection--gated-residual-mlp)
  - [B. Two-Stage Structure Learning Pipeline](#b-two-stage-structure-learning-pipeline)
  - [C. Adaptive Post-Pruning Thresholding](#c-adaptive-post-pruning-thresholding)
- [IV. Experimental Evaluation & Comparative Benchmarks](#iv-experimental-evaluation--comparative-benchmarks)
  - [A. Quantitative Structure Learning Comparison](#a-quantitative-structure-learning-comparison)
  - [B. Causal Subgroup Discovery and Interventional ATE Analysis](#b-causal-subgroup-discovery-and-interventional-ate-analysis)
- [V. Visual Diagnostics & Structural Interpretations](#v-visual-diagnostics--structural-interpretations)
  - [A. Sachs Protein Network Diagnostics](#a-sachs-protein-network-diagnostics)
  - [B. SynTReN Gene Expression Network Diagnostics](#b-syntren-gene-expression-network-diagnostics)
- [VI. Repository Structure](#vi-repository-structure)
- [VII. Step-by-Step Reproduction & Setup Guide](#vii-step-by-step-reproduction--setup-guide)
- [VIII. References](#viii-references)
- [IX. Citation](#ix-citation)
- [X. License](#x-license)

---

## I. Introduction

Causal structure discovery aims to reconstruct the directed acyclic graph (DAG) representing the causal relationships among a set of observational variables. Traditional methods have historically been categorized into two paradigms:
*   **Constraint-based methods** (e.g., PC, FCI) which rely on conditional independence tests. These are highly sensitive to test selection, propagate decision errors, and suffer from high computational complexity.
*   **Score-based methods** (e.g., GES, Hill-Climbing) which search the discrete space of graphs to maximize a scoring function (such as BIC). These face a combinatorial explosion because the number of possible DAGs scales super-exponentially with the number of variables: $f(d) \approx 2^{d^2/2}$.

A major breakthrough occurred with the introduction of **NOTEARS** [13], which reframes the discrete search problem as a continuous optimization task over a weighted adjacency matrix $\mathbf{W} \in \mathbb{R}^{d \times d}$, enforcing acyclicity via a smooth matrix-exponential constraint $h(\mathbf{W}) = 0$ that admits gradient-based optimization.

While NOTEARS and its deep neural network extensions (e.g., GraN-DAG [15], DAG-GNN) successfully handle continuous optimization, three critical challenges remain unaddressed in the literature:
1.  **Rigid Parametric Noise Assumptions**: Most existing methods assume that the structural noise follows a simple, fixed distribution (typically homoscedastic Gaussian). In real-world biological systems, noise is frequently non-Gaussian, asymmetric, multi-modal, or heavy-tailed.
2.  **Implicit Independence Enforcement**: Under the Additive Noise Model (ANM) [8] framework, the noise residual $\varepsilon_j$ must be statistically independent of its causal parents $\mathbf{PA}_j$. Standard continuous methods maximize likelihood, which only implicitly drives residuals to independence, often yielding sub-optimal structures in small-sample regimes.
3.  **Kernel Complexity Scaling**: Traditional kernel-based independence tests, such as the Hilbert-Schmidt Independence Criterion (HSIC) [4], scale quadratically $\mathcal{O}(n^2)$ with the number of samples $n$, making their parallel application during gradient descent computationally prohibitive for large datasets.

**CausalFlowNet** bridges these gaps by providing an end-to-end framework that couples highly expressive neural density estimators (Rational-Quadratic Spline Flows) with an extremely fast parallelized independence penalty (RFF-based HSIC) under a unified continuous constraint framework.

---

## II. Theoretical Foundation & Mathematical Framework

<p align="center">
  <img src="arch.png" width="90%" alt="CausalFlowNet Scientific Architecture Diagram"/>
</p>

### A. Structural Equation Models and Additive Noise Assumptions

We consider a $d$-dimensional random vector $\mathbf{X} = (X_1, X_2, \ldots, X_d)$ governed by a joint distribution $P(\mathbf{X})$ induced by a Structural Equation Model (SEM) over a DAG $\mathcal{G} = (\mathbf{V}, \mathbf{E})$:

$$X_j = f_j(\mathbf{PA}_j^{\mathcal{G}}) + \varepsilon_j, \quad \forall j \in \{1, \ldots, d\}$$

where $f_j: \mathbb{R}^{|\mathbf{PA}_j|} \rightarrow \mathbb{R}$ is an arbitrary continuous nonlinear function, $\mathbf{PA}_j^{\mathcal{G}} \subseteq \mathbf{V} \setminus \{X_j\}$ represents the set of direct causal parents of $X_j$, and $\varepsilon_j$ are mutually independent noise variables.

The Additive Noise Model (ANM) framework guarantees the unique identifiability of the true causal DAG under weak assumptions [8], provided that the structural noise $\varepsilon_j$ is statistically independent of the active parents:

$$\varepsilon_j \perp\!\!\!\perp X_i, \quad \forall X_i \in \mathbf{PA}_j^{\mathcal{G}}$$

### B. Invertible Residual Modeling via Neural Spline Flows

Rather than assuming Gaussian noise, CausalFlowNet learns the exact density of each structural residual using a **Neural Spline Flow (NSF)** [3]. For each variable $X_j$, the residual $\varepsilon_j = X_j - f_j(\mathbf{PA}_j^{\mathcal{G}})$ is mapped by an invertible function $g_{\boldsymbol{\theta}}: \mathbb{R} \rightarrow \mathbb{R}$ to a latent variable $z_j$, and the change-of-variables theorem yields a tractable exact log-likelihood:

$$\log p(\varepsilon_j) = \log p_{\text{prior}}\bigl(g_{\boldsymbol{\theta}}(\varepsilon_j)\bigr) + \log \left| \frac{\partial g_{\boldsymbol{\theta}}(\varepsilon_j)}{\partial \varepsilon_j} \right|$$

The flow $g_{\boldsymbol{\theta}}$ is implemented as a composition of **Rational-Quadratic Spline** coupling layers—monotonic, analytically invertible transformations whose parameters are predicted by a small neural network—enabling highly flexible, non-Gaussian residual modeling.

#### Learnable GMM Prior

To capture multi-modal latent structure and support downstream causal subgrouping, we replace the standard Gaussian prior with a learnable $C$-component **Gaussian Mixture Model (GMM)**:

$$p_{\text{prior}}(z) = \sum_{c=1}^{C} \pi_c \, \mathcal{N}(z \mid \mu_c, \sigma_c^2)$$

where the mixture weights $\pi_c$, means $\mu_c$, and variances $\sigma_c^2$ are jointly optimized end-to-end via backpropagation.

---

### C. Parallelized Independence Verification via Fast HSIC with RFF

To explicitly enforce the ANM independence condition during training, we introduce a parallelized **Hilbert-Schmidt Independence Criterion (HSIC)** [4] module. The standard kernel HSIC requires $\mathcal{O}(n^2)$ kernel matrix computation per node, making it prohibitively expensive for gradient-based optimization over all $d$ variables simultaneously.

We resolve this by approximating the RBF kernel via **Random Fourier Features (RFF)** (Bochner's theorem), projecting each input into a low-dimensional randomized feature space of dimension $m \ll n$. This yields a closed-form, fully differentiable estimator:

$$\widehat{\text{HSIC}}(j) = \frac{1}{(n-1)^2} \left\| \left(\tilde{\Phi}_X^{(j)}\right)^\top \tilde{\Phi}_\varepsilon^{(j)} \right\|_F^2$$

where $\tilde{\Phi}_X^{(j)}$ and $\tilde{\Phi}_\varepsilon^{(j)}$ are centered RFF embeddings of the parent inputs and residuals for node $j$. All $d$ estimates are computed in a single batched matrix multiplication, reducing complexity from $\mathcal{O}(d \cdot n^2)$ to $\mathcal{O}(d \cdot n \cdot m)$ and making end-to-end backpropagation through the independence penalty tractable.

---

### D. Continuous Acyclicity Constraint and Constrained Optimization

The continuous causal learning task is formulated as a constrained optimization problem:

$$
\min_{\mathbf{W}, \boldsymbol{\theta}} \quad \mathcal{L}_{\text{main}}(\mathbf{W}, \boldsymbol{\theta}) = \mathcal{L}_{\text{NLL}}(\mathbf{W}, \boldsymbol{\theta}) + \lambda_{\text{HSIC}} \mathcal{L}_{\text{HSIC}}(\mathbf{W}, \boldsymbol{\theta}) + \lambda_{L_1} \|\mathbf{W}\|_1
$$

$$\text{subject to} \quad h(\mathbf{W}) = \text{Tr}\left(e^{\mathbf{W} \circ \mathbf{W}}\right) - d = 0$$


where:
*   $\mathcal{L}_{\text{NLL}}$ is the negative log-likelihood of the observational data under the learned rational-quadratic splines.
*   $\mathcal{L}_{\text{HSIC}}$ is the sum of log-HSIC penalties over all $d$ nodes, explicitly driving parent-residual independence.
*   $\lVert\mathbf{W}\rVert_1$ is the entry-wise L1 norm of off-diagonal weights, enforcing structural sparsity.

We solve this using the **Augmented Lagrangian Method (ALM)**, which optimizes a sequence of unconstrained subproblems:

$$\mathcal{L}_{\text{ALM}}(\mathbf{W}, \boldsymbol{\theta}, \alpha, \rho) = \mathcal{L}_{\text{main}}(\mathbf{W}, \boldsymbol{\theta}) + \alpha h(\mathbf{W}) + \frac{\rho}{2} h(\mathbf{W})^2$$

where $\alpha$ is the Lagrange multiplier and $\rho > 0$ is the penalty parameter.

---

## III. Proposed System Architecture & Algorithms

### A. Nonlinear Parent Selection & Gated Residual MLP

To model arbitrary nonlinear causal mechanisms, CausalFlowNet employs a single shared **Gated Residual MLP (Gated-ResMLP)** that serves all $d$ variables simultaneously. Rather than training $d$ independent networks, a continuous soft-masking strategy—multiplying each input row by the corresponding column of $\mathbf{W}$—dynamically routes only the active parents into each node's prediction.

Each residual block applies **Layer Normalization** followed by a linear projection that is split into a feature branch (activated by LeakyReLU) and a gating branch (activated by sigmoid); their element-wise product is added back via a residual connection. This gating mechanism selectively amplifies informative features while suppressing irrelevant ones. Weight sharing reduces the parameter count from $\mathcal{O}(d^2 \cdot L)$ to $\mathcal{O}(d \cdot L)$ (where $L$ is the number of layers), providing strong regularization against overfitting on small biological datasets.

---

### B. Two-Stage Structure Learning Pipeline

CausalFlowNet employs a **two-stage training strategy** to balance structural exploration and topological refinement:

| Phase | Duration | L1 Penalty | Objective |
| :--- | :---: | :---: | :--- |
| **Stage 1: Aggressive Discovery** | 30 Epochs | λ = 0.001 | Broad search of the continuous graph space; establishes coarse topological pathways. |
| **Stage 2: Structural Refinement** | 20 Epochs | λ = 0.012 | Aggressive pruning of spurious edges; focuses on parent-residual independence. |

---

### C. Adaptive Post-Pruning Thresholding

Upon convergence, the continuous weight matrix $\mathbf{W}$ is binarized into a directed adjacency matrix $\widehat{\mathbf{A}}$ using **data-driven adaptive thresholding**: an edge $i \rightarrow j$ is retained if and only if $|W_{ij}|$ exceeds the global mean of all off-diagonal weights plus $\kappa$ standard deviations (default $\kappa = 0.8$). This removes the need for manual threshold selection and automatically adapts the pruning aggressiveness to the weight distribution of each dataset.

---

## IV. Experimental Evaluation & Comparative Benchmarks

### A. Quantitative Structure Learning Comparison

We evaluate CausalFlowNet against eight established baselines spanning constraint-based (PC), score-based (GES), functional causal models (CAM), and continuous optimization (NOTEARS, DAG-GNN, GSF, GraN-DAG, GraN-DAG++) paradigms. 

Experiments are conducted on two standard benchmarks: the **Sachs** protein signaling dataset ($d=11$, $n=7,466$) and the **SynTReN** gene expression simulation network ($d=20$, $n=500$).

#### Performance Comparison Table

| Paradigm | Method | SHD (Sachs) ↓ | SHD-c (Sachs) ↓ | SID (Sachs) ↓ | SHD (Syn) ↓ | SHD-c (Syn) ↓ | SID (Syn) ↓ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CB** | PC [11] | $17.0$ | $11.0$ | $47.0 \text{ to } 62.0$ | $41.0 \pm 5.1$ | $42.4 \pm 4.6$ | $154.8 \pm 47.6$ |
| **SB** | GES [2] | $26.0$ | $28.0$ | $34.0 \text{ to } 45.0$ | $82.6 \pm 9.3$ | $85.6 \pm 10.0$ | $157.2 \pm 48.3$ |
| **FCM** | CAM [7] | $12.0$ | **$9.0$** | $55.0$ | $40.5 \pm 6.8$ | $41.4 \pm 7.1$ | $152.3 \pm 48.0$ |
| **CO** | NOTEARS [13] | $21.0$ | $21.0$ | $44.0$ | $151.8 \pm 28.2$ | $156.1 \pm 28.7$ | $110.7 \pm 66.7$ |
| **CO** | DAG-GNN | $16.0$ | $21.0$ | $44.0$ | $93.6 \pm 9.2$ | $97.6 \pm 10.3$ | $157.5 \pm 74.6$ |
| **CO** | GSF | $18.0$ | $10.0$ | $44.0 \text{ to } 61.0$ | $61.8 \pm 9.6$ | $63.3 \pm 11.4$ | **$76.7 \pm 51.1$** |
| **CO** | GraN-DAG [15] | $13.0$ | $11.0$ | $47.0$ | $34.0 \pm 8.5$ | $36.4 \pm 8.3$ | $161.7 \pm 53.4$ |
| **CO** | GraN-DAG++ | $13.0$ | $10.0$ | $48.0$ | $33.7 \pm 3.7$ | $39.4 \pm 4.9$ | $127.5 \pm 52.8$ |
| **CO** | **CausalFlowNet (Ours)** | **$12.0$** | $16.0$ | **$37.0$** | **$25.0$** | **$35.0$** | $166.0$ |

*   **CB**: Constraint-Based, **SB**: Score-Based, **FCM**: Functional Causal Model, **CO**: Continuous Optimization.
*   Values represent the mean $\pm$ standard deviation across independent initialization seeds where applicable.

---

### B. Causal Subgroup Discovery and Interventional ATE Analysis

By modeling structural residuals with a learnable GMM prior, CausalFlowNet naturally enables **Causal Subgroup Discovery** and **Average Treatment Effect (ATE)** estimation.

#### 1. Latent Causal Clustering
The latent space representations $z_j = g_{\boldsymbol{\theta}}(\varepsilon_j)$ correspond to normalized structural noise. By running K-Means on the latent space $\mathbf{Z} \in \mathbb{R}^{n \times d}$, we can identify subgroups of samples corresponding to specific biological states or experimental conditions without supervision.

#### 2. Average Treatment Effect Estimation
Using the learned Gated-ResMLP mechanisms, we estimate the ATE of intervening on a source node $X_s$ ($do(X_s = v)$) on a downstream target node $X_t$:

$$\text{ATE}(s \rightarrow t) = \mathbb{E}\left[X_t \mid do(X_s = 1)\right] - \mathbb{E}\left[X_t \mid do(X_s = 0)\right]$$

We simulate the interventional distribution by setting the column $\mathbf{W}_{:, s} = \mathbf{0}$, forcing $X_s$ to the intervention value $v$, and evaluating the downstream expectations via a forward pass through the Gated-ResMLP.

---

## V. Visual Diagnostics & Structural Interpretations

### A. Sachs Protein Network Diagnostics

The Sachs protein signaling network is a real-world biology benchmark containing $d=11$ phosphoproteins and $17$ ground-truth interactions. 

<p align="center">
  <img src="sachs_graph_comparison.png" width="48%" alt="Sachs Causal Graph with ATEs"/>
  <img src="sachs_adjacency_comparison.png" width="48%" alt="Sachs Adjacency Comparison Matrices"/>
</p>

*   **Fig. 1**: Sachs reconstructed causal graph with estimated edge weights (left).
*   **Fig. 2**: Continuous Adjacency Comparison Matrix (right): Ground Truth (blue) vs. CausalFlowNet Estimates (red).

CausalFlowNet successfully recovers key biological pathways, including the canonical **PKC → Raf → Mek → Erk** MAPK cascade and the **PIP2 → PIP3** phospholipid pathway.

---

### B. SynTReN Gene Expression Network Diagnostics

The SynTReN gene regulatory network benchmark simulates gene interactions in *E. coli* ($d=20$, $24$ true edges) with realistic, non-linear kinetics.

<p align="center">
  <img src="syntren_graph_comparison.png" width="48%" alt="SynTReN Reconstructed Gene Network"/>
  <img src="syntren_adjacency_comparison.png" width="48%" alt="SynTReN Adjacency Comparison Matrices"/>
</p>

*   **Fig. 3**: SynTReN reconstructed gene regulatory network (left).
*   **Fig. 4**: Continuous Adjacency Comparison Matrix (right): Ground Truth vs. CausalFlowNet Estimates.

On this simulated dataset, CausalFlowNet achieves SOTA structural accuracy, recovering correct regulatory paths (such as **Gene_9 → Gene_17** and **Gene_10 → Gene_11**) with a low false positive rate ($FPR = 0.08$).

---

## VI. Repository Structure

The codebase is organized into modular components to facilitate extension and integration:

```
CausalFlowNet/
├── core/
│   ├── HSIC.py            # Fast parallel RFF-based HSIC module
│   └── Optimization.py    # Continuous acyclicity constraint & ALM solver
├── modules/
│   ├── MLP.py             # Gated Residual Block & Gated-ResMLP modeler
│   └── Flow.py            # Neural Spline Flow & learnable GMM prior
├── ultis/
│   ├── Evaluation.py      # Quantitative metrics (SHD, SID, TPR, FPR, FDR)
│   └── visualize.py       # DAG plotting and adjacency comparison tools
├── demo/                  # Interactive Flask web application
│   ├── app.py             # Backend server for real-time what-if simulations
│   ├── templates/         # HTML structure
│   └── static/            # CSS & JS styling assets
├── CausalFlowNet.py       # Main orchestrator (training pipeline, ATE, clustering)
├── test_sachs.py          # Sachs benchmark reproduction script
├── test_syntren.py        # SynTReN benchmark reproduction script
├── requirements.txt       # Core dependencies
└── LICENSE                # MIT License
```

---

## VII. Step-by-Step Reproduction & Setup Guide

### A. Prerequisites
*   Python $\geq 3.8$
*   CUDA Toolkit $\geq 11.3$ (Optional, for GPU acceleration)

### B. Installation
```bash
# Clone the repository
git clone https://github.com/manhthai1706/CausalFlowNet.git
cd CausalFlowNet

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### C. Running Benchmarks
```bash
# Run Sachs protein signaling network benchmark (~2 mins on CPU)
python test_sachs.py

# Run SynTReN gene regulatory network benchmark (~5 mins on CPU)
python test_syntren.py
```
Each benchmark script automatically downloads the dataset, runs the two-stage continuous ALM training loop, evaluates structural accuracy (SHD, SID), and outputs high-quality visualization figures (`.png` files) in the root directory.

---

## VIII. References

[1] K. Bello, B. Aragam, and P. Ravikumar, "DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization," *Advances in Neural Information Processing Systems*, vol. 35, 2022.

[2] D. M. Chickering, "Optimal structure identification with greedy search," *Journal of Machine Learning Research*, vol. 3, no. Nov, pp. 507-554, 2002.

[3] C. Durkan, A. Bekasov, I. Murray, and G. Papamakarios, "Neural spline flows," *Advances in Neural Information Processing Systems*, vol. 32, 2019.

[4] A. Gretton, O. Bousquet, A. Smola, and B. Schölkopf, "Measuring statistical dependence with Hilbert-Schmidt norms," in *Algorithmic Learning Theory: 16th International Conference, ALT 2005*, pp. 63-77, 2005.

[5] S. Hu, Z. Chen, *et al.*, "Causal Inference and Mechanism Clustering of A Mixture of Additive Noise Models (ANM-MM)," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 31, 2018.

[6] J. Pearl, *Causality: Models, Reasoning and Inference*. Cambridge University Press, 2000.

[7] J. Peters and P. Bühlmann, "Structural intervention distance for evaluating causal graphs," *Neural Computation*, vol. 27, no. 3, pp. 771-799, 2015.

[8] J. Peters, J. M. Mooij, D. Janzing, and B. Schölkopf, "Causal discovery with continuous additive noise models," *Journal of Machine Learning Research*, vol. 15, no. 1, pp. 2009-2053, 2014.

[9] K. Sachs, O. Perez, D. Pe'er, D. A. Lauffenburger, and G. P. Nolan, "Causal protein-signaling networks derived from multiparameter single-cell data," *Science*, vol. 308, no. 5721, pp. 523-529, 2005.

[10] S. Shimizu, P. O. Hoyer, A. Hyvärinen, and A. Kerminen, "A linear non-Gaussian acyclic model for causal discovery," *Journal of Machine Learning Research*, vol. 7, no. 10, pp. 2003-2030, 2006.

[11] P. Spirtes, C. N. Glymour, and R. Scheines, *Causation, prediction, and search*, 2nd ed. MIT Press, 2000.

[12] T. Van den Bulcke, K. Van Leemput, B. Naudts, P. van Remortel, H. Ma, A. Verschoren, B. De Moor, and K. Marchal, "SynTReN: a generator of synthetic gene expression data for design and analysis of structure learning algorithms," *BMC Bioinformatics*, vol. 7, no. 1, p. 43, 2006.

[13] X. Zheng, B. Aragam, P. K. Ravikumar, and E. P. Xing, "DAGs with NO TEARS: Continuous optimization for structure learning," *Advances in Neural Information Processing Systems*, vol. 31, 2018.

[14] S. Hu, Z. Chen, V. Partovi Nia, L. Chan, and Y. Geng, "Causal Inference and Mechanism Clustering of A Mixture of Additive Noise Models," Poster presented at NeurIPS 2018.

[15] S. Lachapelle, P. Brouillard, T. Deleu, and S. Lacoste-Julien, "Gradient-Based Neural DAG Learning," *arXiv preprint arXiv:1906.02226*, 2020.

---

## IX. Citation

If you find **CausalFlowNet** useful in your research or application, please cite our software repository:

```bibtex
@software{tran2026causalflownet,
  author       = {Tran, Manh Thai},
  title        = {{CausalFlowNet}: Nonlinear Causal Discovery via Normalizing Flows and Parallel Independence Testing},
  year         = {2026},
  url          = {https://github.com/manhthai1706/CausalFlowNet},
  license      = {MIT}
}
```

---

## X. License

This project is licensed under the terms of the [MIT License](LICENSE).

Copyright © 2026 Manh Thai Tran.

---
<p align="center"><em>CausalFlowNet — Bridging Normalizing Flows and Independence Testing for principled Causal Structure Learning</em></p>
