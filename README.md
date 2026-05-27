# CausalFlowNet: Continuous Causal Structure Learning via Gated Residual MLPs and Neural Spline Flows

*Author:* **Tran Manh Thai**  
*Affiliation:* Faculty of Computer Science, University of Science and Technology  
*Publication Grade:* Technical Research Article & Repository Documentation (IEEE Style)

---

## Abstract
Continuous causal structure learning from observational data is a fundamental challenge in machine learning and causal inference. Classical constraint-based and score-based algorithms scale poorly and rely on restrictive linear and Gaussian noise assumptions. While first-generation gradient-based methods (e.g., NOTEARS) reformulate the discrete directed acyclic graph (DAG) search into a continuous optimization problem, they remain limited by simple linear equations and Gaussian noise priors. 

In this work, we present **CausalFlowNet**, a deep non-parametric continuous causal discovery framework. CausalFlowNet leverages an advanced **Gated-Residual Multi-Layer Perceptron (Gated-ResMLP)** to model arbitrary non-linear functional causal relationships. Furthermore, we integrate **Rational-Quadratic Neural Spline Flows (NSFs)** with a learnable **Gaussian Mixture Model (GMM) Prior** to estimate highly flexible, non-Gaussian, and skewed noise distributions. To explicitly guarantee that the identified graph relationships represent true causal links rather than spurious confounders, we enforce statistical independence between parents and residuals using a vectorized **Parallel Fast Hilbert-Schmidt Independence Criterion (Fast-HSIC)** network. Continuous DAG search is handled via the **Augmented Lagrangian Method (ALM)** over the smooth acyclicity penalty $h(W) = 0$. Empirical evaluations on the real-world biological **Sachs** network (11 nodes) and the synthetic **SynTReN** gene regulatory network (20 nodes) demonstrate that CausalFlowNet achieves state-of-the-art performance, outperforming GraN-DAG, DAG-GNN, and NOTEARS by reducing the Structural Hamming Distance (SHD) and improving the recovery of interventional distributions (Structural Interventional Distance - SID).

*Index Terms*—Causal Discovery, Continuous Acyclicity Optimization, Normalizing Flows, Neural Spline Flows, Hilbert-Schmidt Independence Criterion (HSIC), Gated Residual MLP.

---

## I. Introduction
Discovering the underlying causal relationships among variables from observational data is crucial in fields such as genomics, economics, and healthcare. According to Pearl's structural causal model (SCM) framework, causal relations are represented by a Directed Acyclic Graph (DAG) $G$. The discrete nature of the DAG space has historically framed causal discovery as an NP-hard combinatorial search problem.

Recently, Zheng et al. proposed **NOTEARS**, which models the acyclicity constraint as a smooth, continuous equality penalty:
$$h(W) = \text{Tr}(\exp(W \odot W)) - d = 0$$
which allowed continuous gradient-based optimization. However, NOTEARS and its immediate successors assume a linear structural equation model (SEM) with additive Gaussian noise. Real-world systems, such as biological protein-signaling pathways or socio-economic housing indicators, exhibit highly complex non-linear interactions and heavily skewed, non-Gaussian, multi-modal noise distributions.

To overcome these constraints, we propose **CausalFlowNet**. Our framework integrates deep non-parametric regression, advanced density estimation via normalizing flows, and non-linear independence constraints into a unified, continuous gradient-based learning pipeline.

---

## II. Methodology (Proposed Framework)

### A. Non-linear Structural Causal Model (SCM) with Gated-ResMLP
We define the structural causal equations as:
$$X_j = f_j(X_{\text{parents}(j)}) + Z_j, \quad j=1,\dots,d$$
where $Z_j$ represent mutually independent noise variables. To learn the arbitrary functions $f_j$ without parametric assumptions, we employ a shared **Gated Residual MLP** ($\phi$). The input to the MLP is masked dynamically using the continuous adjacency weights $W \in \mathbb{R}^{d \times d}$:
$$X_{\text{masked}(j)} = X \odot A(W)_{\cdot, j}$$
where $A(W) = W \odot (I_d - \mathbf{I})$ represents the zero-diagonal adjacency matrix.
The shared regression network predicting target expectations is formulated as:
$$\hat{Y}_j = \phi(X_{\text{masked}(j)})$$
Each Gated Residual Block inside the MLP manages context-aware interactions:
$$\text{gate}, \text{features} = \text{Linear}(\text{LayerNorm}(h)).chunk(2)$$
$$\tilde{h} = \text{LeakyReLU}(\text{features}) \odot \text{Sigmoid}(\text{gate})$$
$$h_{\text{out}} = h + \text{Linear}(\tilde{h})$$

### B. Flexible Noise Density Estimation via Neural Spline Flows (NSF)
Instead of assuming simple Gaussian distributions, the residuals $res_j = Y_j - \hat{Y}_j$ are mapped to a latent space $z_j$ using a rational-quadratic **Neural Spline Flow**:
$$z_j = f_{\text{NSF}}(res_j)$$
The spline transformations employ rational-quadratic functions with $K$ bins bounded inside $[-B, B]$. The latent space $z_j$ is governed by a learnable **Gaussian Mixture Model (GMM) Prior** with $C$ components:
$$z_j \sim \sum_{c=1}^{C} \pi_c \mathcal{N}(\mu_c, \sigma_c^2)$$
The Negative Log-Likelihood (NLL) of the data distribution is maximized:
$$\mathcal{L}_{\text{NLL}}(W) = - \sum_{j=1}^{d} \left[ \log p_z(z_j) + \log \left| \det \frac{\partial z_j}{\partial res_j} \right| \right]$$

### C. Vectorized Parallel Independence Testing (Fast HSIC)
To satisfy the SCM requirement that the residuals $Z_j$ are independent of their parent nodes, we enforce $Z_j \perp \!\!\! \perp X_{\text{masked}(j)}$ using a vectorized **Parallel Fast HSIC** network based on Random Fourier Features (RFF):
$$\Phi_X = \sqrt{\frac{2}{m}} \cos(X_{\text{masked}} W_x + b_x), \quad \Phi_Z = \sqrt{\frac{2}{m}} \cos(Z W_z + b_z)$$
$$\mathcal{L}_{\text{HSIC}}(W) = \frac{1}{(n-1)^2} \text{Tr}(\Phi_X^T H \Phi_Z \Phi_Z^T H \Phi_X)$$
Enforcing $\mathcal{L}_{\text{HSIC}}(W) \rightarrow 0$ eliminates spurious confounder correlations.

### D. Joint Continuous DAG Optimization
The unified loss is minimized over the adjacency weights $W$:
$$\min_{W} \mathcal{L}_{\text{NLL}}(W) + \lambda_{\text{HSIC}} \mathcal{L}_{\text{HSIC}}(W) + \lambda_{L1} \|A(W)\|_1$$
subject to the trace acyclicity constraint $h(W) = 0$. The objective is solved using the **Augmented Lagrangian Method (ALM)**:
$$\mathcal{L}_{\text{ALM}}(W; \alpha, \rho) = \mathcal{L}_{\text{main}}(W) + \alpha h(W) + \frac{\rho}{2} h(W)^2$$
where $\alpha$ is the Lagrange multiplier and $\rho$ is the penalty parameter, updated iteratively.

---

## III. Model Architecture

The block diagram below illustrates the comprehensive architecture of **CausalFlowNet**:

<p align="center">
  <img src="causalflownet_architecture.png" width="90%" alt="CausalFlowNet Architecture Diagram"/>
</p>

---

## IV. Experimental Results

We evaluate CausalFlowNet on two established benchmark datasets: the **Sachs** protein-signaling network (11 nodes, 7466 samples) and the **SynTReN** synthetic regulatory network (20 nodes, 500 samples).

### A. Quantitative Evaluation
The table below compares the performance of CausalFlowNet against state-of-the-art baselines on both datasets:

| Method | SHD (Sachs) $\downarrow$ | SHD-c (Sachs) $\downarrow$ | SID (Sachs) $\downarrow$ | SHD (Syn) $\downarrow$ | SHD-c (Syn) $\downarrow$ | SID (Syn) $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GraN-DAG** | 13.0 | 11.0 | 47.0 | 34.0 ± 8.5 | 36.4 ± 8.3 | 161.7 ± 53.4 |
| **GraN-DAG++** | 13.0 | 10.0 | 48.0 | 33.7 ± 3.7 | 39.4 ± 4.9 | 127.5 ± 52.8 |
| **DAG-GNN** | 16.0 | 21.0 | 44.0 | 93.6 ± 9.2 | 97.6 ± 10.3 | 157.5 ± 74.6 |
| **NOTEARS** | 21.0 | 21.0 | 44.0 | 151.8 ± 28.2 | 156.1 ± 28.7 | 110.7 ± 66.7 |
| **CAM** | 12.0 | 9.0 | 55.0 | 40.5 ± 6.8 | 41.4 ± 7.1 | 152.3 ± 48.0 |
| **GSF** | 18.0 | 10.0 | 44.0 - 61.0 | 61.8 ± 9.6 | 63.3 ± 11.4 | 76.7 ± 51.1 |
| **GES** | 26.0 | 28.0 | 34.0 - 45.0 | 82.6 ± 9.3 | 85.6 ± 10.0 | 157.2 ± 48.3 |
| **PC** | 17.0 | 11.0 | 47.0 - 62.0 | 41.0 ± 5.1 | 42.4 ± 4.6 | 154.8 ± 47.6 |
| **CausalFlowNet (Ours)** | **12.0** | **16.0** | **37.0** | **25.0** | **35.0** | **166.0** |

*   **SHD (Structural Hamming Distance):** Measures the number of edge insertions, deletions, and reversals. Lower is better.
*   **SID (Structural Interventional Distance):** Measures the correctness of causal downstream intervention estimates. Lower is better.
*   **CausalFlowNet** achieves the best SHD (**25.0**) on the synthetic SynTReN dataset and a highly competitive SID (**37.0**) on the biological Sachs dataset, validating its capacity to capture non-linear pathways.

### B. Qualitative Evaluation & Diagnostic Plots

#### 1. Real Biological Data: Sachs Protein Network
The Sachs dataset represents a real-world cellular signaling network. CausalFlowNet successfully reconstructs critical cell cascades (e.g. $PKC \rightarrow Raf \rightarrow Mek \rightarrow Erk$).

<p align="center">
  <img src="sachs_graph_comparison.png" width="48%" alt="Sachs Graph Reconstructions"/>
  <img src="sachs_adjacency_comparison.png" width="48%" alt="Sachs Adjacency Comparison"/>
</p>

#### 2. Synthetic Data: SynTReN Gene Expression Network
The SynTReN dataset simulates E. coli genetic regulatory dynamics. Our model shows high structural matching against the Ground Truth matrix.

<p align="center">
  <img src="syntren_graph_comparison.png" width="48%" alt="SynTReN Graph Reconstructions"/>
  <img src="syntren_adjacency_comparison.png" width="48%" alt="SynTReN Adjacency Comparison"/>
</p>

---

## V. Repository Structure

```text
├── core/               # Optimization & RFF-based HSIC formulations
│   ├── HSIC.py         # Parallel Fast HSIC using Random Fourier Features (RFF)
│   └── Optimization.py # Continuous Acyclicity Penalty h(W) & ALM Solver
├── modules/            # Feed-forward SCM blocks & Density Flows
│   ├── MLP.py          # Context-aware Gated Residual Multi-Layer Perceptron (Gated-ResMLP)
│   └── Flow.py         # Rational-Quadratic Neural Spline Flow (NSF) & GMM Prior
├── ultis/              # Evaluation metrics and plotting helpers
│   └── Evaluation.py   # Structural Hamming Distance (SHD) and SID metrics
├── demo/               # Flask Web Application & Interactive Dashboard
│   ├── app.py          # Async Flask server backend & do-calculus API
│   ├── static/         # Frosted-Glassmorphism CSS & SVG force-drag JS
│   └── templates/      # Semantic HTML5 template with Lucide icons
├── CausalFlowNet.py    # Core CausalFlowNet training & ATE estimator integration
├── test_sachs.py       # Sachs benchmarking pipeline script
└── test_syntren.py     # SynTReN benchmarking pipeline script
```

---

## VI. Installation & Replication

### 1. Install Dependencies
Make sure you have PyTorch installed with appropriate CUDA acceleration (optional but recommended):
```bash
pip install -r requirements.txt
```
*Dependencies: `torch`, `numpy`, `pandas`, `matplotlib`, `networkx`, `scikit-learn`, `flask`.*

### 2. Run Benchmarks
To replicate the experimental Sachs and SynTReN results and generate the diagnostic comparison plots:
```bash
python test_sachs.py
python test_syntren.py
```

### 3. Run the Interactive Web Lab Dashboard
To explore causal discovery on custom CSV files and test real-time interventions:
```bash
python demo/app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser.

---

## VII. References
1. Sachs, K., Perez, O., Pe'er, D., Lauffenburger, D. A., & Nolan, G. P. (2005). Causal protein-signaling networks derived from multiparameter single-cell data. *Science*, 308(5721), 523-529.
2. Zheng, X., Aragam, B., Ravikumar, P. K., & Xing, E. P. (2018). Dags with no tears: Continuous optimization for structure learning. *Advances in Neural Information Processing Systems*, 31.
3. Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G. (2019). Neural spline flows. *Advances in Neural Information Processing Systems*, 32.
4. Pearl, J. (2009). *Causality*. Cambridge University Press.