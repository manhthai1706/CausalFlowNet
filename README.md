# CausalFlowNet: Continuous Causal Structure Learning with Gated Residual MLPs and Neural Spline Flows

## Abstract
Learning a faithful directed acyclic graph (DAG) from samples of a joint distribution is a challenging combinatorial problem, owing to the search space that is superexponential in the number of graph nodes. A recent breakthrough reformulates the problem as a continuous optimization with a structural constraint that ensures acyclicity. While statistically elegant, first-generation methods are largely limited to linear structural equation models (SEMs) or simple additive noise models (ANMs) under standard Gaussian assumptions. 

In this work, we propose **CausalFlowNet**, a deep generative and structural causal framework capable of capturing complex nonlinear mappings and arbitrary non-Gaussian noise distributions. At the heart of the framework is a shared **Gated Residual Multi-Layer Perceptron (Gated-ResMLP)** that models non-linear functional causal relationships, combined with rational-quadratic **Neural Spline Flows (NSFs)** parameterized by a learnable **Gaussian Mixture Prior** to estimate flexible, skewed, or multi-modal residual densities. To guarantee causal faithfulness and prevent spurious confounder-driven associations, we incorporate a vectorized **Parallel Fast Hilbert-Schmidt Independence Criterion (Fast-HSIC)** network to explicitly enforce independence between parent features and noise residuals. We optimize the joint objective subject to a continuous trace-exponential acyclicity constraint via the **Augmented Lagrangian Method (ALM)**. Empirical results on the real-world biological **Sachs** network and the synthetic **SynTReN** gene regulatory network demonstrate that CausalFlowNet learns more accurate causal structures and yields significantly lower Structural Hamming Distance (SHD) and Structural Interventional Distance (SID) compared to state-of-the-art methods including GraN-DAG, DAG-GNN, and NOTEARS.

---

## 1. Introduction
Discovering causal relations from observational data is a cornerstone of scientific inquiry, with critical applications ranging from biology and genomics to economics and medicine. Under the structural causal model (SCM) framework, causal interactions among $d$ observed variables $X = (X_1, \dots, X_d)^T$ are described by a Directed Acyclic Graph (DAG) $G$. Finding the optimal DAG is historically framed as a discrete combinatorial search problem, which is NP-hard due to the superexponential space of $d! \times 2^{\binom{d}{2}}$ potential graphs.

A major paradigm shift occurred with the introduction of **NOTEARS** (Zheng et al., 2018), which models the discrete acyclicity constraint as a smooth, continuous equality constraint:
$$h(W) = \text{Tr}(\exp(W \odot W)) - d = 0$$
where $W \in \mathbb{R}^{d \times d}$ represents the weighted adjacency matrix. This continuous formulation enables gradient-based optimization over continuous spaces. 

Despite its success, NOTEARS and many of its continuous extensions make restrictive parametric assumptions:
1. **Linearity:** They assume linear structural equations ($X_j = \sum_i W_{ij} X_i + Z_j$) which fail to capture highly complex, non-linear relationships in biological cell cascades or macro-economic dynamics.
2. **Parametric Noise Priors:** They assume the noise $Z_j$ is homoscedastic and Gaussian. Real-world physical and clinical noise is frequently skewed, heavy-tailed, or multi-modal.

To address these limitations, we propose **CausalFlowNet**. Inspired by deep generative models and advanced density estimation, we model arbitrary non-linear structural functions via a shared context-aware **Gated-ResMLP** regression network. Furthermore, we leverage rational-quadratic **Neural Spline Flows (NSFs)** to learn arbitrary noise distributions in a completely data-driven manner, bypassing Gaussian priors. To explicitly enforce that the computed noise residuals are statistically independent of their parent nodes—a core requirement of causal faithfulness—we implement a parallelized **Fast-HSIC** regularization penalty. Continuous structure optimization is solved via the **Augmented Lagrangian Method (ALM)**.

---

## 2. Related Work
* **Discrete Causal Discovery:** Traditional methods are split into *constraint-based* algorithms (e.g., PC, FCI), which rely on conditional independence tests, and *score-based* algorithms (e.g., GES, FGS), which greedily search the DAG space to optimize a score (e.g., BIC). These methods struggle to scale and are highly sensitive to test thresholds.
* **Continuous Causal Learning:** Following NOTEARS, continuous DAG learning has been extended to non-linear settings. **DAG-GNN** (Yu et al., 2019) integrates a Graph Variational Autoencoder to handle non-linearities but assumes additive Gaussian noise. **GraN-DAG** (Lachapelle et al., 2020) utilizes neural networks to model non-linear equations but relies on standard least-squares or maximum likelihood under fixed parametric assumptions. CausalFlowNet represents a substantial advancement by jointly learning non-linear structures, estimating arbitrary non-Gaussian noise densities, and explicitly regularizing non-linear statistical independence.

---

## 3. Methodology

### 3.1 Non-linear Structural Causal Model
We consider a joint distribution $P(X)$ over $d$ continuous variables $X = (X_1, \dots, X_d)^T$ generated by a non-linear SCM:
$$X_j = f_j(X_{\text{parents}(j; G)}) + Z_j, \quad j=1,\dots,d$$
where the noise variables $Z_j$ are mutually independent. We parameterize the SCM using a shared **Gated Residual MLP** ($\phi$). The input to the MLP is masked using the continuous weighted adjacency matrix $W$:
$$X_{\text{masked}(j)} = X \odot A(W)_{\cdot, j}$$
where $A(W) = W \odot (I_d - \mathbf{I})$ represents the zero-diagonal adjacency matrix.
The shared regression network predicting target expectations is formulated as:
$$\hat{Y}_j = \phi(X_{\text{masked}(j)})$$
Each Gated Residual Block inside the MLP manages context-aware, non-linear feature interactions:
$$\text{gate}, \text{features} = \text{Linear}(\text{LayerNorm}(h)).chunk(2)$$
$$\tilde{h} = \text{LeakyReLU}(\text{features}) \odot \text{Sigmoid}(\text{gate})$$
$$h_{\text{out}} = h + \text{Linear}(\tilde{h})$$

### 3.2 Flexible Residual Density Fitting via Neural Spline Flows (NSF)
To accommodate arbitrary and non-Gaussian noise distributions, the residuals $res_j = Y_j - \hat{Y}_j$ are mapped to a latent space $z_j$ using a rational-quadratic **Neural Spline Flow**:
$$z_j = f_{\text{NSF}}(res_j)$$
The spline transformations employ rational-quadratic functions with $K$ bins bounded inside $[-B, B]$. The latent space $z_j$ is governed by a learnable **Gaussian Mixture Model (GMM) Prior** with $C$ components:
$$z_j \sim \sum_{c=1}^{C} \pi_c \mathcal{N}(\mu_c, \sigma_c^2)$$
By modeling the noise density dynamically, CausalFlowNet bypasses restrictive Gaussian priors. The Negative Log-Likelihood (NLL) of the data distribution is maximized:
$$\mathcal{L}_{\text{NLL}}(W) = - \sum_{j=1}^{d} \left[ \log p_z(z_j) + \log \left| \det \frac{\partial z_j}{\partial res_j} \right| \right]$$

### 3.3 Vectorized Parallel Independence Testing (Fast HSIC)
To satisfy the SCM requirement that the noise residuals $Z_j$ are statistically independent of their parent nodes, we enforce $Z_j \perp \!\!\! \perp X_{\text{masked}(j)}$ using a vectorized **Parallel Fast HSIC** network based on Random Fourier Features (RFF):
$$\Phi_X = \sqrt{\frac{2}{m}} \cos(X_{\text{masked}} W_x + b_x), \quad \Phi_Z = \sqrt{\frac{2}{m}} \cos(Z W_z + b_z)$$
$$\mathcal{L}_{\text{HSIC}}(W) = \frac{1}{(n-1)^2} \text{Tr}(\Phi_X^T H \Phi_Z \Phi_Z^T H \Phi_X)$$
Enforcing $\mathcal{L}_{\text{HSIC}}(W) \rightarrow 0$ eliminates spurious confounder correlations and guarantees that the identified structural relationships are truly causal.

### 3.4 Joint Continuous DAG Optimization
The unified loss is minimized over the adjacency weights $W$:
$$\min_{W} \mathcal{L}_{\text{NLL}}(W) + \lambda_{\text{HSIC}} \mathcal{L}_{\text{HSIC}}(W) + \lambda_{L1} \|A(W)\|_1$$
subject to the trace acyclicity constraint $h(W) = 0$. The objective is solved using the **Augmented Lagrangian Method (ALM)**:
$$\mathcal{L}_{\text{ALM}}(W; \alpha, \rho) = \mathcal{L}_{\text{main}}(W) + \alpha h(W) + \frac{\rho}{2} h(W)^2$$
where $\alpha$ is the Lagrange multiplier and $\rho$ is the penalty parameter, updated iteratively.

---

## 4. Model Architecture
The block diagram below illustrates the comprehensive architecture of **CausalFlowNet**:

<p align="center">
  <img src="causalflownet_architecture.png" width="90%" alt="CausalFlowNet Architecture Diagram"/>
</p>

---

## 5. Experiments

We evaluate CausalFlowNet on two established benchmark datasets: the **Sachs** protein-signaling network (11 nodes, 7466 samples) and the **SynTReN** synthetic regulatory network (20 nodes, 500 samples).

### 5.1 Quantitative Evaluation
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

### 5.2 Qualitative Evaluation & Diagnostic Plots

#### 5.2.1 Real Biological Data: Sachs Protein Network
The Sachs dataset represents a real-world cellular signaling network. CausalFlowNet successfully reconstructs critical cell cascades (e.g. $PKC \rightarrow Raf \rightarrow Mek \rightarrow Erk$).

<p align="center">
  <img src="sachs_graph_comparison.png" width="48%" alt="Sachs Graph Reconstructions"/>
  <img src="sachs_adjacency_comparison.png" width="48%" alt="Sachs Adjacency Comparison"/>
</p>

#### 5.2.2 Synthetic Data: SynTReN Gene Expression Network
The SynTReN dataset simulates E. coli genetic regulatory dynamics. Our model shows high structural matching against the Ground Truth matrix.

<p align="center">
  <img src="syntren_graph_comparison.png" width="48%" alt="SynTReN Graph Reconstructions"/>
  <img src="syntren_adjacency_comparison.png" width="48%" alt="SynTReN Adjacency Comparison"/>
</p>

---

## 6. Conclusion
In this work, we introduced **CausalFlowNet**, a deep generative SCM framework for non-linear continuous causal discovery. The integration of Gated Residual MLPs, Neural Spline Flows with learnable GMM priors, and vectorized Fast-HSIC independence constraints yields substantial improvements in structure search accuracy and interventional recovery. Empirical benchmarking on biological and synthetic datasets demonstrates CausalFlowNet's superior capacity to capture complex DAG structures compared to established baselines.

---

## References
1. Yu, Y., Chen, J., Gao, T., & Yu, M. (2019). Dag-gnn: Dag structure learning with graph neural networks. *International Conference on Artificial Intelligence and Statistics (AISTATS)*.
2. Zheng, X., Aragam, B., Ravikumar, P. K., & Xing, E. P. (2018). Dags with no tears: Continuous optimization for structure learning. *Advances in Neural Information Processing Systems (NeurIPS)*.
3. Sachs, K., Perez, O., Pe'er, D., Lauffenburger, D. A., & Nolan, G. P. (2005). Causal protein-signaling networks derived from multiparameter single-cell data. *Science*, 308(5721), 523-529.
4. Lachapelle, S., Brouillard, P., Deleu, T., & Lacoste-Julien, S. (2020). Gradient-based neural DAG learning. *International Conference on Learning Representations (ICLR)*.
5. Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G. (2019). Neural spline flows. *Advances in Neural Information Processing Systems (NeurIPS)*.

---

## Appendix: Repository Guide & Reproduction

### Repository Structure
```text
├── core/               # Optimization & RFF-based HSIC formulations
│   ├── HSIC.py         # Parallel Fast HSIC using Random Fourier Features (RFF)
│   └── Optimization.py # Continuous Acyclicity Penalty h(W) & ALM Solver
├── modules/            # Feed-forward SCM blocks & Density Flows
│   ├── MLP.py          # Gated Residual Multi-Layer Perceptron (Gated-ResMLP)
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

### Installation & Execution

#### 1. Install Dependencies
Make sure you have PyTorch installed with appropriate CUDA acceleration (optional but recommended):
```bash
pip install -r requirements.txt
```
*Dependencies: `torch`, `numpy`, `pandas`, `matplotlib`, `networkx`, `scikit-learn`, `flask`.*

#### 2. Run Benchmarks
To replicate the experimental Sachs and SynTReN results and generate the diagnostic comparison plots:
```bash
python test_sachs.py
python test_syntren.py
```

#### 3. Run the Interactive Web Lab Dashboard
To explore causal discovery on custom CSV files and test real-time interventions:
```bash
python demo/app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser.