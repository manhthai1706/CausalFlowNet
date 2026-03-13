import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from CausalFlowNet import CausalFlowNet
from ultis.Evaluation import compute_metrics
from ultis.visualize import plot_dag # Import premium visualizer / Nhập bộ vẽ hình cao cấp

# Configuration
CONFIG = {
    'N': 2000,
    'n_vars': 20, # Increased to 20 nodes / Tăng lên 20 nút
    'n_clusters': 4, # More nodes may need more clusters / Nhiều nút hơn cần nhiều cụm hơn
    'flow_bins': 10,
    'lda_hsic': 0.05,
    'stage1_epochs': 40, # Increase epochs for complexity / Tăng vòng lặp cho độ phức tạp
    'stage2_epochs': 25,
    'l1_stage1': 0.001,
    'l1_stage2': 0.012,
    'seed': 42
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_complex_regulatory_data(N=2000):
    """Generates a 20-node complex regulatory network / Tạo mạng lưới điều hòa 20 gene."""
    set_seed(CONFIG['seed'])
    
    # Transcription Factors (Roots) / Các yếu tố phiên mã gốc
    X0 = np.random.gamma(shape=2.0, scale=1.0, size=(N, 1))
    X1 = np.random.normal(0, 1, size=(N, 1))
    
    # Layer 1
    X2 = (4.0 * X0**2) / (1.5**2 + X0**2) + 0.1 * np.random.normal(size=(N, 1))
    X3 = 2.0 / (1.0 + np.exp(-2.0 * X1)) + 0.1 * np.random.normal(size=(N, 1))
    X10 = np.cos(X0) + 0.1 * np.random.normal(size=(N, 1))
    
    # Layer 2
    X4 = 0.5 * X2 + 0.5 * X3 + 0.2 * X2 * X3 + 0.1 * np.random.normal(size=(N, 1))
    X5 = np.sin(X0) + X1**2 + 0.1 * np.random.normal(size=(N, 1))
    X11 = X10 * X2 + 0.1 * np.random.normal(size=(N, 1))
    
    # Layer 3
    X6 = np.tanh(X4) + 0.1 * np.random.normal(size=(N, 1))
    X7 = np.exp(-X5**2) + 0.1 * np.random.normal(size=(N, 1))
    X12 = 0.7 * X11 + 0.3 * X3 + 0.1 * np.random.normal(size=(N, 1))
    X13 = X5 + X6 + 0.1 * np.random.normal(size=(N, 1))
    
    # Layer 4
    X8 = 0.3 * X6 + 0.7 * X7 + 0.1 * np.random.normal(size=(N, 1))
    X9 = X8**3 + 0.1 * np.random.normal(size=(N, 1))
    X14 = np.maximum(0, X12) + 0.1 * np.random.normal(size=(N, 1))
    X15 = np.log1p(np.abs(X13)) + 0.1 * np.random.normal(size=(N, 1))
    
    # Layer 5 (Additional interactions for 20 nodes) / Các tương tác bổ sung
    X16 = X14 * X15 + 0.1 * np.random.normal(size=(N, 1))
    X17 = 0.5 * X9 + 0.5 * X16 + 0.1 * np.random.normal(size=(N, 1))
    X18 = np.sqrt(np.abs(X7)) + 0.1 * np.random.normal(size=(N, 1))
    X19 = X17 + X18 + 0.1 * np.random.normal(size=(N, 1))
    
    data = np.hstack([X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, X18, X19])
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    
    # True Adjacency Matrix (20x20) / Ma trận kề thực tế
    true_adj = np.zeros((20, 20))
    # Original 10 edges / 10 cạnh ban đầu
    edges = [
        (0,2), (1,3), (2,4), (3,4), (0,5), (1,5), (4,6), (5,7), (6,8), (7,8), (8,9),
        # New 13 edges for 20 nodes / 13 cạnh mới cho 20 nút
        (0,10), (10,11), (2,11), (11,12), (3,12), (5,13), (6,13), (12,14), (13,15),
        (14,16), (15,16), (9,17), (16,17), (7,18), (17,19), (18,19)
    ]
    for u, v in edges:
        true_adj[u, v] = 1
        
    return torch.tensor(data, dtype=torch.float32, device=device), true_adj

def run_experiment():
    print("=== CausalFlowNet Experiment: Complex Regulatory (SynTReN-like) ===")
    set_seed(CONFIG['seed'])
    
    # 1. Generate Data
    X, true_adj = generate_complex_regulatory_data(N=CONFIG['N'])
    n_vars = X.shape[1]
    node_names = [f"Gene_{i}" for i in range(n_vars)]

    # 2. Model Initialization (Gated-ResMLP)
    model = CausalFlowNet(
        n_vars=n_vars, 
        lda_hsic=CONFIG['lda_hsic'], 
        n_clusters=CONFIG['n_clusters'], 
        flow_bins=CONFIG['flow_bins']
    )

    # 3. Two-Stage Training
    print(f"\n[Step 1] Aggressive Discovery (L1={CONFIG['l1_stage1']})")
    model.fit(X.cpu().numpy(), outer_epochs=CONFIG['stage1_epochs'], l1_reg=CONFIG['l1_stage1'])
    
    print(f"\n[Step 2] Structural Refinement (L1={CONFIG['l1_stage2']})")
    adj_weights = model.fit(X.cpu().numpy(), outer_epochs=CONFIG['stage2_epochs'], l1_reg=CONFIG['l1_stage2'])

    # 4. Adaptive Thresholding
    abs_weights = np.abs(adj_weights)
    mask = ~np.eye(n_vars, dtype=bool)
    significant_weights = abs_weights[mask]
    threshold = np.mean(significant_weights) + 1.0 * np.std(significant_weights)
    
    est_adj = (abs_weights > threshold).astype(int)
    metrics = compute_metrics(true_adj, est_adj)
    
    # 5. Display Results / Hiển thị kết quả
    display_results(model, X.cpu().numpy(), metrics, true_adj, est_adj, adj_weights, node_names, threshold)
    
    # 6. Compute ATE Matrix for visualization / Tính toán ma trận ATE để trực quan hóa
    ate_matrix = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(n_vars):
            if est_adj[i, j] == 1:
                ate_matrix[i, j] = model.estimate_ate(X.cpu().numpy(), i, j)
    
    # 7. Visual Comparison / So sánh bằng hình ảnh
    visualize_comparison(true_adj, est_adj, node_names, adj_weights, ate_matrix, metrics)

def visualize_comparison(true_adj, est_adj, node_names, adj_weights, ate_matrix, metrics):
    """
    Generate professional visualizations for SynTReN.
    Tạo hình ảnh chuyên nghiệp cho SynTReN.
    """
    # 1. Adjacency Matrix Visualization / Trực quan hóa ma trận kề
    plt.figure(figsize=(8, 7))
    sns.heatmap(est_adj, annot=False, cbar=False, cmap="Reds", 
                xticklabels=False, yticklabels=False)
    plt.title("SynTReN Estimated Matrix")
    
    plt.tight_layout()
    plt.savefig("syntren_adjacency_comparison.png")
    print(f"\n[Artifact] Estimated adjacency matrix saved to syntren_adjacency_comparison.png")
    plt.close()

    # 2. Premium Causal Graph with ATE Labels
    plot_dag(
        W_matrix=adj_weights, 
        labels=node_names, 
        GT_matrix=true_adj, 
        ate_matrix=ate_matrix,
        metrics=metrics,
        title="SynTReN-20 Discovery",
        threshold=0.08, # Discovery threshold
        save_path="syntren_graph_comparison.png",
        figure_size=(16, 12),
        node_size=1200
    )
    print(f"[Artifact] Professional graph with ATE labels and Metrics saved to syntren_graph_comparison.png")

def display_results(model, data, metrics, true_adj, est_adj, weights, node_names, threshold):
    n_vars = true_adj.shape[0]
    print("\n" + "="*40)
    print("           SYNTREN EXPERIMENT SUMMARY")
    print("="*40)
    print(f"TPR: {metrics['tpr']:.2f} | FPR: {metrics['fpr']:.2f} | FDR: {metrics['fdr']:.2f}")
    print(f"SHD: {metrics['shd']} | SHD-c: {metrics['shd_c']} | SID: {metrics['sid']}")
    print(f"Total True Edges: {int(np.sum(true_adj))}")
    print(f"Total Estimated Edges: {int(np.sum(est_adj))}")
    print(f"Adaptive Threshold: {threshold:.4f}")
    print("="*40)
    
    print("\nTop Discovered Regulatory Edges (with ATE):")
    for i in range(n_vars):
        for j in range(n_vars):
            if est_adj[i, j] == 1:
                # Estimate ATE for regulatory impact / Ước lượng ATE cho tác động điều hòa
                ate = model.estimate_ate(data, i, j)
                status = "[V]" if true_adj[i, j] == 1 else "[X]"
                print(f"  {status} {node_names[i]:>8} -> {node_names[j]:<8} (w: {weights[i, j]:+.3f} | ATE: {ate:+.3f})")

if __name__ == "__main__":
    run_experiment()
