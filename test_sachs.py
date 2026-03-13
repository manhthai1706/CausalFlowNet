import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from CausalFlowNet import CausalFlowNet
from ultis.Evaluation import compute_metrics
from ultis.visualize import plot_dag # Import premium visualizer / Nhập bộ vẽ hình cao cấp

# Configuration
CONFIG = {
    'data_path': 'https://raw.githubusercontent.com/FenTechSolutions/CausalDiscoveryToolbox/master/cdt/data/resources/cyto_full_data.csv',
    'target_path': 'https://raw.githubusercontent.com/FenTechSolutions/CausalDiscoveryToolbox/master/cdt/data/resources/cyto_full_target.csv', 
    'n_clusters': 5,
    'flow_bins': 12,
    'lda_hsic': 0.03,
    'stage1_epochs': 30,
    'stage2_epochs': 20,
    'l1_stage1': 0.001,
    'l1_stage2': 0.012,
    'threshold': 0.05
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data():
    """Load and prepare Sachs dataset from remote URLs."""
    print(f"Loading data from: {CONFIG['data_path']}")
    df = pd.read_csv(CONFIG['data_path'])
    node_names = list(df.columns)
    data = df.values
    data = (data - data.mean(axis=0)) / data.std(axis=0) # Standardize
    
    col_to_idx = {col: i for i, col in enumerate(node_names)}
    true_adj = np.zeros((len(node_names), len(node_names)))
    
    print(f"Loading target from: {CONFIG['target_path']}")
    target_df = pd.read_csv(CONFIG['target_path'])
    for _, row in target_df.iterrows():
        cause, effect = row['Cause'], row['Effect']
        if cause in col_to_idx and effect in col_to_idx:
            true_adj[col_to_idx[cause], col_to_idx[effect]] = 1
                
    return torch.tensor(data, dtype=torch.float32, device=device), true_adj, node_names

def run_experiment():
    print("=== CausalFlowNet V2 Experiment: Sachs Dataset ===")
    set_seed(42)
    
    # 1. Load Data
    X, true_adj, node_names = load_data()
    n_vars = X.shape[1]
    print(f"Nodes: {node_names}")

    # 2. Model Initialization
    model = CausalFlowNet(
        n_vars=n_vars, 
        lda_hsic=CONFIG['lda_hsic'], 
        n_clusters=CONFIG['n_clusters'], 
        flow_bins=CONFIG['flow_bins']
    )

    # 3. Two-Stage Training Pipeline
    print(f"\n[Step 1] Aggressive Discovery (L1={CONFIG['l1_stage1']})")
    model.fit(X.cpu().numpy(), outer_epochs=CONFIG['stage1_epochs'], l1_reg=CONFIG['l1_stage1'])
    
    print(f"\n[Step 2] Structural Refinement (L1={CONFIG['l1_stage2']})")
    adj_weights = model.fit(X.cpu().numpy(), outer_epochs=CONFIG['stage2_epochs'], l1_reg=CONFIG['l1_stage2'])

    # 4. Clustering Analysis
    labels = model.predict_clusters(X.cpu().numpy(), n_clusters=CONFIG['n_clusters'])
    print(f"\nCausal Clusters: {np.unique(labels, return_counts=True)}")

    # 5. Adaptive Thresholding
    abs_weights = np.abs(adj_weights)
    # Exclude diagonal
    mask = ~np.eye(n_vars, dtype=bool)
    significant_weights = abs_weights[mask]
    
    # Adaptive Threshold: Mean + 0.8 * Std of absolute weights
    threshold = np.mean(significant_weights) + 0.8 * np.std(significant_weights)
    print(f"\n[Step 3] Adaptive Threshold Calculated: {threshold:.4f}")
    
    est_adj = (abs_weights > threshold).astype(int)
    metrics = compute_metrics(true_adj, est_adj)
    
    display_results(model, X.cpu().numpy(), metrics, true_adj, est_adj, adj_weights, node_names)

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
    Generate professional visualizations using the unified visualize.py suite.
    Tạo hình ảnh chuyên nghiệp bằng bộ công cụ visualize.py thống nhất.
    """
    # 1. Adjacency Matrix Comparison (Heatmaps) / So sánh ma trận kề (Bản đồ nhiệt)
    plt.figure(figsize=(16, 7))
    plt.subplot(1, 2, 1)
    sns.heatmap(true_adj, annot=True, cbar=False, cmap="Blues", 
                xticklabels=node_names, yticklabels=node_names)
    plt.title("Ground Truth Adjacency Matrix\n(Ma trận kề thực tế)")
    
    plt.subplot(1, 2, 2)
    sns.heatmap(est_adj, annot=True, cbar=False, cmap="Reds", 
                xticklabels=node_names, yticklabels=node_names)
    plt.title("Estimated Adjacency Matrix\n(Ma trận kề ước tính)")
    
    plt.tight_layout()
    plt.savefig("sachs_adjacency_comparison.png")
    print(f"\n[Artifact] Adjacency comparison saved to sachs_adjacency_comparison.png")
    plt.close()

    # 2. Premium Causal Graph with ATE Labels and Metrics
    # We no longer use plot_structure_comparison to focus only on Discovered Graph
    # Chúng ta không dùng plot_structure_comparison nữa để tập trung vào Đồ thị khám phá được
    plot_dag(
        W_matrix=adj_weights, 
        labels=node_names, 
        GT_matrix=true_adj, 
        ate_matrix=ate_matrix,
        metrics=metrics,
        title="Sachs Dataset: Discovered Causal Structure (with ATE)",
        threshold=0.1, 
        save_path="sachs_graph_comparison.png",
        figure_size=(14, 10),
        node_size=2200
    )
    print(f"[Artifact] Professional causal graph with ATE and Metrics saved to sachs_graph_comparison.png")

def display_results(model, data, metrics, true_adj, est_adj, weights, node_names):
    print("\n" + "="*40)
    print("           EXPERIMENT SUMMARY")
    print("="*40)
    print(f"TPR: {metrics['tpr']:.2f} | FPR: {metrics['fpr']:.2f} | FDR: {metrics['fdr']:.2f}")
    print(f"SHD: {metrics['shd']} | SHD-c: {metrics['shd_c']} | SID: {metrics['sid']}")
    print(f"Edges: True={int(np.sum(true_adj))}, Estimated={int(np.sum(est_adj))}")
    print("="*40)
    
    print("\nTop Discovered Causal Edges (with ATE):")
    n_vars = len(node_names)
    for i in range(n_vars):
        for j in range(n_vars):
            if est_adj[i, j] == 1:
                # Estimate ATE / Ước lượng ATE
                ate = model.estimate_ate(data, i, j)
                status = "[V]" if true_adj[i, j] == 1 else "[X]"
                print(f"  {status} {node_names[i]:>8} -> {node_names[j]:<8} (w: {weights[i, j]:+.3f} | ATE: {ate:+.3f})")

if __name__ == "__main__":
    run_experiment()
