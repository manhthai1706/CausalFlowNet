import numpy as np # Import NumPy / Nhập thư viện NumPy
import networkx as nx # Import NetworkX for SID / Nhập NetworkX để tính SID

def compute_metrics(B_true, B_est):
    """
    Compute TPR, FPR, and SHD for causal discovery evaluation.
    Tính toán các chỉ số TPR, FPR và SHD để đánh giá khám phá nhân quả.
    B_true: True binary adjacency matrix. / Ma trận kề nhị phân thực tế (nhãn).
    B_est: Estimated binary adjacency matrix. / Ma trận kề nhị phân ước tính (kết quả mô hình).
    """
    # True Positives, False Positives, etc. / Các chỉ số Dương tính thật, Dương tính giả, v.v.
    # tp: i->j exists in both true and estimated / tp: cạnh i->j tồn tại ở cả thực tế và ước tính
    tp = np.sum((B_true == 1) & (B_est == 1))
    # fp: i->j exists in estimated but not in true / fp: cạnh i->j có trong ước tính nhưng không có trong thực tế
    fp = np.sum((B_true == 0) & (B_est == 1))
    # fn: i->j exists in true but not in estimated / fn: cạnh i->j có trong thực tế nhưng không có trong ước tính
    fn = np.sum((B_true == 1) & (B_est == 0))
    # tn: i->j correctly identified as non-existent / tn: xác định đúng là không có cạnh i->j
    tn = np.sum((B_true == 0) & (B_est == 0))
    
    # tpr: True Positive Rate (Recall) / tpr: Tỷ lệ dương tính thật (Độ gợi nhớ)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    # fpr: False Positive Rate / fpr: Tỷ lệ dương tính giả
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Structural Hamming Distance / Khoảng cách Hamming cấu trúc (SHD)
    reversed_edges = np.sum((B_true == 1) & (B_est.T == 1))
    shd = np.sum(np.abs(B_true - B_est)) - reversed_edges
    
    # Structural Intervention Distance / Khoảng cách Can thiệp Cấu trúc (SID)
    sid = compute_sid(B_true, B_est)
    
    return {
        'tpr': tpr, 
        'fpr': fpr, 
        'shd': int(shd), 
        'sid': int(sid), 
        'tp': int(tp), 
        'fp': int(fp), 
        'fn': int(fn), 
        'tn': int(tn)  
    }

def compute_sid(B_true, B_est):
    """
    Compute Structural Intervention Distance (SID) for two DAGs.
    If cycles exist in B_est, it greedily breaks them to allow evaluation.
    """
    G_true = nx.from_numpy_array(B_true, create_using=nx.DiGraph)
    G_est = nx.from_numpy_array(B_est, create_using=nx.DiGraph)
    
    # Greedily break cycles to ensure DAGs for SID calculation / Phá vỡ chu trình để đảm bảo DAG
    for G in [G_true, G_est]:
        while not nx.is_directed_acyclic_graph(G):
            cycle = nx.find_cycle(G)
            G.remove_edge(*cycle[0])
            
    nodes = list(G_true.nodes())
    sid_score = 0
    
    true_descendants = {n: nx.descendants(G_true, n) for n in nodes}
    est_descendants = {n: nx.descendants(G_est, n) for n in nodes}
    
    for i in nodes:
        for j in nodes:
            if i == j: continue
            
            # Simplified SID: Count pairs (i, j) with incorrect causal relationship
            # If j is descendant in true but not in est (or vice versa)
            if (j in true_descendants[i]) != (j in est_descendants[i]):
                sid_score += 1
                
    return sid_score
