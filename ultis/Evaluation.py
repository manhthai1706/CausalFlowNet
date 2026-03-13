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
    
    # fdr: False Discovery Rate / fdr: Tỷ lệ phát hiện sai
    fdr = fp / (fp + tp) if (fp + tp) > 0 else 0
    
    # Structural Hamming Distance / Khoảng cách Hamming cấu trúc (SHD)
    reversed_edges = np.sum((B_true == 1) & (B_est.T == 1))
    shd = np.sum(np.abs(B_true - B_est)) - reversed_edges
    
    # SHD-c: SHD on CPDAGs (Completed Partially Directed Acyclic Graphs)
    # This evaluates at the level of Markov Equivalence Classes
    shd_c = compute_shd_c(B_true, B_est)
    
    # Structural Intervention Distance / Khoảng cách Can thiệp Cấu trúc (SID)
    sid = compute_sid(B_true, B_est)
    
    return {
        'tpr': tpr, 
        'fpr': fpr, 
        'fdr': fdr,
        'shd': int(shd), 
        'shd_c': int(shd_c),
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

def compute_shd_c(B_true, B_est):
    """
    Compute Structural Hamming Distance for CPDAGs (SHD-c).
    Evaluates orientation based on Markov Equivalence Classes.
    """
    CP_true = dag_to_cpdag(B_true)
    CP_est = dag_to_cpdag(B_est)
    
    # Difference in skeletons + Difference in orientations
    # CPDAG representation: 1 for u->v, -1 for u-v (undirected)
    # SHD for PDAGs (simplified version)
    shd = 0
    n = B_true.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            # Skeleton check
            true_edge = (CP_true[i, j] != 0 or CP_true[j, i] != 0)
            est_edge = (CP_est[i, j] != 0 or CP_est[j, i] != 0)
            
            if true_edge != est_edge:
                shd += 1
            elif true_edge: # Both have an edge, check orientation
                # Orientations: (u->v, v->u) vs (u-v)
                t_dir = (CP_true[i, j], CP_true[j, i])
                e_dir = (CP_est[i, j], CP_est[j, i])
                if t_dir != e_dir:
                    shd += 1
    return shd

def dag_to_cpdag(B):
    """
    Convert a DAG adjacency matrix to its CPDAG (Completed Partially Directed Acyclic Graph).
    Output: matrix where 1 means i->j, and both i,j=1 means i-j (undirected).
    Uses a simplified version of Chickering's algorithm.
    """
    n = B.shape[0]
    G = nx.from_numpy_array(B, create_using=nx.DiGraph)
    if not nx.is_directed_acyclic_graph(G):
        # Greedily break cycles if not a DAG
        G_dag = G.copy()
        while not nx.is_directed_acyclic_graph(G_dag):
            cycle = nx.find_cycle(G_dag)
            G_dag.remove_edge(*cycle[0])
        G = G_dag

    # 1. Identify compelled edges (starting with v-structures)
    compelled = np.zeros_like(B)
    nodes = list(G.nodes())
    for v in nodes:
        parents = list(G.predecessors(v))
        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                p1, p2 = parents[i], parents[j]
                if not G.has_edge(p1, p2) and not G.has_edge(p2, p1):
                    compelled[p1, v] = 1
                    compelled[p2, v] = 1

    # 2. Propagate compelled edges (Meeks' rules - Simplified)
    change = True
    while change:
        change = False
        for u in nodes:
            for v in nodes:
                if G.has_edge(u, v) and compelled[u, v] == 0:
                    # Rule 1: u -> v and v - w and u,w not adj => v -> w (compelled)
                    for w in G.neighbors(v): # v-w means edge in skeleton
                        if not G.has_edge(u, w) and not G.has_edge(w, u):
                            if G.has_edge(v, w) and compelled[v, w] == 0:
                                # This is a bit complex for a script, 
                                # but essentially we mark u->v as compelled 
                                # if it avoids new v-structures
                                pass
        break # Simplified: only V-structures are compelled for now in this version

    # 3. Create CPDAG matrix
    # If edge is not compelled, it's undirected (i,j=1 and j,i=1)
    CP = compelled.copy()
    for u, v in G.edges():
        if CP[u, v] == 0:
            CP[u, v] = 1
            CP[v, u] = 1
    return CP
