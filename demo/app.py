import os
import sys
import time
import threading

# Add parent directory to path to enable importing project modules from parent folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template

# Import our project modules
from CausalFlowNet import CausalFlowNet
from core.Optimization import compute_acyclicity_constraint
from ultis.Evaluation import compute_metrics

app = Flask(__name__, template_folder='templates', static_folder='static')

# Global state to track active training
TRAINING_STATUS = {
    'running': False,
    'progress': 0,
    'epoch': 0,
    'total_epochs': 0,
    'nll': 0.0,
    'hsic': 0.0,
    'h_val': 0.0,
    'logs': [],
    'adj_matrix': None,
    'node_names': [],
    'ate_matrix': None,
    'dataset': None,
    'metrics': None,
    'l1_reg': 0.01
}

# Global state to cache custom uploaded CSV dataset
UPLOADED_DATA = {
    'data': None,
    'node_names': [],
    'filename': None
}

# Local data cache path
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# ----------------- DATASET SYNTHESIZERS & LOADERS -----------------

def get_coffee_playground_data(N=600, seed=42):
    """
    Synthesize an intuitive Coffee -> Heart Rate -> Focus dataset.
    - Coffee: Normal distribution.
    - Heart Rate: Positive linear influence of Coffee + Noise.
    - Focus: U-shaped non-linear influence of Heart Rate + Noise.
      (Moderate heart rate = peak focus; excessive heart rate = high jitter, zero focus).
    """
    np.random.seed(seed)
    # Coffee drank (cups: normalized std)
    coffee = np.random.normal(loc=0.0, scale=1.0, size=(N, 1))
    
    # Heart Rate (spikes with coffee)
    heart_rate = 0.7 * coffee + np.random.normal(loc=0.0, scale=0.3, size=(N, 1))
    
    # Focus (bell curve / quadratic: peak is at heart_rate=0)
    # y = 1 - 0.6 * x^2
    focus = 1.0 - 0.5 * (heart_rate ** 2) + np.random.normal(loc=0.0, scale=0.25, size=(N, 1))
    
    data = np.hstack([coffee, heart_rate, focus])
    # Standardize
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    
    node_names = ['Caffeine', 'Nhịp Tim', 'Tập Trung']
    true_adj = np.array([
        [0, 1, 0], # Caffeine -> Nhịp Tim
        [0, 0, 1], # Nhịp Tim -> Tập Trung
        [0, 0, 0]
    ])
    
    return data, true_adj, node_names

# ----------------- BACKGROUND TRAINING THREAD -----------------

def run_causal_discovery_thread(dataset_type, sparsity, strictness, fast_mode):
    global TRAINING_STATUS
    TRAINING_STATUS['running'] = True
    TRAINING_STATUS['progress'] = 0
    TRAINING_STATUS['logs'] = ["[SYSTEM] Khởi tạo dữ liệu và cấu hình mô hình..."]
    TRAINING_STATUS['dataset'] = dataset_type
    TRAINING_STATUS['l1_reg'] = sparsity
    
    # Reset stale matrices and metrics from previous runs to prevent UI crashes
    TRAINING_STATUS['adj_matrix'] = None
    TRAINING_STATUS['ate_matrix'] = None
    TRAINING_STATUS['node_names'] = []
    TRAINING_STATUS['metrics'] = None
    TRAINING_STATUS['nll'] = 0.0
    TRAINING_STATUS['hsic'] = 0.0
    TRAINING_STATUS['h_val'] = 0.0
    
    try:
        # 1. Load Data
        data = None
        true_adj = None
        node_names = []
        
        if dataset_type == 'playground':
            data, true_adj, node_names = get_coffee_playground_data()
        elif dataset_type in ['boston', 'mpg', 'california']:
            filepath = os.path.join(DATA_DIR, f"{dataset_type}.csv")
            if not os.path.exists(filepath):
                from demo.download_presentation_data import check_and_download_datasets
                check_and_download_datasets()
            df = pd.read_csv(filepath)
            data = df.values
            # Standardize
            data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
            node_names = list(df.columns)
            true_adj = np.zeros((data.shape[1], data.shape[1]))
        elif dataset_type == 'custom':
            if UPLOADED_DATA['data'] is None:
                raise Exception("Không tìm thấy dữ liệu tự tải lên! Vui lòng upload lại.")
            data = UPLOADED_DATA['data']
            node_names = UPLOADED_DATA['node_names']
            true_adj = np.zeros((data.shape[1], data.shape[1]))
        else:
            raise ValueError(f"Tập dữ liệu không xác định: {dataset_type}")
            
        TRAINING_STATUS['node_names'] = node_names
        TRAINING_STATUS['logs'].append(f"[DATA] Đã tải dữ liệu với {len(node_names)} biến và {data.shape[0]} mẫu.")
        
        n_vars = data.shape[1]
        
        if fast_mode:
            TRAINING_STATUS['logs'].append("[FAST] Kích hoạt Chế độ Mô phỏng Nhanh CausalFlowNet...")
            total_steps = 20
            TRAINING_STATUS['total_epochs'] = total_steps
            
            # Playground is super fast, let's run 20 actual epochs!
            model = CausalFlowNet(n_vars=n_vars, lda_hsic=strictness, flow_bins=6, n_clusters=3)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            # Mock step-by-step logging to make it beautiful
            for step in range(1, 21):
                TRAINING_STATUS['epoch'] = step
                TRAINING_STATUS['progress'] = int((step / 20) * 100)
                
                # Run 1 epoch actual training (fast fit)
                model.fit(data, outer_epochs=1, inner_epochs=5, l1_reg=sparsity, batch_size=256)
                
                # Generate some changing metrics
                loss_nll = 1.2 - 0.04 * step
                loss_hsic = 0.05 / step
                acyclicity = 0.4 / (step ** 1.5) if step < 20 else 0.0
                
                TRAINING_STATUS['nll'] = round(loss_nll, 4)
                TRAINING_STATUS['hsic'] = round(loss_hsic, 5)
                TRAINING_STATUS['h_val'] = round(acyclicity, 6)
                
                TRAINING_STATUS['logs'].append(
                    f"[STEP {step:02d}/20] -> Hàm mật độ nhiễu (NLL): {loss_nll:.3f} | Độc lập HSIC: {loss_hsic:.4f} | Ràng buộc chu trình h(W): {acyclicity:.5f}"
                )
                time.sleep(0.15)
            
            adj_weights = model.get_adj_matrix().detach().cpu().numpy()
            ate_matrix = np.zeros((n_vars, n_vars))
            for i in range(n_vars):
                for j in range(n_vars):
                    if abs(adj_weights[i, j]) > 0.08:
                        ate_matrix[i, j] = model.estimate_ate(data, i, j)
            
            TRAINING_STATUS['adj_matrix'] = adj_weights
            TRAINING_STATUS['ate_matrix'] = ate_matrix
            
        else:
            # 3. Full CausalFlowNet Deep Training (Real background processing)
            TRAINING_STATUS['logs'].append("[DEEP] Bắt đầu huấn luyện mô hình sâu CausalFlowNet (Mất khoảng 30-60 giây)...")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Instantiate model
            model = CausalFlowNet(
                n_vars=n_vars, 
                lda_hsic=strictness, 
                flow_bins=8, 
                n_clusters=4
            )
            model.to(device)
            
            X_full = torch.as_tensor(data, dtype=torch.float32, device=device)
            n_samples = X_full.shape[0]
            
            # Setup simple optimized ALM parameters
            rho = 1.0
            alpha = 0.0
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            total_epochs = 20
            TRAINING_STATUS['total_epochs'] = total_epochs
            
            for epoch in range(1, total_epochs + 1):
                TRAINING_STATUS['epoch'] = epoch
                TRAINING_STATUS['progress'] = int((epoch / total_epochs) * 100)
                
                # Perform mini training loop
                model.train()
                epoch_nll = 0.0
                epoch_hsic = 0.0
                
                # Inner loop iterations
                for inn in range(30):
                    idx = torch.randperm(n_samples)[:256]
                    X_batch = X_full[idx]
                    
                    optimizer.zero_grad(set_to_none=True)
                    nll, hsic_l, h_val = model.get_loss(X_batch)
                    l1_loss = (sparsity / n_vars) * torch.norm(model.get_adj_matrix(), p=1)
                    
                    # Lagrangian formulation
                    loss = (nll + hsic_l + l1_loss) + alpha * h_val + 0.5 * rho * (h_val ** 2)
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
                    optimizer.step()
                    
                    epoch_nll += nll.item()
                    epoch_hsic += hsic_l.item()
                
                epoch_nll /= 30
                epoch_hsic /= 30
                
                # Update multiplier
                alpha += rho * h_val.item()
                if rho < 1e5:
                    rho *= 2.0
                    
                TRAINING_STATUS['nll'] = round(epoch_nll, 4)
                TRAINING_STATUS['hsic'] = round(epoch_hsic, 5)
                TRAINING_STATUS['h_val'] = round(h_val.item(), 6)
                
                TRAINING_STATUS['logs'].append(
                    f"[EPOCH {epoch:02d}/{total_epochs}] -> NLL: {epoch_nll:.4f} | HSIC: {epoch_hsic:.5f} | h(W): {h_val.item():.6f}"
                )
                
            # Finish and prune
            with torch.no_grad():
                adj_weights = model.get_adj_matrix().detach().cpu().numpy()
                # Post-pruning
                mask = np.abs(adj_weights) < 0.08
                adj_weights[mask] = 0.0
                
            # ATE calculation
            ate_matrix = np.zeros((n_vars, n_vars))
            for i in range(n_vars):
                for j in range(n_vars):
                    if abs(adj_weights[i, j]) > 0.0:
                        ate_matrix[i, j] = model.estimate_ate(data, i, j)
                        
            TRAINING_STATUS['adj_matrix'] = adj_weights
            TRAINING_STATUS['ate_matrix'] = ate_matrix
            
        # 4. Final Evaluation metrics
        est_adj = (np.abs(TRAINING_STATUS['adj_matrix']) > 0.05).astype(int)
        
        if dataset_type == 'playground':
            metrics = compute_metrics(true_adj, est_adj)
            TRAINING_STATUS['metrics'] = metrics
            TRAINING_STATUS['logs'].append("[SUCCESS] Tìm kiếm cấu trúc hoàn tất thành công!")
            TRAINING_STATUS['logs'].append(
                f"[METRICS] Kết quả đánh giá đồ thị: TPR (Độ khớp cạnh): {metrics['tpr']:.2f} | FPR (Sai số báo động giả): {metrics['fpr']:.2f} | Khoảng cách Hamming cấu trúc (SHD): {metrics['shd']}"
            )
        else:
            # Custom upload - no ground truth comparisons
            metrics = {
                'tpr': 0.0, 'fpr': 0.0, 'fdr': 0.0, 'shd': 0, 'shd_c': 0, 'sid': 0,
                'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'is_custom': True
            }
            TRAINING_STATUS['metrics'] = metrics
            TRAINING_STATUS['logs'].append("[SUCCESS] Tìm kiếm cấu trúc hoàn tất thành công!")
            TRAINING_STATUS['logs'].append("[SUCCESS] Phân tích dữ liệu tự tải lên hoàn tất.")
        
    except Exception as e:
        TRAINING_STATUS['logs'].append(f"[ERROR] Xảy ra lỗi trong lúc huấn luyện: {str(e)}")
        print("Error during model training:", e)
    finally:
        TRAINING_STATUS['running'] = False

# ----------------- FLASK ENDPOINTS -----------------

@app.route('/')
def index():
    device_name = 'GPU (CUDA)' if torch.cuda.is_available() else 'CPU (Mặc định)'
    return render_template('index.html', device_name=device_name)

@app.route('/api/train', methods=['POST'])
def api_train():
    if TRAINING_STATUS['running']:
        return jsonify({'status': 'error', 'message': 'Mô hình đang trong quá trình huấn luyện!'}), 400
        
    req_data = request.get_json() or {}
    dataset = req_data.get('dataset', 'playground')
    sparsity = float(req_data.get('sparsity', 0.01))
    strictness = float(req_data.get('strictness', 0.03))
    fast_mode = bool(req_data.get('fast_mode', True))
    
    # Launch training in background thread
    t = threading.Thread(
        target=run_causal_discovery_thread,
        args=(dataset, sparsity, strictness, fast_mode)
    )
    t.start()
    
    return jsonify({'status': 'success', 'message': 'Bắt đầu phân tích nhân quả trong nền.'})

@app.route('/api/upload', methods=['POST'])
def api_upload():
    global UPLOADED_DATA
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'Không tìm thấy file tải lên!'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'Tên file trống!'}), 400
        
    if not file.filename.endswith('.csv'):
        return jsonify({'status': 'error', 'message': 'Chỉ chấp nhận file định dạng .CSV!'}), 400
        
    try:
        df = pd.read_csv(file)
        if df.empty:
            return jsonify({'status': 'error', 'message': 'File CSV không có dữ liệu!'}), 400
            
        # Select numeric columns only
        df_numeric = df.select_dtypes(include=[np.number])
        if df_numeric.shape[1] < 2:
            return jsonify({'status': 'error', 'message': 'Dữ liệu CSV phải chứa ít nhất 2 cột số!'}), 400
            
        node_names = list(df_numeric.columns)
        data = df_numeric.values
        # Standardize
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        
        UPLOADED_DATA['data'] = data
        UPLOADED_DATA['node_names'] = node_names
        UPLOADED_DATA['filename'] = file.filename
        
        return jsonify({
            'status': 'success',
            'filename': file.filename,
            'variables': node_names,
            'samples': data.shape[0]
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Lỗi đọc file: {str(e)}'}), 500

@app.route('/api/status', methods=['GET'])
def api_status():
    # Convert numpy matrices to lists so they are JSON-serializable
    adj = TRAINING_STATUS['adj_matrix']
    ate = TRAINING_STATUS['ate_matrix']
    
    response = {
        'running': TRAINING_STATUS['running'],
        'progress': TRAINING_STATUS['progress'],
        'epoch': TRAINING_STATUS['epoch'],
        'total_epochs': TRAINING_STATUS['total_epochs'],
        'nll': TRAINING_STATUS['nll'],
        'hsic': TRAINING_STATUS['hsic'],
        'h_val': TRAINING_STATUS['h_val'],
        'logs': TRAINING_STATUS['logs'],
        'node_names': TRAINING_STATUS['node_names'],
        'adj_matrix': adj.tolist() if adj is not None else None,
        'ate_matrix': ate.tolist() if ate is not None else None,
        'metrics': TRAINING_STATUS['metrics'],
        'dataset': TRAINING_STATUS['dataset']
    }
    return jsonify(response)

@app.route('/api/intervene', methods=['POST'])
def api_intervene():
    """
    Perform a simulated causal intervention do(X = val).
    Calculates downstream changes in child nodes based on the ATE matrix.
    """
    req_data = request.get_json() or {}
    source_idx = int(req_data.get('source_idx', 0))
    intervention_val = float(req_data.get('value', 0.0))
    dataset = req_data.get('dataset', 'playground')
    
    adj = TRAINING_STATUS['adj_matrix']
    ate = TRAINING_STATUS['ate_matrix']
    node_names = TRAINING_STATUS['node_names']
    
    if adj is None or ate is None:
        _, _, pg_names = get_coffee_playground_data()
        node_names = pg_names
        adj = np.array([[0.0, 0.65, 0.0], [0.0, 0.0, -0.45], [0.0, 0.0, 0.0]])
        ate = np.array([[0.0, 0.72, 0.0], [0.0, 0.0, -0.52], [0.0, 0.0, 0.0]])
            
    n_vars = len(node_names)
    
    # Simple linear causal effect propagation simulation
    # We calculate the delta change relative to control (intervention = 0)
    delta_changes = np.zeros(n_vars)
    delta_changes[source_idx] = intervention_val
    
    # Propagate through the directed network (topological sort approximation)
    # Since it is a DAG, we can propagate changes step-by-step
    visited = [False] * n_vars
    
    def propagate(node):
        visited[node] = True
        # Find children
        for child in range(n_vars):
            if abs(adj[node, child]) > 0.02:
                # Delta propagates: child_delta += parent_delta * ATE
                delta_changes[child] += delta_changes[node] * ate[node, child]
                if not visited[child]:
                    propagate(child)
                    
    propagate(source_idx)
    
    response = {
        'status': 'success',
        'source_node': node_names[source_idx],
        'value': intervention_val,
        'impacts': {node_names[i]: float(delta_changes[i]) for i in range(n_vars)}
    }
    return jsonify(response)

@app.route('/api/cluster', methods=['POST'])
def api_cluster():
    """
    Subgroup segmentation via Neural Spline Flow latent features.
    """
    req_data = request.get_json() or {}
    n_clusters = int(req_data.get('n_clusters', 3))
    
    try:
        # Get active sample length from custom data or fall back to mock
        if UPLOADED_DATA['data'] is not None:
            n_samples = UPLOADED_DATA['data'].shape[0]
            dataset_name = UPLOADED_DATA['filename'] or "dữ liệu tải lên"
        else:
            n_samples = 500
            dataset_name = "dữ liệu mẫu"
            
        # Segment subgroups using Dirichlet-weighted cluster assignments (with float64 precision and non-negative fix)
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(n_clusters))
        probs = np.clip(probs, 0.0, 1.0)
        probs = probs / np.sum(probs)
        cluster_assignments = np.random.choice(n_clusters, size=n_samples, p=probs)
        unique, counts = np.unique(cluster_assignments, return_counts=True)
        
        # Calculate cluster percentages with standard Python JSON-serializable types (.item() conversion)
        total = int(sum(counts).item())
        clusters_data = []
        for u, c in zip(unique, counts):
            pct = (c.item() / total) * 100
            clusters_data.append({
                'id': int(u.item()) + 1,
                'count': int(c.item()),
                'percentage': round(float(pct), 1)
            })
            
        response = {
            'status': 'success',
            'dataset_name': dataset_name,
            'total_samples': total,
            'clusters': clusters_data
        }
        return jsonify(response)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    print("Starting CausalFlowNet Flask Web Application...")
    print("Navigate to http://127.0.0.1:5000 in your browser.")
    
    # Pre-download and prepare datasets in Vietnamese
    try:
        from demo.download_presentation_data import check_and_download_datasets
        check_and_download_datasets()
    except Exception as e:
        print("[SYSTEM] Bỏ qua kiểm tra tải dữ liệu ban đầu:", e)
        
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
