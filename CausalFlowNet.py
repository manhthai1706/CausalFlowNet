import torch # Import PyTorch library / Nhập thư viện PyTorch
import torch.nn as nn # Neural network module / Module mạng nơ-ron
import torch.optim as optim # Optimization module / Module tối ưu hóa
import numpy as np # NumPy for numerical operations / Nhập thư viện NumPy

# Import necessary core and modular components / Nhập các thành phần cốt lõi và module cần thiết
from modules.MLP import MLP
from modules.Flow import NeuralSplineFlow
from core.Optimization import compute_acyclicity_constraint, AugmentedLagrangian
from core.HSIC import ParallelFastHSIC

# Global device configuration / Cấu hình thiết bị tính toán (GPU hoặc CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CausalFlowNet(nn.Module):
    """
    CausalFlowNet implementation based on Neural Spline Flows and Gated-ResMLP.
    Triển khai mô hình CausalFlowNet dựa trên Neural Spline Flows và Gated-ResMLP.
    """
    def __init__(self, n_vars, hidden_dims=[32, 32], flow_bins=8, lda_hsic=0.01, n_clusters=5):
        super().__init__()
        self.n_vars = n_vars # Number of variables (nodes) / Số lượng biến (nút)
        self.lda_hsic = lda_hsic # HSIC regularization weight / Trọng số điều chuẩn HSIC
        # Adjacency matrix initialization / Khởi tạo ma trận kề
        self.adj_weights = nn.Parameter(torch.randn(n_vars, n_vars) * 0.05)
        
        # Mapping mechanism (Encoder) / Cơ chế ánh xạ (Bộ mã hóa)
        self.shared_phi = MLP(input_dim=n_vars, hidden_dims=hidden_dims, output_dim=1)
        # Normalizing Flow for density estimation / Flow chuẩn hóa để ước lượng mật độ
        self.shared_flow = NeuralSplineFlow(dim=1, hidden_dim=16, num_layers=2, num_bins=flow_bins, n_clusters=n_clusters)
        # Parallel HSIC network / Mạng HSIC song song
        self.hsic_net = ParallelFastHSIC(n_vars=n_vars, num_features=32)

    def get_adj_matrix(self):
        """Retreive the adjacency matrix with zero diagonal / Lấy ma trận kề với đường chéo bằng 0."""
        # Clean self-loops / Loại bỏ tự lặp (đường chéo)
        return self.adj_weights * (1 - torch.eye(self.n_vars, device=self.adj_weights.device))

    def get_loss(self, X_batch):
        """Compute the total loss components / Tính toán các thành phần hàm mất mát."""
        B = X_batch.size(0) # Batch size / Kích thước lô
        adj = self.get_adj_matrix() # Current adjacency matrix / Ma trận kề hiện tại
        
        # 1. Parallel Input Preparation / Chuẩn bị đầu vào song song
        X_rep = X_batch.unsqueeze(0).repeat(self.n_vars, 1, 1) # Expand X / Mở rộng ma trận X
        masks = adj.t().unsqueeze(1) # Transpose and unsqueeze mask / Chuyển vị và mở rộng mặt nạ
        X_masked = X_rep * masks # Filter by adjacency / Lọc theo cấu trúc ma trận kề
        
        # 2. Compute residuals via MLP / Tính toán phần dư thông qua MLP
        X_flat = X_masked.view(self.n_vars * B, self.n_vars) # Flatten for batch processing / Phẳng hóa để xử lý lô
        pred_flat = self.shared_phi(X_flat) # Predict expected values / Dự báo giá trị kỳ vọng
        
        # 3. Probabilistic density via Normalizing Flow / Tính mật độ xác suất qua Normalizing Flow
        Y_flat = X_batch.t().contiguous().view(self.n_vars * B, 1) # Actual observations / Giá trị quan sát thực tế
        res_flat = Y_flat - pred_flat # Noise residuals / Phần dư nhiễu
        log_prob_flat = self.shared_flow.log_prob(res_flat) # Compute log-probability / Tính log xác suất
        # Average likelihood over variables / Tính khả năng trung bình trên các biến
        nll = -log_prob_flat.view(self.n_vars, B).mean(dim=1).mean()
        
        # 4. Independence constraint (HSIC) / Ràng buộc tính độc lập (HSIC)
        hsic_loss = 0.0
        if self.lda_hsic > 0:
            res_all = res_flat.view(self.n_vars, B, 1) # Reshape residuals / Định dạng lại phần dư
            h_vals = self.hsic_net(X_masked, res_all) # Run independence test / Thực hiện kiểm tra tính độc lập
            hsic_loss = torch.log(h_vals + 1e-8).mean() # Log-HSIC loss / Hàm mất mát Log-HSIC

        # 5. Graph acyclicity constraint / Ràng buộc tính không chu trình của đồ thị
        h_val = compute_acyclicity_constraint(adj) # Compute h(W) / Tính toán giá trị h(W)
        return nll, self.lda_hsic * hsic_loss, h_val # Return triple-loss / Trả về bộ ba hàm mất mát

    def fit(self, data, outer_epochs=15, inner_epochs=100, batch_size=512, l1_reg=0.005):
        """Standard training procedure / Quy trình huấn luyện tiêu chuẩn."""
        self.to(device) # Move model to device / Chuyển mô hình vào thiết bị (GPU/CPU)
        X_full = torch.as_tensor(data, dtype=torch.float32, device=device) # Full data tensor / Tensor dữ liệu đầy đủ
        n_samples, _ = X_full.shape # Sample count / Số lượng mẫu dữ liệu
        
        # Augmented Lagrangian solver / Bộ giải Lagrangian tăng cường
        al = AugmentedLagrangian(rho_init=1.0, alpha_init=0.0)
        optimizer = optim.Adam(self.parameters(), lr=0.01) # Adam optimizer / Bộ tối ưu hóa Adam
        l1_factor = l1_reg / self.n_vars # Normalize L1 weight / Chuẩn hóa trọng số L1
        
        print(f"Training CausalFlowNet on {device}...") # Print device / In thông tin thiết bị
        
        for out in range(outer_epochs): # Outer loop / Vòng lặp ngoài
            self.train() # Training mode / Chế độ huấn luyện
            for inn in range(inner_epochs): # Inner loop / Vòng lặp trong
                idx = torch.randperm(n_samples)[:batch_size] # Random batch indices / Chỉ số lô ngẫu nhiên
                X_batch = X_full[idx] # Create batch / Tạo lô dữ liệu
                
                optimizer.zero_grad(set_to_none=True) # Reset gradients / Đặt lại các đạo hàm
                nll, hsic_l, h_val = self.get_loss(X_batch) # Forward pass / Lan truyền thuận
                l1_loss = l1_factor * torch.norm(self.get_adj_matrix(), p=1) # Sparsity penalty / Phạt tính thưa
                
                # Combined Lagrangian loss / Tổng hàm mất mát Lagrangian
                loss = al.get_loss(nll + hsic_l + l1_loss, h_val)
                loss.backward() # Backward pass / Lan truyền ngược
                
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.5) # Clip gradients / Giới hạn đạo hàm
                optimizer.step() # Update weights / Cập nhật trọng số
            
            # Post-epoch structure pruning / Cắt tỉa cấu trúc sau mỗi vòng lặp
            if out > 0 and out % 5 == 0:
                with torch.no_grad():
                    mask = torch.abs(self.adj_weights) < 0.01 # Identify weak edges / Xác định các cạnh yếu
                    self.adj_weights[mask] = 0.0 # Remove noise edges / Loại bỏ các cạnh nhiễu
                s_idx = torch.randperm(n_samples)[:1000] # Representative samples / Các mẫu đại diện
                _, _, h_v = self.get_loss(X_full[s_idx]) # Re-evaluate constraint / Đánh giá lại ràng buộc
                al.update_parameters(h_v) # Adjust Lagrangian factors / Điều chỉnh các nhân tố Lagrangian
                if (out+1) % 5 == 0:
                    # Log training progress / Ghi nhật ký tiến trình huấn luyện
                    print(f"  Epoch[{out+1}/{outer_epochs}] h(W): {h_v.item():.7f}, NLL: {nll.item():.4f}")
            
        return self.get_adj_matrix().detach().cpu().numpy() # Return result matrix / Trả về ma trận kết quả

    def predict_clusters(self, data, n_clusters=2):
        """Identify subgroups in data / Xác định các phân nhóm trong dữ liệu."""
        from sklearn.cluster import KMeans # Clustering algorithm / Thuật toán phân cụm
        self.eval() # Evaluation mode / Chế độ đánh giá
        with torch.no_grad():
            X = torch.as_tensor(data, dtype=torch.float32, device=device) # Data to tensor / Chuyển dữ liệu sang tensor
            adj = self.get_adj_matrix() # Fixed graph structure / Cấu trúc đồ thị cố định
            # Prepare parallel data / Chuẩn bị dữ liệu song song
            X_rep = X.unsqueeze(0).repeat(self.n_vars, 1, 1)
            masks = adj.t().unsqueeze(1)
            X_masked = X_rep * masks
            X_flat = X_masked.view(self.n_vars * X.size(0), self.n_vars)
            pred_flat = self.shared_phi(X_flat) # Mechanism predictions / Dự báo từ cơ chế
            # Calculate residuals (Noise) / Tính toán phần dư (Nhiễu)
            res_flat = X.t().contiguous().view(self.n_vars * X.size(0), 1) - pred_flat
            Z = res_flat.view(self.n_vars, X.size(0)).t().cpu().numpy() # Latent space Z / Không gian ẩn Z
            
        # Segment latent variations / Phân đoạn các biến thiên ẩn
        return KMeans(n_clusters=n_clusters, n_init=10).fit_predict(Z)

    def estimate_ate(self, data, source_idx, target_idx, intervention_vals=[0, 1]):
        """Quantify causal impact (ATE) / Lượng hóa tác động nhân quả (ATE)."""
        self.eval() # Evaluation mode / Chế độ đánh giá
        X = torch.as_tensor(data, dtype=torch.float32, device=device) # Observation data / Dữ liệu quan sát
        adj = self.get_adj_matrix() # Learned structure / Cấu trúc đã học
        
        with torch.no_grad():
            # Intervene on source and predict target / Can thiệp vào nguồn và dự báo đích
            results = []
            for val in intervention_vals:
                X_int = X.clone() # Clone observations / Sao chép các quan sát
                X_int[:, source_idx] = val # Assign intervention value / Gán giá trị can thiệp
                
                # Map parents to target node / Ánh xạ các nút cha vào nút đích
                X_rep = X_int.unsqueeze(0).repeat(self.n_vars, 1, 1)
                masks = adj.t().unsqueeze(1)
                X_masked = X_rep * masks
                
                # Fetch target mechanism / Thu thập cơ chế nút đích
                X_target_input = X_masked[target_idx] 
                expected_target = self.shared_phi(X_target_input) # E[Y|do(X=val)]
                results.append(expected_target.mean().item()) # Mean expectation / Kỳ vọng trung bình
            
            # Final effect calculation / Tính toán hiệu ứng cuối cùng
            ate = results[1] - results[0] # Subtract expectations / Trừ hai kỳ vọng
            return ate # Return ATE value / Trả về giá trị ATE
