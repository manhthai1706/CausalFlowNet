import torch # Import PyTorch library / Nhập thư viện PyTorch
import torch.nn as nn # Neural network module / Module mạng nơ-ron
import numpy as np # NumPy for numerical operations / NumPy cho các toán tử số học

class ParallelFastHSIC(nn.Module):
    """
    Super-fast Parallel HSIC using Random Fourier Features.
    HSIC song song siêu nhanh sử dụng Đặc trưng Fourier ngẫu nhiên (RFF).
    Processes all N causal nodes in a single vectorized operation.
    Xử lý tất cả N nút nhân quả trong một toán tử vectơ hóa duy nhất.
    """
    def __init__(self, n_vars, num_features=32):
        super(ParallelFastHSIC, self).__init__()
        self.K = n_vars # Number of variables (nodes) / Số lượng biến (nút)
        self.m = num_features # Number of random features / Số lượng đặc trưng ngẫu nhiên
        
        # All-in-one buffers for K nodes / Bộ đệm tổng hợp cho K nút
        # Weights for input X / Trọng số cho đầu vào X
        self.register_buffer('Wx', torch.randn(self.K, self.K, self.m))
        # Weights for residual Y / Trọng số cho phần dư Y
        self.register_buffer('Wy', torch.randn(self.K, 1, self.m))
        # Bias for X / Độ lệch cho X
        self.register_buffer('Bx', torch.rand(self.K, 1, self.m) * 2 * np.pi)
        # Bias for Y / Độ lệch cho Y
        self.register_buffer('By', torch.rand(self.K, 1, self.m) * 2 * np.pi)

    def forward(self, X_all, Y_all):
        """
        X_all: (K, B, K) - All inputs for K tests / Tất cả đầu vào cho K kiểm thử
        Y_all: (K, B, 1) - All residuals for K tests / Tất cả phần dư cho K kiểm thử
        """
        n = X_all.size(1) # Batch size / Kích thước lô
        if n < 2: return torch.zeros(self.K, device=X_all.device) # Need at least 2 samples / Cần ít nhất 2 mẫu

        # Batch matrix multiplication / Nhân ma trận theo lô: (K, B, K) @ (K, K, M) -> (K, B, M)
        proj_x = torch.bmm(X_all, self.Wx) + self.Bx # Linear projection for X / Ánh xạ tuyến tính cho X
        proj_y = torch.bmm(Y_all, self.Wy) + self.By # Linear projection for Y / Ánh xạ tuyến tính cho Y
        
        # Random Features / Các đặc trưng ngẫu nhiên: (K, B, M)
        phi_x = torch.cos(proj_x) * (2.0 / self.m)**0.5 # Cosine transform for X / Biến đổi Cosine cho X
        phi_y = torch.cos(proj_y) * (2.0 / self.m)**0.5 # Cosine transform for Y / Biến đổi Cosine cho Y
        
        # Centering across batch B / Chuẩn hóa tâm qua lô B: (K, B, M)
        phi_x = phi_x - phi_x.mean(dim=1, keepdim=True) # Remove mean of X / Loại bỏ giá trị trung bình của X
        phi_y = phi_y - phi_y.mean(dim=1, keepdim=True) # Remove mean of Y / Loại bỏ giá trị trung bình của Y
        
        # Covariance / Hiệp biến: (K, M, B) @ (K, B, M) -> (K, M, M)
        cov = torch.bmm(phi_x.transpose(1, 2), phi_y) / (n - 1) # Compute covariance matrix / Tính ma trận hiệp phương sai
        
        # Parallel sum of squared covariances / Tổng song song bình phương các hiệp biến
        return torch.sum(cov**2, dim=(1, 2)) # Return HSIC values / Trả về các giá trị HSIC
