import torch # Import PyTorch library / Nhập thư viện PyTorch
import torch.nn as nn # Neural network module / Module mạng nơ-ron
import torch.nn.functional as F # Functional interface / Giao diện chức năng
import numpy as np # NumPy for numerical operations / NumPy cho các toán tử số học

def searchsorted(bin_locations, inputs, eps=1e-6):
    """Search for bin indices / Tìm kiếm chỉ số thùng (bin)."""
    bin_locations[..., -1] += eps # Add epsilon to avoid boundary issues / Thêm epsilon để tránh lỗi biên
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1 # Return indices / Trả về các chỉ số

def unconstrained_rational_quadratic_spline(inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives, tail_bound=3.0):
    """
    Memory-optimized Rational-Quadratic Splines.
    Spline hữu tỉ bậc hai tối ưu bộ nhớ.
    """
    mask = (inputs >= -tail_bound) & (inputs <= tail_bound) # Create mask for inside-bound values / Tạo mặt nạ cho các giá trị trong biên
    outputs, logabsdet = torch.zeros_like(inputs), torch.zeros_like(inputs) # Initialize outputs and log-det / Khởi tạo đầu ra và log-det
    outputs[~mask], logabsdet[~mask] = inputs[~mask], 0. # Outside bound: identity mapping / Ngoài biên: phép chiếu đồng nhất
    
    if not torch.any(mask): return outputs, logabsdet # If no values in mask, return early / Nếu không có giá trị nào trong mặt nạ, kết thúc sớm

    # Process insidebound values / Xử lý các giá trị trong biên
    ins, u_w, u_h, u_d = inputs[mask], unnormalized_widths[mask], unnormalized_heights[mask], unnormalized_derivatives[mask]
    num_bins = u_w.shape[-1] # Number of bins / Số lượng thùng
    
    def normalize(u, min_v):
        """Normalize bin widths/heights / Chuẩn hóa độ rộng/độ cao của thùng."""
        v = F.softmax(u, dim=-1) # Apply softmax / Áp dụng softmax
        v = min_v + (1 - min_v * num_bins) * v # Ensure min value / Đảm bảo giá trị tối thiểu
        cum_v = torch.cumsum(v, dim=-1) # Cumulative sum / Tính tổng tích lũy
        cum_v = F.pad(cum_v, (1, 0), value=0.0) # Pad with zero / Đệm với giá trị 0
        return (cum_v - 0.5) * 2 * tail_bound, v # Return locations and sizes / Trả về vị trí và kích thước

    cum_w, w = normalize(u_w, 1e-3) # Normalize widths / Chuẩn hóa độ rộng
    cum_h, h = normalize(u_h, 1e-3) # Normalize heights / Chuẩn hóa độ cao
    d = F.softplus(u_d) + 1e-3 # Ensure positive derivatives / Đảm bảo đạo hàm dương
    
    # Find the bin index for each input / Tìm chỉ số thùng cho mỗi đầu vào
    idx = searchsorted(cum_w, ins)[..., None]
    # Gather bin properties / Thu thập các thuộc tính của thùng
    cw, ww, ch, hh = cum_w.gather(-1, idx)[..., 0], w.gather(-1, idx)[..., 0], cum_h.gather(-1, idx)[..., 0], h.gather(-1, idx)[..., 0]
    di, di1 = d.gather(-1, idx)[..., 0], d.gather(-1, idx + 1)[..., 0]
    
    # Stability: Clamp derivatives / Độ ổn định: Giới hạn giá trị đạo hàm
    di = torch.clamp(di, min=1e-3, max=1e3)
    di1 = torch.clamp(di1, min=1e-3, max=1e3)
    
    delta = hh / ww # Calculate delta / Tính toán delta
    
    # Stability: Clamp theta / Độ ổn định: Giới hạn giá trị theta
    theta = torch.clamp((ins - cw) / ww, 0.0, 1.0)
    
    # Formula components / Các thành phần của công thức
    num_term = di1 * theta**2 + delta * theta * (1 - theta) # Numerator / Tử số
    den = delta + (di + di1 - 2 * delta) * theta * (1 - theta) # Denominator / Mẫu số
    den = torch.clamp(den, min=1e-6) # Prevent division by zero / Ngăn chặn chia cho 0
    
    outputs[mask] = ch + (delta * num_term / den) * ww # Compute final outputs / Tính toán đầu ra cuối cùng
    
    # Derivative log-det / Log-định thức đạo hàm
    deriv_num = delta**2 * (di1 * theta**2 + 2 * delta * theta * (1 - theta) + di * (1 - theta)**2)
    logabsdet[mask] = torch.log(torch.clamp(deriv_num / (den**2), min=1e-12))
    return outputs, logabsdet # Return results / Trả về kết quả

class SplineCouplingLayer(nn.Module):
    """Spline Coupling Layer for Flow / Lớp ghép Spline cho Flow."""
    def __init__(self, dim, hidden_dim, mask, num_bins=4, tail_bound=3.0):
        super().__init__()
        self.register_buffer('mask', mask) # Register binary mask / Đăng ký mặt nạ nhị phân
        self.total_params = num_bins * 3 + 1 # Parameters per dimension / Số tham số cho mỗi chiều
        self.num_bins, self.tail_bound = num_bins, tail_bound # Bin count and bound / Số thùng và biên
        self.net = nn.Sequential( # Neural network for parameters / Mạng nơ-ron cho các tham số
            nn.Linear(dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, dim * self.total_params)
        )

    def forward(self, x):
        """Forward pass through coupling layer / Lan truyền thuận qua lớp ghép."""
        x_static, x_dynamic = x * self.mask, x * (1 - self.mask) # Split by mask / Chia theo mặt nạ
        # Generate spline parameters / Tạo các tham số spline
        params = self.net(x_static).view(x.shape[0], -1, self.total_params)
        # Apply RQ Spline / Áp dụng Rational-Quadratic Spline
        new_x_dynamic, log_det = unconstrained_rational_quadratic_spline(
            x_dynamic, params[..., :self.num_bins], params[..., self.num_bins:2*self.num_bins], params[..., 2*self.num_bins:], tail_bound=self.tail_bound
        )
        # Combine and return / Kết hợp và trả về
        return x_static + new_x_dynamic * (1 - self.mask), torch.sum(log_det * (1 - self.mask), dim=1)

class GaussianMixturePrior(nn.Module):
    """Learnable GMM Prior for the Flow Latent Space. / Phân phối ưu tiên hỗn hợp Gaussian có thể học được."""
    def __init__(self, dim, n_components=3):
        super().__init__()
        self.dim = dim # Dimension / Số chiều
        self.n_components = n_components # Number of components / Số lượng thành phần (cụm)
        self.locs = nn.Parameter(torch.randn(n_components, dim) * 0.5) # Means / Trung trị (tâm cụm)
        self.log_scales = nn.Parameter(torch.zeros(n_components, dim)) # Log standard deviations / Log độ lệch chuẩn
        self.pi = nn.Parameter(torch.ones(n_components) / n_components) # Mixing weights / Trọng số pha trộn

    def log_prob(self, x):
        """Compute log probability of x / Tính log xác suất của x."""
        B = x.size(0) # Batch size / Kích thước lô
        # Expansion for GMM calculation / Mở rộng chiều cho tính toán GMM
        x_exp = x.unsqueeze(0).expand(self.n_components, -1, -1)
        locs_exp = self.locs.unsqueeze(1).expand(-1, B, -1)
        scales_exp = torch.exp(self.log_scales).unsqueeze(1).expand(-1, B, -1)
        
        # Log-prob for each component / Log xác suất cho mỗi thành phần cụm
        log_probs = -0.5 * (np.log(2 * np.pi) + 2 * torch.log(scales_exp) + (x_exp - locs_exp)**2 / scales_exp**2)
        log_probs = log_probs.sum(dim=-1) # (n_components, B)
        
        # Mixing weights / Trọng số pha trộn
        pi_normalized = F.softmax(self.pi, dim=0) # Normalize weights / Chuẩn hóa trọng số
        weighted_log_probs = log_probs + torch.log(pi_normalized.unsqueeze(1) + 1e-8) # Weighted probabilities / Xác suất có trọng số
        
        # LogSumExp over components / Tính LogSumExp qua các thành phần cụm
        return torch.logsumexp(weighted_log_probs, dim=0)

class NeuralSplineFlow(nn.Module):
    """Lightweight NSF with GMM Prior for clustering. / Neural Spline Flow nhẹ với GMM Prior cho phân cụm."""
    def __init__(self, dim, hidden_dim=16, num_layers=2, num_bins=4, n_clusters=3):
        super().__init__()
        self.prior = GaussianMixturePrior(dim, n_components=n_clusters) # GMM Prior / Phân phối ưu tiên GMM
        self.layers = nn.ModuleList([ # Flow layers / Các lớp Flow
            SplineCouplingLayer(dim, hidden_dim, torch.tensor([(i+j)%dim for j in range(dim)]).gt(0).float()) 
            for i in range(num_layers)
        ])

    def forward(self, x):
        """Map x to latent space z / Ánh xạ x sang không gian ẩn z."""
        total_log_det = 0 # Initialize log-det sum / Khởi tạo tổng log-det
        for layer in self.layers: # Iterate through layers / Lặp qua các lớp
            x, log_det = layer(x)
            total_log_det += log_det
        return x, total_log_det # Return latent z and log-det / Trả về không gian ẩn z và log-det

    def log_prob(self, x):
        """Direct log probability density estimation / Ước tính mật độ log xác suất trực tiếp."""
        z, log_det = self.forward(x) # Forward mapping / Phép ánh xạ thuận
        # Return probability using change of variables formula / Trả về xác suất sử dụng công thức đổi biến
        return self.prior.log_prob(z) + log_det
