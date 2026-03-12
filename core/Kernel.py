# NOTE: This module is currently not used by the primary ParallelFastHSIC engine.
# LƯU Ý: Module này hiện không được sử dụng bởi bộ máy ParallelFastHSIC chính.
# It is kept for legacy support or future kernel-based extensions.
# Nó được giữ lại để hỗ trợ các phiên bản cũ hoặc các mở rộng dựa trên kernel trong tương lai.

import torch # Import PyTorch library / Nhập thư viện PyTorch

def rbf_kernel(x, y=None, sigma=1.0):
    """Vectorized RBF Kernel using torch.cdist (Faster). / Nhân RBF vectơ hóa sử dụng torch.cdist (Nhanh hơn)."""
    if y is None: y = x # If y is not provided, use x / Nếu không có y, sử dụng x
    dist = torch.cdist(x, y, p=2)**2 # Square of Euclidean distance / Bình phương khoảng cách Euclidean
    return torch.exp(-dist / (2 * sigma**2)) # Exponential of negative distance / Hàm mũ của khoảng cách âm

def polynomial_kernel(x, y=None, degree=3, gamma=1.0, coef0=1.0):
    """Vectorized Polynomial Kernel. / Nhân Đa thức vectơ hóa."""
    if y is None: y = x # If y is not provided, use x / Nếu không có y, sử dụng x
    # (gamma * <x,y> + coef0)^degree / (gamma * tích vô hướng + hệ số)^bậc
    return (gamma * torch.mm(x, y.t()) + coef0) ** degree

def linear_kernel(x, y=None):
    """Vectorized Linear Kernel. / Nhân Tuyến tính vectơ hóa."""
    if y is None: y = x # If y is not provided, use x / Nếu không có y, sử dụng x
    return torch.mm(x, y.t()) # Matrix multiplication (dot product) / Nhân ma trận (tích vô hướng)

def get_kernel_matrix(x, y=None, kernel_type='rbf', **kwargs):
    """Wrapper for kernel matrix computation. / Lớp bao cho tính toán ma trận nhân."""
    if kernel_type == 'rbf': # Gaussian Kernel / Nhân Gaussian
        return rbf_kernel(x, y, **kwargs)
    elif kernel_type in ['poly', 'polynomial']: # Polynomial Kernel / Nhân Đa thức
        return polynomial_kernel(x, y, **kwargs)
    elif kernel_type == 'linear': # Linear Kernel / Nhân Tuyến tính
        return linear_kernel(x, y)
    raise ValueError(f"Unknown kernel type: {kernel_type}") # Handle unknown types / Xử lý loại không xác định
