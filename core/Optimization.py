import torch # Import PyTorch library / Nhập thư viện PyTorch

def compute_acyclicity_constraint(adj_matrix):
    """
    Computes h(W) = Tr(exp(W * W)) - d.
    Tính toán h(W) = Tr(exp(W * W)) - d.
    This constraint is 0 if and only if the graph is a DAG.
    Ràng buộc này bằng 0 khi và chỉ khi đồ thị là DAG (hướng không chu trình).
    """
    d = adj_matrix.shape[0] # Number of variables / Số lượng biến (chiều của ma trận)
    # Use squared matrix for the DAG constraint / Sử dụng bình phương ma trận cho ràng buộc DAG
    M = torch.matrix_exp(adj_matrix * adj_matrix) # Matrix exponential / Tính hàm mũ ma trận
    return torch.trace(M) - d # Return trace minus dimension / Trả về vết của ma trận trừ đi số chiều

class AugmentedLagrangian:
    """
    Optimized Augmented Lagrangian Solver for constrained optimization.
    Bộ giải Augmented Lagrangian tối ưu cho bài toán tối ưu hóa có ràng buộc.
    Subject to h(x) = 0. / Điều kiện ràng buộc h(x) = 0.
    """
    def __init__(self, rho_init=1.0, alpha_init=0.0, rho_max=1e10, gamma=5.0):
        self.rho = rho_init # Penalty parameter / Tham số phạt (rho)
        self.alpha = alpha_init # Lagrange multiplier / Nhân tử Lagrange (alpha)
        self.rho_max = rho_max # Max penalty value / Giá trị phạt tối đa
        self.gamma = gamma # Growth factor for rho / Hệ số tăng trưởng của rho

    def get_loss(self, main_loss, constraint_val):
        # Return total loss with penalty terms / Trả về tổng hàm mất mát kèm các thành phần phạt
        return main_loss + self.alpha * constraint_val + 0.5 * self.rho * (constraint_val ** 2)

    def update_parameters(self, constraint_val):
        """Updates parameters with improved stability. / Cập nhật các tham số với độ ổn định được cải thiện."""
        self.alpha += self.rho * constraint_val.item() # Update multiplier / Cập nhật nhân tử Lagrange
        if self.rho < self.rho_max: # Increase penalty if not at max / Tăng tham số phạt nếu chưa đạt tối đa
            self.rho *= self.gamma # Multiply rho by gamma / Nhân rho với hệ số gamma
        return self.alpha, self.rho # Return updated values / Trả về các giá trị đã cập nhật
