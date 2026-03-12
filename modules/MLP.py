import torch # Import PyTorch library / Nhập thư viện PyTorch
import torch.nn as nn # Neural network module / Module mạng nơ-ron

class GatedResBlock(nn.Module):
    """
    Advanced Gated Residual Block.
    Khối Residual có cổng (Gated) tiên tiến.
    Implements a context-aware 'on/off' switch for protein interactions.
    Triển khai cơ chế 'bật/tắt' theo ngữ cảnh cho các tương tác protein.
    """
    def __init__(self, dim):
        super().__init__()
        self.layernorm = nn.LayerNorm(dim) # Layer Normalization for stability / Chuẩn hóa lớp để ổn định
        # Gating: Linear to 2 * dim (one for features, one for gate)
        # Cổng: Ánh xạ tuyến tính lên 2 lần số chiều (một cho đặc trưng, một cho cổng)
        self.gate_linear = nn.Linear(dim, dim * 2)
        self.output_linear = nn.Linear(dim, dim) # Final output projection / Phép chiếu đầu ra cuối cùng
        self.act = nn.LeakyReLU(0.2, inplace=False) # Non-linear activation / Hàm kích hoạt phi tuyến

    def forward(self, x):
        """Forward pass through gated block / Lan truyền thuận qua khối có cổng."""
        residual = x # Identity skip connection / Kết nối tắt (residual)
        x = self.layernorm(x) # Apply normalization / Áp dụng chuẩn hóa
        
        # Split into features and gating signal / Chia thành các đặc trưng và tín hiệu cổng
        gated = self.gate_linear(x)
        features, gate = gated.chunk(2, dim=-1)
        
        # Gating mechanism: features * sigmoid(gate)
        # Cơ chế cổng: đặc trưng * sigmoid(cổng)
        x = self.act(features) * torch.sigmoid(gate)
        
        x = self.output_linear(x) # Project back to original dim / Chiếu lại về số chiều ban đầu
        return x + residual # Return sum (residual connection) / Trả về tổng (kết nối residual)

class MLP(nn.Module):
    """
    Super-fast Gated-ResMLP.
    Mạng MLP Residual có cổng siêu nhanh.
    Mimics biological gating mechanisms (Context-dependent interactions).
    Mô phỏng cơ chế cổng sinh học (Các tương tác phụ thuộc vào ngữ cảnh).
    """
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        # Initial projection to hidden state / Phép chiếu ban đầu vào trạng thái ẩn
        layers = [nn.Linear(input_dim, hidden_dims[0]), nn.LeakyReLU(0.2, inplace=False)]
        
        # Stack Gated Residual Blocks / Xếp chồng các khối Residual có cổng
        for h in hidden_dims:
            layers.append(GatedResBlock(h))
            
        # Final prediction layer / Tầng dự báo cuối cùng
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.net = nn.Sequential(*layers) # Sequence container / Bộ chứa chuỗi các tầng
        self._init_weights() # Initialize weights / Khởi tạo trọng số

    def _init_weights(self):
        """Orthogonal weight initialization for faster convergence / Khởi tạo trọng số trực giao để hội tụ nhanh hơn."""
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear): # Initialize Linear layers / Khởi tạo các tầng Tuyến tính
                    nn.init.orthogonal_(m.weight, gain=1.4)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0) # Zero initialization for bias / Khởi tạo 0 cho độ lệch

    def forward(self, x):
        """Main forward pass / Lan truyền thuận chính."""
        return self.net(x) # Execute network / Thực thi mạng

def test_mlp():
    """Simple test function for the MLP / Hàm kiểm thử đơn giản cho MLP."""
    X = torch.randn(100, 5) # Create random input / Tạo đầu vào ngẫu nhiên
    model = MLP(5, [16, 16], 1) # Initialize model / Khởi tạo mô hình
    y = model(X) # Forward pass / Chạy thử mẫu
    print(f"MLP Test Success. Output shape: {y.shape}") # Verify shape / Xác nhận kích thước đầu ra

if __name__ == "__main__":
    test_mlp() # Execute test / Chạy kiểm thử
