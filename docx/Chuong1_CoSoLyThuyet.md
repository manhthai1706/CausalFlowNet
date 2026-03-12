# CHƯƠNG 1: CƠ SỞ LÝ THUYẾT

## 1.1. Đồ thị Nhân quả và Mô hình Phương trình Cấu trúc

### 1.1.1. Đồ thị Có hướng Không chu trình (DAG)

Một **Đồ thị Có hướng Không chu trình** (Directed Acyclic Graph – DAG) là một cấu trúc toán học G = (V, E), trong đó V là tập hợp các nút (biến ngẫu nhiên) và E là tập hợp các cạnh có hướng, với điều kiện không tồn tại bất kỳ chu trình có hướng nào trong đồ thị. Trong bối cảnh khám phá nhân quả, mỗi cạnh có hướng i → j trong DAG biểu thị rằng biến X_i là nguyên nhân trực tiếp của biến X_j.

Tính chất quan trọng nhất của DAG là tính acyclicity (không chu trình). Để đảm bảo đồ thị học được từ dữ liệu là một DAG hợp lệ, CausalFlowNet sử dụng hàm ràng buộc sau:

**h(W) = Tr(exp(W ⊙ W)) − d = 0**

Trong đó W ∈ ℝ^(d×d) là ma trận trọng số kề, ⊙ là phép nhân từng phần tử (Hadamard product), và d là số lượng biến. Hàm h(W) = 0 khi và chỉ khi đồ thị biểu diễn bởi W là một DAG (Zheng et al., 2018). Việc tính hàm mũ ma trận (matrix exponential) đảm bảo tất cả các đường đi có độ dài bất kỳ đều được xét đến trong việc phát hiện chu trình.

### 1.1.2. Mô hình Phương trình Cấu trúc (SEM)

**Mô hình Phương trình Cấu trúc** (Structural Equation Model – SEM) mô tả cơ chế sinh ra dữ liệu thông qua hệ phương trình:

**X_i = f_i(PA_i, ε_i), ∀i = 1, ..., d**

Trong đó PA_i là tập hợp các biến cha (parents) của X_i trong DAG, f_i là hàm cấu trúc nhân quả, và ε_i là nhiễu độc lập.

Trong CausalFlowNet, hàm cấu trúc f_i được xấp xỉ bởi một mạng Perceptron Nhiều Lớp (MLP) dùng chung cho tất cả các nút. Phần dư sau khi trừ giá trị dự báo chính là phần nhiễu ε_i, và tính độc lập của ε_i với các biến đầu vào là điều kiện cần để xác nhận cấu trúc nhân quả tìm được.

---

## 1.2. Mạng Perceptron Nhiều Lớp Tăng cường (Gated-ResMLP)

### 1.2.1. Kiến trúc Cơ bản của MLP

**Mạng Perceptron Nhiều Lớp** (Multi-Layer Perceptron – MLP) là nền tảng của mọi mạng học sâu, bao gồm các lớp tuyến tính và hàm kích hoạt phi tuyến xếp chồng lên nhau. Mỗi lớp thực hiện phép biến đổi:

**h^(l) = σ(W^(l) · h^(l-1) + b^(l))**

### 1.2.2. Khối Residual có Cổng (Gated Residual Block)

Để tăng cường khả năng biểu diễn và giúp mô hình học được các cơ chế nhân quả phi tuyến phức tạp — điều mà MLP đơn giản khó đạt được do vấn đề gradient vanishing và thiếu cơ chế lọc thông tin — CausalFlowNet triển khai kiến trúc **Gated Residual Block**. Cơ chế hoạt động của khối này gồm ba bước:

1. **Layer Normalization:** Chuẩn hóa đầu vào để ổn định quá trình huấn luyện.
2. **Cơ chế Cổng (Gating):** Ánh xạ đặc trưng lên không gian 2D, sau đó tách thành hai thành phần — features và gate — điều tiết lẫn nhau theo công thức **h = σ_act(features) · σ(gate)**. Cơ chế này cho phép mô hình chủ động "bật/tắt" các tín hiệu đặc trưng tùy theo ngữ cảnh dữ liệu.
3. **Kết nối Residual:** Cộng kết quả với đầu vào gốc để tránh mất mát thông tin và giúp mô hình hội tụ nhanh hơn.

Trọng số được khởi tạo theo phương pháp **Orthogonal Initialization** với hệ số gain = 1.4, giúp gradient lan truyền ổn định qua nhiều lớp.

---

## 1.3. Normalizing Flows – Ước lượng Mật độ Xác suất

### 1.3.1. Khái niệm Normalizing Flows

**Normalizing Flows** là một họ mô hình xác suất sử dụng chuỗi các phép biến đổi khả nghịch (invertible transformations) để ánh xạ một phân phối đơn giản z (thường là Gauss) sang phân phối dữ liệu phức tạp x:

**x = f_K ∘ f_(K-1) ∘ ... ∘ f_1(z)**

Xác suất log của x được tính thông qua định lý đổi biến:

**log p(x) = log p(z) + Σ log |det(∂f_k/∂z_k)|**

Số hạng **log |det J|** (log của định thức Jacobian) đo lường sự co giãn thể tích của phép biến đổi — về mặt trực quan, nếu phép biến đổi làm "nén" không gian thì log-prob tăng lên, và ngược lại. Điều này đảm bảo tổng xác suất luôn bảo toàn và phân phối học được có tính chuẩn tắc.

### 1.3.2. Neural Spline Flow với Rational-Quadratic Splines

CausalFlowNet sử dụng **Neural Spline Flow** (NSF) với lớp ghép **Rational-Quadratic Splines** (RQS). Thay vì các phép biến đổi affine đơn giản, RQS chia miền đầu vào thành nhiều thùng (bins) và xấp xỉ từng đoạn bằng một hàm hữu tỷ bậc hai, cho phép biểu diễn phân phối phức tạp và đa dạng một cách linh hoạt.

**Cơ chế hoạt động của lớp ghép Spline:**
- Đầu vào x được chia thành hai phần theo mặt nạ nhị phân: phần không biến đổi và phần được biến đổi.
- Phần không biến đổi được đưa qua mạng nơ-ron để tạo ra các tham số đặc trưng của spline (bao gồm độ rộng, độ cao và đạo hàm của từng đoạn).
- Phần được biến đổi được áp dụng RQS sử dụng các tham số trên, sinh ra giá trị mới và log-determinant tương ứng.

### 1.3.3. Phân phối Ưu tiên Hỗn hợp Gaussian (Gaussian Mixture Prior)

Thay vì dùng Gauss đơn giản làm phân phối ưu tiên, CausalFlowNet sử dụng **Gaussian Mixture Model (GMM)** có thể học được. GMM là tổ hợp tuyến tính của K thành phần Gaussian:

**p(z) = Σ_k π_k · N(z | μ_k, σ_k)**

Trong đó π_k, μ_k, σ_k là các tham số có thể học được. GMM Prior cho phép mô hình nắm bắt cấu trúc đa cụm trong không gian ẩn, đặc biệt phù hợp với dữ liệu sinh học có nhiều phân tầng (như tập Sachs).

---

## 1.4. Toán tử HSIC và Kiểm định Độc lập Thống kê

### 1.4.1. Tiêu chuẩn Độc lập Hilbert-Schmidt (HSIC)

**Hilbert-Schmidt Independence Criterion** (HSIC) là một tiêu chuẩn đo lường sự phụ thuộc thống kê giữa hai biến ngẫu nhiên X và Y dựa trên lý thuyết không gian Hilbert tái sinh nhân (RKHS). HSIC = 0 khi và chỉ khi X và Y độc lập thống kê.

Trong SEM, nhiễu ε_i phải độc lập với các biến cha PA_i. CausalFlowNet sử dụng HSIC như một hạng phạt trong hàm mất mát để thúc đẩy tính độc lập này, từ đó xác nhận tính đúng đắn của chiều nhân quả được học.

### 1.4.2. Xấp xỉ Nhanh bằng Random Fourier Features

Tính toán HSIC chính xác có độ phức tạp O(N²), không khả thi với tập dữ liệu lớn. CausalFlowNet giải quyết vấn đề này bằng cơ chế tính toán HSIC song song siêu nhanh, sử dụng **Random Fourier Features** (RFF) để xấp xỉ hàm nhân Gauss:

**k(x, y) ≈ φ(x)ᵀ φ(y)**

Trong đó φ(x) = √(2/m) · cos(Wᵀx + b) là đặc trưng Fourier ngẫu nhiên với W ∼ N(0, I) và b ∼ Uniform(0, 2π). Phép xấp xỉ này hạ độ phức tạp tính toán xuống còn **O(N · m)** (với m là số lượng đặc trưng ngẫu nhiên), cho phép xử lý song song tất cả N nút nhân quả trong một phép nhân ma trận theo lô (batch matrix multiplication).

---

## 1.5. Tối ưu hóa Có ràng buộc bằng Augmented Lagrangian

### 1.5.1. Bài toán Tối ưu hóa Có ràng buộc

Bài toán học cấu trúc nhân quả trong CausalFlowNet được phát biểu dưới dạng bài toán tối ưu hóa có ràng buộc:

**min_{W} L(W) = NLL(W) + λ_HSIC · L_HSIC(W) + λ_L1 · ‖W‖₁**

**subject to: h(W) = Tr(exp(W ⊙ W)) − d = 0**

Trong đó NLL là hàm mất mát âm log-likelihood từ Normalizing Flow, L_HSIC là hạng phạt HSIC, ‖W‖₁ là chuẩn L1 khuyến khích ma trận kề thưa (sparse), và h(W) = 0 là ràng buộc acyclicity.

### 1.5.2. Phương pháp Augmented Lagrangian (ALM)

**Augmented Lagrangian Method** chuyển bài toán có ràng buộc sang bài toán không ràng buộc bằng cách đưa ràng buộc vào hàm mục tiêu:

**L_aug(W, α, ρ) = L(W) + α · h(W) + (ρ/2) · h(W)²**

Trong đó α là nhân tử Lagrange và ρ là tham số phạt. Sau mỗi vòng lặp ngoài, các tham số được cập nhật theo quy tắc:

- **α ← α + ρ · h(W)**
- **ρ ← min(γ · ρ, ρ_max)**

Với γ là hệ số tăng trưởng và ρ_max là giá trị phạt tối đa (được giới hạn để đảm bảo ổn định số học). Phương pháp này đảm bảo mô hình hội tụ về một DAG hợp lệ.

---

## 1.6. Các Chỉ số Đánh giá Cấu trúc Nhân quả

### 1.6.1. Tỷ lệ Dương tính thật (TPR) và Tỷ lệ Dương tính giả (FPR)

Hai chỉ số cơ bản để đánh giá chất lượng đồ thị nhân quả được phục hồi:

- **TPR (True Positive Rate)** = TP / (TP + FN): Tỷ lệ các cạnh thực sự tồn tại mà mô hình phát hiện đúng. TPR cao cho thấy mô hình không bỏ sót các mối quan hệ nhân quả quan trọng.
- **FPR (False Positive Rate)** = FP / (FP + TN): Tỷ lệ các cạnh không tồn tại nhưng bị mô hình phát hiện nhầm. FPR thấp cho thấy mô hình ít tạo ra cạnh giả.

### 1.6.2. Khoảng cách Hamming Cấu trúc (SHD)

**Structural Hamming Distance (SHD)** đếm số lượng chỉnh sửa tối thiểu cần thiết để biến đồ thị ước tính thành đồ thị thực tế. Các chỉnh sửa bao gồm thêm cạnh, xóa cạnh hoặc đảo chiều cạnh. SHD = 0 có nghĩa là mô hình phục hồi hoàn toàn chính xác cấu trúc đồ thị.

### 1.6.3. Khoảng cách Can thiệp Cấu trúc (SID)

**Structural Intervention Distance (SID)** (Peters & Bühlmann, 2015) đo lường sự khác biệt giữa hai DAG dưới góc độ **can thiệp nhân quả**. SID đếm số cặp (i, j) mà quan hệ nhân quả "j có phải là hậu duệ của i không?" bị phán đoán sai. SID phản ánh trực tiếp chất lượng mô hình khi được dùng để trả lời các câu hỏi can thiệp (do-calculus), đây là mục tiêu ứng dụng cuối cùng của khám phá nhân quả.

Cả ba chỉ số trên đều được tính toán và báo cáo trong mọi thực nghiệm của đề tài.
