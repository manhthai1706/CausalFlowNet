# CHƯƠNG 1: CƠ SỞ LÝ THUYẾT

Chương này trình bày các nền tảng lý thuyết cần thiết để xây dựng và hiểu rõ mô hình CausalFlowNet. Nội dung bao quát từ lý thuyết đồ thị nhân quả và mô hình phương trình cấu trúc, các kỹ thuật học sâu hiện đại như Gated Residual MLP và Normalizing Flows, cho đến phương pháp kiểm định độc lập thống kê bằng HSIC và tối ưu hóa có ràng buộc bằng Augmented Lagrangian. Phần cuối chương giới thiệu các chỉ số định lượng được dùng để đánh giá chất lượng đồ thị nhân quả học được.

## 1.1. Đồ thị Nhân quả và Mô hình Phương trình Cấu trúc

### 1.1.1. Đồ thị Có hướng Không chu trình (DAG)

Một **Đồ thị Có hướng Không chu trình** (Directed Acyclic Graph – DAG) là một cấu trúc toán học G = (V, E), trong đó V là tập hợp các nút (biến ngẫu nhiên) và E là tập hợp các cạnh có hướng, với điều kiện không tồn tại bất kỳ chu trình có hướng nào trong đồ thị. Trong bối cảnh khám phá nhân quả, mỗi cạnh có hướng $X_i \rightarrow X_j$ trong DAG biểu thị rằng biến $X_i$ là nguyên nhân trực tiếp của biến $X_j$ trong phạm vi các biến quan sát được (tức là không có biến trung gian nào khác nằm trong mô hình đóng vai trò trung chuyển tác động từ $X_i$ sang $X_j$).

Tính chất quan trọng nhất của DAG là tính acyclicity (không chu trình). Để đảm bảo đồ thị học được từ dữ liệu là một DAG hợp lệ, CausalFlowNet sử dụng hàm ràng buộc sau:

$$ h(W) = \text{Tr}(e^{W \circ W}) - d = 0 $$

Trong đó $W \in \mathbb{R}^{d \times d}$ là ma trận trọng số kề kích thước $d \times d$, $W \circ W$ là phép nhân từng phần tử (Hadamard product), $d$ là số lượng biến, và $\text{Tr}(\cdot)$ là Vết của ma trận (tổng các phần tử trên đường chéo chính). Hàm $h(W) = 0$ khi và chỉ khi đồ thị biểu diễn bởi $W$ là một DAG (Zheng et al., 2018). Việc tính hàm mũ ma trận (matrix exponential) đảm bảo tất cả các đường đi có độ dài bất kỳ đều được xét đến trong việc phát hiện chu trình.

### 1.1.2. Mô hình Nhân quả Cấu trúc (SCM)

**Mô hình Nhân quả Cấu trúc** (Structural Causal Model – SCM), được hệ thống hóa bởi Judea Pearl (2000), là một khung lý thuyết tổng quát để biểu diễn và suy luận về quan hệ nhân quả. Một SCM bao gồm ba thành phần:

1. **Tập biến ngoại sinh $U$** (exogenous variables): Các biến nhiễu độc lập, đại diện cho các yếu tố bên ngoài mô hình không thể quan sát trực tiếp.
2. **Tập biến nội sinh $X$** (endogenous variables): Các biến quan sát được, mỗi biến được xác định bởi các biến cha và nhiễu tương ứng thông qua một hàm cấu trúc.
3. **Đồ thị nhân quả $G$**: Biểu diễn quan hệ nhân quả giữa các biến dưới dạng DAG.

Điểm mạnh cốt lõi của SCM so với các mô hình thống kê thông thường là khả năng trả lời câu hỏi **can thiệp (interventional)** thông qua toán tử $do(\cdot)$ của Pearl. Khi thực hiện can thiệp $do(X_i = v)$, tức là gán cưỡng bức giá trị cho biến $X_i$, toàn bộ cơ chế sinh dữ liệu thay đổi theo cấu trúc nhân quả, khác hoàn toàn so với việc chỉ điều kiện hóa thống kê thông thường (conditioning). Khả năng phân biệt này là nền tảng để CausalFlowNet học được cấu trúc nhân quả thực sự từ dữ liệu quan sát.

### 1.1.3. Mô hình Phương trình Cấu trúc (SEM)

**Mô hình Phương trình Cấu trúc** (Structural Equation Model – SEM) mô tả cơ chế sinh ra dữ liệu thông qua hệ phương trình:

$$ X_i = f_i(PA_i, \epsilon_i), \quad \forall i = 1, \dots, d $$

Trong đó $PA_i$ là tập hợp các biến cha (parents) của $X_i$ trong DAG, $f_i$ là hàm cấu trúc nhân quả, và $\epsilon_i$ là nhiễu độc lập.

Trong CausalFlowNet, hàm cấu trúc $f_i$ được xấp xỉ bởi một mạng Perceptron Nhiều Lớp (MLP) dùng chung cho tất cả các nút. Phần dư sau khi trừ giá trị dự báo chính là phần nhiễu $\epsilon_i$, và tính độc lập của $\epsilon_i$ với các biến đầu vào là điều kiện cần để xác nhận cấu trúc nhân quả tìm được.

---

## 1.2. Mạng Perceptron Nhiều Lớp Tăng cường (Gated-ResMLP)

### 1.2.1. Kiến trúc Cơ bản của MLP

**Mạng Perceptron Nhiều Lớp** (Multi-Layer Perceptron – MLP) là nền tảng của mọi mạng học sâu, bao gồm các lớp tuyến tính và hàm kích hoạt phi tuyến xếp chồng lên nhau. Mỗi lớp thực hiện phép biến đổi:

$$ h^{(l)} = \sigma(W^{(l)} \cdot h^{(l-1)} + b^{(l)}) $$

Trong đó $W^{(l)}$ và $b^{(l)}$ lần lượt là ma trận trọng số và vectơ độ lệch của lớp thứ $l$, còn $\sigma$ là hàm kích hoạt phi tuyến.

### 1.2.2. Khối Residual có Cổng (Gated Residual Block)

Để tăng cường khả năng biểu diễn và giúp mô hình học được các cơ chế nhân quả phi tuyến phức tạp — điều mà MLP đơn giản khó đạt được do vấn đề gradient vanishing và thiếu cơ chế lọc thông tin — CausalFlowNet triển khai kiến trúc **Gated Residual Block**. Cơ chế hoạt động của khối này gồm ba bước:

1. **Layer Normalization:** Chuẩn hóa đầu vào để ổn định quá trình huấn luyện.
2. **Cơ chế Cổng (Gating):** Ánh xạ đặc trưng lên không gian 2D, sau đó tách thành hai thành phần — features và gate — điều tiết lẫn nhau theo công thức **$h = \sigma_{\text{act}}(\text{features}) \circ \sigma(\text{gate})$**. Trong đó $\sigma_{\text{act}}$ là hàm kích hoạt đặc trưng (như SiLU), $\sigma$ là hàm Sigmoid ép giá trị cổng về khoảng $(0,1)$, và $\circ$ là phép nhân từng phần tử. Cơ chế này cho phép mô hình chủ động "bật/tắt" các tín hiệu đặc trưng tùy theo ngữ cảnh dữ liệu.
3. **Kết nối Residual:** Cộng kết quả với đầu vào gốc để tránh mất mát thông tin và giúp mô hình hội tụ nhanh hơn.

Trọng số được khởi tạo theo phương pháp **Orthogonal Initialization** với hệ số gain = 1.4, giúp gradient lan truyền ổn định qua nhiều lớp.

---

## 1.3. Normalizing Flows – Ước lượng Mật độ Xác suất

### 1.3.1. Khái niệm Normalizing Flows

**Normalizing Flows** là một họ mô hình xác suất sử dụng chuỗi các phép biến đổi khả nghịch (invertible transformations) để ánh xạ biến dữ liệu gốc $x$ thành một biến trong không gian ẩn $z$ có phân phối biết trước (thường là Gauss hoặc GMM):

$$ z = f_K \circ f_{K-1} \circ \dots \circ f_1(x) $$

Xác suất log của dữ liệu quan sát $x$ được tính chuẩn xác thông qua định lý đổi biến, dựa trên phân phối ưu tiên $p(z)$ và đạo hàm của phép biến đổi:

$$ \log p(x) = \log p(z) + \sum_{k=1}^K \log \left| \det \left( \frac{\partial f_k}{\partial x_{k-1}} \right) \right| $$

Số hạng **$\log |\det J|$** (log của định thức ma trận Jacobian) đo lường sự thay đổi thể tích cục bộ của phép biến đổi. Về mặt trực quan, nếu phép biến đổi làm "nén" không gian thì mật độ xác suất tại đó tăng lên, và ngược lại. Điều này giúp CausalFlowNet đánh giá chính xác mật độ xác suất của từng nhiễu $\epsilon_i$ khi đưa chúng qua luồng Spline.

### 1.3.2. Neural Spline Flow với Rational-Quadratic Splines

CausalFlowNet sử dụng **Neural Spline Flow** (NSF) với lớp ghép **Rational-Quadratic Splines** (RQS). Thay vì các phép biến đổi affine đơn giản, RQS chia miền đầu vào thành nhiều thùng (bins) và xấp xỉ từng đoạn bằng một hàm hữu tỷ bậc hai, cho phép biểu diễn phân phối phức tạp và đa dạng một cách linh hoạt.

**Cơ chế hoạt động của lớp ghép Spline:**
- Đầu vào x được chia thành hai phần theo mặt nạ nhị phân: phần không biến đổi và phần được biến đổi.
- Phần không biến đổi được đưa qua mạng nơ-ron để tạo ra các tham số đặc trưng của spline (bao gồm độ rộng, độ cao và đạo hàm của từng đoạn).
- Phần được biến đổi được áp dụng RQS sử dụng các tham số trên, sinh ra giá trị mới và log-determinant tương ứng.

### 1.3.3. Phân phối Ưu tiên Hỗn hợp Gaussian (Gaussian Mixture Prior)

Thay vì dùng Gauss đơn giản làm phân phối ưu tiên, CausalFlowNet sử dụng **Gaussian Mixture Model (GMM)** có thể học được. GMM là tổ hợp tuyến tính của K thành phần Gaussian:

$$ p(z) = \sum_{k=1}^K \pi_k \mathcal{N}(z | \mu_k, \Sigma_k) $$

Trong đó $\pi_k, \mu_k, \Sigma_k$ là các tham số có thể học được. GMM Prior cho phép mô hình nắm bắt cấu trúc đa cụm trong không gian ẩn, đặc biệt phù hợp với dữ liệu sinh học có nhiều phân tầng (như tập Sachs).

---

## 1.4. Toán tử HSIC và Kiểm định Độc lập Thống kê

### 1.4.1. Tiêu chuẩn Độc lập Hilbert-Schmidt (HSIC)

**Hilbert-Schmidt Independence Criterion** (HSIC) là một tiêu chuẩn đo lường sự phụ thuộc thống kê giữa hai biến ngẫu nhiên X và Y dựa trên lý thuyết không gian Hilbert tái sinh nhân (RKHS). HSIC = 0 khi và chỉ khi X và Y độc lập thống kê.

Trong SEM, nhiễu $\epsilon_i$ phải độc lập với các biến cha $PA_i$. CausalFlowNet sử dụng HSIC như một hạng phạt trong hàm mất mát để thúc đẩy tính độc lập này, từ đó xác nhận tính đúng đắn của chiều nhân quả được học.

### 1.4.2. Xấp xỉ Nhanh bằng Random Fourier Features

Tính toán HSIC chính xác có độ phức tạp O(N²), không khả thi với tập dữ liệu lớn. CausalFlowNet giải quyết vấn đề này bằng cơ chế tính toán HSIC song song siêu nhanh, sử dụng **Random Fourier Features** (RFF) để xấp xỉ hàm nhân Gauss:

$$ k(x, y) \approx \phi(x)^\top \phi(y) $$

Trong đó $\phi(x) = \sqrt{\frac{2}{m}} \cos(W^\top x + b)$ là đặc trưng Fourier ngẫu nhiên với $W \sim \mathcal{N}(0, I)$ và $b \sim \text{Uniform}(0, 2\pi)$. Phép xấp xỉ này hạ độ phức tạp tính toán xuống còn **$\mathcal{O}(N \cdot m)$** (với $m$ là số lượng đặc trưng ngẫu nhiên), cho phép xử lý song song tất cả N nút nhân quả trong một phép nhân ma trận theo lô (batch matrix multiplication).

---

## 1.5. Tối ưu hóa Có ràng buộc bằng Augmented Lagrangian

### 1.5.1. Bài toán Tối ưu hóa Có ràng buộc

Bài toán học cấu trúc nhân quả trong CausalFlowNet được phát biểu dưới dạng bài toán tối ưu hóa có ràng buộc:

$$ \min_W L(W) = \text{NLL}(W) + \lambda_{HSIC} L_{HSIC}(W) + \lambda_{L1} \|W\|_1 $$

**Điều kiện:** $\quad h(W) = \text{Tr}(e^{W \circ W}) - d = 0$

Trong đó NLL là hàm mất mát âm log-likelihood từ Normalizing Flow, $L_{HSIC}$ là hạng phạt HSIC, $\|W\|_1$ là chuẩn L1 khuyến khích ma trận kề thưa (sparse), $\lambda_{HSIC}$ và $\lambda_{L1}$ là các hệ số điều chuẩn, và $h(W) = 0$ là ràng buộc acyclicity.

### 1.5.2. Phương pháp Augmented Lagrangian (ALM)

**Augmented Lagrangian Method** chuyển bài toán có ràng buộc sang bài toán không ràng buộc bằng cách đưa ràng buộc vào hàm mục tiêu:

$$ L_{\text{aug}}(W, \alpha, \rho) = L(W) + \alpha h(W) + \frac{\rho}{2} h(W)^2 $$

Trong đó $\alpha$ là nhân tử Lagrange và $\rho$ là tham số phạt. Sau mỗi vòng lặp ngoài, các tham số được cập nhật theo quy tắc:

- $\alpha \leftarrow \alpha + \rho h(W)$
- $\rho \leftarrow \min(\gamma \rho, \rho_{\text{max}})$

Với $\gamma$ là hệ số tăng trưởng và $\rho_{\text{max}}$ là giá trị phạt tối đa (được giới hạn để đảm bảo ổn định số học). Phương pháp này đảm bảo mô hình hội tụ về một DAG hợp lệ.

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

---

## 1.7. Tiểu kết Chương 1

Chương 1 đã trình bày các nền tảng lý thuyết cốt lõi phục vụ cho đề tài, bao gồm: khái niệm DAG và SEM làm ngôn ngữ biểu diễn quan hệ nhân quả; kiến trúc Gated Residual MLP để xấp xỉ hàm cấu trúc phi tuyến; Neural Spline Flow để ước lượng mật độ phân phối nhiễu linh hoạt; toán tử HSIC để kiểm định tính độc lập thống kê; và phương pháp Augmented Lagrangian để tối ưu hóa có ràng buộc acyclicity. Cuối cùng, bộ ba chỉ số TPR, SHD và SID cung cấp công cụ đánh giá đa chiều cho đồ thị nhân quả được phục hồi. Những lý thuyết này là nền tảng trực tiếp cho kiến trúc CausalFlowNet sẽ được trình bày trong Chương 2.

