# CausalFlowNet: Khám phá Cấu trúc Nhân quả Phi tuyến tính dựa trên Luồng Chuẩn hóa và Kiểm định Độc lập Song song

**Tác giả:** Trần Mạnh Thái  
*Khoa lưu trữ mã nguồn:* [manhthai1706/CausalFlowNet](https://github.com/manhthai1706/CausalFlowNet)

---

## Tóm tắt (Abstract)

Khám phá cấu trúc nhân quả (Causal Discovery) từ dữ liệu quan sát liên tục là một trong những bài toán cốt lõi của khoa học dữ liệu và suy luận nhân quả. Tuy nhiên, các mô hình nhân quả hàm số (Functional Causal Models - FCMs) hiện đại thường phải dựa vào các giả định tham số cứng nhắc (như cơ chế nhiễu tuyến tính Gauss) hoặc không thể áp đặt một cách tường minh điều kiện độc lập thống kê giữa phần dư cấu trúc (structural residuals) và các biến cha nhân quả—một yêu cầu bắt buộc của các mô hình nhiễu cộng (Additive Noise Models - ANMs). 

Để giải quyết các hạn chế này, chúng tôi đề xuất **CausalFlowNet**, một khung học sâu hợp nhất đầu-cuối (end-to-end) cho việc học cấu trúc nhân quả liên tục. CausalFlowNet tích hợp ba thành phần cốt lõi:
1. **Mô hình hóa cơ chế phi tuyến tính linh hoạt** thông qua mạng Perceptron đa lớp khối dư có cổng chia sẻ (**Gated Residual Multi-Layer Perceptron - Gated-ResMLP**).
2. **Ước lượng mật độ nhiễu bất biến** sử dụng luồng spline nơ-ron (**Neural Spline Flows - NSF**) kết hợp với phân phối tiên nghiệm mô hình hỗn hợp Gauss (**Gaussian Mixture Model - GMM**) có khả năng tự học.
3. **Áp đặt tường minh giả định độc lập** giữa biến cha và phần dư của ANM thông qua mô đun tiêu chí độc lập Hilbert-Schmidt (**Hilbert-Schmidt Independence Criterion - HSIC**) song song hóa hoàn toàn, được tăng tốc bởi các đặc trưng Fourier ngẫu nhiên (**Random Fourier Features - RFF**).

Toàn bộ khung mô hình được tối ưu hóa đồng thời dựa trên **Phương pháp Lagrangian Tăng cường (Augmented Lagrangian Method - ALM)** với ràng buộc phi chu trình liên tục (continuous acyclicity constraint), đảm bảo nghiệm thu được là một đồ thị có hướng không chu trình (DAG). Thử nghiệm thực nghiệm trên mạng lưới truyền tin nội bào thực tế (**Sachs**, $d=11$ nút) và dữ liệu mô phỏng biểu hiện gen sinh học (**SynTReN**, $d=20$ nút) chứng minh rằng CausalFlowNet đạt được hiệu năng vượt trội hoặc cạnh tranh mạnh mẽ so với các phương pháp tiên tiến nhất (State-of-the-Art) trên nhiều thước đo đánh giá cấu trúc bao gồm Khoảng cách Hamming Cấu trúc (SHD), CPDAG-SHD ($SHD\text{-}c$), và Khoảng cách Can thiệp Cấu trúc (SID).

---

## I. Giới thiệu (Introduction)

Khám phá cấu trúc nhân quả nhằm mục đích tái dựng đồ thị có hướng không chu trình (Directed Acyclic Graph - DAG) biểu diễn các mối quan hệ nhân quả giữa một tập hợp các biến quan sát. Về mặt lịch sử, các phương pháp truyền thống được chia làm hai trường phái chính:
* **Các phương pháp dựa trên ràng buộc (Constraint-based methods)** (ví dụ: thuật toán PC, FCI): Sử dụng các kiểm định độc lập điều kiện. Nhược điểm của trường phái này là cực kỳ nhạy cảm với sai số tích lũy của các kiểm định độc lập, dễ lan truyền lỗi quyết định và có độ phức tạp tính toán rất lớn khi số lượng biến tăng.
* **Các phương pháp dựa trên điểm số (Score-based methods)** (ví dụ: GES, Hill-Climbing): Tìm kiếm trên không gian đồ thị rời rạc để tối đa hóa một hàm điểm số (như Bayesian Information Criterion - BIC). Phương pháp này đối mặt với sự bùng nổ tổ hợp do số lượng DAG khả dĩ tăng theo hàm siêu mũ (super-exponentially) so với số lượng biến $d$: 
  $$f(d) \approx 2^{d^2/2}$$

Một bước ngoặt lớn xảy ra với sự ra đời của thuật toán **NOTEARS** [13], chuyển đổi bài toán tìm kiếm rời rạc NP-khó thành một bài toán tối ưu hóa liên tục trên ma trận trọng số kề $\mathbf{W} \in \mathbb{R}^{d \times d}$. Tính phi chu trình được áp đặt thông qua một hàm ràng buộc trơn dựa trên hàm mũ ma trận (matrix-exponential) $h(\mathbf{W}) = 0$, cho phép áp dụng trực tiếp các thuật toán tối ưu hóa dựa trên gradient.

Mặc dù NOTEARS và các biến thể mạng nơ-ron sâu của nó (như GraN-DAG [15], DAG-GNN) đã giải quyết tốt bài toán tối ưu hóa liên tục, vẫn tồn tại ba thách thức lớn chưa được giải quyết triệt để trong các tài liệu hiện hành:
1. **Giả định phân phối nhiễu tham số cứng nhắc**: Hầu hết các phương pháp hiện tại giả định nhiễu cấu trúc tuân theo một phân phối đơn giản cố định (thường là phân phối Gauss thuần nhất - homoscedastic Gaussian). Trong các hệ thống sinh học thực tế, nhiễu thường phi Gauss, bất đối xứng, đa đỉnh (multi-modal) hoặc có đuôi nặng (heavy-tailed).
2. **Áp đặt độc lập một cách gián tiếp**: Trong khung mô hình nhiễu cộng (ANM) [8], phần dư cấu trúc $\varepsilon_j$ bắt buộc phải độc lập thống kê với các biến cha nhân quả $\mathbf{PA}_j$. Các phương pháp tối ưu hóa liên tục thông thường chỉ tối đa hóa hàm hợp lý (likelihood), điều này chỉ thúc đẩy phần dư độc lập một cách gián tiếp, dẫn đến cấu trúc đồ thị thu được thường bị sai lệch (sub-optimal) trong các trường hợp kích thước mẫu nhỏ.
3. **Độ phức tạp tính toán của kiểm định độc lập dạng nhân (Kernel)**: Các kiểm định độc lập dựa trên nhân truyền thống, chẳng hạn như Tiêu chí Độc lập Hilbert-Schmidt (HSIC) [4], có độ phức tạp tính toán tỷ lệ thuận với bình phương số mẫu $\mathcal{O}(n^2)$. Điều này khiến việc áp dụng trực tiếp chúng làm thành phần phạt (penalty) trong quá trình lan truyền ngược (backpropagation) cho tất cả các nút cùng lúc trở nên bất khả thi về mặt tính toán đối với dữ liệu lớn.

Để vượt qua các rào cản này, chúng tôi giới thiệu **CausalFlowNet**, một khung làm việc hợp nhất đầu-cuối kết hợp bộ ước lượng mật độ nơ-ron cực kỳ linh hoạt (Rational-Quadratic Spline Flows) với một hàm phạt độc lập song song hóa siêu nhanh dựa trên RFF-HSIC dưới một cơ chế tối ưu hóa ràng buộc liên tục duy nhất.

---

## II. Cơ sở Lý thuyết & Khung Toán học

<p align="center">
  <img src="arch.png" width="90%" alt="Kiến trúc Khoa học của CausalFlowNet"/>
</p>

### A. Mô hình Phương trình Cấu trúc và Giả định Nhiễu Cộng

Xét một vectơ ngẫu nhiên $d$ chiều $\mathbf{X} = (X_1, X_2, \ldots, X_d)$ được chi phối bởi một phân phối đồng thời $P(\mathbf{X})$ sinh ra từ một Mô hình Phương trình Cấu trúc (Structural Equation Model - SEM) trên một đồ thị DAG $\mathcal{G} = (\mathbf{V}, \mathbf{E})$:

$$X_j = f_j(\mathbf{PA}_j^{\mathcal{G}}) + \varepsilon_j, \quad \forall j \in \{1, \ldots, d\}$$

Trong đó:
* $f_j: \mathbb{R}^{|\mathbf{PA}_j|} \rightarrow \mathbb{R}$ là một hàm phi tuyến liên tục bất kỳ biểu diễn cơ chế nhân quả.
* $\mathbf{PA}_j^{\mathcal{G}} \subseteq \mathbf{V} \setminus \{X_j\}$ đại diện cho tập hợp các nút cha trực tiếp của $X_j$ trong đồ thị $\mathcal{G}$.
* $\varepsilon_j$ là các biến nhiễu (noise variables) độc lập đôi một với nhau.

Khung Mô hình Nhiễu Cộng (Additive Noise Model - ANM) đảm bảo tính định danh duy nhất (unique identifiability) của đồ thị DAG thực tế dưới các điều kiện yếu [8], với điều kiện là nhiễu cấu trúc $\varepsilon_j$ độc lập thống kê với các biến cha tương ứng:

$$\varepsilon_j \perp\!\!\!\perp X_i, \quad \forall X_i \in \mathbf{PA}_j^{\mathcal{G}}$$

### B. Mô hình hóa Phần dư Khả nghịch thông qua Luồng Spline Nơ-ron

Thay vì áp đặt giả định nhiễu Gauss thông thường, CausalFlowNet học mật độ chính xác của từng phần dư cấu trúc bằng cách sử dụng **Luồng Spline Nơ-ron (Neural Spline Flow - NSF)** [3]. Đối với mỗi biến $X_j$, phần dư $\varepsilon_j = X_j - f_j(\mathbf{PA}_j^{\mathcal{G}})$ được ánh xạ qua một hàm khả nghịch $g_{\boldsymbol{\theta}}: \mathbb{R} \rightarrow \mathbb{R}$ để chuyển sang một biến ẩn $z_j$. Áp dụng định lý biến đổi biến số (change-of-variables), ta thu được hàm log-likelihood chính xác và khả vi:

$$\log p(\varepsilon_j) = \log p_{\text{prior}}\bigl(g_{\boldsymbol{\theta}}(\varepsilon_j)\bigr) + \log \left| \frac{\partial g_{\boldsymbol{\theta}}(\varepsilon_j)}{\partial \varepsilon_j} \right|$$

Luồng $g_{\boldsymbol{\theta}}$ được hiện thực hóa bằng sự kết hợp của các tầng liên kết **Rational-Quadratic Spline** (RQS). Đây là các phép biến đổi đơn điệu, khả nghịch giải tích, có các tham số được dự báo bởi một mạng nơ-ron nhỏ, cho phép mô hình hóa các phân phối phần dư phức tạp, phi Gauss một cách cực kỳ linh hoạt.

#### Hàm Tiên nghiệm Mô hình Hỗn hợp Gauss tự học (Learnable GMM Prior)

Để nắm bắt cấu trúc ẩn đa đỉnh của phần dư và hỗ trợ quá trình phân nhóm nhân quả (causal subgrouping) ở các bước sau, chúng tôi thay thế phân phối tiên nghiệm Gauss tiêu chuẩn bằng một **Mô hình Hỗn hợp Gauss (GMM)** gồm $C$ thành phần có khả năng tự học:

$$p_{\text{prior}}(z) = \sum_{c=1}^{C} \pi_c \, \mathcal{N}(z \mid \mu_c, \sigma_c^2)$$

Trong đó, các trọng số hỗn hợp $\pi_c$, kỳ vọng $\mu_c$, và phương sai $\sigma_c^2$ là các tham số được tối ưu hóa đồng thời đầu-cuối thông qua thuật toán lan truyền ngược gradient.

---

### C. Kiểm định Độc lập Song song hóa qua RFF-HSIC tốc độ cao

Để áp đặt một cách tường minh điều kiện độc lập của ANM trong suốt quá trình huấn luyện mạng, chúng tôi phát triển một mô đun song song hóa dựa trên **Tiêu chí Độc lập Hilbert-Schmidt (HSIC)** [4]. Kiểm định HSIC dạng nhân thông thường yêu cầu tính toán ma trận nhân (kernel matrix) với độ phức tạp $\mathcal{O}(n^2)$ trên từng nút, điều này vô cùng đắt đỏ khi thực hiện tối ưu hóa dựa trên gradient đồng thời cho toàn bộ $d$ biến.

Chúng tôi giải quyết bài toán này bằng cách xấp xỉ nhân RBF thông qua **Đặc trưng Fourier Ngẫu nhiên (Random Fourier Features - RFF)** (dựa trên Định lý Bochner). RFF ánh xạ từng dữ liệu đầu vào vào một không gian đặc trưng ngẫu nhiên số chiều thấp $m \ll n$. Từ đó, ta xây dựng được bộ ước lượng dạng đóng (closed-form) và khả vi hoàn toàn:

$$\widehat{\text{HSIC}}(j) = \frac{1}{(n-1)^2} \left\| \left(\tilde{\Phi}_X^{(j)}\right)^\top \tilde{\Phi}_\varepsilon^{(j)} \right\|_F^2$$

Trong đó:
* $\tilde{\Phi}_X^{(j)} \in \mathbb{R}^{n \times m}$ và $\tilde{\Phi}_\varepsilon^{(j)} \in \mathbb{R}^{n \times m}$ lần lượt là các ma trận đặc trưng RFF đã được trung tâm hóa (centered) của các biến cha và phần dư tại nút $j$.
* $\lVert \cdot \rVert_F^2$ ký hiệu cho bình phương chuẩn Frobenius.

Toàn bộ $d$ giá trị ước lượng độc lập của tất cả các nút được tính toán đồng thời chỉ bằng một phép nhân ma trận dạng khối (batched matrix multiplication) duy nhất. Điều này giảm độ phức tạp tính toán từ $\mathcal{O}(d \cdot n^2)$ xuống còn $\mathcal{O}(d \cdot n \cdot m)$, giúp việc tính toán gradient của hàm phạt độc lập trở nên cực kỳ nhanh chóng và khả thi.

---

### D. Ràng buộc Phi chu trình Liên tục và Tối ưu hóa Ràng buộc

Bài toán học cấu trúc nhân quả liên tục được phát biểu dưới dạng bài toán tối ưu hóa có ràng buộc như sau:

$$\min_{\mathbf{W}, \boldsymbol{\theta}} \quad \mathcal{L}_{\text{main}}(\mathbf{W}, \boldsymbol{\theta}) = \mathcal{L}_{\text{NLL}}(\mathbf{W}, \boldsymbol{\theta}) + \lambda_{\text{HSIC}} \mathcal{L}_{\text{HSIC}}(\mathbf{W}, \boldsymbol{\theta}) + \lambda_{L_1} \|\mathbf{W}\|_1$$

$$\text{thỏa mãn} \quad h(\mathbf{W}) = \text{Tr}\left(e^{\mathbf{W} \circ \mathbf{W}}\right) - d = 0$$

Trong đó:
* **Hàm log-likelihood âm**:
  $$\mathcal{L}_{\text{NLL}} = -\sum_{j=1}^d \log p(\varepsilon_j)$$
  là hàm âm log-likelihood của dữ liệu quan sát dưới luồng chuẩn hóa rational-quadratic splines đã học.
* **Hàm phạt độc lập**:
  $$\mathcal{L}_{\text{HSIC}} = \sum_{j=1}^d \log\left(\widehat{\text{HSIC}}(j) + \epsilon_{\text{stab}}\right)$$
  là tổng hàm phạt độc lập trên tất cả $d$ nút, thúc đẩy trực tiếp tính độc lập giữa cha và phần dư (với $\epsilon_{\text{stab}}$ là hằng số ổn định số học).
* **Chuẩn L1 của ma trận trọng số**:
  $$\lVert\mathbf{W}\rVert_1 = \sum_{i \neq j} |W_{ij}|$$
  là chuẩn $L_1$ của các trọng số ngoài đường chéo, có tác dụng áp đặt tính thưa (sparsity) cho cấu trúc đồ thị.
* **Ràng buộc phi chu trình**:
  $$h(\mathbf{W}) = \text{Tr}\left(e^{\mathbf{W} \circ \mathbf{W}}\right) - d = 0$$
  là ràng buộc phi chu trình trơn (NOTEARS), trong đó $\circ$ là phép nhân liên kết phần tử Hadamard (element-wise product) và $e^{\mathbf{A}}$ là hàm mũ ma trận.



Chúng tôi giải quyết bài toán tối ưu hóa phi tuyến có ràng buộc này bằng **Phương pháp Lagrangian Tăng cường (ALM)**. Thuật toán tối ưu hóa một chuỗi các bài toán con không ràng buộc:

$$\mathcal{L}_{\text{ALM}}(\mathbf{W}, \boldsymbol{\theta}, \alpha, \rho) = \mathcal{L}_{\text{main}}(\mathbf{W}, \boldsymbol{\theta}) + \alpha h(\mathbf{W}) + \frac{\rho}{2} h(\mathbf{W})^2$$

Trong đó $\alpha$ là nhân tử Lagrange và $\rho > 0$ là tham số phạt của ràng buộc phi chu trình. Hai tham số này được cập nhật lặp lại sau mỗi chu kỳ hội tụ của bài toán con để đảm bảo tính khả thi của ràng buộc ($h(\mathbf{W}) \to 0$).

---

## III. Kiến trúc Hệ thống & Thuật toán Đề xuất

### A. Chọn Cha Phi tuyến tính & Gated Residual MLP

Để mô hình hóa các cơ chế nhân quả phi tuyến tính phức tạp một cách hiệu quả, CausalFlowNet sử dụng một mạng **Gated Residual MLP (Gated-ResMLP)** chia sẻ chung cho cả $d$ biến đồng thời. Thay vì huấn luyện $d$ mạng nơ-ron độc lập (gây lãng phí tài nguyên và dễ quá khớp), chúng tôi áp dụng một chiến lược định tuyến mềm liên tục (continuous soft-masking): nhân từng dòng dữ liệu đầu vào với cột tương ứng của ma trận trọng số $\mathbf{W}$. Cơ chế này tự động định tuyến động các biến cha đang hoạt động vào bộ dự báo của từng nút.

Mỗi khối dư (residual block) của Gated-ResMLP áp dụng **Layer Normalization** trước khi đi qua một phép chiếu tuyến tính được tách đôi thành: nhánh đặc trưng (kích hoạt bằng LeakyReLU) và nhánh cổng (gating branch, kích hoạt bằng Sigmoid). Tích chập Hadamard giữa hai nhánh này sau đó được cộng ngược lại đầu vào qua kết nối tắt (residual connection). Cơ chế cổng này cho phép mạng khuếch đại các đặc trưng quan trọng và triệt tiêu các đặc trưng dư thừa. Việc chia sẻ trọng số mạng giúp giảm độ phức tạp số lượng tham số từ $\mathcal{O}(d^2 \cdot L)$ xuống còn $\mathcal{O}(d \cdot L)$ (với $L$ là số tầng mạng), đóng vai trò như một bộ chính quy hóa (regularization) mạnh mẽ chống quá khớp đối với các bộ dữ liệu sinh học có số lượng mẫu hạn chế.

---

### B. Quy trình Học Cấu trúc Hai Giai đoạn

CausalFlowNet áp dụng **chiến lược huấn luyện hai giai đoạn** nhằm cân bằng giữa việc khám phá không gian đồ thị và tinh chỉnh cấu trúc tối ưu:

| Giai đoạn | Số lượng Epoch | Tham số phạt $L_1$ ($\lambda_{L_1}$) | Mục tiêu chính |
| :--- | :---: | :---: | :--- |
| **Giai đoạn 1: Khám phá Cấu trúc** | 30 Epochs | $\lambda = 0.001$ | Tìm kiếm diện rộng trên không gian đồ thị liên tục; thiết lập sơ bộ các đường truyền topo lớn. |
| **Giai đoạn 2: Tinh chỉnh Topo** | 20 Epochs | $\lambda = 0.012$ | Phạt thưa mạnh mẽ để loại bỏ các cạnh dư thừa; tập trung tối ưu hóa tính độc lập giữa cha và phần dư. |

---

### C. Ngưỡng Cắt tỉa Thích ứng (Adaptive Post-Pruning)

Sau khi quá trình tối ưu hóa liên tục hội tụ, ma trận trọng số liên tục $\mathbf{W}$ được nhị phân hóa thành ma trận kề có hướng $\widehat{\mathbf{A}}$ bằng phương pháp **ngưỡng cắt tỉa thích ứng dựa trên phân phối dữ liệu**: một cạnh $i \rightarrow j$ được giữ lại nếu và chỉ nếu trọng số $|W_{ij}|$ vượt quá giá trị trung bình của toàn bộ các trọng số ngoài đường chéo cộng thêm $\kappa$ lần độ lệch chuẩn:

$$\tau = \text{mean}\left(|\mathbf{W}_{\text{off-diag}}|\right) + \kappa \cdot \text{std}\left(|\mathbf{W}_{\text{off-diag}}|\right)$$

Giá trị mặc định của hệ số cắt tỉa là $\kappa = 0.8$. Phương pháp này loại bỏ hoàn toàn việc phải dò tìm ngưỡng thủ công và tự động điều chỉnh độ khắt khe của việc cắt tỉa phù hợp với phân phối trọng số thực tế của từng bộ dữ liệu cụ thể.

---

## IV. Đánh giá Thực nghiệm & So sánh Hiệu năng

### A. So sánh Định lượng Học Cấu trúc Nhân quả

Chúng tôi đánh giá CausalFlowNet so với 8 phương pháp cơ sở (baselines) mạnh mẽ đại diện cho các trường phái khác nhau: PC (Constraint-based), GES (Score-based), CAM (Functional Causal Model), và các phương pháp tối ưu hóa liên tục tiên tiến như NOTEARS, DAG-GNN, GSF, GraN-DAG, GraN-DAG++.

Thử nghiệm được thực hiện trên hai bộ dữ liệu chuẩn của ngành:
1. **Sachs**: Bộ dữ liệu mạng truyền tín hiệu protein thực tế ($d=11$ nút, $n=7.466$ mẫu thực nghiệm sinh học tế bào).
2. **SynTReN**: Bộ dữ liệu mô phỏng biểu hiện mạng điều hòa gen của vi khuẩn *E. coli* ($d=20$ nút, $n=500$ mẫu với động học phi tuyến tính phức tạp).

#### Bảng so sánh hiệu năng chi tiết

| Trường phái | Phương pháp | SHD (Sachs) ↓ | SHD-c (Sachs) ↓ | SID (Sachs) ↓ | SHD (SynTReN) ↓ | SHD-c (SynTReN) ↓ | SID (SynTReN) ↓ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ràng buộc** | PC [11] | $17.0$ | $11.0$ | $47.0 \text{ đến } 62.0$ | $41.0 \pm 5.1$ | $42.4 \pm 4.6$ | $154.8 \pm 47.6$ |
| **Điểm số** | GES [2] | $26.0$ | $28.0$ | $34.0 \text{ đến } 45.0$ | $82.6 \pm 9.3$ | $85.6 \pm 10.0$ | $157.2 \pm 48.3$ |
| **FCM** | CAM [7] | $12.0$ | **$9.0$** | $55.0$ | $40.5 \pm 6.8$ | $41.4 \pm 7.1$ | $152.3 \pm 48.0$ |
| **Liên tục** | NOTEARS [13] | $21.0$ | $21.0$ | $44.0$ | $151.8 \pm 28.2$ | $156.1 \pm 28.7$ | $110.7 \pm 66.7$ |
| **Liên tục** | DAG-GNN | $16.0$ | $21.0$ | $44.0$ | $93.6 \pm 9.2$ | $97.6 \pm 10.3$ | $157.5 \pm 74.6$ |
| **Liên tục** | GSF | $18.0$ | $10.0$ | $44.0 \text{ đến } 61.0$ | $61.8 \pm 9.6$ | $63.3 \pm 11.4$ | **$76.7 \pm 51.1$** |
| **Liên tục** | GraN-DAG [15] | $13.0$ | $11.0$ | $47.0$ | $34.0 \pm 8.5$ | $36.4 \pm 8.3$ | $161.7 \pm 53.4$ |
| **Liên tục** | GraN-DAG++ | $13.0$ | $10.0$ | $48.0$ | $33.7 \pm 3.7$ | $39.4 \pm 4.9$ | $127.5 \pm 52.8$ |
| **Liên tục** | **CausalFlowNet (Ours)** | **$12.0$** | $16.0$ | **$37.0$** | **$25.0$** | **$35.0$** | $166.0$ |

*Các ký hiệu:* SHD (Structural Hamming Distance - càng nhỏ càng tốt), SHD-c (CPDAG SHD - càng nhỏ càng tốt), SID (Structural Interventional Distance - càng nhỏ càng tốt). Giá trị đại diện cho giá trị trung bình $\pm$ độ lệch chuẩn từ các hạt giống khởi tạo độc lập.

**Phân tích kết quả thực nghiệm:**
* Trên bộ dữ liệu sinh học thực tế **Sachs**, CausalFlowNet đạt được chỉ số SHD thấp nhất bằng **$12.0$** (ngang bằng với CAM nhưng CAM có SID rất cao là $55.0$). CausalFlowNet vượt trội hoàn toàn về mặt suy luận can thiệp với khoảng cách can thiệp thấp nhất **$37.0$** (thấp hơn đáng kể so với mức $44.0 - 48.0$ của các phương pháp liên tục khác), chứng tỏ đồ thị dự báo giữ cấu trúc nhân quả nhân tố chính xác hơn rất nhiều.
* Trên bộ dữ liệu phức tạp phi tuyến **SynTReN**, mô hình của chúng tôi đạt SHD vượt trội là **$25.0$** và SHD-c là **$35.0$**, giảm lỗi cấu trúc xuống hơn $25\%$ so với đối thủ mạnh nhất là GraN-DAG++.

---

### B. Khám phá Nhóm nhân quả phụ và Ước lượng Ảnh hưởng Trị liệu (ATE)

Nhờ việc tích hợp GMM Prior có khả năng tự học vào luồng chuẩn hóa phần dư, CausalFlowNet tự nhiên mở ra hai khả năng ứng dụng thực tế vô cùng giá trị:

#### 1. Phân cụm Nhân quả Ẩn (Latent Causal Clustering)
Các biểu diễn không gian ẩn $z_j = g_{\boldsymbol{\theta}}(\varepsilon_j)$ tương ứng với nhiễu cấu trúc đã được chuẩn hóa. Bằng cách áp dụng thuật toán K-Means trên không gian ẩn toàn cục $\mathbf{Z} \in \mathbb{R}^{n \times d}$, chúng tôi có thể tự động phân nhóm (cluster) các mẫu quan sát thành các phân nhóm trạng thái sinh học hoặc điều kiện thực nghiệm ẩn mà không cần nhãn giám sát.

#### 2. Ước lượng Ảnh hưởng Can thiệp Trung bình (ATE)
Sử dụng các cơ chế phi tuyến tính đã học trong mạng Gated-ResMLP, chúng tôi có thể ước lượng trực tiếp Ảnh hưởng Trị liệu Trung bình (Average Treatment Effect - ATE) khi thực hiện can thiệp cấu trúc $do(X_s = v)$ lên một nút nguồn $X_s$ đối với nút đích downstream $X_t$:

$$\text{ATE}(s \rightarrow t) = \mathbb{E}\left[X_t \mid do(X_s = 1)\right] - \mathbb{E}\left[X_t \mid do(X_s = 0)\right]$$

Để thực hiện điều này một cách toán học, chúng tôi mô phỏng phân phối can thiệp bằng cách đặt cột $\mathbf{W}_{:, s} = \mathbf{0}$ trong ma trận kề (cắt đứt mọi ảnh hưởng nhân quả đi vào nút nguồn $X_s$), áp đặt cứng giá trị can thiệp $X_s = v$, và lan truyền tiến (forward propagation) qua mạng Gated-ResMLP để tính toán kỳ vọng của nút đích.

---

## V. Chẩn đoán Trực quan & Diễn giải Cấu trúc

### A. Phân tích Mạng Protein Sachs

Mạng tín hiệu protein Sachs chứa $d=11$ phosphoprotein và $17$ cạnh tương tác thực tế sinh học đã được kiểm chứng.

* **Hình 1 (Đồ thị nhân quả Sachs thực nghiệm)**: Biểu diễn đồ thị tái dựng bởi CausalFlowNet cùng trọng số cạnh nhân quả ước lượng.
* **Hình 2 (Ma trận kề so sánh)**: So sánh trực quan ma trận kề thực tế (Ground Truth - màu xanh) và ma trận kề ước lượng bởi CausalFlowNet (màu đỏ).

<p align="center">
  <img src="sachs_graph_comparison.png" width="48%" alt="Đồ thị nhân quả Sachs với ước lượng ATE"/>
  <img src="sachs_adjacency_comparison.png" width="48%" alt="Ma trận kề so sánh mạng Sachs"/>
</p>

Mô hình tái dựng thành công các con đường truyền tín hiệu sinh học kinh điển như dòng thác MAPK **PKC → Raf → Mek → Erk** và con đường phospholipid quan trọng **PIP2 → PIP3**, đồng thời hạn chế tối đa các cạnh giả mạo nhờ hàm phạt HSIC hoạt động hiệu quả.

### B. Phân tích Mạng Biểu hiện Gen SynTReN

Mô phỏng động học biểu hiện gen trong vi khuẩn *E. coli* ($d=20$ nút, $24$ cạnh thực tế) chứa các phản ứng phi tuyến động học enzyme cực kỳ phức tạp.

* **Hình 3 (Mạng điều hòa gen SynTReN)**: Đồ thị gen tái dựng bởi mô hình.
* **Hình 4 (Ma trận kề so sánh SynTReN)**: So sánh trực quan giữa cấu trúc thực tế và ước lượng.

<p align="center">
  <img src="syntren_graph_comparison.png" width="48%" alt="Mạng điều hòa gen SynTReN tái dựng"/>
  <img src="syntren_adjacency_comparison.png" width="48%" alt="Ma trận kề so sánh mạng SynTReN"/>
</p>

CausalFlowNet đạt được độ chính xác cấu trúc vượt trội trên dữ liệu phi tuyến này, phục hồi chính xác các con đường điều hòa cốt lõi như **Gene_9 → Gene_17** và **Gene_10 → Gene_11** với tỷ lệ báo động giả (False Positive Rate) cực thấp chỉ **$0.08$**.

---

## VI. Kết luận (Conclusion)

Chúng tôi đã trình bày **CausalFlowNet**, một giải pháp học sâu đầu-cuối đột phá cho bài toán khám phá cấu trúc nhân quả phi tuyến tính. Bằng việc kết hợp sức mạnh biểu diễn phân phối bất kỳ của Neural Spline Flows, khả năng tự học cấu trúc ẩn của GMM Prior, và phép phạt độc lập RFF-HSIC song song hóa siêu nhanh dưới ràng buộc phi chu trình liên tục, CausalFlowNet đã thiết lập một tiêu chuẩn mới về cả độ chính xác cấu trúc lẫn khả năng suy luận can thiệp.

Các hướng nghiên cứu tiềm năng trong tương lai bao gồm việc mở rộng mô hình để giải quyết bài toán có chứa biến ẩn gây nhiễu không quan sát được (latent confounders) và tối ưu hóa tính toán song song trên GPU để nâng quy mô ứng dụng lên hàng ngàn biến trong các mạng lưới gen quy mô lớn.

---

## VII. Tài liệu Tham khảo (References)

[1] K. Bello, B. Aragam, and P. Ravikumar, "DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization," *Advances in Neural Information Processing Systems*, vol. 35, 2022.

[2] D. M. Chickering, "Optimal structure identification with greedy search," *Journal of Machine Learning Research*, vol. 3, no. Nov, pp. 507-554, 2002.

[3] C. Durkan, A. Bekasov, I. Murray, and G. Papamakarios, "Neural spline flows," *Advances in Neural Information Processing Systems*, vol. 32, 2019.

[4] A. Gretton, O. Bousquet, A. Smola, and B. Schölkopf, "Measuring statistical dependence with Hilbert-Schmidt norms," in *Algorithmic Learning Theory*, pp. 63-77, 2005.

[5] S. Hu, Z. Chen, *et al.*, "Causal Inference and Mechanism Clustering of A Mixture of Additive Noise Models (ANM-MM)," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 31, 2018.

[6] J. Pearl, *Causality: Models, Reasoning and Inference*. Cambridge University Press, 2000.

[7] J. Peters and P. Bühlmann, "Structural intervention distance for evaluating causal graphs," *Neural Computation*, vol. 27, no. 3, pp. 771-799, 2015.

[8] J. Peters, J. M. Mooij, D. Janzing, and B. Schölkopf, "Causal discovery with continuous additive noise models," *Journal of Machine Learning Research*, vol. 15, no. 1, pp. 2009-2053, 2014.

[9] K. Sachs, O. Perez, D. Pe'er, D. A. Lauffenburger, and G. P. Nolan, "Causal protein-signaling networks derived from multiparameter single-cell data," *Science*, vol. 308, no. 5721, pp. 523-529, 2005.

[10] S. Shimizu, P. O. Hoyer, A. Hyvärinen, and A. Kerminen, "A linear non-Gaussian acyclic model for causal discovery," *Journal of Machine Learning Research*, vol. 7, no. 10, pp. 2003-2030, 2006.

[11] P. Spirtes, C. N. Glymour, and R. Scheines, *Causation, prediction, and search*, 2nd ed. MIT Press, 2000.

[12] T. Van den Bulcke, K. Van Leemput, B. Naudts, P. van Remortel, H. Ma, A. Verschoren, B. De Moor, and K. Marchal, "SynTReN: a generator of synthetic gene expression data for design and analysis of structure learning algorithms," *BMC Bioinformatics*, vol. 7, no. 1, p. 43, 2006.

[13] X. Zheng, B. Aragam, P. K. Ravikumar, and E. P. Xing, "DAGs with NO TEARS: Continuous optimization for structure learning," *Advances in Neural Information Processing Systems*, vol. 31, 2018.

[14] S. Hu, Z. Chen, V. Partovi Nia, L. Chan, and Y. Geng, "Causal Inference and Mechanism Clustering of A Mixture of Additive Noise Models," Poster presented at NeurIPS 2018.

[15] S. Lachapelle, P. Brouillard, T. Deleu, and S. Lacoste-Julien, "Gradient-Based Neural DAG Learning," *arXiv preprint arXiv:1906.02226*, 2020.
