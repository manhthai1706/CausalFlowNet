# PHẦN MỞ ĐẦU

## 1. Lý do chọn đề tài

Trong bối cảnh bùng nổ của Khoa học dữ liệu (Data Science) và Học máy (Machine Learning), các mô hình Học sâu (Deep Learning) ngày một thể hiện năng lực cao trong việc giải quyết những bài toán phức tạp từ nhận dạng hình ảnh, xử lý ngôn ngữ tự nhiên cho đến dự báo xu hướng. Hầu hết sự thành công này được xây dựng trên năng lực nắm bắt sự **Tương quan (Correlation)** tốt của mô hình học sâu ẩn dưới các tập dữ liệu khổng lồ. 

Tuy nhiên, trong các bài toán đưa ra quyết định thực tiễn (Ví dụ: Y sinh, Kinh tế, Công nghiệp), sự tương quan thống kê không phải lúc nào cũng mang lại giá trị định hướng. Câu nói nổi tiếng "Correlation does not imply Causation" (Tương quan không có nghĩa là Nhân quả) nhấn mạnh rằng hai sự kiện có thể cùng xảy ra do một yếu tố ẩn (Confounder) chứ không nhất thiết có quan hệ trực tiếp. Việc chỉ dựa vào phân tích tương quan dẫn đến những quyết sách sai lầm, điều này là cốt lõi của **"Khám phá Nhân quả (Causal Discovery)"**.

Tìm kiếm được cấu trúc nguyên nhân - kết quả (Cause-Effect Topology) là chìa khóa để trả lời các câu hỏi về **Sự can thiệp (Intervention)** (*"Điều gì xảy ra nếu thay đổi A?"*) thay vì chỉ tập trung vào **Khả năng quan sát (Observation)**. Trong khi các thử nghiệm lâm sàng (RCT) thường tốn kém hoặc bất khả thi vì lý do đạo đức, khả năng học cấu trúc nhân quả từ dữ liệu quan sát (Observational Data) mở ra tiềm năng cực lớn. Thấu hiểu nhu cầu này, tôi quyết định thực hiện đề tài **"Khám phá cấu trúc nhân quả phi tuyến bằng mô hình Causal Flow Network (CausalFlowNet)"** nhằm kết hợp sức mạnh biểu diễn của Normalizing Flows và sự linh hoạt của Mạng Neural để khai phá quan hệ nhân quả với độ chính xác và tính giải thích cao.

## 2. Tổng quan lịch sử nghiên cứu của đề tài

### a) Tại thế giới
Bài toán khám phá nhân quả xuất hiện từ sớm với các thuật toán dựa trên ràng buộc (Constraint-based) như PC Algorithm (Spirtes et al., 2000) hoặc tính điểm số (Score-based) như GES (Chickering, 2002). Một hướng đi khác là Functional Causal Models (FCM) như LiNGAM (Shimizu, 2006), ANM (Peters et al., 2014) lợi dụng tính bất đối xứng của nhiễu để tìm chiều tác động.
  * Sự bùng nổ đến từ **Causal Continuous Optimization** với công trình **NOTEARS** (Zheng et al., 2018), chuyển đổi bài toán tìm đồ thị DAG từ tổ hợp sang tối ưu hóa liên tục. Sau đó, **DAGMA** (2022) và các mô hình dựa trên mạng neural như **DECI/Causica** đã đẩy mạnh khả năng xử lý dữ liệu phi tuyến.
  * Xu hướng mới nhất tập trung vào việc xử lý các phân phối phức tạp, không chỉ dừng lại ở nhiễu cộng (Additive Noise), thông qua các kỹ thuật như **Normalizing Flows** và biến đổi biến tiềm ẩn (Latent Variable Models).

### b) Tại Việt Nam và cơ sở giáo dục
Tại Việt Nam, các nghiên cứu về AI hiện tại phần lớn vẫn tập trung vào Computer Vision và NLP. Mảng **Causal AI** còn khá mới mẻ, chủ yếu dừng lại ở mức độ ứng dụng các thư viện sẵn có hoặc dùng mạng Bayes truyền thống trong Thống kê y tế. Việc tự xây dựng và tối ưu một kiến trúc mạng Neural chuyên biệt, tích hợp các ràng buộc toán học khắt khe để học cấu trúc nhân quả như CausalFlowNet là một hướng đi mang tính đón đầu tại cấp độ học thuật.

### c) Tính mới và sự nổi bật của đề tài (CausalFlowNet)
CausalFlowNet khắc phục những hạn chế của các mô hình truyền thống bằng cách tích hợp các công nghệ tiên tiến:
- **Normalizing Flows**: Sử dụng cơ chế biến đổi mật độ (Density Estimation) để mô hình hóa những phân phối dữ liệu phức tạp, phi tuyến và không nhất thiết phải là Gauss.
- **Self-Attention & Gumbel-Softmax**: Tích hợp cơ chế Attention để nắm bắt phụ thuộc đặc trưng và Gumbel-Softmax để mô hình hóa các cơ chế đa dạng (Mixed Mechanisms).
- **Quy trình tối ưu hóa hai giai đoạn**: Phân tách quá trình huấn luyện thành giai đoạn **Khám phá (Discovery)** (với trọng số L1 mạnh) và giai đoạn **Tinh chỉnh cấu trúc (Refinement)** (sử dụng Augmented Lagrangian) để giảm thiểu tối đa tỷ lệ cạnh giả (False Positives).
- **Hệ thống đánh giá đa chiều**: Không chỉ dùng SHD, đề tài sử dụng thêm **SID (Structural Intervention Distance)** để đánh giá hiệu quả của mô hình dưới góc độ can thiệp nhân quả thực tế.

## 3. Mục tiêu đồ án tốt nghiệp

Đề tài hướng tới việc thực hiện các mục tiêu cốt lõi sau:
1. Hệ thống hóa lý thuyết về học cấu trúc nhân quả (Causal Structure Learning), mô hình phương trình cấu trúc (SEM) và các ràng buộc phi chu trình liên tục (Acyclicity Constraints).
2. Xây dựng mô hình **CausalFlowNet** có khả năng học quan hệ nhân quả trên dữ liệu đa biến phi tuyến và phân phối phức tạp.
3. Triển khai thuật toán tối ưu hóa **Augmented Lagrangian (ALM)** để đảm bảo tính hợp lệ (DAG) của đồ thị nhân quả tìm được.
4. Đạt được kết quả vượt trội về độ chính xác (TPR), độ tin cậy (SID/SHD) trên các bộ dữ liệu Benchmark tiêu chuẩn (Sachs, SynTReN-20).

## 4. Đối tượng và phạm vi nghiên cứu

**Đối tượng nghiên cứu:**
- Các mô hình học sâu nhân quả (Deep Causal Discovery).
- Kỹ thuật Normalizing Flows để ước lượng mật độ xác suất.
- Thuật toán tối ưu hóa trên không gian đồ thị liên tục.

**Phạm vi nghiên cứu:**
- **Về lý thuyết:** Tập trung vào mạng Perceptron (MLP) với tích hợp Self-Attention, cơ chế Gumbel-Softmax, và toán tử HSIC (Hilbert-Schmidt Independence Criterion).
- **Về dữ liệu:** Xử lý dữ liệu dạng bảng (Tabular Data) quan sát đa biến liên tục. Thử nghiệm trên hai bộ dữ liệu Benchmark gồm dữ liệu Y sinh học thực tế (Sachs - 11 nodes) và dữ liệu mạng gen giả lập (SynTReN-20 - 20 nodes).
- **Giới hạn:** Đề tài làm việc trên dữ liệu quan sát tĩnh, chưa xét đến dữ liệu chuỗi thời gian hay các biến ẩn không quan sát được (Unobserved Confounders).

## 5. Phương pháp nghiên cứu

- **Nghiên cứu tài liệu:** Phân tích các công nghệ từ NOTEARS, DAGMA đến các kiến trúc Normalizing Flows mới nhất, đồng thời nghiên cứu lý thuyết về toán tử HSIC (Hilbert-Schmidt Independence Criterion) phục vụ đo lường độc lập thống kê.
- **Xây dựng kiến trúc (Architectural Design):** Thiết kế Pipeline tích hợp từ tiền xử lý, định tuyến dòng (Flow), kiểm định độc lập HSIC đến lớp ràng buộc Augmented Lagrangian.
- **Lập trình thực nghiệm:** Sử dụng Python, PyTorch để huấn luyện mô hình với sự tối ưu từ GPU. Đóng gói mã nguồn theo chuẩn module hóa, dễ dàng mở rộng.
- **Đánh giá và Đối chiếu:** So sánh trực quan (Heatmaps, Causal Graphs) và định lượng (TPR, SHD, SID) giữa kết quả dự đoán của mô hình và Ground-truth thực tế.

## 6. Đóng góp mới của đề tài và những vấn đề chưa thực hiện được

### 6.1 Những đóng góp thiết thực
1. **Lớp Khai phá dựa trên Dòng chảy (Flow-based Discovery)**: Đề xuất cách tiếp cận dùng Normalizing Flows để xử lý dữ liệu có độ lệch lớn và nhiễu phi chuẩn, giúp mô hình hóa phân phối chính xác hơn các phương pháp MSE truyền thống.
2. **Cơ chế Ngưỡng thích nghi (Adaptive Thresholding)**: Tự động tính toán ngưỡng loại bỏ cạnh dựa trên phân phối trọng số thực tế thay vì một hằng số cứng, giúp tăng TPR và hạ SHD.
3. **Trực quan hóa cao cấp**: Xây dựng bộ công cụ `visualize.py` hỗ trợ so sánh song song đồ thị nhân quả thực tế (Ground-truth) và đồ thị dự đoán, kết hợp hiển thị ma trận trọng số (Heatmap) để kiểm tra chéo định tính các cạnh đúng/sai (TP/FP).

### 6.2 Những vấn đề chưa thực hiện được 
1. Chi phí tính toán tăng nhanh khi số lượng biến (nodes) tăng lên hàng trăm do độ phức tạp của ma trận trọng số và các vòng lặp tối ưu hóa Augmented Lagrangian.
2. Độ nhạy của mô hình vẫn phụ thuộc vào việc điều chỉnh các siêu tham số (Hyperparameters) như `l1_reg` trong từng giai đoạn huấn luyện, đòi hỏi kinh nghiệm tinh chỉnh thực tế.
3. Mô hình chưa xử lý được trường hợp dữ liệu chứa các biến ngoại lai ẩn không quan sát được (Latent Confounders), dẫn đến khả năng nhầm lẫn cạnh giả khi có yếu tố gây nhiễu chung.

## 7. Kết cấu của đề tài

Với nội dung và mục tiêu trên, báo cáo được tổ chức thành Phần Mở Đầu và 4 chương chính:
- **Phần mở đầu:** Lý do chọn đề tài, mục tiêu, đối tượng, phương pháp và khung tóm tắt đề tài. 
- **Chương 1 – Cơ sở lý thuyết:** Trình bày nền tảng cốt lõi về đồ thị nhân quả DAG, mô hình phương trình cấu trúc (SEM), lý thuyết về Normalizing Flows và hàm phạt tối ưu phi chu trình liên tục. 
- **Chương 2 – Mô hình CausalFlowNet:** Trình bày chi tiết về kiến trúc hệ thống, Pipeline xử lý, kiến trúc MLP-Attention, cơ chế Gumbel-Softmax và quy trình tối ưu hóa hai giai đoạn. 
- **Chương 3 – Thử nghiệm và Đánh giá (Experimental Results):** Giới thiệu các bộ dữ liệu được dùng để benchmark (Sachs, SynTReN-20), so sánh các chỉ số SHD, SID, TPR và trình bày bộ công cụ trực quan hóa. 
- **Chương 4 – Kết luận:** Đúc kết lại những đóng góp của đề tài CausalFlowNet so với các phương pháp truyền thống và đề xuất hướng mở rộng. 
