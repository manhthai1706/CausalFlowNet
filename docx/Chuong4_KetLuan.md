# CHƯƠNG 4: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

Đề tài nghiên cứu đã tập trung vào bài toán khám phá cấu trúc nhân quả từ dữ liệu quan sát phi tuyến tính - một trong những thách thức cốt lõi của khoa học dữ liệu hiện đại. Thông qua việc đề xuất và triển khai mô hình **CausalFlowNet**, nghiên cứu này đã đạt được những mục tiêu đặt ra ban đầu và mở ra những hướng tiếp cận mới trong lĩnh vực tin sinh học và suy luận nhân quả.

## 4.1. Những kết quả đạt được

Đề tài đã hoàn thành các mục tiêu chính bao gồm:

1.  **Xây dựng thành công kiến trúc CausalFlowNet:** Mô hình đã kết hợp khéo léo giữa Normalizing Flows (cụ thể là Neural Spline Flows) với mạng Neural Gated-ResMLP để mô phỏng các cơ chế nhân quả phi tuyến phức tạp. Việc tích hợp ràng buộc độc lập HSIC vào hàm mục tiêu Likelihood đã giúp mô hình khắc phục nhược điểm của các phương pháp chỉ dựa trên điểm số truyền thống.
2.  **Lượng hóa hiệu ứng nhân quả:** Không chỉ dừng lại ở việc phát hiện các cạnh, mô hình đã triển khai thành công việc ước lượng Hiệu ứng can thiệp trung bình (ATE). Điều này giúp chuyển đổi các đồ thị nhân quả từ dạng cấu trúc thuần túy sang các mô hình định lực có khả năng dự báo tác động của các can thiệp thực tế.
3.  **Khám phá đa cơ chế ngầm định:** Qua cơ chế phân cụm dựa trên phần dư (residual clustering), nghiên cứu đã chỉ ra khả năng nhận diện các "causal contexts" khác nhau trong cùng một tập dữ liệu, một đặc điểm cực kỳ quan trọng đối với các dữ liệu sinh học có tính nhiễu và biến động cao như tập Sachs.

## 4.2. Đánh giá tổng quát từ thực nghiệm

Thông qua các thử nghiệm trên tập dữ liệu thực tế Sachs và mô phỏng SynTReN, mô hình đã thể hiện các ưu điểm thực tế:

-   **Hiệu năng cạnh tranh:** Kết quả cho thấy CausalFlowNet đạt được các chỉ số SID và SHD nằm trong nhóm dẫn đầu so với các thuật toán SOTA hiện nay như GraN-DAG, CAM hay GES. Đặc biệt, sai số can thiệp (SID) thấp trên tập Sachs khẳng định giá trị ứng dụng của mô hình trong việc xây dựng các giả thuyết sinh học.
-   **Tính ổn định và tính mở rộng:** Mô hình vận hành ổn định trên các mạng lưới có số lượng nút tăng dần (từ 11 lên 20 nút) mà không làm tăng đáng kể tỷ lệ dương tính giả (FPR). Khả năng tính toán song song HSIC cũng đảm bảo thời gian huấn luyện tối ưu trên các thiết bị phần cứng phổ thông.

## 4.3. Hạn chế và Hướng phát triển tương lai

Mặc dù đạt được những kết quả khả quan, đề tài vẫn còn tồn tại một số hạn chế nhất định cần được tiếp tục hoàn thiện trong tương lai:

### 4.3.1. Hạn chế
-   **Sự tương đương Markov:** Trong các trường hợp dữ liệu có tương quan cực mạnh, mô hình đôi khi vẫn gặp khó khăn trong việc phân biệt chính xác hướng nhân quả trong lớp tương đương Markov, dẫn đến chỉ số SHD-c còn dư địa để cải thiện.
-   **Độ nhạy siêu tham số:** Hiệu năng của mô hình phụ thuộc khá nhiều vào việc tinh chỉnh các trọng số L1 và HSIC, đòi hỏi quy trình tối ưu hóa tham số nhạy bén hơn cho từng miền dữ liệu cụ thể.

### 4.3.2. Hướng phát triển
-   **Mở rộng quy mô mạng lưới:** Áp dụng mô hình vào các mạng điều hòa gen quy mô lớn (hàng trăm đến hàng nghìn gen) thông qua các kỹ thuật học sâu cắt tỉa (pruning) và tối ưu hóa bộ nhớ.
-   **Kết hợp dữ liệu can thiệp:** Tích hợp thêm các dữ liệu từ các thí nghiệm can thiệp thực tế (perturbation data) cùng với dữ liệu quan sát để phá vỡ các lớp tương đương Markov một cách triệt để.
-   **Tự động hóa tìm kiếm siêu tham số:** Triển khai các thuật toán tối ưu hóa như Bayesian Optimization hoặc Genetic Algorithms để tự động tìm kiếm bộ tham số tối ưu cho từng ứng dụng cụ thể.

Tóm lại, CausalFlowNet đã chứng minh được tính đúng đắn và hiệu quả trong việc khám phá cấu trúc nhân quả với vai trò là một hướng tiếp cận triển vọng. Kết quả của đề tài không chỉ đóng góp về mặt thuật toán mà còn có tiềm năng hỗ trợ đắc lực trong việc phân tích các hệ thống phức tạp, đặc biệt là trong lĩnh vực phân tích mạng lưới sinh học và y tế.
