# CHƯƠNG 4: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

Đề tài nghiên cứu đã tập trung vào bài toán khám phá cấu trúc nhân quả từ dữ liệu quan sát phi tuyến tính - một trong những thách thức quan trọng của khoa học dữ liệu hiện nay. Thông qua việc đề xuất và triển khai mô hình **CausalFlowNet**, nghiên cứu này đã đạt được những kết quả bước đầu và đề xuất một hướng tiếp cận khả thi trong lĩnh vực tin sinh học và suy luận nhân quả.

## 4.1. Những kết quả đạt được

Đề tài đã tập trung hoàn thành các nội dung chính bao gồm:

1.  **Xây dựng mô hình CausalFlowNet:** Mô hình kết hợp giữa Normalizing Flows (Neural Spline Flows) với mạng Neural Gated-ResMLP nhằm mô phỏng các cơ chế nhân quả phi tuyến. Việc tích hợp ràng buộc độc lập HSIC vào hàm mục tiêu Likelihood đã góp phần hạn chế một số nhược điểm của các phương pháp chỉ dựa trên điểm số truyền thống.
2.  **Bước đầu lượng hóa hiệu ứng nhân quả:** Bên cạnh việc phát hiện các cạnh, mô hình đã thực hiện ước lượng Hiệu ứng can thiệp trung bình (ATE). Điều này giúp chuyển đổi các đồ thị nhân quả từ dạng cấu trúc sang các mô hình định lượng, bước đầu cho thấy khả năng dự báo tác động của các can thiệp thực tế.
3.  **Thử nghiệm khám phá đa cơ chế ngầm định:** Qua cơ chế phân cụm dựa trên phần dư (residual clustering), nghiên cứu đã cho thấy khả năng nhận diện các ngữ cảnh nhân quả khác nhau trong cùng một tập dữ liệu, đây có thể xem là một đặc điểm cần thiết đối với các dữ liệu sinh học có tính nhiễu như tập Sachs.

## 4.2. Đánh giá tổng quát từ thực nghiệm

Thông qua các thử nghiệm trên tập dữ liệu thực tế Sachs và mô phỏng SynTReN, mô hình đã đạt được một số kết quả khả quan:

-   **Hiệu năng thực nghiệm:** Các chỉ số SID và SHD của CausalFlowNet có tính cạnh tranh khi so sánh với một số thuật toán hiện nay như GraN-DAG, CAM hay GES. Đặc biệt, sai số can thiệp (SID) trên tập Sachs thể hiện tính khả thi của mô hình trong việc hỗ trợ xây dựng các giả thuyết sinh học.
-   **Tính ổn định bước đầu:** Mô hình duy trì được sự ổn định tương đối trên các mạng lưới khi số lượng nút tăng dần (từ 11 lên 20 nút) mà không làm tăng quá mức tỷ lệ dương tính giả (FPR). Khả năng tính toán song song HSIC cũng góp phần tối ưu thời gian huấn luyện trên các thiết bị phần cứng phổ thông.

## 4.3. Hạn chế và Hướng phát triển tương lai

Mặc dù đạt được những kết quả nhất định, đề tài vẫn còn tồn tại các hạn chế cần được tiếp tục hoàn thiện trong tương lai:

### 4.3.1. Hạn chế
-   **Sự tương đương Markov:** Trong các trường hợp dữ liệu có tương quan mạnh, mô hình vẫn gặp khó khăn trong việc phân biệt chính xác hướng nhân quả trong lớp tương đương Markov, dẫn đến chỉ số SHD-c còn dư địa để cải thiện.
-   **Sự phụ thuộc vào siêu tham số:** Hiệu năng của mô hình còn phụ thuộc vào việc tinh chỉnh các trọng số L1 và HSIC, đòi hỏi quy trình tối ưu hóa tham số phù hợp hơn cho từng loại dữ liệu cụ thể.

### 4.3.2. Hướng phát triển
-   **Thử nghiệm với mạng lưới quy mô lớn hơn:** Áp dụng mô hình vào các mạng điều hòa gen quy mô lớn hơn thông qua các kỹ thuật tối ưu hóa bộ nhớ và cấu trúc mạng.
-   **Tích hợp thêm các nguồn dữ liệu bổ trợ:** Tìm cách kết hợp dữ liệu từ các thí nghiệm can thiệp thực tế cùng với dữ liệu quan sát để cải thiện khả năng xác định hướng cạnh.
-   **Nghiên cứu tự động hóa tìm kiếm siêu tham số:** Tìm hiểu các phương pháp tối ưu hóa để tự động xác định bộ tham số phù hợp cho từng ứng dụng cụ thể.

Tóm lại, nghiên cứu về CausalFlowNet đã cho thấy một hướng tiếp cận có tiềm năng trong việc khám phá cấu trúc nhân quả. Kết quả của đề tài có thể đóng góp phần nào vào việc phân tích các hệ thống phức tạp, đặc biệt là trong lĩnh vực phân tích mạng lưới sinh học và y tế.
