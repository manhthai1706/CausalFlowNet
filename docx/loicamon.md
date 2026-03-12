LỜI CẢM ƠN

Lời đầu tiên, chúng em xin gửi lời chào trân trọng và lời chúc sức khỏe đến quý Thầy Cô cùng toàn thể quý độc giả. Trong quá trình học tập tại trường Đại học Lạc Hồng, chúng em đã thực hiện dự án tốt nghiệp thuộc lĩnh vực Trí tuệ nhân tạo với đề tài: **"Nghiên cứu và Xây dựng kiến trúc mô hình để khám phá cấu trúc nhân quả phi tuyến tính từ dữ liệu"**. Đây là kết quả của sự nỗ lực nghiên cứu, tích lũy kiến thức và trải nghiệm thực tiễn của chúng em trong suốt thời gian qua.

Mục tiêu cốt lõi của đề tài là thiết kế và triển khai mô hình **CausalFlowNet** — một hệ thống học sâu tiên tiến nhằm tự động trích xuất các đồ thị có hướng không chu trình (DAG) từ dữ liệu quan sát. CausalFlowNet hướng tới việc giải quyết các mối quan hệ phi tuyến tính phức tạp và nhiễu không định dạng bằng cách kết hợp sức mạnh của **các mô hình xác suất dựa trên dòng chảy chuẩn hóa (Normalizing Flows)** và các ràng buộc toán học khắt khe về tính nhân quả.

Đồ án được hoàn thiện dựa trên các trụ cột công nghệ cốt lõi:
- **Neural Spline Flows & GMM Prior:** Sử dụng dòng chảy chuẩn hóa với các hàm bậc hai hữu tỷ và phân phối ưu tiên hỗn hợp Gaussian để ước lượng mật độ nhiễu một cách linh hoạt và chính xác.
- **Gated Residual MLP:** Kiến trúc mạng nơ-ron học các phương trình cấu trúc (SEM) với cơ chế cổng (Gating) giúp lọc thông tin và xử lý các tương tác phi tuyến phức tạp.
- **Toán tử HSIC với Đặc trưng Fourier Ngẫu nhiên:** Kiểm định tính độc lập thống kê siêu nhanh để xác nhận chiều nhân quả giữa các biến.
- **Tối ưu hóa Augmented Lagrangian:** Đảm bảo đồ thị học được hội tụ về một cấu trúc DAG hợp lệ thông qua các ràng buộc liên tục.

Trong báo cáo này, quý Thầy Cô sẽ tìm thấy chi tiết về thực nghiệm định lượng trên các tập dữ liệu sinh học chuẩn như Sachs và SynTReN. Chúng em hy vọng nghiên cứu này sẽ đóng góp một hướng tiếp cận hiệu quả và ổn định cho bài toán khám phá nhân quả, hỗ trợ việc đưa ra các quyết định dựa trên bản chất nhân quả thực sự của dữ liệu.

Chúng em xin gửi lời cảm ơn sâu sắc nhất đến giảng viên hướng dẫn là **Thầy Trần Thanh Phương** vì sự dẫn dắt tận tình, những định hướng chuyên môn quý báu và sự khích lệ của Thầy trong suốt quá trình phát triển đồ án. Sự đồng hành của Thầy là yếu tố then chốt giúp chúng em hoàn thiện đề tài này.

Sau cùng, chúng em xin kính chúc quý Thầy Cô trong khoa Công nghệ Thông tin thật dồi dào sức khỏe và hạnh phúc để tiếp tục thực hiện sứ mệnh cao cả là truyền đạt tri thức và niềm đam mê khoa học cho thế hệ mai sau.

Chúng em xin chân thành cảm ơn!
