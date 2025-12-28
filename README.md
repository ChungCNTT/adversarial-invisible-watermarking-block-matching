# adversarial-invisible-watermarking-block-matching
"Phương pháp nhúng thủy vân đối kháng vô hình dựa trên thuật toán khớp khối"
# Nhúng Thủy Vân Đối Kháng Vô Hình Dựa Trên Thuật Toán Khớp Khối

<p align="center">
  <img src="https://via.placeholder.com/1280x640/2F4F4F/FFFFFF?text=Nhúng+Thủy+Vân+Đối+Kháng+Vô+Hình" alt="Banner Dự Án" width="100%"/>
</p>

**Nhúng Thủy Vân Đối Kháng Vô Hình Dựa Trên Thuật Toán Khớp Khối** là dự án nghiên cứu chuyên sâu nhằm phát triển kỹ thuật bảo mật thông tin số trong lĩnh vực xử lý ảnh. Phương pháp kết hợp thủy vân số (digital watermarking) với tấn công đối kháng (adversarial attacks), cho phép nhúng thông tin ẩn vào ảnh mà không làm giảm chất lượng hình ảnh đáng kể, đồng thời thêm nhiễu tinh vi để đánh lừa các mô hình mạng nơ-ron sâu (Deep Neural Networks - DNN).

Dự án được thực hiện trong khuôn khổ học phần **Xử lý Ảnh** tại Trường Đại học Thủy Lợi, do sinh viên **Đào Việt Chung** (Lớp 64CNTT2) thực hiện dưới sự hướng dẫn tận tình của **TS. Đinh Phú Hùng**.

## 📄 Tổng quan dự án

Trong bối cảnh số hóa thông tin phát triển mạnh mẽ, việc bảo vệ quyền sở hữu trí tuệ đối với ảnh số trở nên cấp bách do tính dễ sao chép và chỉnh sửa. Dự án đề xuất giải pháp nhúng thủy vân đối kháng vô hình, kết hợp lý thuyết mật mã học, xử lý tín hiệu số và học sâu. Thủy vân không chỉ xác minh tính xác thực và quyền sở hữu mà còn tích hợp nhiễu đối kháng để gây phân loại sai trong các mô hình DNN, từ đó tăng cường bảo mật.

Các mục tiêu cụ thể:
- Phát triển thuật toán nhúng thủy vân vô hình với độ bền cao trước các tấn công thông thường (nén JPEG, thêm nhiễu Gaussian, cắt xén, xoay).
- Tích hợp tấn công đối kháng để đánh lừa DNN mà không ảnh hưởng đến chất lượng hình ảnh.
- Triển khai chương trình thực nghiệm với giao diện đồ họa thân thiện, dễ dàng kiểm chứng kết quả.
- Đánh giá hiệu suất qua các chỉ số tiêu chuẩn và phân tích ưu nhược điểm.

Cấu trúc báo cáo chi tiết:
- Chương 1: Tổng quan về tấn công đối kháng và thủy vân số (khái niệm, phân loại, ví dụ thực tế).
- Chương 2: Thuật toán nhúng thủy vân đối kháng vô hình (cơ sở lý thuyết khớp khối, kỹ thuật nhúng/trích xuất với SimBA).
- Chương 3: Ví dụ minh họa (các case study cụ thể).
- Chương 4: Xây dựng chương trình thử nghiệm (mô tả quy trình và giao diện phần mềm).
- Chương 5: Đánh giá kết quả (phân tích chỉ số, ưu nhược điểm).
- Chương 6: Kết luận và hướng phát triển.

## ✨ Tính năng chính

- **Nhúng thủy vân vô hình**: Quy trình bao gồm phóng to ảnh gốc bằng nội suy song tuyến, chia khối 4x4/8x8 pixel, sử dụng độ tương đồng cosin dựa trên độ lệch chuẩn để ánh xạ khối thủy vân, và điều chỉnh dịch chuyển trung bình (Δ) để đảm bảo tính vô hình.
- **Tích hợp nhiễu đối kháng**: Áp dụng SimBA (hộp đen) với biên độ nhiễu ϵ=0.001, lặp tối đa 10 lần, nhằm giảm độ tin cậy dự đoán của DNN.
- **Trích xuất thủy vân**: Blind extraction sử dụng ánh xạ khối đã lưu để tái tạo thủy vân mà không cần ảnh gốc.
- **Đánh giá chỉ số**: Tích hợp tính toán PSNR (>60 dB), SSIM (≈0.95), Pearson (>0.95 mục tiêu), ASR (98% mô phỏng).
- **Giao diện Tkinter**: 4 tab chuyên biệt (Nhúng, Trích xuất, Demo ma trận 4x4, Chỉ số đánh giá).
- **Hỗ trợ ảnh màu (RGB)**: Xử lý ảnh màu với chuẩn hóa RGB; hỗ trợ ảnh xám dự kiến trong phiên bản nâng cấp.

## 🛠 Công nghệ sử dụng

| Công nghệ          | Mô tả chi tiết                                                                 |
|---------------------|-------------------------------------------------------------------------------|
| **Python**          | Ngôn ngữ chính (3.8+), mã modular với hàm riêng biệt cho từng bước.            |
| **NumPy**           | Tính toán ma trận pixel, độ lệch chuẩn, tương đồng cosin.                     |
| **Pillow (PIL)**    | Tải, phóng to, chuyển RGB và lưu ảnh (PNG/JPG).                               |
| **Tkinter**         | Xây dựng GUI với Notebook, Canvas hiển thị ảnh và chỉ số thời gian thực.     |
| **Cơ sở toán học**  | Nội suy song tuyến, tương đồng cosin, nhiễu SimBA.                           |

## 📊 Thông tin dự án

| Thông tin                  | Chi tiết                                                                 |
|----------------------------|--------------------------------------------------------------------------|
| **Tên đề tài**             | Phương pháp nhúng thủy vân đối kháng vô hình dựa trên thuật toán khớp khối |
| **Sinh viên**              | Đào Việt Chung (Lớp 64CNTT2)                                             |
| **Giảng viên hướng dẫn**   | TS. Đinh Phú Hùng                                                        |
| **Học phần**               | Xử lý Ảnh                                                               |
| **Đơn vị**                 | Khoa Công nghệ Thông tin, Trường Đại học Thủy Lợi                        |
| **Năm**                    | 2025                                                                     |

## 🚀 Hướng dẫn sử dụng

### Yêu cầu hệ thống
- Python 3.8+
- Thư viện: numpy, pillow
  ```bash
  pip install numpy pillow
Cài đặt và chạy

Clone repository:Bashgit clone https://github.com/ChungCNTT/adversarial-invisible-watermarking-block-matching.git
cd adversarial-invisible-watermarking-block-matching
Chạy script chính:Bashpython demochinhthuyvanhocsau.py
Sử dụng giao diện:
Tab Nhúng: Tải ảnh gốc + thủy vân → Nhấn "Nhúng" → Lưu anh_nhung.png.
Tab Trích xuất: Tải ảnh nhúng → Nhấn "Trích xuất" → Lưu anh_nuoc_khoi_phuc.png.
Tab Demo Ma trận: Minh họa trên ma trận 4x4.
Tab Chỉ số: Xem PSNR, SSIM, Pearson, ASR.


Kích thước khuyến nghị: Ảnh gốc 256×256 hoặc 512×512; thủy vân 32×32 hoặc 128×128.
📈 Kết quả thực nghiệm

PSNR: ≈60 dB (tính vô hình cao).
SSIM: 0.9529 (bảo toàn cấu trúc tốt).
Pearson: 0.5882 (cần cải tiến).
ASR: 98% (mô phỏng, hiệu quả cao với VGG19, SqueezeNet).

Hình ảnh minh họa (sẽ cập nhật ảnh thực tế):

  Ảnh Nhúng
  Thủy Vân Tái Tạo
  Giao Diện

Ưu điểm

Tính vô hình và độ bền cao.
Triển khai đơn giản, dễ mở rộng.
Giao diện trực quan, hỗ trợ minh họa hiệu quả.

Hạn chế

Tái tạo kém do thiếu thông tin phụ.
Chưa tích hợp DNN thực tế.
Chỉ miền không gian; chưa hỗ trợ ảnh xám.

Hướng phát triển

Tích hợp DNN để tính ASR thực tế.
Nhúng trong miền tần số (DCT/DWT).
Lưu thông tin phụ để tăng Pearson.
Hỗ trợ ảnh xám và video/audio.
Thêm Matplotlib cho biểu đồ và CSV.

📚 Tài liệu

Báo cáo: BTL_BaocaoXLA_Đào Việt Chung_64CNTT2_12.docx
Thuyết trình: PPT_XLA_DAOVIETCHUNG.pptx
Ảnh mẫu: Lena2.png, Pepper.png, baboon1.png, Logo-Thuy_Loi.png, logo128s128.png
Mã nguồn: demochinhthuyvanhocsau.py, test2.py

📜 Giấy phép
Dự án phục vụ mục đích học tập và nghiên cứu, không sử dụng thương mại.
🙏 Lời tri ân
Xin chân thành cảm ơn TS. Đinh Phú Hùng đã hướng dẫn tận tình. Cảm ơn gia đình, bạn bè và các thầy cô Khoa Công nghệ Thông tin – Trường Đại học Thủy Lợi đã hỗ trợ.
Dự án góp phần nhỏ vào bảo mật thông tin số trong kỷ nguyên AI.
text
