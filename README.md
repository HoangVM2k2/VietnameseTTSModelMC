# TTS Model Tiếng Việt

## Giới Thiệu
Dự án này triển khai một mô hình TTS (Text-to-Speech) cho tiếng Việt, cho phép chuyển đổi văn bản thành âm thanh. Mô hình được xây dựng trên nền tảng PyTorch và sử dụng kiến trúc Glow-TTS.

## Cài Đặt

### Yêu Cầu
- Python 3.x
- Các thư viện Python: PyTorch, onnxruntime, piper-phonemize

```bash
pip install torch
pip install onnxruntime
pip install piper-phonemize
Tải Mô Hình
Tải mô hình Glow-TTS đã được fine-tuned cho tiếng Việt từ đường dẫn và giải nén vào thư mục models.

Chạy Dự Án
Chạy script text_to_speech.py để chuyển đổi văn bản thành âm thanh.

bash
Copy code
python text_to_speech.py --text "Xin chào, đây là ví dụ về âm thanh được tạo bởi mô hình TTS tiếng Việt." --speed "normal"
Cấu Trúc Dự Án
text_to_speech.py
Script chính để chuyển đổi văn bản thành âm thanh sử dụng mô hình đã được fine-tuned.

models/
Thư mục chứa các mô hình đã được fine-tuned.

data/
Thư mục chứa dữ liệu đầu vào hoặc dữ liệu mẫu nếu có.


less
Copy code

