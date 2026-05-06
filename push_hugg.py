
from huggingface_hub import HfApi

# 1. Điền token MỚI của bạn vào đây (Nhớ giữ bí mật, không chia sẻ lên mạng)
TOKEN = "YOUR_HF_TOKEN_HERE"

# 2. Định nghĩa thông tin repo
REPO_ID = "Thang1703/msc-segmentation" # Sửa lại nếu tên repo của bạn khác
FILE_PATH = "resunet50_aspp_best.weights.h5"

# Khởi tạo API và truyền trực tiếp token vào (bỏ qua bước login)
api = HfApi(token=TOKEN)

print("Đang tải model lên Hugging Face... Vui lòng đợi (có thể mất vài phút tùy mạng).")

# 3. Tải file lên
api.upload_file(
    path_or_fileobj=FILE_PATH,
    path_in_repo=FILE_PATH,
    repo_id=REPO_ID,
    repo_type="model",
)

print("Tải lên thành công! Bạn đã có thể dùng model này trên Streamlit.")
