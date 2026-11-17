import sys
import pathlib

# Thêm thư mục gốc CoDeX-Net/ vào sys.path để import được model/
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torchvision.transforms as transforms
from PIL import Image

from model.codex_net import CoDeX_Net

# Thiết bị sử dụng
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Đang sử dụng:", device)

# Đường dẫn trọng số và ảnh cần dự đoán
weights_path = 'F:\THANHDAT\\final_model_epoch_53_acc_0.9020.pt'
sample_path = pathlib.Path('samples')  # Có thể đổi thành đường dẫn 1 ảnh cụ thể (vd: 'samples/test.png')

# Danh sách 12 lớp (thay bằng tên 12 dạng sóng thực tế theo đúng thứ tự train_dataset.class_to_idx)
class_names = [
    'Class_0', 'Class_1', 'Class_2', 'Class_3',
    'Class_4', 'Class_5', 'Class_6', 'Class_7',
    'Class_8', 'Class_9', 'Class_10', 'Class_11'
]

# Transform giống hệt val_transform lúc train
val_transform = transforms.Compose([
    #transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Khởi tạo model và nạp trọng số
channels = (8, 8, 16, 16, 32, 32)
model = CoDeX_Net(channels_per_stage=channels, num_classes=len(class_names), drop_path_rate=0.0)

checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

# Hàm suy diễn 1 ảnh
def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    tensor = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, predicted = probs.max(1)

    pred_class = class_names[predicted.item()]
    return pred_class, conf.item()

if __name__ == "__main__":
    if sample_path.is_file():
        pred_class, conf = predict(sample_path)
        print(f"{sample_path.name} -> {pred_class} ({conf * 100:.2f}%)")
    elif sample_path.is_dir():
        for img_file in sorted(sample_path.iterdir()):
            if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                pred_class, conf = predict(img_file)
                print(f"{img_file.name} -> {pred_class} ({conf * 100:.2f}%)")
