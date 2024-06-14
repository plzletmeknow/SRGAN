import os
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from model import Generator, Discriminator  # 원래 코드에서 가져온 모델
import torch.nn as nn
import torch.optim as optim

# 데이터셋 경로 설정
root_dir = "D:\\바탕 화면\\KJM\\대학원\\딥러닝\\24-1\\기말발표\\final_presentation\\SRGAN\\TB_Chest_Radiography_Database\\Normal"
image_paths = [
    os.path.join(root, file)
    for root, _, files in os.walk(root_dir)
    for file in files
    if file.endswith((".png", ".jpg", ".jpeg"))
]

# 데이터 섞기
random.shuffle(image_paths)

# 학습용, 검증용, 테스트용으로 나누기 (60% 학습용, 20% 검증용, 20% 테스트용)
train_paths, test_paths = train_test_split(image_paths, test_size=0.2, random_state=42)
train_paths, val_paths = train_test_split(
    train_paths, test_size=0.25, random_state=42
)  # 전체의 20%를 검증용으로 분할


# 데이터셋 클래스
class MedicalImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


transform = transforms.Compose([transforms.Resize((96, 96)), transforms.ToTensor()])

# 학습용 데이터셋과 데이터로더
train_dataset = MedicalImageDataset(train_paths, transform)
val_dataset = MedicalImageDataset(val_paths, transform)
test_dataset = MedicalImageDataset(test_paths, transform)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# SRGAN 모델 초기화 함수
def init_model():
    generator = Generator(scale_factor=4).to(device)
    discriminator = Discriminator().to(device)
    optimizer_generator = optim.Adam(
        generator.parameters(), lr=1e-4, betas=(0.9, 0.999)
    )
    optimizer_discriminator = optim.Adam(
        discriminator.parameters(), lr=1e-4, betas=(0.9, 0.999)
    )
    return generator, discriminator, optimizer_generator, optimizer_discriminator


generator_criterion = nn.MSELoss()
discriminator_criterion = nn.BCELoss()


# Perceptual Loss 클래스
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features[:16].eval().to(device)
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.criterion = nn.MSELoss()

    def forward(self, fake, real):
        fake_features = self.vgg(fake)
        real_features = self.vgg(real)
        loss = self.criterion(fake_features, real_features)
        return loss


perceptual_criterion = PerceptualLoss()
l1_criterion = nn.L1Loss()


# 조기 종료 클래스
class EarlyStopping:
    def __init__(self, patience=20, delta=0.001):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """모델을 저장합니다."""
        torch.save(model.state_dict(), "checkpoint.pth")
        self.val_loss_min = val_loss


# 다양한 배치 사이즈 테스트
batch_sizes = [8, 16, 32]  # 배치 사이즈를 더 작게 설정
num_epochs_test = 5
best_val_loss = float("inf")
best_batch_size = None

for batch_size in batch_sizes:
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    generator, discriminator, optimizer_generator, optimizer_discriminator = (
        init_model()
    )

    for epoch in range(num_epochs_test):
        generator.train()
        discriminator.train()
        for i, data in enumerate(train_dataloader):
            hr_images = data.to(device)
            lr_images = nn.functional.interpolate(
                hr_images, scale_factor=0.25, mode="bicubic"
            ).to(device)

            optimizer_discriminator.zero_grad()
            real_labels = torch.ones((hr_images.size(0)), device=device)
            fake_labels = torch.zeros((hr_images.size(0)), device=device)

            outputs = discriminator(hr_images).view(-1)
            real_loss = discriminator_criterion(outputs, real_labels)

            fake_images = generator(lr_images)
            outputs = discriminator(fake_images.detach()).view(-1)
            fake_loss = discriminator_criterion(outputs, fake_labels)

            discriminator_loss = real_loss + fake_loss
            discriminator_loss.backward()
            optimizer_discriminator.step()

            optimizer_generator.zero_grad()
            fake_images = generator(lr_images)
            outputs = discriminator(fake_images).view(-1)
            adversarial_loss = generator_criterion(outputs, real_labels)

            perceptual_loss = perceptual_criterion(fake_images, hr_images)
            l1_loss = l1_criterion(fake_images, hr_images)

            generator_loss = l1_loss + 1e-3 * adversarial_loss + 1e-2 * perceptual_loss
            generator_loss.backward()
            optimizer_generator.step()

        # 검증 손실 계산
        val_loss = 0.0
        generator.eval()
        with torch.no_grad():
            for data in val_dataloader:
                hr_images = data.to(device)
                lr_images = nn.functional.interpolate(
                    hr_images, scale_factor=0.25, mode="bicubic"
                ).to(device)
                fake_images = generator(lr_images)
                val_loss += generator_criterion(fake_images, hr_images).item()

        val_loss /= len(val_dataloader)
        print(
            f"Batch size: {batch_size}, Epoch {epoch+1}, Validation Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_batch_size = batch_size

print(f"Best Batch size: {best_batch_size} with Validation Loss: {best_val_loss:.4f}")

# 최적의 배치 사이즈로 학습
train_dataloader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=best_batch_size, shuffle=False)

generator, discriminator, optimizer_generator, optimizer_discriminator = init_model()
early_stopping = EarlyStopping(patience=20, delta=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    generator.train()
    discriminator.train()
    for i, data in enumerate(train_dataloader):
        hr_images = data.to(device)
        lr_images = nn.functional.interpolate(
            hr_images, scale_factor=0.25, mode="bicubic"
        ).to(device)

        optimizer_discriminator.zero_grad()
        real_labels = torch.ones((hr_images.size(0)), device=device)
        fake_labels = torch.zeros((hr_images.size(0)), device=device)

        outputs = discriminator(hr_images).view(-1)
        real_loss = discriminator_criterion(outputs, real_labels)

        fake_images = generator(lr_images)
        outputs = discriminator(fake_images.detach()).view(-1)
        fake_loss = discriminator_criterion(outputs, fake_labels)

        discriminator_loss = real_loss + fake_loss
        discriminator_loss.backward()
        optimizer_discriminator.step()

        optimizer_generator.zero_grad()
        fake_images = generator(lr_images)
        outputs = discriminator(fake_images).view(-1)
        adversarial_loss = generator_criterion(outputs, real_labels)

        perceptual_loss = perceptual_criterion(fake_images, hr_images)
        l1_loss = l1_criterion(fake_images, hr_images)

        generator_loss = l1_loss + 1e-3 * adversarial_loss + 1e-2 * perceptual_loss
        generator_loss.backward()
        optimizer_generator.step()

        # 메모리 초기화
        torch.cuda.empty_cache()

    # 검증 손실 계산
    val_loss = 0.0
    generator.eval()
    with torch.no_grad():
        for data in val_dataloader:
            hr_images = data.to(device)
            lr_images = nn.functional.interpolate(
                hr_images, scale_factor=0.25, mode="bicubic"
            ).to(device)
            fake_images = generator(lr_images)
            l1_loss = l1_criterion(fake_images, hr_images)
            perceptual_loss = perceptual_criterion(fake_images, hr_images)
            val_loss += l1_loss.item() + 1e-2 * perceptual_loss.item()

    val_loss /= len(val_dataloader)
    print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")
# 최종 모델 로드 이후 코드 수정
generator.load_state_dict(torch.load("checkpoint.pth"))

# 테스트 코드
generator.eval()

# 테스트용 데이터에서 랜덤으로 이미지 선택
test_img_path = random.choice(test_paths)
print(f"Test image path: {test_img_path}")

img = Image.open(test_img_path).convert("RGB")
lr_img = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    sr_img = generator(lr_img).squeeze(0).cpu()
    sr_img = sr_img * 0.5 + 0.5
    sr_img = sr_img.numpy().transpose(1, 2, 0)

plt.imshow(sr_img)
plt.axis("off")  # 축 숨기기

# 이미지 파일로 저장
save_path = os.path.join(root_dir, "sr_img.png")
plt.imsave(save_path, sr_img)
plt.show()

print(f"Super-resolution image saved at: {save_path}")
