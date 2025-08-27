from Recongnition.model import PMG 
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import json
import timm

import torch
import torch.nn as nn

class PMGWithClinical(nn.Module):
    def __init__(self, pmg_backbone, clinical_size=5, num_classes=2, sample_input=(1,3,224,224)):
        """
        PMG + Clinical Features

        Args:
            pmg_backbone: PMG model, returns image feature tensor
            clinical_size: number of clinical features
            num_classes: number of classes
            sample_input: sample input tensor shape to compute PMG output feature size
        """
        super(PMGWithClinical, self).__init__()
        self.pmg = pmg_backbone

        # 自动计算 PMG 输出特征维度
        with torch.no_grad():
            dummy_input = torch.randn(sample_input)
            img_feat = self.pmg(dummy_input)
            if isinstance(img_feat, tuple) or isinstance(img_feat, list):
                img_feat = img_feat[-1]  # 取最后输出特征
            self.img_feature_dim = img_feat.view(img_feat.size(0), -1).shape[1]

        # 临床特征全连接
        self.clinical_fc = nn.Sequential(
            nn.Linear(clinical_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 融合特征分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.img_feature_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, clinical_data):
        """
        x: [B, C, H, W] image tensor
        clinical_data: [B, clinical_size] tensor
        """
        img_feat = self.pmg(x)
        if isinstance(img_feat, tuple) or isinstance(img_feat, list):
            img_feat = img_feat[-1]
        img_feat = img_feat.view(img_feat.size(0), -1)  # flatten

        clinical_feat = self.clinical_fc(clinical_data)
        fused = torch.cat([img_feat, clinical_feat], dim=1)
        out = self.classifier(fused)
        return out



# ================================
# Dataset for PMG with clinical features using full images
# ================================
class CancerPatchClinicalDataset(Dataset):
    def __init__(self, img_dir, label_file, clinical_json, transform=None):
        """
        Args:
            img_dir (str): Directory containing all images.
            label_file (str): Path to the file with image labels (image_name label).
            clinical_json (str): Path to JSON file with clinical features.
            transform (callable, optional): Transform to apply to images.
        """
        self.img_dir = img_dir
        self.transform = transform

        # Load image labels
        self.samples = []
        with open(label_file, "r") as f:
            for line in f:
                img_name, label = line.strip().rsplit(maxsplit=1)
                self.samples.append((img_name, int(label)))

        # Load clinical features
        with open(clinical_json, "r") as f:
            self.clinical_dict = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # Read image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)

        # Apply transformation if provided
        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0  # C,H,W normalized to [0,1]

        # Load clinical features
        clinical_feat = torch.tensor(list(map(float, self.clinical_dict[os.path.splitext(img_name)[0]])), dtype=torch.float32)

        return img_tensor, clinical_feat, label

# ================================
# Data preparation for PMG + clinical features
# ================================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Load training and validation datasets
train_dataset = CancerPatchClinicalDataset(
    img_dir="./train_images",
    label_file="train_labels.txt",
    clinical_json="train_clinical.json",
    transform=transform
)
val_dataset = CancerPatchClinicalDataset(
    img_dir="./val_images",
    label_file="val_labels.txt",
    clinical_json="val_clinical.json",
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

# ================================
# Initialize PMG backbone and PMGWithClinical model
# ================================
backbone = timm.create_model('resnet50', pretrained=True, num_classes=0)
pmg = PMG(backbone, feature_size=512, classes_num=2)
model = PMGWithClinical(pmg_backbone=pmg, clinical_size=5, num_classes=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ================================
# Training loop for PMG + clinical features
# ================================
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for imgs, clinical, labels in train_loader:
        imgs, clinical, labels = imgs.to(device), clinical.to(device), labels.to(device)
        outputs = model(imgs, clinical)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print("[Epoch {}/{}] Training Loss: {:.4f}".format(epoch+1, num_epochs, total_loss/len(train_loader)))


    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, clinical, labels in val_loader:
            imgs, clinical, labels = imgs.to(device), clinical.to(device), labels.to(device)
            outputs = model(imgs, clinical)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print("[Epoch {}/{}] Validation Accuracy: {:.4f}".format(epoch+1, num_epochs, correct/total))


# ================================
# Save trained PMG model
# ================================
os.makedirs("weights", exist_ok=True)
torch.save(model.state_dict(), "weights/pmg_with_clinical_yolo.pth")
print("PMG model training is completed and saved to weights/pmg_with_clinical_yolo.pth")
