import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# 1. Custom Dataset for Orthodontic Landmarks (e.g. ISBI Cephalometric Dataset)
class CephalometricLandmarksDataset(Dataset):
    """
    Dataset loader for ISBI Cephalometric/Dental radiograph landmarks.
    Expects images directory and a CSV file containing:
    image_name, landmark_0_x, landmark_0_y, landmark_1_x, landmark_1_y, ...
    """
    def __init__(self, csv_file: str, img_dir: str, num_landmarks: int = 19, img_size: int = 512, heatmap_size: int = 128, sigma: float = 3.0):
        self.df = pd.read_csv(csv_file) if os.path.exists(csv_file) else None
        self.img_dir = img_dir
        self.num_landmarks = num_landmarks
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma

    def __len__(self):
        if self.df is None:
            return 100 # Mock length if CSV not present
        return len(self.df)

    def _generate_heatmap(self, x: float, y: float) -> np.ndarray:
        """Generates a 2D Gaussian heatmap centered at (x, y)"""
        hm = np.zeros((self.heatmap_size, self.heatmap_size), dtype=np.float32)
        if x < 0 or y < 0 or x >= self.heatmap_size or y >= self.heatmap_size:
            return hm
            
        mu_x = int(x)
        mu_y = int(y)
        
        # Grid range around centroid
        radius = int(self.sigma * 3)
        x_min = max(0, mu_x - radius)
        x_max = min(self.heatmap_size, mu_x + radius + 1)
        y_min = max(0, mu_y - radius)
        y_max = min(self.heatmap_size, mu_y + radius + 1)
        
        for cur_y in range(y_min, y_max):
            for cur_x in range(x_min, x_max):
                dist_sq = (cur_x - mu_x)**2 + (cur_y - mu_y)**2
                hm[cur_y, cur_x] = np.exp(-dist_sq / (2 * self.sigma**2))
                
        return hm

    def __getitem__(self, idx):
        # 1. Handle mock mode
        if self.df is None:
            # Return synthetic image and heatmaps
            img = torch.randn(3, self.img_size, self.img_size)
            heatmaps = torch.zeros(self.num_landmarks, self.heatmap_size, self.heatmap_size)
            for i in range(self.num_landmarks):
                rx = np.random.uniform(20, self.heatmap_size - 20)
                ry = np.random.uniform(20, self.heatmap_size - 20)
                heatmaps[i] = torch.tensor(self._generate_heatmap(rx, ry))
            return img, heatmaps

        # 2. Real dataset mode
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_name"])
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        
        # Resize image
        image = image.resize((self.img_size, self.img_size))
        img_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0

        # Read landmark points and project onto heatmap coordinates
        heatmaps = np.zeros((self.num_landmarks, self.heatmap_size, self.heatmap_size), dtype=np.float32)
        for i in range(self.num_landmarks):
            x_col = f"lm_{i}_x"
            y_col = f"lm_{i}_y"
            if x_col in row and y_col in row:
                px = (row[x_col] / orig_w) * self.heatmap_size
                py = (row[y_col] / orig_h) * self.heatmap_size
                heatmaps[i] = self._generate_heatmap(px, py)
                
        return img_tensor, torch.tensor(heatmaps)

# 2. PyTorch HRNet-like Backbone
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class SimpleHRNetRegress(nn.Module):
    """
    Simplified HRNet-W48 heatmap regressor.
    Extracts multi-resolution features and predicts landmark coordinates heatmaps.
    """
    def __init__(self, num_landmarks=19, base_channels=48):
        super().__init__()
        self.stage1 = DoubleConv(3, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        
        self.stage2 = DoubleConv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.stage3 = DoubleConv(base_channels * 2, base_channels * 4)
        
        # Up-sampling blocks to project back to heatmap scale (e.g. 1/4 resolution)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.head = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, num_landmarks, 1) # Output logits
        )

    def forward(self, x):
        x1 = self.stage1(x) # 512x512
        x2 = self.pool1(x1) # 256x256
        x3 = self.stage2(x2) # 256x256
        x4 = self.pool2(x3) # 128x128
        x5 = self.stage3(x4) # 128x128 -> heatmap scale
        
        out = self.head(x5) # 128x128
        return out

# 3. Training Script Execution
def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "gpu" else "cpu")
    print(f"Training on device: {device}")

    # Initialize dataloader
    dataset = CephalometricLandmarksDataset(
        csv_file=args.csv_file,
        img_dir=args.img_dir,
        num_landmarks=args.num_landmarks,
        img_size=args.img_size,
        heatmap_size=args.heatmap_size
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Initialize Model, Loss, Optimizer
    model = SimpleHRNetRegress(num_landmarks=args.num_landmarks).to(device)
    criterion = nn.MSELoss() # Heatmap MSE loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("Starting Heatmap Regression Training Loop...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * images.size(0)
            
        scheduler.step()
        avg_loss = epoch_loss / len(dataset)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {avg_loss:.6f} - Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

    # Save final model
    torch.save(model.state_dict(), args.save_path)
    print(f"Model checkpoint successfully saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HRNet Landmark Detector Model")
    parser.add_argument("--csv_file", type=str, default="landmarks_annotations.csv", help="Path to landmark CSV annotations")
    parser.add_argument("--img_dir", type=str, default="./images", help="Path to images directory")
    parser.add_argument("--num_landmarks", type=int, default=19, help="Number of orthodontic landmarks to predict")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--img_size", type=int, default=512, help="Input size")
    parser.add_argument("--heatmap_size", type=int, default=128, help="Output heatmap size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, gpu)")
    parser.add_argument("--save_path", type=str, default="hrnet_landmarks.pth", help="Output path for weights")
    
    args = parser.parse_args()
    
    # Create a template csv file if not exists
    if not os.path.exists(args.csv_file):
        print(f"Template CSV '{args.csv_file}' not found. Generating a mock dataset file...")
        cols = ["image_name"]
        for idx in range(args.num_landmarks):
            cols.extend([f"lm_{idx}_x", f"lm_{idx}_y"])
        df = pd.DataFrame(columns=cols)
        # Add 5 dummy rows
        for i in range(5):
            row = { "image_name": f"patient_{i}.png" }
            for idx in range(args.num_landmarks):
                row[f"lm_{idx}_x"] = np.random.randint(100, 400)
                row[f"lm_{idx}_y"] = np.random.randint(100, 400)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(args.csv_file, index=False)
        print("Generated mock CSV with 5 patient entries.")
        
    run_training(args)
