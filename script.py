import os
import csv
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

WEIGHTS_PATH = r"C:\Users\Lenovo\OneDrive\Desktop\SC_2\best_vit_finetuned_full.pth"
TEST_FOLDER = r"C:\Users\Lenovo\OneDrive\Desktop\SC_2\Practical_Test_Samples"
OUTPUT_CSV = "predictions.csv"


IMG_SIZE = 224
NUM_CLASSES = 17
PATCH_SIZE = 16
EMBED_DIM = 768
DEPTH = 12
NUM_HEADS = 12
MLP_RATIO = 4.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ManualViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed(
            IMG_SIZE, PATCH_SIZE, 3, EMBED_DIM
        )
        num_patches = (IMG_SIZE // PATCH_SIZE) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, EMBED_DIM)
        )

        self.blocks = nn.ModuleList([
            Block(EMBED_DIM, NUM_HEADS, MLP_RATIO)
            for _ in range(DEPTH)
        ])

        self.norm = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, NUM_CLASSES)

    def forward(self, x):
        x = self.patch_embed(x)
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return self.head(x[:, 0])


model = ManualViT().to(DEVICE)
print("Loading weights...")
state = torch.load(WEIGHTS_PATH, map_location=DEVICE)
model.load_state_dict(state, strict=True)
model.eval()
print("Weights loaded successfully.")


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


predictions = []

with torch.no_grad():
    for name in sorted(os.listdir(TEST_FOLDER)):
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img = Image.open(
            os.path.join(TEST_FOLDER, name)
        ).convert("RGB")

        img = transform(img).unsqueeze(0).to(DEVICE)
        logits = model(img)
        pred = torch.argmax(logits, dim=1).item()

        predictions.append([name, pred])


with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ImageName", "ClassLabel"])
    writer.writerows(predictions)

print("CSV saved:", OUTPUT_CSV)
