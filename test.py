import torch
import clip
from PIL import Image
import numpy as np
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# 你的 3 張資料庫圖片（自行修改檔名）
image_files = {
    "bird": "bird1.jpg",
    "fish": "fish1.jpg",
    "dog" : "dog1.jpg"
}

embeddings = []
labels = []

with torch.no_grad():
    for label, path in image_files.items():
        image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        emb = model.encode_image(image)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        vec = emb.squeeze(0).cpu().numpy()

        embeddings.append(vec)
        labels.append(label)

# 存檔
np.save("db_embeddings.npy", np.stack(embeddings, axis=0))  # shape = (3, 512)
np.save("db_labels.npy", np.array(labels))                  # ['bird','fish','dog']

print("✅ 資料庫已建立，包含", len(labels), "張圖片")
