#你的程式 = 這張圖裡的 「patch → Transformer Encoder → [CLS] token」部分。
import torch
import clip
from PIL import Image
import numpy as np
import sys

def vit(img_path):
    # 1) 選 GPU / CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 2) 載入 CLIP 模型 (ViT-B/32 版本；你也可以換 ViT-B/16)
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    # 3) 讀取圖片
    img_path = img_path
    image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device) #負責 resize / normalize（圖上沒畫，但必要的前處理）。

    # 4) 轉向量 (embedding)   CLIP 沒有走到「MLP Head」，因為我們不是要分類
    with torch.no_grad():
        emb = model.encode_image(image)   #直接做了圖裡 patch embedding → [CLS] token → Transformer Encoder
        emb = emb / emb.norm(dim=-1, keepdim=True)  # 正規化，方便做相似度

    # 5) 輸出結果  
    vec = emb.squeeze(0).cpu().numpy()  #embedding 向量，對應到圖裡的 [CLS] token(整張圖的代表向量)輸出
    print("Embedding shape:", vec.shape)   # CLIP ViT-B/32 → 512 維
    print("前 10 維:", vec[:10])
    out_path = img_path.split("\\")[-1] + "_embedding.npy"
    np.save(out_path, vec)
    return vec,out_path