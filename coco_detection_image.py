import sys, os
sys.dont_write_bytecode = True

import cv2
import numpy as np
from PIL import Image
import pathlib

import torch
import torchvision

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
image_path = sys.argv[1] # 入力画像のパス
file_name = pathlib.Path(sys.argv[1])

thDetection = 0.6

# フォント、枠の設定
font_scale = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_DUPLEX, 11, 1)
colors = [(255, 100, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
clr_num = len(colors)
class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet", "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "hair brush"]

# モデルの定義と読み込みおよび評価用のモードにセットする
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights = "DEFAULT") # こちらだとCPU環境でもいける
# model = torchvision.models.detection.fcos_resnet50_fpn(weights="DEFAULT") # fcos
model.to(DEVICE)
model.eval()

data_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# 画像の読み込み・変換
img = Image.open(image_path).convert("RGB") # カラー指定で開く
i_w, i_h = img.size
data = data_transforms(img)
data = data.unsqueeze(0) # テンソルに変換してから1次元追加

data = data.to(DEVICE)
outputs = model(data) # 推定処理
# print(outputs)
bboxs = outputs[0]["boxes"].detach().cpu().numpy()
scores = outputs[0]["scores"].detach().cpu().numpy()
labels = outputs[0]["labels"].detach().cpu().numpy()
# print(bboxs, scores, labels)

img = cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_RGB2BGR)
for i in range(len(scores)):
    b = bboxs[i]
    # print(b)
    prd_val = scores[i]
    if prd_val < thDetection: continue # 閾値以下は飛ばす
    prd_cls = labels[i]

    x0, y0 = int(b[0]), int(b[1])
    p0, p1 = (x0, y0), (int(b[2]), int(b[3]))
    print(prd_cls, prd_val, p0, p1)
    
    box_col = colors[prd_cls % clr_num]

    # text = f" {prd_cls}  {prd_val:.3f} " # クラスIDと確率を表示させる場合
    text = f" {class_names[prd_cls - 1]} " # クラス名のみを表示させる場合
    (t_w, t_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1) # テキスト部の矩形サイズ取得
    cv2.rectangle(img, p0, p1, box_col, thickness = 2) # 検出領域の矩形
    cv2.rectangle(img, (x0, y0 - t_h), (x0 + t_w, y0), box_col, thickness = -1) # テキストの背景の矩形
    cv2.putText(img, text, p0, cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

cv2.imwrite(str(file_name.parent / f"{file_name.stem}_det.png"), img)
