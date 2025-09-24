# mediapipe旧IFのselfie_segmentationお試しプログラム
# 背景と人物をリアルタイムで分離する

import sys
sys.dont_write_bytecode = True
import cv2
import numpy as np
import mediapipe as mp

cap = cv2.VideoCapture(int(sys.argv[1])) # キャプチャ開始(複数カメラはIDを追加)

cw, ch = 640, 480
# cw, ch = 1280, 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cw)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ch)

mp_selfie_segmentation = mp.solutions.selfie_segmentation # セグメンテーション機能の初期化
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation()

ret, frame = cap.read()
i_h, i_w, _ = frame.shape

tick_meter = cv2.TickMeter() # FPS表示用のタイマー
tick_meter.start() # 計測開始

while True:
    ret, frame = cap.read() # キャプチャ
    if not ret: # フレームの読み込みに失敗したらループ終了
        break

    results = selfie_segmentation.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # セグメンテーション処理
    
    condition = 0.1 < np.stack((results.segmentation_mask,) * 3, axis = -1) # マスクの閾値は0.1〜0.5くらいで

    bg_image = cv2.GaussianBlur(frame, (55, 55), 0)  # 背景をぼかす

    img_disp = np.where(condition, frame, bg_image) # 背景と人物を合成

    tick_meter.stop() # 計測終了
    cv2.putText(img_disp, f"{tick_meter.getFPS():.1f}", (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    tick_meter.reset() # 次フレーム用に時計をリセット
    tick_meter.start() # 計測開始

    cv2.imshow("Frame segmentation", img_disp)

    k = cv2.waitKey(1)
    if k == 27:
        break # escキーでプログラム終了