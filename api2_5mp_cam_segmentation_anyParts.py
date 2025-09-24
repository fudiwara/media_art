# mediapipe新IFのセマンティックセグメンテーションお試しプログラム
# 髪だけに色を付加するサンプル
# 予め以下のURLから特徴点ファイルをダウンロードしておく必要がある
# https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter

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

options = mp.tasks.vision.ImageSegmenterOptions( # セグメンテーションのオプション
    base_options = mp.tasks.BaseOptions(model_asset_path = "selfie_multiclass_256x256.tflite"), # 人用
    output_category_mask = True)
segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(options) # 推定器の初期化

ret, frame = cap.read()
i_h, i_w, _ = frame.shape

tick_meter = cv2.TickMeter()
tick_meter.start() # 計測開始

while True:
    ret, frame = cap.read() # キャプチャ
    if not ret: # フレームの読み込みに失敗したらループ終了
        break
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # MediaPipe用にRGBに
    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = img_rgb) # MediaPipeのImageオブジェクトへ

    segmented_masks = segmenter.segment(mp_image) # セグメンテーション処理
    category_masks = segmented_masks.category_mask.numpy_view() # ndarrayに変換
    binary_mask = (category_masks == 1) # 特定のIDを持つマスクのみを作る

    mask_color = np.zeros((i_h, i_w, 3), dtype = np.uint8)
    color = [0, 255, 255] # 黄色
    mask_color[:, :, 0] = np.where(binary_mask, color[0], 0)
    mask_color[:, :, 1] = np.where(binary_mask, color[1], 0)
    mask_color[:, :, 2] = np.where(binary_mask, color[2], 0)

    img_mask = binary_mask.astype(np.uint8) * 255 # 表示用の2値画像

    img_disp = cv2.addWeighted(frame, 1, mask_color, 0.2, 0) # カラー付けしたマスクと入力フレームを合成
    
    tick_meter.stop() # 計測終了
    cv2.putText(frame, f"{tick_meter.getFPS():.1f}", (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    tick_meter.reset() # 次フレーム用に時計をリセット
    tick_meter.start() # 計測開始

    cv2.imshow("input frame", frame) # 入力フレーム
    cv2.imshow("segmented mask", img_mask) # 特定のマスクの表示
    cv2.imshow("segmented result", img_disp) # 合成結果の表示

    key = cv2.waitKey(1)
    if key == 27:
        break
