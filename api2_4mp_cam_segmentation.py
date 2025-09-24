# mediapipe新IFのセマンティックセグメンテーションお試しプログラム
# 認識対象のモデルが決められている(参照：color_map)
# 予め以下のURLから特徴点ファイルをダウンロードしておく必要がある
# https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter
# segmented_masksがピクセル単位で領域分割された結果になる
# 何らかの表示等で利用する際にはある程度の加工処理が必要になる

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
    # base_options = mp.tasks.BaseOptions(model_asset_path = "deeplab_v3.tflite"), # VOCカテゴリ用
    output_category_mask = True)
segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(options) # 推定器の初期化
num_categories = 6 # selfie_multiclass_256x256.tflite の場合は6
color_map = {
    0: (0, 0, 0),       # 背景 (黒)
    1: (0, 0, 255),     # 髪 (青)
    2: (0, 255, 0),     # 顔 (緑)
    3: (255, 0, 0),     # 肌 (赤)
    4: (255, 255, 0),   # 服 (シアン)
    5: (0, 255, 255)    # アクセサリ (黄色)
}

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

    segmented_colored_image = np.zeros((i_h, i_w, 3), dtype = np.uint8) # セグメンテーション結果の合成用画像
    for idx in range(num_categories):
        binary_mask = (category_masks == idx) # IDと同じ値となる画素から得るブールマスク

        mask_buf = np.zeros((i_h, i_w, 3), dtype = np.uint8)
        color = color_map[idx]
        mask_buf[:, :, 0] = np.where(binary_mask, color[0], 0)
        mask_buf[:, :, 1] = np.where(binary_mask, color[1], 0)
        mask_buf[:, :, 2] = np.where(binary_mask, color[2], 0)

        segmented_colored_image = cv2.addWeighted(segmented_colored_image, 1, mask_buf, 1, 0) # 合成用プレーンに追加

    img_disp = cv2.addWeighted(frame, 0.6, segmented_colored_image, 0.4, 0) # 全体のマスクと入力フレームを合成
    
    tick_meter.stop() # 計測終了
    cv2.putText(frame, f"{tick_meter.getFPS():.1f}", (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    tick_meter.reset() # 次フレーム用に時計をリセット
    tick_meter.start() # 計測開始

    cv2.imshow("input frame", frame) # 入力フレーム
    cv2.imshow("segmented mask", segmented_colored_image) # 色づけされたマスクの表示
    cv2.imshow("segmented result", img_disp) # 結果の表示

    key = cv2.waitKey(1)
    if key == 27:
        break
