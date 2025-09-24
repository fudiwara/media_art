# mediapipe新IFのhand_detectionお試しプログラム
# 手の特徴点の座標を表示する
# 予め以下のURLから特徴点ファイルをダウンロードしておく必要がある
# https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
# 手の左右の認識もできる
# カメラ中心の距離情報も推定できる

import sys
sys.dont_write_bytecode = True

import cv2
import mediapipe as mp

cap = cv2.VideoCapture(int(sys.argv[1])) # キャプチャ開始(複数カメラはIDを追加)

cw, ch = 640, 480
# cw, ch = 1280, 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cw)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ch)

options = mp.tasks.vision.HandLandmarkerOptions( # 検出器のオプション
    base_options = mp.tasks.BaseOptions(model_asset_path = "hand_landmarker.task"), # 手ランドマーク用モデル
    num_hands = 2) # 検出できる手の数
landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options) # 検出器の初期化

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

    hand_landmarker_result = landmarker.detect(mp_image) # 手ランドマーク検出

    if hand_landmarker_result:
        for i, hand in enumerate(hand_landmarker_result.hand_landmarks): # 各手に対するループ
            for j, p in enumerate(hand):
                x = int(p.x * i_w)
                y = int(p.y * i_h)
                cv2.circle(frame, (x, y), 5, (0, 0, 255), 2) # 赤色の点

                if j == 0: # 0(手のつけね)の位置で情報を表示する
                    hand_side = hand_landmarker_result.handedness[i][0].category_name # 左か右か
                    cv2.putText(frame, hand_side, (x, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    tick_meter.stop() # 計測終了
    cv2.putText(frame, f"{tick_meter.getFPS():.1f}", (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    tick_meter.reset() # 次フレーム用に時計をリセット
    tick_meter.start() # 計測開始

    cv2.imshow("image", frame) # 結果の表示

    key = cv2.waitKey(1)
    if key == 27:
        break

