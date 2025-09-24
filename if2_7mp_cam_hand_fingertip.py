# mediapipe新IFのhand_detectionお試しプログラム
# 指先の特徴点のみ表示するサンプル
# 予め以下のURLから特徴点ファイルをダウンロードしておく必要がある
# https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

import sys
sys.dont_write_bytecode = True

import cv2
import mediapipe as mp

cap = cv2.VideoCapture(int(sys.argv[1])) # キャプチャ開始(複数カメラはIDを追加)

options = mp.tasks.vision.HandLandmarkerOptions( # 検出器のオプション
    base_options = mp.tasks.BaseOptions(model_asset_path = "hand_landmarker.task"), # 手ランドマーク用モデル
    num_hands = 2) # 検出できる手の数
landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options) # 検出器の初期化

posHandsIds = [4, 8, 12, 16, 20] # 各指の先端

ret, frame = cap.read()
i_h, i_w, _ = frame.shape

tick_meter = cv2.TickMeter()

while True:
    ret, frame = cap.read()

    if not ret: # フレームの読み込みに失敗したらループ終了
        break
    
    tick_meter.start() # 計測開始
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # MediaPipe用にRGBに
    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = img_rgb) # MediaPipeのImageオブジェクトへ

    hand_landmarker_result = landmarker.detect(mp_image) # 手ランドマーク検出

    if hand_landmarker_result:
        for i, hand in enumerate(hand_landmarker_result.hand_landmarks): # 各手に対するループ
            for j in range(len(posHandsIds)):
                x = int(hand[posHandsIds[j]].x * i_w)
                y = int(hand[posHandsIds[j]].y * i_h)
                cv2.circle(frame, (x, y), 11, (0, 255, 0), 3) # 赤色の点
            

            hand_side = hand_landmarker_result.handedness[i][0].category_name # 左か右か
            x = int(hand[0].x * i_w)
            y = int(hand[0].y * i_h)
            cv2.putText(frame, hand_side, (x, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    tick_meter.stop() # 計測終了
    fps = tick_meter.getFPS() # FPSの計算
    cv2.putText(frame, f"{fps:.1f}", (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    tick_meter.reset() # 次フレーム用に時計をリセット

    cv2.imshow("image", frame) # 結果の表示

    key = cv2.waitKey(1)
    if key == 27:
        break

