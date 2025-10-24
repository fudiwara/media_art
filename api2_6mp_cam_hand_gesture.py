# mediapipe新IFのhand_gestureお試しプログラム
# Macの場合はOS側のジェスチャー認識をオフしてから使った方がよい
# 予め以下のURLから特徴点ファイルをダウンロードしておく必要がある
# https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer
# ジェスチャーというよりハンドサインぽい感じ
# 1: グー、2: パー、3: 上指さし、4: だめね、5: いいね、6: チョキ、7: ILoveYouサイン

import sys
sys.dont_write_bytecode = True

import cv2
import mediapipe as mp

cap = cv2.VideoCapture(int(sys.argv[1])) # キャプチャ開始(複数カメラはIDを追加)

cw, ch = 640, 480
# cw, ch = 1280, 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cw)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ch)

options = mp.tasks.vision.GestureRecognizerOptions( # 検出器のオプション
    base_options = mp.tasks.BaseOptions(model_asset_path = "gesture_recognizer.task"), # 手ジェスチャー用モデル
    num_hands = 2) # 検出できる手の数
recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)

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

    gesture_recognition_result = recognizer.recognize(mp_image) # ハンドジェスチャーの認識

    if gesture_recognition_result:
        for i, hand in enumerate(gesture_recognition_result.hand_landmarks): # 各手に対するループ
            for j, p in enumerate(hand):
                x = int(p.x * i_w)
                y = int(p.y * i_h)
                cv2.circle(frame, (x, y), 5, (0, 0, 255), 2) # 赤色の点

                if j == 0: # 0(手のつけね)の位置で情報を表示する
                    hand_side = gesture_recognition_result.handedness[i][0].category_name # 左か右か
                    cv2.putText(frame, hand_side, (x, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

                    hand_gesture = gesture_recognition_result.gestures[i][0].category_name # ジェスチャーの種類
                    cv2.putText(frame, hand_gesture, (x, y + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    tick_meter.stop() # 計測終了
    cv2.putText(frame, f"{tick_meter.getFPS():.1f}", (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    tick_meter.reset() # 次フレーム用に時計をリセット
    tick_meter.start() # 計測開始

    cv2.imshow("image", frame) # 結果の表示

    key = cv2.waitKey(1)
    if key == 27:
        break
