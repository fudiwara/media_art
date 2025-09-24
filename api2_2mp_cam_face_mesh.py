# mediapipe新IFのfaceお試しプログラム
# 顔メッシュの特徴点の座標に○を表示する
# 予め以下のURLから特徴点ファイルをダウンロードしておく必要がある
# https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

import sys
sys.dont_write_bytecode = True

import cv2
import mediapipe as mp

cap = cv2.VideoCapture(int(sys.argv[1])) # キャプチャ開始(複数カメラはIDを追加)

cw, ch = 640, 480
# cw, ch = 1280, 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cw)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ch)

options = mp.tasks.vision.FaceLandmarkerOptions( # 検出器のオプション
    base_options = mp.tasks.BaseOptions(model_asset_path = "face_landmarker.task"), # モデル
    num_faces = 2) # 検出できる顔の数
detector = mp.tasks.vision.FaceLandmarker.create_from_options(options) # 検出器の初期化

# 点の色を変えるパーツ
posFacesParts = [7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246, 249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466,
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415]
# https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py
parts_color = [(0, 0, 255), (0, 255, 0)]

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

    detection_result = detector.detect(mp_image) # ランドマーク検出

    for face_lms in detection_result.face_landmarks: # 各顔のループ
        for i in range(len(face_lms)): # 各landmarkに対するループ
            x = int(face_lms[i].x * i_w)
            y = int(face_lms[i].y * i_h)

            if i in posFacesParts:
                c = 0
            else:
                c = 1
            cv2.circle(frame, (x, y), 3, parts_color[c], 1)

    tick_meter.stop() # 計測終了
    cv2.putText(frame, f"{tick_meter.getFPS():.1f}", (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    tick_meter.reset() # 次フレーム用に時計をリセット
    tick_meter.start() # 計測開始

    cv2.imshow("image", frame) # 結果の表示

    key = cv2.waitKey(1)
    if key == 27:
        break
