# mediapipe旧IFのface_detectionお試しプログラム
# 顔の矩形と特徴点の座標を表示
# model_selectionの指定で小さい顔も検出できるようになる

import sys
sys.dont_write_bytecode = True
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(int(sys.argv[1])) # キャプチャ開始(複数カメラはIDを追加)

cw, ch = 640, 480
# cw, ch = 1280, 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cw)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ch)

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection = 1, min_detection_confidence = 0.5) # 小さい顔も検出
# face_detection = mp_face_detection.FaceDetection(min_detection_confidence = 0.5)

ret, frame = cap.read()
i_h, i_w, _ = frame.shape

tick_meter = cv2.TickMeter() # FPS表示用のタイマー
tick_meter.start() # 計測開始

while True:
    ret, frame = cap.read() # キャプチャ
    if not ret: # フレームの読み込みに失敗したらループ終了
        break

    results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # mediapipeに処理を渡す
    if results.detections:
        for i in range(len(results.detections)):
            # 顔領域の検出結果描画
            b = results.detections[i].location_data.relative_bounding_box
            x0 = int(b.xmin * i_w)
            y0 = int(b.ymin * i_h)
            x1 = int((b.xmin + b.width) * i_w)
            y1 = int((b.ymin + b.height) * i_h)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

            score = results.detections[i].score[0] * 100 # 検出スコアを%で表示
            text = f"{i}  {score:.0f}%"
            cv2.putText(frame, text, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 顔特徴点の検出結果描画
            k = results.detections[i].location_data.relative_keypoints
            for j in range(len(k)):
                kx = int(k[j].x * i_w)
                ky = int(k[j].y * i_h)
                cv2.circle(frame, (kx, ky), 1, (0, 255, 0))
            # print(len(k))

    tick_meter.stop() # 計測終了
    cv2.putText(frame, f"{tick_meter.getFPS():.1f}", (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    tick_meter.reset() # 次フレーム用に時計をリセット
    tick_meter.start() # 計測開始

    cv2.imshow("Frame pose", frame) # 顔検出した結果の表示

    k = cv2.waitKey(1)
    if k == 27:
        break # escキーでプログラム終了
