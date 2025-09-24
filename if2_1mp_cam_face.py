# mediapipe新IFのface_detectionお試しプログラム
# 顔の矩形と特徴点の座標を表示する
# 予め以下のURLから特徴点ファイルをダウンロードしておく必要がある
# https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector
# 新IFの顔検出はスマホのセルフィーに最適化されている感じなので利用時には色々と注意

import sys
sys.dont_write_bytecode = True

import cv2
import mediapipe as mp

cap = cv2.VideoCapture(int(sys.argv[1])) # キャプチャ開始(複数カメラはIDを追加)

options = mp.tasks.vision.FaceDetectorOptions(mp.tasks.BaseOptions("blaze_face_short_range.tflite")) # 検出器のオプション
detector = mp.tasks.vision.FaceDetector.create_from_options(options) # 検出器の初期化

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

    detection_result = detector.detect(mp_image) # 顔検出

    if detection_result.detections:
        for i, detection in enumerate(detection_result.detections): # 各顔に対するループ
            b = detection.bounding_box
            print(b)
            p0 = (b.origin_x, b.origin_y) # バウンディングボックスは入力画像の座標
            p1 = (b.origin_x + b.width, b.origin_y + b.height)
            cv2.rectangle(frame, p0, p1, (0, 255, 0), 2) # 緑色の矩形

            score = int(detection.categories[0].score * 100) # 検出スコアを%で表示
            text = f"{i}  {score}%"
            cv2.putText(frame, text, (b.origin_x, b.origin_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            for keypoint in detection.keypoints: # キーポイントの描画
                keypoint_x = int(keypoint.x * i_w) # キーポイントはUV座標系
                keypoint_y = int(keypoint.y * i_h)
                cv2.circle(frame, (keypoint_x, keypoint_y), 5, (0, 255, 0), 1) # 緑色の点

    tick_meter.stop() # 計測終了
    fps = tick_meter.getFPS() # FPSの計算
    cv2.putText(frame, f"{fps:.1f}", (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    tick_meter.reset() # 次フレーム用に時計をリセット

    cv2.imshow("image", frame) # 結果の表示

    key = cv2.waitKey(1)
    if key == 27:
        break

