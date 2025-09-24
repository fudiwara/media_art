# mediapipe新IFの一般物体検出お試しプログラム
# 80種のオブジェクト検出ができる
# 予め以下のURLから特徴点ファイルをダウンロードしておく必要がある
# https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector
# 80種のカテゴリ等もURL要参照(それぞれの検出性能についてはそこまで期待しない方がよい)

import sys
sys.dont_write_bytecode = True

import cv2
import mediapipe as mp

cap = cv2.VideoCapture(int(sys.argv[1])) # キャプチャ開始(複数カメラはIDを追加)

options = mp.tasks.vision.ObjectDetectorOptions( # 検出器のオプション
    base_options = mp.tasks.BaseOptions(model_asset_path = "efficientdet_lite2.tflite"), # モデル
    max_results = 5) # 検出される最大オブジェクト数
detector = mp.tasks.vision.ObjectDetector.create_from_options(options) # 検出器の初期化

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

    detection_result = detector.detect(mp_image) # オブジェクト検出

    if detection_result.detections:
        for i, detection in enumerate(detection_result.detections): # 各オブジェクトに対するループ
            b = detection.bounding_box
            p0 = (b.origin_x, b.origin_y)
            p1 = (b.origin_x + b.width, b.origin_y + b.height)
            cv2.rectangle(frame, p0, p1, (0, 255, 0), 2) # 緑色の矩形

            score = int(detection.categories[0].score * 100) # 検出スコアを%で表示
            text = f"{detection.categories[0].category_name}  {score}%"
            cv2.putText(frame, text, (b.origin_x, b.origin_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    tick_meter.stop() # 計測終了
    fps = tick_meter.getFPS() # FPSの計算
    cv2.putText(frame, f"{fps:.1f}", (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    tick_meter.reset() # 次フレーム用に時計をリセット

    cv2.imshow("image", frame) # 結果の表示

    key = cv2.waitKey(1)
    if key == 27:
        break
