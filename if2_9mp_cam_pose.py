# mediapipe新IFのposeお試しプログラム
# 関節のの場所に○を表示する
# 予め以下のURLから特徴点ファイルをダウンロードしておく必要がある
# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
# 新IFだと複数人検出ができる
# 新IFだとカメラ中心の距離情報も推定できる

import sys
sys.dont_write_bytecode = True

import cv2
import mediapipe as mp

cap = cv2.VideoCapture(int(sys.argv[1])) # キャプチャ開始(複数カメラはIDを追加)

options = mp.tasks.vision.PoseLandmarkerOptions( # 検出器のオプション
    base_options = mp.tasks.BaseOptions(model_asset_path = "models/pose_landmarker_full.task"),
    num_poses = 2) # 検出できる人数
landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options) # 検出器の初期化

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

    pose_landmarker_result = landmarker.detect(mp_image) # poseランドマーク検出

    if pose_landmarker_result:
        for i, pose in enumerate(pose_landmarker_result.pose_landmarks): # 各人に対するループ
            for j, p in enumerate(pose): # 関節の各点に対するループ
                x = int(p.x * i_w)
                y = int(p.y * i_h)
                cv2.circle(frame, (x, y), 11, (0, 255, 0), 3) # 緑色の円で関節を表示

    tick_meter.stop() # 計測終了
    fps = tick_meter.getFPS() # FPSの計算
    cv2.putText(frame, f"{fps:.1f}", (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    tick_meter.reset() # 次フレーム用に時計をリセット

    cv2.imshow("image", frame) # 結果の表示

    key = cv2.waitKey(1)
    if key == 27:
        break

