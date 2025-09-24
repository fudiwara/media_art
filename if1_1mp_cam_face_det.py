# mediapipe旧IFのface_detectionお試しプログラム
# 顔の矩形と特徴点の座標を表示します

import sys
sys.dont_write_bytecode = True
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(int(sys.argv[1])) # キャプチャ開始(複数カメラはIDを追加)

cw, ch = 640, 480
# cw, ch = 1280, 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cw)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ch)

mp_face_detection = mp.solutions.face_detection # model_selectionは定義できないもよう
# face_detection = mp_face_detection.FaceDetection(model_selection = 1, min_detection_confidence = 0.5)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence = 0.5)

tm = cv2.TickMeter()
tm.start()

while True:
    ret, frame = cap.read() # キャプチャ
    if not ret: # フレームの読み込みに失敗したらループ終了
        break
    cam_height, cam_width, _ = frame.shape # フレームサイズ取得(一回やればほんとうはいいのだけど)

    results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # mediapipeに処理を渡す
    if results.detections:
        for i in range(len(results.detections)):
            # 顔領域の検出結果描画
            b = results.detections[i].location_data.relative_bounding_box
            x0 = int(b.xmin * cam_width)
            y0 = int(b.ymin * cam_height)
            x1 = int((b.xmin + b.width) * cam_width)
            y1 = int((b.ymin + b.height) * cam_height)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 3)

            # 顔特徴点の検出結果描画
            k = results.detections[i].location_data.relative_keypoints
            for j in range(len(k)):
                kx = int(k[j].x * cam_width)
                ky = int(k[j].y * cam_height)
                cv2.circle(frame, (kx, ky), 1, (0, 255, 0))
            # print(len(k))

    tm.stop()
    disp_fps = f"{1000 / tm.getTimeMilli():.1f}"
    tm.reset()
    tm.start()
    cv2.putText(frame, disp_fps, (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow("Frame pose", frame) # 顔検出した結果の表示

    k = cv2.waitKey(1)
    if k == 27:
        break # escキーでプログラム終了
