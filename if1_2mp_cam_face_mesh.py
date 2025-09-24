# mediapipe旧IFのfaceお試しプログラム
# 顔メッシュの特徴点の座標に○を表示します

import sys
sys.dont_write_bytecode = True
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(int(sys.argv[1])) # キャプチャ開始(複数カメラはIDを追加)

mp_face_mesh = mp.solutions.face_mesh # face trackの初期化
# メッシュは各パラメタを明示する：最大検出数、検出信頼度、追跡信頼度
face_mesh = mp_face_mesh.FaceMesh(max_num_faces = 1, min_detection_confidence = 0.5, min_tracking_confidence = 0.5)

# 赤い線にする特徴点 (詳細は以下のURLから)
posFacesParts = [7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246, 249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466,
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415]
# https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py

fCol = [(0, 0, 255), (0, 255, 0)]

while True:
    ret, frame = cap.read() # キャプチャ
    if not ret: # フレームの読み込みに失敗したらループ終了
        break
    cam_height, cam_width, _ = frame.shape # フレームサイズ取得(一回やればほんとうはいいのだけど)

    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # mediapipeに処理を渡す
    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            for i in range(468):
                x = int(face.landmark[i].x * cam_width)
                y = int(face.landmark[i].y * cam_height)
                if i in posFacesParts:
                    c = 0
                else:
                    c = 1
                cv2.circle(frame, (x, y), 3, fCol[c], thickness = 1)

    cv2.imshow("Frame pose", frame) # 円描画した結果の表示

    k = cv2.waitKey(1)
    if k == 27: break # escキーでプログラム終了
