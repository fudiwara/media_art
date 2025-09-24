# mediapipe旧IFのposeお試しプログラム
# 鼻、左手、右手、左足、右足の場所に○を表示
# 例外処理をしていないので体全体が全て写っているという前提で安定動作する感じです(入っていないパーツは荒ぶるカモ)
# 旧IFの場合は検出可能人数は一人

import sys
sys.dont_write_bytecode = True
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(int(sys.argv[1])) # キャプチャ開始(複数カメラはIDを追加)

cw, ch = 640, 480
# cw, ch = 1280, 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cw)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ch)

mp_pose = mp.solutions.pose # ポーズの初期化(変数はこのままでいいかな？)
pose = mp_pose.Pose(min_detection_confidence = 0.7, min_tracking_confidence = 0.5)

posName = [0, 19, 20, 31, 32] # 鼻、左手、右手、左足、右足
# https://google.github.io/mediapipe/solutions/pose.html

ret, frame = cap.read()
i_h, i_w, _ = frame.shape

tick_meter = cv2.TickMeter() # FPS表示用のタイマー
tick_meter.start() # 計測開始

while True:
    ret, frame = cap.read() # キャプチャ
    if not ret: # フレームの読み込みに失敗したらループ終了
        break

    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # mediapipeに処理を渡す
    if results.pose_landmarks is None: continue # 未検出なら再ループ

    pl = results.pose_landmarks
    for n in range(len(posName)): # 上記で設定したパーツ分ループする
        x = int(pl.landmark[posName[n]].x * i_w)
        y = int(pl.landmark[posName[n]].y * i_h)
        cv2.circle(frame, (x, y), 15, (0, 255, 0), thickness = 3)
        # print(x, y) # デバッグ用

    tick_meter.stop() # 計測終了
    cv2.putText(frame, f"{tick_meter.getFPS():.1f}", (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    tick_meter.reset() # 次フレーム用に時計をリセット
    tick_meter.start() # 計測開始

    cv2.imshow("Frame pose", frame) # 人検出結果の各関節へ円描画

    k = cv2.waitKey(1)
    if k == 27:
        break # escキーでプログラム終了
