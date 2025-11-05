# mediapipe旧IFのhandお試しプログラム
# 各指の先端の位置に○を表示
# 例外処理をしていないので手が全て写っているという前提で安定動作する感じです(入っていないパーツは荒ぶるカモ)

import sys
sys.dont_write_bytecode = True
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(int(sys.argv[1])) # キャプチャ開始(複数カメラはIDを追加)

cw, ch = 640, 480
# cw, ch = 1280, 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cw)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ch)

mp_hands = mp.solutions.hands # handsの初期化
# 最大検出数、検出信頼度、追跡信頼度 (変数はこのままでいいかな？)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

posHandsIds = [4, 8, 12, 16, 20] # 各指の先端
# https://chuoling.github.io/mediapipe/solutions/hands.html

ret, frame = cap.read()
i_h, i_w, _ = frame.shape

tick_meter = cv2.TickMeter() # FPS表示用のタイマー
tick_meter.start() # 計測開始

while True:
    ret, frame = cap.read() # キャプチャ
    if not ret: # フレームの読み込みに失敗したらループ終了
        break
    
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # mediapipeに処理を渡す
    if results.multi_hand_landmarks:

        # 検出した手の数分繰り返し
        for h_id, hand_landmarks in enumerate(results.multi_hand_landmarks):

            hand_side = results.multi_handedness[h_id].classification[0].label # 左か右か
            x = int(hand_landmarks.landmark[0].x * i_w)
            y = int(hand_landmarks.landmark[0].y * i_h)
            cv2.putText(frame, hand_side, (x, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            if hand_side == "Right":
                col = (0, 255, 0)
            elif hand_side == "Left":
                col = (255, 0, 0)
            else:
                col = (0, 0, 0)

            for n in range(len(posHandsIds)): # 上記で設定したパーツ分ループする
                x = int(hand_landmarks.landmark[posHandsIds[n]].x * i_w)
                y = int(hand_landmarks.landmark[posHandsIds[n]].y * i_h)
            # for n in range(21): # 手の各特徴点に対してループする
            #     x = int(hand_landmarks.landmark[n].x * i_w)
            #     y = int(hand_landmarks.landmark[n].y * i_h)
                cv2.circle(frame, (x, y), 15, col, thickness = 3)
                # print(x, y) # デバッグ用

    tick_meter.stop() # 計測終了
    cv2.putText(frame, f"{tick_meter.getFPS():.1f}", (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    tick_meter.reset() # 次フレーム用に時計をリセット
    tick_meter.start() # 計測開始

    cv2.imshow("Frame pose", frame) # 手検出された関節へ円描画した結果の表示

    k = cv2.waitKey(1)
    if k == 27:
        break # escキーでプログラム終了
