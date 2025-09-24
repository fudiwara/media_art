# mediapipe旧IFのhandお試しプログラム
# 各指の先端の位置に○を表示します
# 例外処理をしていないので手が全て写っているという前提で安定動作する感じです(入っていないパーツは荒ぶるカモ)

import sys
sys.dont_write_bytecode = True
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(int(sys.argv[1])) # キャプチャ開始(複数カメラはIDを追加)

mp_hands = mp.solutions.hands # handsの初期化
# 最大検出数、検出信頼度、追跡信頼度 (変数はこのままでいいかな？)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

posHandsIds = [4, 8, 12, 16, 20] # 各指の先端
# https://google.github.io/mediapipe/solutions/hands

handsCol = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]

while True:
    ret, frame = cap.read() # キャプチャ
    if not ret: # フレームの読み込みに失敗したらループ終了
        break
    cam_height, cam_width, _ = frame.shape # フレームサイズ取得(一回やればほんとうはいいのだけど)

    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # mediapipeに処理を渡す
    if results.multi_hand_landmarks:

        # 検出した手の数分繰り返し
        for h_id, hand_landmarks in enumerate(results.multi_hand_landmarks):

            for n in range(len(posHandsIds)): # 上記で設定したパーツ分ループする
                x = int(hand_landmarks.landmark[posHandsIds[n]].x * cam_width)
                y = int(hand_landmarks.landmark[posHandsIds[n]].y * cam_height)
            # for n in range(21): # 手の各特徴点に対してループする
            #     x = int(hand_landmarks.landmark[n].x * cam_width)
            #     y = int(hand_landmarks.landmark[n].y * cam_height)
                cv2.circle(frame, (x, y), 15, handsCol[h_id], thickness = 5)
                # print(x, y) # デバッグ用

    cv2.imshow("Frame pose", frame) # 手検出された関節へ円描画した結果の表示

    k = cv2.waitKey(1)
    if k == 27:
        break # escキーでプログラム終了
