import pygame
import cv2 as cv
import numpy as np
import mediapipe as mp

pygame.init() # pygame全体の初期化
clock = pygame.time.Clock() # FPS制御用
WIDTH, HEIGHT = 800, 600 # ウィンドウサイズ 横x縦
screen = pygame.display.set_mode((WIDTH, HEIGHT)) # ウィンドウscreenを作る
pygame.display.set_caption("e4 hand detection") # タイトルバーの設定

options = mp.tasks.vision.GestureRecognizerOptions( # 手の検出器のオプション
    base_options = mp.tasks.BaseOptions(model_asset_path = "gesture_recognizer.task"), # 手ジェスチャー用モデル
    num_hands = 2) # 検出できる手の数
hand_detection = mp.tasks.vision.GestureRecognizer.create_from_options(options)

cap = cv.VideoCapture(0) # キャプチャ開始 (カメラが複数ある場合は数値を変えてみる)
cw, ch = 640, 480 # 画面サイズを指定する
cap.set(cv.CAP_PROP_FRAME_WIDTH, cw)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, ch)

running = True
while running: # メインループ
    for event in pygame.event.get(): # イベント取得のループ
        if event.type == pygame.QUIT:
            running = False

    ret, frame = cap.read() # カメラからのキャプチャ
    if not ret:
        break # キャプチャが失敗したらメインループを終了する
    
    frame = cv.flip(frame, 1) # 左右方向を反転
    img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB) # 色空間を変更する
    img_pg = pygame.surfarray.make_surface(np.swapaxes(img_rgb, 0, 1)) # pygame用に画像変換
    img_disp = pygame.transform.scale(img_pg, (WIDTH, HEIGHT)) # 表示用にサイズ調整
    screen.blit(img_disp, (0, 0)) # 背景としてキャプチャ結果を描画

    img_qvga = cv.resize(img_rgb, None, fx=0.5, fy=0.5) # 処理高速化のために1/4サイズにする
    img_mp = mp.Image(image_format = mp.ImageFormat.SRGB, data = img_qvga) # MediaPipe用に変換
    hand_landmarker_result = hand_detection.recognize(img_mp) # 手の検出

    if hand_landmarker_result: # 手検出されてたら各処理をする
        for i, hand in enumerate(hand_landmarker_result.hand_landmarks): # 各手に対するループ
            hand_side = hand_landmarker_result.handedness[i][0].category_name
            if hand_side == "Right": # 左右属性に応じて色をつける
                color = (0, 255, 0) # 事前にflipしているので反転した色が付く
            elif hand_side == "Left":
                color = (255, 0, 0) # 事前にflipしているので反転した色が付く
            else:
                color = (0, 0, 0)

            for p in hand: # 手のランドマークの各点に対してループする
                x = int(p.x * WIDTH)
                y = int(p.y * HEIGHT)
                pygame.draw.circle(screen, color, (x, y), 5, 2) # 円の描画

    pygame.display.flip() # 画面の更新
    clock.tick(60) # FPSのために時計調整
pygame.quit()