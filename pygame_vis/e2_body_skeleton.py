import pygame
import cv2 as cv
import numpy as np
import mediapipe as mp

pygame.init() # pygame全体の初期化
clock = pygame.time.Clock() # FPS制御用
WIDTH, HEIGHT = 800, 600 # ウィンドウサイズ 横x縦
screen = pygame.display.set_mode((WIDTH, HEIGHT)) # ウィンドウscreenを作る
pygame.display.set_caption("e2 body skeleton") # タイトルバーの設定

options = mp.tasks.vision.PoseLandmarkerOptions( # ボディの検出器のオプション
    base_options = mp.tasks.BaseOptions(model_asset_path = "pose_landmarker_full.task"),
    num_poses = 1) # 検出できる人数 (最大人数は制御PCのスペックに依存)
landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options) # 検出器の初期化
body_line = [ [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10], [11, 12], [11, 23], [12, 24], [23, 24], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19], [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20], [23, 25], [25, 27], [27, 29], [27, 31], [29, 31], [24, 26], [26, 28], [28, 30], [28, 32], [30, 32] ]

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
    pose_landmarker_result = landmarker.detect(img_mp) # poseランドマーク検出

    if pose_landmarker_result:
        for i, pose in enumerate(pose_landmarker_result.pose_landmarks): # 各人に対するループ

            for connection in body_line: # 節のリストに対するループ
                start_idx, end_idx = connection # 始点と終点のランドマークのインデックス
                x1 = int(pose[start_idx].x * WIDTH) # 端点1の座標
                y1 = int(pose[start_idx].y * HEIGHT)
                x2 = int(pose[end_idx].x * WIDTH) # 端点2の座標
                y2 = int(pose[end_idx].y * HEIGHT)
                pygame.draw.line(screen, (0, 0, 255), (x1, y1), (x2, y2), 3)
            
            for p in pose: # 関節の各点に対するループ
                x = int(p.x * WIDTH)
                y = int(p.y * HEIGHT)
                pygame.draw.circle(screen, (0, 255, 0), (x, y), 9) # 円の描画

    pygame.display.flip() # 画面の更新
    clock.tick(60) # FPSのために時計調整
pygame.quit()