import pygame
import cv2 as cv
import numpy as np
import mediapipe as mp

pygame.init() # pygame全体の初期化
clock = pygame.time.Clock() # FPS制御用
WIDTH, HEIGHT = 800, 600 # ウィンドウサイズ 横x縦
screen = pygame.display.set_mode((WIDTH, HEIGHT)) # ウィンドウscreenを作る
pygame.display.set_caption("e3 body location") # タイトルバーの設定

options = mp.tasks.vision.PoseLandmarkerOptions( # ボディの検出器のオプション
    base_options = mp.tasks.BaseOptions(model_asset_path = "pose_landmarker_full.task"),
    num_poses = 1) # 検出できる人数 (最大人数は制御PCのスペックに依存)
landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options) # 検出器の初期化

cap = cv.VideoCapture(0) # キャプチャ開始 (カメラが複数ある場合は数値を変えてみる)
cw, ch = 640, 480 # 画面サイズを指定する
cap.set(cv.CAP_PROP_FRAME_WIDTH, cw)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, ch)

X_LEFT, X_RIGHT = 500, 600 # この場所にいることを見つけたいという範囲の左右座標

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

    flag_body_location = False # ボディがとある場所にいるかどうかというフラグ
    if pose_landmarker_result:
        for i, pose in enumerate(pose_landmarker_result.pose_landmarks): # 各人に対するループ
            x_center = int(0.5 * (pose[23].x + pose[24].x) * WIDTH)
            if X_LEFT < x_center < X_RIGHT:
                flag_body_location = True
    
    if flag_body_location:
        pygame.draw.circle(screen, (255, 0, 0), (75, 75), 70, 5) # いるよ！を示す円の描画

    pygame.display.flip() # 画面の更新
    clock.tick(60) # FPSのために時計調整
pygame.quit()