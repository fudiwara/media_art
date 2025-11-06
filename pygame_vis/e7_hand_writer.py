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

drawing_canvas = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA) # お絵かき用のキャンバス
drawing_canvas.fill((255, 255, 255, 0)) # 透明な色で初期化

drawing_l, drawing_r = False, False # 描画モード中か
last_pos_l, last_pos_r = None, None # 一つ前のフレームの座標

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

    flag_l, flag_r = False, False # 左右でそれぞれグーになっているか
    num_victory = 0 # グーの数を数える変数
    if hand_landmarker_result: # 手検出されてたら各処理をする
        for i, hand in enumerate(hand_landmarker_result.hand_landmarks): # 各手に対するループ
            x = int(0.5 * (hand[0].x + hand[9].x) * WIDTH)
            y = int(0.5 * (hand[0].y + hand[9].y) * HEIGHT)

            hand_side = hand_landmarker_result.handedness[i][0].category_name
            hand_gesture = hand_landmarker_result.gestures[i][0].category_name
            if hand_side == "Right": # 左右属性に応じた処理
                if hand_gesture == "Closed_Fist":
                    flag_l = True # flipしているので逆で
                    pos_l = (x, y)
                elif hand_gesture == "Victory":
                    num_victory += 1
            elif hand_side == "Left":
                if hand_gesture == "Closed_Fist":
                    flag_r = True # flipしているので逆で
                    pos_r = (x, y)
                elif hand_gesture == "Victory":
                    num_victory += 1
            else:
                color = (0, 0, 0)
            
    if drawing_l: # お絵かき中で
        if flag_l: # グーなら線を描き続けて
            pygame.draw.line(drawing_canvas, (0, 255, 0), pos_l, last_pos_l, 9)
            last_pos_l = pos_l # 今の座標を次のフレームに
        else:
            drawing_l = False # グーじゃなくなったらお絵かき中フラグを解除
    else:
        if flag_l: # お絵かき中じゃない時に、グーになったら
            drawing_l = True # お絵かき中フラグを付けて
            last_pos_l = pos_l # 今の座標を次のフレームに

    # 後は右手で同じ
    if drawing_r:
        if flag_r:
            pygame.draw.line(drawing_canvas, (255, 0, 0), pos_r, last_pos_r, 9)
            last_pos_r = pos_r
        else:
            drawing_r = False
    else:
        if flag_r:
            drawing_r = True
            last_pos_r = pos_r
    
    if num_victory == 2: # グーを数える変数が2の時に
        drawing_canvas.fill((255, 255, 255, 0)) # 両手がチョキになったら履歴を消す

    screen.blit(drawing_canvas, (0, 0)) # カメラ映像の上に描画履歴を重ねて表示

    pygame.display.flip() # 画面の更新
    clock.tick(60) # FPSのために時計調整
pygame.quit()