import math
import pygame

pygame.init() # pygame全体の初期化
clock = pygame.time.Clock() # FPS制御用
WIDTH, HEIGHT = 800, 600 # ウィンドウサイズ 横x縦
screen = pygame.display.set_mode((WIDTH, HEIGHT)) # ウィンドウscreenを作る
pygame.display.set_caption("e1 Mouse Example") # タイトルバーの設定
screen.fill((100, 100, 100)) # 背景の塗りつぶし

pmouse_x, pmouse_y = 0, 0 # pマウス座標の初期化

def speed_to_color(mouse_pressed, x, y, px, py): # 移動量から色を計算する関数
    if mouse_pressed: # 左ボタンを押してたらスピードに合わせて色を決める
        dist_m = min(7 * math.dist((x, y), (px, py)), 250)
    else:
        dist_m = 0 # ボタンを押していない時は黒(0)にする
    return (int(dist_m), int(dist_m), int(dist_m))

running = True
while running: # メインループ
    for event in pygame.event.get(): # イベント取得のループ
        if event.type == pygame.QUIT:
            running = False

    mouse_x, mouse_y = pygame.mouse.get_pos() # マウスの位置情報取得
    color = speed_to_color(pygame.mouse.get_pressed()[0], mouse_x, mouse_y, pmouse_x, pmouse_y)
    pygame.draw.circle(screen, color, (mouse_x, mouse_y), 50) # 円の描画

    pmouse_x, pmouse_y = mouse_x, mouse_y # 次フレームのためにpマウスへ代入
    pygame.display.flip() # 画面の更新
    clock.tick(60) # FPSのために時計調整
pygame.quit()