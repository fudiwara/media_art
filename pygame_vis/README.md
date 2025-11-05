表示機能をpygameで実装したプログラム

#### マウスだけのサンプルプログラム
##### [e1_mouse_circle.py](e1_mouse_circle.py)
カメラやMediaPipeは使っていません。

#### ポーズ検出
##### [e2_body_skeleton.py](e2_body_skeleton.py)
次のプログラムも実行時にはポーズ検出のモデルファイルを用意する必要があります。
[MediaPipeのポーズ推定のページ](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)から pose_landmarker_full.task を予めダウンロードしておいてください。

#### ポーズ検出の応用例
##### [e3_body_location.py](e3_body_location.py)
ポーズの横移動に対して、とある場所に来たら何か反応させたいというサンプルプログラムです。

#### 手検出
##### [e4_hand_detection.py](e4_hand_detection.py)
e7のプログラムまで手検出のモデルファイルを用意する必要があります。
[MediaPipeの手のジェスチャーのページ](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer)から gesture_recognizer.task を予めダウンロードしておいてください。
なお、このプログラムは手の検出のみをしていますが、次のプログラムからはハンドサインの認識もするので、全て手検出ではなくジェスチャー認識のモデルファイルに統一しています。

#### 手検出結果から代表の座標を得るサンプルプログラム
##### [e5_hand_position.py](e5_hand_position.py)
左右の手を代表する座標を1点ずつのみ得るというサンプルプログラムです。
何のことはなく手の付けねと中指の付けねの中間点を採用しているというだけです。
経験則ですが、なんだかんだと、この点が最も安定していて楽に得られる座標です。

#### ハンドサイン認識
##### [e6_hand_gestures.py](e6_hand_gestures.py)

#### ハンドサイン認識の応用例：お絵かきアプリ
##### [e7_hand_writer.py](e7_hand_writer.py)
グーをボタン押下に見立てて、左右の手でお絵かき的な事ができるサンプルプログラムです。