# Pictogrammer

もしリモート会議で顔を出すことを求められたとき，このツールを使うことでピクトグラマーに変装することができます！

When you are asked to show your face in a remote meeting, you can use this tool to show yourself as a pictogrammer!

![2021-06-20-15-22-05_Trim](https://user-images.githubusercontent.com/56689497/122668562-ab87e480-d1f3-11eb-8338-deb70cedd482.gif)

# 準備
## OPENVINOのインストール
本ツールは，Intel社のOpenVINOを用いています．まず，下記URLから，OpenVINO toolkitを導入してください．実装環境では，OpenVINO 2021.3 ver.を使用しています．
[OpenVINO公式HP](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html#configure-model-optimizer)

## モジュールのインストール
使用するモジュールのインストールを行います．
```
pip install numpy
pip install openvino-python
pip install opencv-python
pip install pyvirtualcam
```
を実行してください．

## 仮想カメラのインストール
本ツールはOBS Studioの仮想環境を想定して制作しています．そのため，OBS Studioのインストール，仮想カメラのインストールが必要です．
次のURLより，ダウンロードを行ってください．
[参考にしたNote](https://note.com/shitaper/n/n0e5154f6d786)

# 使用方法
1. Zoom等オンライン会議ツールで，ビデオの設定から仮想カメラ(OBS Virtual Camera)を選択する．
2. `demo.py`を実行する．
3. 数秒後にピクトグラムが表示されます．
