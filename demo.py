import time
import cv2
import numpy as np
from openvino.inference_engine import IECore, IENetwork
import matplotlib.pyplot as plt
import pyvirtualcam

def softmax(x):
    u = np.sum(np.exp(x))
    ans = []
    for i in x:
        ans.append(np.exp(i) / u)
    return ans

# Inference Engineコアオブジェクトの生成
ie = IECore()

# IRモデルファイルの読み込み
net_position = ie.read_network(model="./models/human-pose-estimation-0001.xml",  weights="./models/human-pose-estimation-0001.bin")
exec_net_position = ie.load_network(network=net_position, device_name='CPU', num_requests=1)

# モデルの読み込み（顔検出） 
net_face = ie.read_network(model='./models/face-detection-retail-0004.xml', weights='./models/face-detection-retail-0004.bin')
exec_net_face = ie.load_network(network=net_face, device_name='CPU', num_requests=1)

# モデルの読み込み（感情分類） 
net_emotion = ie.read_network(model='./models/emotions-recognition-retail-0003.xml', weights='./models/emotions-recognition-retail-0003.bin')
exec_net_emotion = ie.load_network(network=net_emotion, device_name='CPU', num_requests=1)

# 入出力blobの名前の取得、入力blobのシェイプの取得
input_blob_name  = list(net_position.input_info.keys())[0]
output_blob_name = list(net_position.outputs.keys())[0]
input_face_name  = list(net_face.input_info.keys())[0]
output_face_name = list(net_face.outputs.keys())[0]
input_emotion_name  = list(net_emotion.input_info.keys())[0]
output_emotion_name = list(net_emotion.outputs.keys())[0]

batch, channel, height, width = net_position.input_info[input_blob_name].tensor_desc.dims

# カメラ設定
cv2.namedWindow("video",cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(0)

while True:
	ret, img = cap.read()

	# 姿勢検出の設定
	img_origin = img.copy()
	back = np.zeros((img.shape[0], img.shape[1], 3))
	back += 255
	img = cv2.resize(img, (width, height)) #サイズ変更
	img = img.transpose((2, 0, 1)) #Hight, Width, Color -> Color, Hight, Width
	img = np.expand_dims(img, axis=0) # 次元合せ
	# PositionEncoderの計算
	res_position = exec_net_position.infer(inputs={input_blob_name: img})

	# 顔検出の設定
	img_face = img_origin.copy()
	img_face = cv2.resize(img_face, (300, 300)) #サイズ変更
	# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # OpenCV は色がGBR順なのでRGB順に並べ替える
	img_face = img_face.transpose((2, 0, 1)) #Hight, Width, Color -> Color, Hight, Width
	img_face = np.expand_dims(img_face, axis=0) # 次元合せ
	# FaceEncoderの計算
	res_face = exec_net_face.infer(inputs={input_face_name: img_face})
	res_face = res_face[output_face_name]
	res_face = np.squeeze(res_face)

	# 検出されたすべての顔領域に対して１つずつ処理 
	for detection in res_face:
		# 信頼値の取得 
		confidence = float(detection[2])

		# バウンディングボックス座標を入力画像のスケールに変換 
		xmin = int(detection[3] * img_origin.shape[1])
		ymin = int(detection[4] * img_origin.shape[0])
		xmax = int(detection[5] * img_origin.shape[1])
		ymax = int(detection[6] * img_origin.shape[0])

		# conf値が0.5より大きい場合のみ感情推論とバウンディングボックス表示 
		if confidence > 0.5:
			# 顔検出領域はカメラ範囲内に補正する。特にminは補正しないとエラーになる 
			if xmin < 0:
				xmin = 0
			if ymin < 0:
				ymin = 0
			if xmax > img_origin.shape[1]:
				xmax = img_origin.shape[1]
			if ymax > img_origin.shape[0]:
				ymax = img_origin.shape[0]

			# 顔領域のみ切り出し 
			img_face = img_origin[ ymin:ymax, xmin:xmax ]

			# 入力データフォーマットへ変換 
			img_face = cv2.resize(img_face, (64, 64))   # サイズ変更 
			img_face = img_face.transpose((2, 0, 1))    # HWC > CHW 
			img_face = np.expand_dims(img_face, axis=0) # 次元合せ 

			# 推論実行 
			res_emotion = exec_net_emotion.infer(inputs={input_emotion_name: img_face})

			# 出力から必要なデータのみ取り出し 
			res_emotion = res_emotion[output_emotion_name]
			res_emotion = np.squeeze(res_emotion) #不要な次元の削減 

			# 出力値が最大のインデックスを得る 
			index_max = np.argmax(res_emotion)
			probability = softmax(res_emotion)
			if (max(probability) < 0.25):
				index_max = 0

			# 各感情の文字列をリスト化 
			list_emotion = ['neutral', 'happy', 'sad', 'surprise', 'anger']
			# 各感情を色指定
			list_emotion_color = [(0,0,0), (0,255,255), (255,255,0), (0,128,0), (0,0,255)]

			# 文字列描画 
			cv2.putText(back, list_emotion[index_max], (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, list_emotion_color[index_max], 4)
 
			# バウンディングボックス表示 
			#cv2.rectangle(img_origin, (xmin, ymin), (xmax, ymax), color=(240, 180, 0), thickness=3)
			cv2.circle(back, ((xmin+xmax)//2, (ymin+ymax)//2), (xmax-xmin)//2, list_emotion_color[index_max], thickness=-1, lineType=cv2.LINE_AA)

			# 姿勢推定
			neck = res_position['Mconv7_stage2_L2'][0][1]
			#肩
			right_sholder = res_position['Mconv7_stage2_L2'][0][2]
			index_rs = np.unravel_index(np.argmax(right_sholder), right_sholder.shape)

			left_sholder = res_position['Mconv7_stage2_L2'][0][5]
			index_ls = np.unravel_index(np.argmax(left_sholder), left_sholder.shape)

			if (right_sholder[index_rs[0]][index_rs[1]] > 0.1) and (left_sholder[index_ls[0]][index_ls[1]] > 0.1):
				pos = np.array(((int(index_rs[1] *(img_origin.shape[1] / 57.0)),int(index_rs[0] * (img_origin.shape[0]) / 32.0)), (int(index_ls[1] *(img_origin.shape[1] / 57.0)),int(index_ls[0] * (img_origin.shape[0]) / 32.0)), (int(index_ls[1] *(img_origin.shape[1] / 57.0) +20),img_origin.shape[0]), (int(index_rs[1] *(img_origin.shape[1] / 57.0)-20),img_origin.shape[0])))
				cv2.drawMarker(back ,(int(index_rs[1] *(img_origin.shape[1] / 57.0)), int(index_rs[0] * (img_origin.shape[0]) / 32.0)),(255, 0, 0), markerType=cv2.MARKER_STAR, markerSize=10)
				cv2.drawMarker(back ,(int(index_ls[1] *(img_origin.shape[1] / 57.0)), int(index_ls[0] * (img_origin.shape[0]) / 32.0)),(255, 0, 0), markerType=cv2.MARKER_STAR, markerSize=10)
				# 多角形描写（首とか使って滑らかにしたい）
				cv2.fillConvexPoly(back, pos, list_emotion_color[index_max])

				#腕の処理
				arm1 = res_position['Mconv7_stage2_L2'][0][4]
				index_a1 = np.unravel_index(np.argmax(arm1), arm1.shape)

				arm2 = res_position['Mconv7_stage2_L2'][0][7]
				index_a2 = np.unravel_index(np.argmax(arm2), arm1.shape)
				
				arm_width = (xmax - xmin) // 2
				arm1_pos = [int(index_a1[1] *(img_origin.shape[1] / 57.0)), int(index_a1[0] *(img_origin.shape[0] / 32.0))]
				
				arm2_pos = [int(index_a2[1] *(img_origin.shape[1] / 57.0)), int(index_a2[0] *(img_origin.shape[0] / 32.0))]

				face_center = (xmin+xmax)//2
				if ((arm1_pos[0] <= face_center and arm2_pos[0] <= face_center) or (arm1_pos[0] > face_center and arm2_pos[0] > face_center)):
					drow_arm = np.argmax(np.array(arm1[index_a1[0]][index_a1[1]], arm2[index_a2[0]][index_a2[1]]))
					drow_pos = [arm1_pos, arm2_pos][drow_arm]
					if(max(arm1[index_a1[0]][index_a1[1]], arm2[index_a2[0]][index_a2[1]]) > 0.05):
						cv2.rectangle(back, (max(drow_pos[0] - arm_width // 2,0), drow_pos[1]),(min(drow_pos[0] + arm_width // 2, img_origin.shape[1] - 1), img_origin.shape[0] - 1), list_emotion_color[index_max], thickness=-1)
				else:
					if(arm1[index_a1[0]][index_a1[1]] > 0.05):
						cv2.rectangle(back, (max(arm1_pos[0] - arm_width // 2,0), arm1_pos[1]),(min(arm1_pos[0] + arm_width // 2, img_origin.shape[1] - 1), img_origin.shape[0] - 1), list_emotion_color[index_max], thickness=-1)
					if(arm2[index_a2[0]][index_a2[1]] > 0.05):
						cv2.rectangle(back, (max(arm2_pos[0] - arm_width // 2,0), arm2_pos[1]),(min(arm2_pos[0] + arm_width // 2, img_origin.shape[1] - 1), img_origin.shape[0] - 1), list_emotion_color[index_max], thickness=-1)
						
				# 1つの顔で終了 
ret, img = cap.read()
with pyvirtualcam.Camera(width=img.shape[1], height=img.shape[0], fps=30.0) as cam:
	while True:
		ret, img = cap.read()

		# 姿勢検出の設定
		img_origin = img.copy()
		back = np.zeros((img.shape[0], img.shape[1], 3))
		back += 255
		img = cv2.resize(img, (width, height)) #サイズ変更
		img = img.transpose((2, 0, 1)) #Hight, Width, Color -> Color, Hight, Width
		img = np.expand_dims(img, axis=0) # 次元合せ
		# PositionEncoderの計算
		res_position = exec_net_position.infer(inputs={input_blob_name: img})

		# 顔検出の設定
		img_face = img_origin.copy()
		img_face = cv2.resize(img_face, (300, 300)) #サイズ変更
		# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # OpenCV は色がGBR順なのでRGB順に並べ替える
		img_face = img_face.transpose((2, 0, 1)) #Hight, Width, Color -> Color, Hight, Width
		img_face = np.expand_dims(img_face, axis=0) # 次元合せ
		# FaceEncoderの計算
		res_face = exec_net_face.infer(inputs={input_face_name: img_face})
		res_face = res_face[output_face_name]
		res_face = np.squeeze(res_face)

		# 検出されたすべての顔領域に対して１つずつ処理 
		for detection in res_face:
			# 信頼値の取得 
			confidence = float(detection[2])

			# バウンディングボックス座標を入力画像のスケールに変換 
			xmin = int(detection[3] * img_origin.shape[1])
			ymin = int(detection[4] * img_origin.shape[0])
			xmax = int(detection[5] * img_origin.shape[1])
			ymax = int(detection[6] * img_origin.shape[0])

			# conf値が0.5より大きい場合のみ感情推論とバウンディングボックス表示 
			if confidence > 0.5:
				# 顔検出領域はカメラ範囲内に補正する。特にminは補正しないとエラーになる 
				if xmin < 0:
					xmin = 0
				if ymin < 0:
					ymin = 0
				if xmax > img_origin.shape[1]:
					xmax = img_origin.shape[1]
				if ymax > img_origin.shape[0]:
					ymax = img_origin.shape[0]

				# 顔領域のみ切り出し 
				img_face = img_origin[ ymin:ymax, xmin:xmax ]

				# 入力データフォーマットへ変換 
				img_face = cv2.resize(img_face, (64, 64))   # サイズ変更 
				img_face = img_face.transpose((2, 0, 1))    # HWC > CHW 
				img_face = np.expand_dims(img_face, axis=0) # 次元合せ 

				# 推論実行 
				res_emotion = exec_net_emotion.infer(inputs={input_emotion_name: img_face})

				# 出力から必要なデータのみ取り出し 
				res_emotion = res_emotion[output_emotion_name]
				res_emotion = np.squeeze(res_emotion) #不要な次元の削減 

				# 出力値が最大のインデックスを得る 
				index_max = np.argmax(res_emotion)

				# 各感情の文字列をリスト化 
				list_emotion = ['neutral', 'happy', 'sad', 'surprise', 'anger']
				# 各感情を色指定
				list_emotion_color = [(0,0,0), (0,255,255), (255,255,0), (0,128,0), (0,0,255)]

				# 文字列描画 
				cv2.putText(back, list_emotion[index_max], (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, list_emotion_color[index_max], 4)
	
				# バウンディングボックス表示 
				#cv2.rectangle(img_origin, (xmin, ymin), (xmax, ymax), color=(240, 180, 0), thickness=3)
				cv2.circle(back, ((xmin+xmax)//2, (ymin+ymax)//2), (xmax-xmin)//2, list_emotion_color[index_max], thickness=-1, lineType=cv2.LINE_AA)

				# 姿勢推定
				neck = res_position['Mconv7_stage2_L2'][0][1]

				# 右肩
				right_sholder = res_position['Mconv7_stage2_L2'][0][2]
				index_rs = np.unravel_index(np.argmax(right_sholder), right_sholder.shape)

				# 左肩
				left_sholder = res_position['Mconv7_stage2_L2'][0][5]
				index_ls = np.unravel_index(np.argmax(left_sholder), left_sholder.shape)

				if (right_sholder[index_rs[0]][index_rs[1]] > 0.1) and (left_sholder[index_ls[0]][index_ls[1]] > 0.1):
					pos = np.array(((int(index_rs[1] *(img_origin.shape[1] / 57.0)),int(index_rs[0] * (img_origin.shape[0]) / 32.0)), (int(index_ls[1] *(img_origin.shape[1] / 57.0)),int(index_ls[0] * (img_origin.shape[0]) / 32.0)), (int(index_ls[1] *(img_origin.shape[1] / 57.0) +20),img_origin.shape[0]), (int(index_rs[1] *(img_origin.shape[1] / 57.0)-20),img_origin.shape[0])))
					# 多角形描写（首とか使って滑らかにしたい）
					cv2.fillConvexPoly(back, pos, list_emotion_color[index_max])

					# 腕の処理
					arm1 = res_position['Mconv7_stage2_L2'][0][4]
					index_a1 = np.unravel_index(np.argmax(arm1), arm1.shape)

					arm2 = res_position['Mconv7_stage2_L2'][0][7]
					index_a2 = np.unravel_index(np.argmax(arm2), arm1.shape)
					
					arm_width = (xmax - xmin) // 2
					arm1_pos = [int(index_a1[1] *(img_origin.shape[1] / 57.0)), int(index_a1[0] *(img_origin.shape[0] / 32.0))]
					
					arm2_pos = [int(index_a2[1] *(img_origin.shape[1] / 57.0)), int(index_a2[0] *(img_origin.shape[0] / 32.0))]

					face_center = (xmin+xmax)//2
					if ((arm1_pos[0] <= face_center and arm2_pos[0] <= face_center) or (arm1_pos[0] > face_center and arm2_pos[0] > face_center)):
						drow_arm = np.argmax(np.array(arm1[index_a1[0]][index_a1[1]], arm2[index_a2[0]][index_a2[1]]))
						drow_pos = [arm1_pos, arm2_pos][drow_arm]
						if(max(arm1[index_a1[0]][index_a1[1]], arm2[index_a2[0]][index_a2[1]]) > 0.05):
							cv2.rectangle(back, (max(drow_pos[0] - arm_width // 2,0), drow_pos[1]),(min(drow_pos[0] + arm_width // 2, img_origin.shape[1] - 1), img_origin.shape[0] - 1), list_emotion_color[index_max], thickness=-1)
					else:
						if(arm1[index_a1[0]][index_a1[1]] > 0.05):
							cv2.rectangle(back, (max(arm1_pos[0] - arm_width // 2,0), arm1_pos[1]),(min(arm1_pos[0] + arm_width // 2, img_origin.shape[1] - 1), img_origin.shape[0] - 1), list_emotion_color[index_max], thickness=-1)
						if(arm2[index_a2[0]][index_a2[1]] > 0.05):
							cv2.rectangle(back, (max(arm2_pos[0] - arm_width // 2,0), arm2_pos[1]),(min(arm2_pos[0] + arm_width // 2, img_origin.shape[1] - 1), img_origin.shape[0] - 1), list_emotion_color[index_max], thickness=-1)
							
					# 1つの顔で終了 
				break

		cv2.imshow("video", back)
		# sendメソッドが使えるように，キャスト
		back_int = back.astype(np.uint8)
		# 仮想カメラに合わせてBGR->RGBに
		back_int = cv2.cvtColor(back_int, cv2.COLOR_BGR2RGB)
		# 画像を仮想カメラに流す
		cam.send(back_int)
		# 次のフレームまで待機する
		cam.sleep_until_next_frame()
		# 終了判定
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
cv2.destroyAllWindows()
cap.release()
cv2.waitKey(0)
'''
nose i=0
neck i=1
right sholder i=2
left sholder i=5 
hand_necks i=4, 7
eyes i=16, 17 
'''