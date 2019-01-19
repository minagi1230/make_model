# codeing: utf-8

import argparse, glob, sys, os, shutil
from tqdm import tqdm
import numpy as np
from PIL import Image

# CNNによって分類予測する場合
from keras.models import model_from_json


# オプション付
def make_args():
    parser = argparse.ArgumentParser(usage = 
            '''
            [概要]
                作ったモデルを用いて指定したフォルダ内の画像が好みかを判定する
            [出力]
                再現率(recall), 精度(precision), 正解率(accuracy)の出力, モデルを保存(CNNの場合は'cnn_model.json','cnn_model.hdf5', それ以外は''として出力).
            ''',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter, #default値を表示
            )
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="判定基準となる確率値")
    parser.add_argument("-i", "--imagefl", type=str, help="判定したい画像フォルダのpathを指定")
    parser.add_argument("-s", "--savedir", type=str, help="モデルに関するデータを保存したフォルダのpathを指定")
    parser.add_argument("-r", "--rm", type=bool, default=False, choices=[True, False], help='好みでないと判定された画像を別フォルダに保存するか削除するかを指定')
    args = parser.parse_args()

    return args


# 画像データの読み込み
def getXTest(folderpath):
    # ファイルの存在を確認
    if not os.path.exists(folderpath):
        print('The folder is NOT found.....')
        sys.exit()

    print('Images loading.....')
    img_size = 100
    x_img = []
    imgpaths = glob.glob(folderpath+'/*')
    for imgpath in tqdm(imgpaths):
        img = Image.open(imgpath).convert("RGB").resize((img_size, img_size))
        rgb_data = np.asarray(img)
        x_img.append(rgb_data)

    # 画素値の正規化
    x_test = np.array(x_img).astype('float32')/255

    return x_test


# 既存のモデルの読み込み
def loadModel(save_dir):
    # ファイルの存在を確認
    if not os.path.exists(save_dir+'/'):
        print("The 'save_dir' is NOT found.....")
        sys.exit()

    print('Model loading.....', end='')
    model_filename = save_dir+'/model/cnn_model.json'              # モデル
    weights_filename = save_dir+'/model/cnn_weights.hdf5'          # 重み
    json_string = open(model_filename).read()
    model = model_from_json(json_string)
    model.load_weights(weights_filename)
    print('Finish!')

    return model


# 画像移動(好みの画像と好みでない画像をそれぞれフォルダに分ける)
def moveImages(folderpath, save_dir, y_predict, threshold):
    like_imgs_path = save_dir+'/image/like_imgs'
    notlike_imgs_path = save_dir+'/image/notlike_imgs'
    if not os.path.exists(save_dir+'/image/'): os.mkdir(save_dir+'/image/')
    if not os.path.exists(like_imgs_path): os.mkdir(like_imgs_path)
    if not os.path.exists(notlike_imgs_path) and not rm: os.mkdir(notlike_imgs_path)
    probability_list = []
    print('Moving images.....')
    for i, imgpath in tqdm(enumerate(glob.glob(folderpath+'/*'))):
        # 各画像の確率をlistにまとめる
        data = []
        data.append(os.path.basename(imgpath))
        prob = y_predict[i][1]
        data.append(prob)
        probability_list.append(data)

        # 確率に応じて画像を移動
        if prob >= threshold: shutil.move(imgpath, like_imgs_path)
        else: shutil.move(imgpath, notlike_imgs_path)
    
    return probability_list

    
def main():
    args = make_args()

    # 閾値の確認
    threshold = args.threshold
    if threshold <= 0.0 or 1.0 <= threshold:
        print('Threshold Error!')
        sys.exit()

    # テストしたい画像フォルダを入力, 取得
    folderpath = args.i
    x_test = getXTest(folderpath)

    # モデルをload
    save_dir = args.savedir
    model = loadModel(save_dir)

    # 予測
    y_predict = model.predict(x_test)

    # 予測に基づいて画像を移動させる
    probability_list = moveImages(folderpath, save_dir, y_predict, threshold)
    if args.rm: os.remove(save_dir+'/image/notlike_imgs')

    # 各画像の確率をtxtファイルに保存
    with open(save_dir+'/image/probability.txt', 'w') as f:
        f.write('\n'.join(map(str, probability_list)))

    # 各画像の確率を標準出力
    for data_prob in probability_list:
        print(data_prob[0]+': '+str(data_prob[1]))

    print('All done!')
        

if __name__ == '__main__':
    main()
