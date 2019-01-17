# codeing: utf-8

import argparse, glob, os, shutil
from tqdm import tqdm
import numpy as np
from PIL import Image

# CNN以外の分類方法で予測する場合
from sklearn.externals import joblib

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
    parser.add_argument("-c", "--classifier", type=str, default='CNN', choices=['CNN', 'GPC', 'Logistic', 'SVM'], help="用いた分類器の指定")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="判定基準となる確率値")
    parser.add_argument("-s", "--savedir", type=str, help="モデルに関するデータを保存したフォルダのpathを指定")
    parser.add_argument("-r", "--rm", type=bool, default=False, choices=[True, False], help='好みでないと判定された画像を別フォルダに保存するか削除するかを指定')
    args = parser.parse_args()

    return args


# 画像データの読み込み
def getXTest(folderpath, classifier):
    print('Images loading.....')
    img_size = 100
    x_img = []
    imgpaths = glob.glob(folderpath+'/*')
    for imgpath in tqdm(imgpaths):
        img = Image.open(imgpath).convert("RGB")
        img = img.resize((img_size, img_size))
        rgb_data = np.asarray(img)
        if classifier != 'CNN': rgb_data = rgb_data.reshape(-1)
        x_img.append(rgb_data)

    # 画素値の正規化
    x_test = np.array(x_img).astype('float32')/255

    return x_test


# 既存のモデルの読み込み
def loadModel(classifier, save_dir):
    print('Model loading.....', end='')
    if classifier == 'CNN':                                                               # 任意のモデル
        model_filename = save_dir+'/model_CNN/CNN_model.json'              # モデル
        weights_filename = save_dir+'/model_CNN/CNN_weights.hdf5'          # 重み
        json_string = open(model_filename).read()
        model = model_from_json(json_string)
        model.load_weights(weights_filename)
    else:
        model = joblib.load(save_dir+'/model_{}/{}_model.pickle'.format(classifier, classifier))
    print('Finish!')

    return model

    
def main():
    args = make_args()
    save_dir = args.savedir
    classifier = args.classifier
    if 0.0 <= args.threshold <= 1.0:
        # テストしたい画像フォルダを入力, 取得
        print('Please input the folder of images:')
        folderpath = input()
        if not os.path.exists(folderpath): print('The folder is NOT found.....')
        else:
            x_test = getXTest(folderpath, classifier)

            if not os.path.exists(save_dir+'/'): print("The 'saving_directory' is NOT found.....")
            else:
                # モデルをload
                model = loadModel(classifier, save_dir)

                # 予測
                y_predict = model.predict(x_test)


                # 画像移動(好みの画像と好みでない画像をそれぞれフォルダに分ける)
                if not os.path.exists(save_dir+'/like_imgs'): os.mkdir(save_dir+'/like_imgs')
                if not os.path.exists(save_dir+'/notlike_imgs'): os.mkdir(save_dir+'/notlike_imgs')
                probability_list = []
                print('Moving images.....')
                for i, imgpath in tqdm(enumerate(glob.glob(folderpath+'/*'))):
                    # 各画像の確率をlistにまとめる
                    data = []
                    data.append(os.path.basename(imgpath))
                    if classifier == 'CNN':
                        prob = y_predict[i][1]
                        data.append(prob)
                        probability_list.append(data)
                    else: prob = y_predict[i]

                    # 確率に応じて画像を移動
                    if prob >= args.threshold:
                        shutil.move(imgpath, save_dir+'/like_imgs')
                    else:
                        # 好みでない画像を破棄するか別フォルダにまとめるか
                        if args.rm: os.remove(imgpath)
                        else: shutil.move(imgpath, save_dir+'/notlike_imgs')
                
                if classifier == 'CNN':
                    # 各画像の確率をtxtファイルに保存
                    with open(save_dir+'/probability.txt', 'w') as f:
                        f.write('\n'.join(map(str, probability_list)))
                
                    # 各画像の確率を標準出力
                    for data_prob in probability_list:
                        print(data_prob[0]+': '+str(data_prob[1]))

                print('All done!')
    
    else: print('Threshold Error!')
        

if __name__ == '__main__':
    main()