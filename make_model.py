# codeing: utf-8

import argparse, glob, os, pickle
import numpy as np
from PIL import Image
from tqdm import tqdm

# CNN以外の分類方法で予測する場合
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.externals import joblib

# CNNによって分類予測する場合
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# オプション付
def make_args():
    parser = argparse.ArgumentParser(usage = 
            '''
            [概要]
                画像が好みかどうかを判定するモデルを作り評価する
            [出力]
                再現率(recall), 精度(precision), 正解率(accuracy)の出力, モデルを保存(CNNの場合は'cnn_model.json','cnn_model.hdf5', それ以外は''として出力).
            ''',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter, #default値を表示
            )
    parser.add_argument("-c", "--classifier", type=str, default='CNN', choices=['CNN', 'GPC', 'Logistic', 'SVC'], help="分類器の指定")
    parser.add_argument("-e", "--epoch", type=int, default=15, choices=range(10, 100), help="epoch数を指定")
    parser.add_argument("-t", "--tuning", type=bool, default=False, choices=[True, False], help="ハイパーパラメータをチューニングするかを指定")
    parser.add_argument("-n", "--trials", type=int, default=5, choices=range(1,51), help="チューニングする際の探索回数")
    parser.add_argument("-d", "--datamake", type=bool, default=False, choices=[True, False], help="データを作るか,pickleファイルから読み込むかを指定")
    parser.add_argument("-s", "--savedir", type=str, help="モデルに関するデータを保存するpathを指定")
    args = parser.parse_args()

    return args


# 画像データを読み込んで加工して訓練データ,テストデータを作る
def makeDatasets(like_imgs_flpath, notlike_imgs_flpath, clf):
    print('Dataset making.....')
    img_size = 100
    x_img, y_label = [], []

    # 好みの画像を読み込む
    print('Loading '+os.path.basename(like_imgs_flpath)+'.....')
    like_imgs = glob.glob(like_imgs_flpath+'/*')
    for imgpath in tqdm(like_imgs):
        img = Image.open(imgpath).convert("RGB")
        img = img.resize((img_size, img_size))
        rgb_data = np.asarray(img)
        if clf != 'CNN': rgb_data = rgb_data.reshape(-1)
        x_img.append(rgb_data)
        y_label.append(1)

    # 好みでない画像を読み込む
    print('Loading '+os.path.basename(notlike_imgs_flpath)+'.....')
    notlike_imgs = glob.glob(notlike_imgs_flpath+'/*')
    for imgpath in tqdm(notlike_imgs):
        img = Image.open(imgpath).convert("RGB")
        img = img.resize((img_size, img_size))
        rgb_data = np.asarray(img)
        if clf != 'CNN': rgb_data = rgb_data.reshape(-1)
        x_img.append(rgb_data)
        y_label.append(0)
    
    # 画素値の正規化
    x_img = np.array(x_img).astype('float32')/255.0

    # CNNの場合はlist(One_hot表現), GPC,Logistic,SVMはfloatのまま
    if clf == 'CNN':
        # One_hot表現へ変換
        y_label = np_utils.to_categorical(np.array(y_label), 2)

    # 訓練とテストに分割
    x_train, x_test, y_train, y_test = train_test_split(x_img, y_label, test_size=0.20)
    print('Finish making dataset!')

    return x_train, y_train, x_test, y_test


# 巨大なpickleをdumpする際の解決のためのクラス
class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size

def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))


# ハイパーパラメータチューニングのためのclass
class Objective_Classifir(object):
    def __init__(x_train, y_train, x_test, y_test, clf, epoch):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.clf = clf
        self.epoch = epoch

    def __call__(self, trial):
        # チューニングしたいパラメータの設定とモデルの構築
        if self.clf == 'CNN':
            learning_rate = trial.suggest_uniform('learning_rate', 0, 1.0)
            model = Sequential()
            model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
            model.add(Activation('relu'))
            model.add(Conv2D(32, (3, 3)))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            model.add(Conv2D(64, (3, 3), padding='same'))
            model.add(Activation('relu'))
            model.add(Conv2D(64, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            model.add(Flatten())
            model.add(Dense(512))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(4))
            model.add(Activation('softmax'))
            model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate), metrics=['accuracy'])
        
        elif self.clf == 'GPC':
            kernel_scale = trial.suggest_discrete_uniform('kernel_scale', 0.1, 5.0, 0.2)
            length_scale = trial.suggest_discrete_uniform('length_scale', 0.1, 5.0, 0.2)
            model = GaussianProcessClassifier(kernel=kernel_scale * RBF(length_scale=length_scale))

        elif self.clf == 'Logistic':
            C = trial.suggest_loguniform('C', 2**(-5), 2**(5))
            model = LogisticRegression(C=C,solver = 'newton-cg',class_weight='balanced')

        elif self.clf == 'SVC':
            C = trial.suggest_loguniform('C', 2**(-5), 2**(5))
            gamma = trial.suggest_loguniform('gamma', 2**(-10), 2**(3))
            model = SVC(C = C, gamma = gamma,class_weight='balanced')

        # モデルを学習
        if self.clf == 'CNN': model.fit(self.x_train, self.y_train, epochs=self.epoch)
        else: model.fit(self.x_train, self.y_train)

        # 学習したモデルの評価
        score = model.evaluate(self.x_test, self.y_test, verbose=0)

        return 1.0 - score[1]


# ハイパーパラメータをチューニングする
def getBestHyperParams(x_train, y_train, x_test, y_test, clf, epoch, trials):
    print('Hyperparams tuning.....', end='')
    study = optuna.creat_study()
    optuna.logging.disable_default_handler()
    study.optimize(Objective_Classifir(x_train, y_train, x_test, y_test, clf, epoch), n_trials=trials)
    print('Finish!')

    return study.best_params


# 学習過程の様子を可視化
def plotHistory(hist, save_dir):
    # 正解率の履歴をプロット
    plt.subplot(211)
    plt.plot(hist.history['acc'], "o-", label="train_acc")
    plt.plot(hist.history['val_acc'], "o-", label="test_acc")
    plt.title('model-accuracy,loss')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")

    # 損失の履歴をプロット
    plt.subplot(212)
    plt.plot(hist.history['loss'], "o-", label="train_loss")
    plt.plot(hist.history['val_loss'], "o-", label="test_loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc="lower right")
    plt.savefig(save_dir+'epoch_AccLoss.png')       # プロットしたグラフを画像として保存
    plt.show()


# CNNを適用
def classifyByCNN(x_train, y_train, x_test, y_test, hyper_params, save_dir, epoch):
    # モデルの構築
    model = Sequential()

    # (入力層)
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))

    # (中間層)
    model.add(Conv2D(32, (3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # (出力層)
    model.add(Dense(2))
    model.add(Activation('softmax'))

    #model.summary()

    # 学習過程の設定
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=hyper_params['learning_rate']), metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

    # 学習
    print('Now learning.....')
    hist = model.fit(x_train, y_train, epochs=epoch, validation_data=(x_test, y_test))
    print('Finish learning!')

    # epoch-acc, epoch-lossのグラフ結果をpngファイルに保存
    plotHistory(hist, save_dir)

    return model


# GPCを適用
def classifyByGPC(x_train, y_train, x_test, y_test, hyper_params):
    model = GaussianProcessClassifier(kernel=hyper_params['kernel_scale'] * RBF(length_scale=hyper_params['length_scale']))
    model.fit(x_train, y_train)

    return model


# Logisticを適用
def classifyByLogistic(x_train, y_train, x_test, y_test, hyper_params):
    model = LogisticRegression(C=hyper_params['C'],solver = 'newton-cg',class_weight='balanced')
    model.fit(x_train, y_train)

    return model


# SVCを適用
def classifyBySVC(x_train, y_train, x_test, y_test, hyper_params):
    model = SVC(C = hyper_params['C'], gamma = hyper_params['gamma'], class_weight='balanced', probability=True)
    model.fit(x_train, y_train)

    return model


# モデルの保存
def saveModel(model, clf, save_dir):
    print('Model saving.....', end='')
    if clf == 'CNN':
        json_string = model.to_json()
        open(save_dir+'{}_model.json'.format(clf), 'w').write(json_string)     # モデル
        model.save_weights(save_dir+'{}_weights.hdf5'.format(clf))             # 重み

    else:
        joblib.dump(model, save_dir+'{}_model.pickle'.format(clf))
    print('Finish!')


# 混同行列を生成する
def getConfusionMatrix(y_test, y_predict, clf):
    if clf == 'CNN':
        # 各ラベルに対しての確率表現になっているy_predictをOne_hot表現へ変換する
        for k in range(len(y_predict)):
            if y_predict[k][0] > y_predict[k][1]: real_predict = [1, 0]
            else: real_predict = [0, 1]
            y_predict[k] = real_predict
        
        # 混同行列を生成
        confusion_matrix = np.zeros((2, len(y_predict[0])))
        for k in range(len(y_predict)):
            for i in range(len(y_test[0])):
                for j in range(len(y_predict[0])):
                    confusion_matrix[i][j] += y_test[k][i]*y_predict[k][j]
    
    else:
        confusion_matrix = np.zeros((2, 2))
        for i in range(len(y_test)):
            for j in range(len(y_predict)):
                confusion_matrix[y_test[i]][y_predict[j]] += 1
    
    return confusion_matrix


# 混同行列から検出率(recall), 精度(precision), 正解率(accuracy)を算出
def getResult(confusion_matrix):
    true_posi = confusion_matrix[1][1]
    true_nega = confusion_matrix[0][0]
    false_posi = confusion_matrix[0][1]
    false_nega = confusion_matrix[1][0]
    recall = [float(true_nega)/(true_nega+false_posi), float(true_posi)/(true_posi+false_nega)]
    precision = [float(true_nega)/(true_nega+false_nega), float(true_posi)/(true_posi+false_posi)]
    accuracy = float(true_posi+true_nega)/(true_posi+true_nega+false_posi+false_nega)
    f_score = 2*precision*recall/(precision+recall)
        
    print('---------------------------------------------------------------')
    print('* Result *')
    print('Recall(not_like, like) :'+str(recall))
    print('Precision(not_like, like) :'+str(precision))
    print('Accuracy :'+str(accuracy))
    print('F-score :'+str(f_score))
    print('---------------------------------------------------------------')

    return recall, precision, accuracy, f_score


def main():
    args = make_args()
    print(args)
    clf = args.classifier
    print(clf)

    # 諸々のデータを保存するディレクトリ
    save_dir = args.savedir+"/model_"+clf+'/'
    if not os.path.exists(save_dir): os.mkdir(save_dir)

    # 訓練データ,テストデータを作るか既存のものを読み込むか
    if args.datamake:
        # 各クラスの画像フォルダのpath
        print("The PATH of the folder in which the images you like are saved:", end="")
        like_imgs_flpath = input()
        print("The PATH of the folder which the images you DON'T like are saved:", end="")
        notlike_imgs_flpath = input()

        # 作った訓練データ,テストデータをpickleに保存
        x_train, y_train, x_test, y_test = makeDatasets(like_imgs_flpath, notlike_imgs_flpath, clf)
        data = {}
        data['x_train'] = x_train
        data['y_train'] = y_train
        data['x_test'] = x_test
        data['y_test'] = y_test

        print('Dumping pickle.....', end='')
        pickle_dump(data, save_dir+'data.pickle')
        print('Finish!')

    else:
        # pickleから訓練データ,テストデータを読み込む
        print('Loading pickle.....', end='')
        data = pickle_load(save_dir+'data.pickle')
        print('Finish!')

        x_train = data['x_train']
        y_train = data['y_train']
        x_test = data['x_test']
        y_test = data['y_test']

    # ハイパーパラメータを取得
    if args.tuning:
        hyper_params = getBestHyperParams(x_train, y_train, x_test, y_test, clf, args.epoch, args.trials)
    else:
        if clf == 'CNN':
            hyper_params = {'learning_rate': 1e-2}
        elif clf == 'GPC':
            hyper_params = {'kernel_scale': 1.0, 'length_scale': 1.0}
        elif clf == 'Logistic':
            hyper_params = {'C': 10}
        elif clf == 'SVC':
            hyper_params = {'C':10, 'gamma':10}

    # モデルを作る
    print('Model making.....', end='')
    if clf == 'CNN':
        model = classifyByCNN(x_train, y_train, x_test, y_test, hyper_params, save_dir, args.epoch)
    elif clf == 'GPC':
        model = classifyByGPC(x_train, y_train, x_test, y_test, hyper_params)
    elif clf == 'Logistic':
        model = classifyByLogistic(x_train, y_train, x_test, y_test, hyper_params)
    elif clf == 'SVC':
        model = classifyBySVC(x_train, y_train, x_test, y_test, hyper_params)
    print('Finish!')
    
    # モデルをファイルに保存
    saveModel(model, clf, save_dir)

    # 予測
    y_predict = model.predict(x_test)

    # 予測結果から混同行列を生成, 評価
    print('Model evaluating.....', end='')
    confusion_matrix = getConfusionMatrix(y_test, y_predict, clf)
    recall, precision, accuracy, f_score = getResult(confusion_matrix)
    print('Finish!')

    # 結果をtxtファイルに保存
    result = ['recall: '+str(recall), 'precision: '+str(precision), 'accuracy: '+str(accuracy), 'f_score: '+str(f_score)]
    with open(save_dir+'result.txt', 'w') as f:
        f.write('\n'.join(map(str, result)))

    print('All Done!')


if __name__ == '__main__':
    main()