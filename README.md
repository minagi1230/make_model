# make_model
## Dockerfile : 以下の2つのスクリプトの実行環境構築のためのdockerファイル
    概要
    1. 以下の2つのスクリプトをCOPY
    2. 必要なモジュールをインストール
    
## make_model.py : 判定モデルを作るスクリプト
    ○ input 
        オプション
        * -e, --epoch
            ... エポック数(1~100) (default値:15)

        * -t, --tuning
            ... ハイパーパラメータをチューニングするか否か(True, False) (default値: False)

        * -n, --trials
            ... チューニングする際の探索回数(1~50) (default値:5)

        * -d, --datamake(初回のみTrueで指定必須)
            ... データを学習用に加工したものを保存するか否か(True, False) (default値: False)
            
        * -i1, --imagefl1(指定必須)
            ... 好みである画像フォルダ
            
        * -i2, --imagefl2(指定必須)
            ... 好みでない画像フォルダ

        * -s, --savedir(指定必須)
            ... モデル, モデルの性能情報に関するファイルを保存するディレクトリ


    ○ output(フォルダmodelに入っている)
        * cnn_model.json, cnn_weights.hdf5
            ... モデルとその重み
        * data.pickle
            ... 学習に用いた加工したデータ(エポック数等を変えてまたモデルを作りたい時に手間を省くもの)
        * epoch_accuracy.png, epoch_loss.png
            ... 学習過程のグラフ
        * result.txt
            ... 生成されたモデルのaccuracy, recall, precision, f1_scoreをまとめたもの


[補足]
* 最初にモデルを生成する場合はオプション'-d'をTrueで指定必須. 同じ分類機でパラメータを変えたモデルを別に生成したい際には指定しなくて良い.
* CNNモデルを作る場合,optimizerにAdamを指定している(255行目)が, 学習が停滞してval_accやval_lossが変化しないことがあり, その時は代わりにSGDを使用(代わりに256行目のコメントアウトを外す)することをオススメする. ただ, 時間はかかるもののAdamの方が学習は安定する.

## use_model.py : make_model.pyで生成した判定モデルを使うスクリプト
    ○ input
        オプション
        * -t, --threshold
            ... 好みであるかの基準となる確率値(0.0~1.0) (default値: 0.5)
            
        * -i, --imagefl
            ... 判定したい画像フォルダ

        * -s, --savedir(指定必須)
            ... モデル, モデルの性能情報に関するファイルが保存されているディレクトリ

        * -r, --rm
            ... 好みでないと判定された画像を,破棄するか,別フォルダ'notlike_imgs'に保存するかを指定(default値: False(破棄しない))
            

    ◯ output(フォルダimageに入っている)
        * like_imgs, notlike_imgs
            ... モデルによって分類されてまとめられた画像フォルダ
        * probability.txt
            ... 判定された各画像の,好みである確率をまとめたファイル
