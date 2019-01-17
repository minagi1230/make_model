# make_model
## make_model.py : 判定モデルを作るスクリプト
    input 
        オプション
        * -c, --classifier
            ... 分類手法(CNN, GPC, Logistic, SVC) (default値:CNN)

        * -e, --epoch
            ... エポック数(10~100) (default値:15) ※CNNでしか機能しない

        * -t, --tuning
            ... ハイパーパラメータをチューニングするかを指定(True, False) (default値: False)

        * -n, --trials
            ... チューニングする際の探索回数(1~50) (default値:5)

        * -d, --datamake(初回のみTrueで指定必須)
            ... データを学習用に加工したものを保存するか否か(True, False) (default値: False)

        * -s, --savedir(指定必須)
            ... モデル, モデルの性能情報に関するファイル保存するディレクトリ

        スクリプト起動後に入力を求められるもの
        * 好みである画像をまとめたフォルダのpath
        * 好みでない画像をまとめたフォルダのpath

    output
        CNNの場合
        * CNN_model.json, CNN_weights.hdf5
            ... モデルとその重み
        * data.pickle
            ... 学習に用いた加工したデータ(エポック数等を変えてまたモデルを作りたい時に手間を省くもの)
        * epoch_AccLoss.png
            ... 学習過程のグラフ
        * result.txt
            ... 生成されたモデルのrecall, precision, accuracy, f_scoreをまとめたもの

        CNN以外の場合
        * (分類機)_model.pickle
            ...モデル
        * data.pickle
            ... 学習に用いた加工したデータ(ハイパーパラメータをチューニングし直したりする際に手間を省くもの)

[補足]
* 大抵の場合,CNNが性能が良いのでオプション'-c'は特に指定しなくても良い.
* 出力されるファイルは,オプション'-s'で指定したディレクトリに生成されるフォルダ'model_(分類機)'に保存される.
* 最初にモデルを生成する場合はオプション'-d'をTrueで指定必須. 同じ分類機でパラメータを変えたモデルを別に生成したい際には指定しなくて良い.
* CNNモデルを作る場合,optimizerにAdamを指定している(255行目)が, val_accやval_lossが変化しないことがあるのでその時は代わりにSGDを使用(代わりに256行目のコメントアウトを外す)して欲しい.

## use_model.py : make_model.pyで生成した判定モデルを使うスクリプト
    input
        <dt>オプション</dt>
        * -c, --classifier
            ... 生成したモデルの分類手法(CNN, GPC, Logistic, SVC) (default値:CNN)

        * -t, --threshold
            ... 好みであるかの基準となる確率値(0.0~1.0) (default値: 0.5) ※CNNでしか機能しない

        * -s, --savedir(指定必須)
            ... モデル, モデルの性能情報に関するファイル保存されているディレクトリ

        * -r, --rm
            ... 好みでないと判定された画像を,破棄するか,別フォルダ'notlike_imgs'に保存するかを指定(default値: False(破棄しない))

        <dt>スクリプト起動後に入力を求められるもの</dt>
        * 判定したい画像をまとめたフォルダのpath

    output
        CNNの場合
        * like_imgs, notlike_imgs
            ... モデルによって分類されてまとめられた画像フォルダ
        * probability.txt
            ... 判定された各画像の,好みである確率をまとめたファイル

        CNN以外の場合
        * like_imgs, notlike_imgs
            ... モデルによって分類されてまとめられた画像フォルダ
