# 演習問題

このリポジトリは，共立出版社が出版する「マテリアルズインフォマティクス」の演習問題1~22をまとめている．
その内，演習16と17は，現在訓練済みモデルのダウンロードサービスのメンテナンスにより実行できない状態になってる．サービス開始後演習16と17を追加する．

## 実行環境

実行環境の構築に関しては，我々は[conda](https://docs.conda.io/en/latest/miniconda.html)の利用をおすすめします．
condaを利用した場合は，[XenonPy installation](https://xenonpy.readthedocs.io/en/latest/installation.html#using-conda-and-pip)を参考してください．

独自で実行環境を構築する場合，以下の条件を満たすこと．

* Python >= 3.7
* Pymatgen >= 2020.10.9
* rdkit == 2020.09
* xenonpy >= 0.6
* Pytorch >= 1.7.0

## 演習用データの獲得

これらの演習を実行する前に，下記のデータセットを用意してください．

* `retrieve_materials_project.ipynb`を実行して，Materials Projectから無機結晶データをダウンロード．
* [In house data](https://github.com/yoshida-lab/XenonPy/releases/download/v0.6.5/data.zip)をダウンロードして，README.mdと同じフォルダに解凍してください．

フォルダのストラクチャーのイメージ：
```
data
  |- QC_AC_data.pd.xz
  |- mp_ids.txt
  |- ...
output
  |- <演習中の出力>
  |- ...
exercise_2-10.ipynb
exercise...
```

## 目次

### 1.3

* 演習1）[XenonPy installation](https://xenonpy.readthedocs.io/en/latest/installation.html#using-conda-and-pip)を参考にXenonPyをインストールせよ．

### 1.3.1

Notebook: [Exercise 2~10](exercise_2-10.ipynb)

* 演習2）配布したSMILES形式の化学構造をMOLオブジェクトという形式に変換し，化学構造を描画せよ．

* 演習3）例題の化学構造の内，GetSubstructMatchを用いてベンゼン’c1ccccc1’にマッチする原子のインデックスを取得し，RDKitのモジュールを使用して，ベンゼンをカラーハイライトして化学構造を図示せよ．

* 演習4）アスピリンCC(=O)Oc1ccccc1C(=O)OのECFPフィンガープリント（Morganフィンガープリント）を計算せよ．ビット数をB=2048，半径をR=2とする．

* 演習5）演習4の各ビットの部分構造を取り出し，化学構造を描画せよ．

* 演習6）カウント型のECFPフィンガープリントとアトムペアフィンガープリントを計算し，それぞれのベクトルの数値を可視化せよ．

* 演習7）分子動力学シミュレーションのサンプルデータを用いて，モノマー構造のECFPフィンガープリント記述子から熱伝導率を予測するモデルを構築せよ．ここでは，ランダムフォレスト回帰とニューラルネットワークを用いる．また，訓練データの数やフィンガープリントの種類等を変更し，予測精度の変化を調べて解析結果をまとめよ．

* 演習8）ポリエチレンCCC(=O)Oの単量体，二量体，三量体のSMILES文字列を生成し．ECFP記述子を計算せよ（B=2048,R=2）．

* 演習9）演習8の各ビットの部分構造を描画し，比較せよ．

* 演習10）サンプルデータの10化合物のRDKitの2次元記述子とmordredの2次元記述子を計算せよ．

### 1.3.2

Notebook: [Exercise 11](exercise_11.ipynb)

* 演習11）XenonPy の 58 種類の元素特徴量を抽出せよ．

Notebook: [Exercise 12](exercise_12.ipynb)

* 演習12）サンプルデータの化学組成の記述子を計算せよ．

### 1.4.1

Notebook: [Exercise 13](exercise_13.ipynb)

* 演習13）分子動力学シミュレーションのサンプルデータにある300種類のポリマーの構造物性相関データを用いて，密度，定圧熱容量，熱伝導率，線膨張係数をの予測モデルを構築せよ．以下の説明やサンプルコードを参考にモデルの作成方法を自ら工夫し，その結果を考察せよ．

### 1.4.2

Notebook: [Exercise 14](exercise_14.ipynb)

* 演習14）以下の説明とサンプルコードを参考にして，Materials Project に収録されているデータを用いて，化学組成や結晶構造から形成エネルギー，バンドギャップ，密度を予測するモデルを構築せよ．

### 1.4.3

Notebook: [Exercise 15](exercise_15.ipynb)

* 演習15）以下の説明とサンプルコードを参考に，任意の化学組成が形成する三つの構造クラス（準結晶・近似結晶・通常の周期結晶）を判別するモデルを構築せよ（3 クラス分類問題）．

### 1.5.2

Notebook: [修正中](/)

* 演習16）API を用いて，XenonPy.MDL から組成から形成エネルギーを予測するモデル集合を抽出せよ．

* 演習17）同じく，XenonPy.MDL からポリマーの繰り返し単位の化学構造から誘電率を予測するモデル集合を抽出せよ．

### 1.5.3

Notebook: [Exercise 18](exercise_18.ipynb)

* 演習18）以下の説明やサンプルコードを参考にし，転移学習を適用して無機化合物の格子熱伝導率の予測モデルを構築せよ．

### 1.5.4

Notebook: [Exercise 19](exercise_19.ipynb)

* 演習19）以下の説明とサンプルコードを参考にし，ポリマーと無機化合物の屈折率の間で転移学習を実行せよ．

### 1.6.2

Notebook: [Exercise 20](exercise_20.ipynb)

* 演習20）サンプルコードを実行し，エピガロカテキン(Epigallocatechingallate)に変異・挿入・欠失・伸長の操作を施し，生成される化学構造を可視化せよ.

Notebook: [Exercise 21](exercise_21.ipynb)

* 演習21）サンプルコードを実行し，フラグメントの確率的な組み換えを行い，所望のHOMO（highest occupied molecular orbital），LUMO（lowest unoccupied molecular orbital）を有する分子（有機薄膜太陽電池のドナー分子）を生成せよ．計算のフローについては，Algorithm3を参照せよ．

### 1.7.5

Notebook: [Exercise 22](exercise_22.ipynb)

* 演習22）以下の解説とサンプルコードを参考に，所望の熱伝導率と線膨張係数を持つ高分子のモノマーを設計せよ．物性の目標範囲を変化させて，生成されるモノマーの構造的な違いを考察せよ．
