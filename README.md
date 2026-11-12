# セミナーII 2026 実践教材

Google Colaboratory で動かす PyTorch の実践教材です。回ごとにフォルダを分けています。Colab で GPU が使えないときは、Kaggle でも動かせます（第1回）。

| 回 | フォルダ | 内容 | Colabで開く | Kaggleで開く |
| --- | --- | --- | --- | --- |
| 第1回 | `lesson01/` | 犬と猫の写真の分類（CNN、転移学習） | [開く](https://colab.research.google.com/github/r-terasawa457/seminar2-2026/blob/main/lesson01/01_image_classification.ipynb) | [開く](https://kaggle.com/kernels/welcome?src=https://github.com/r-terasawa457/seminar2-2026/blob/main/lesson01/01_image_classification_kaggle.ipynb) |

## 使い方

1. リンクからノートブックを開き、「ドライブにコピーを保存」します。
2. メニューの「ランタイム」→「ランタイムのタイプを変更」で GPU を選びます。
3. 上のセルから順に実行します。最初のセルで、このリポジトリが Colab に取得されます。ログインやトークンの設定は必要ありません。

詳しい手順（アカウントの用意、保存、課題の PDF の作り方、困ったとき）は [`lesson01/colab_guide.md`](lesson01/colab_guide.md) にあります。

**Kaggle で動かすとき**（Colab で GPU が使えないときの予備）：「Kaggleで開く」のリンクから Kaggle 版のノートブックを開き、右の欄の「Session options」で Accelerator を **GPU T4 x2**、**Internet** をオンにして、上から順に実行します。Kaggle のアカウントと電話番号の認証が必要です。手順は [`lesson01/kaggle_guide.md`](lesson01/kaggle_guide.md) にあります。

## 第1回のファイル

| ファイル | 中身 |
| --- | --- |
| `01_image_classification.ipynb` | 本編。スライドと対応した節に分けて、学習を1つずつ動かす |
| `01_image_classification_kaggle.ipynb` | Kaggle 版（Colab で GPU が使えないときの予備）。中身は本編と同じで、実行環境の説明と自分の画像を入れるセルだけが違う |
| `colab_guide.md` | Colab で動かす手順 |
| `kaggle_guide.md` | Kaggle で動かす手順（アカウントの登録から） |
| `data.py` | データの読み込み、train / validation / test への分割、前処理、データ拡張、DataLoader |
| `models.py` | モデルの作成（timm） |
| `engine.py` | 1 epoch の学習、評価、推論 |
| `train.py` | コマンドラインで学習する（`python train.py --help`） |
| `predict.py` | 保存した重みで推論する |
| `viz.py` | ノートブックの表示用 |

## 取得するアセットと権利

| アセット | 取得元 | ライセンス |
| --- | --- | --- |
| Oxford-IIIT Pet | HF Hub [`timm/oxford-iiit-pet`](https://huggingface.co/datasets/timm/oxford-iiit-pet)（原典：[Oxford VGG](https://www.robots.ox.ac.uk/~vgg/data/pets/)） | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)。画像の著作権は元の撮影者にある |
| ResNet-18 学習済み重み | HF Hub [`timm/resnet18.a1_in1k`](https://huggingface.co/timm/resnet18.a1_in1k)（ImageNet-1k で学習） | Apache-2.0 |
| `lesson01/assets/chelsea.png` | scikit-image の `data.chelsea()` を書き出したもの（撮影 Stefan van der Walt） | CC0 |
| `lesson01/assets/coffee.png` | scikit-image の `data.coffee()` を書き出したもの（撮影 Rachel Michetti、Pikolo Espresso Bar の提供） | CC0 |

ライセンスは各配布元の表示による（2026-11-10 確認）。

Oxford-IIIT Pet の出典：O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar, "Cats and Dogs," IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012.
