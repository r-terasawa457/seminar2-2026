# セミナーII 2026 実践教材

Google Colaboratory で動かす PyTorch の実践教材です。回ごとにフォルダを分けています。

| 回 | フォルダ | 内容 | Colabで開く |
| --- | --- | --- | --- |
| 第1回 | `lesson01/` | 犬と猫の写真の分類（CNN、転移学習） | [開く](https://colab.research.google.com/github/r-terasawa457/seminar2-2026/blob/main/lesson01/01_image_classification.ipynb) |

## 使い方

1. リンクからノートブックを開き、「ドライブにコピーを保存」します。
2. メニューの「ランタイム」→「ランタイムのタイプを変更」で GPU を選びます。
3. 上のセルから順に実行します。最初のセルで、このリポジトリが Colab に取得されます。ログインやトークンの設定は必要ありません。

## 第1回のファイル

| ファイル | 中身 |
| --- | --- |
| `01_image_classification.ipynb` | 本編。スライドと対応した節に分けて、学習を1つずつ動かす |
| `data.py` | データの読み込み、train / validation / test への分割、前処理、データ拡張、DataLoader |
| `models.py` | モデルの作成（timm） |
| `engine.py` | 1 epoch の学習、評価、推論 |
| `train.py` | コマンドラインで学習する（`python train.py --help`） |
| `predict.py` | 保存した重みで推論する |
| `viz.py` | ノートブックの表示用 |

## 取得するアセットと権利

| アセット | 取得元 | ライセンス |
| --- | --- | --- |
| Oxford-IIIT Pet | HF Hub [`timm/oxford-iiit-pet`](https://huggingface.co/datasets/timm/oxford-iiit-pet) | CC BY-SA 4.0（HF Hub の表示による） |
| ResNet-18 学習済み重み | HF Hub [`timm/resnet18.a1_in1k`](https://huggingface.co/timm/resnet18.a1_in1k) | Apache-2.0（HF Hub の表示による） |
| `lesson01/assets/chelsea.png`、`coffee.png` | scikit-image の `data.chelsea()`、`data.coffee()` を書き出したもの | CC0（scikit-image のデータ一覧を公開前に再確認） |
