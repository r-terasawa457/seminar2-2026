# セミナーII 2026 実践教材

Google Colaboratory で動かす PyTorch の実践教材です。回ごとにフォルダを分けています。Colab で GPU が使えないときは、Kaggle でも動かせます（第1回）。

## 第1回（`lesson01/`）

| 題材 | 内容 | Colabで開く | Kaggleで開く |
| --- | --- | --- | --- |
| ResNet-18 | 犬と猫の写真の分類（CNN、転移学習） | [開く](https://colab.research.google.com/github/r-terasawa457/seminar2-2026/blob/main/lesson01/01_image_classification.ipynb) | [開く](https://kaggle.com/kernels/welcome?src=https://github.com/r-terasawa457/seminar2-2026/blob/main/lesson01/01_image_classification_kaggle.ipynb) |
| ResNet-18（ダイジェスト版） | 第1回の要点を短時間で動かす。プログラミングや機械学習に慣れていない人向け | [開く](https://colab.research.google.com/github/r-terasawa457/seminar2-2026/blob/main/lesson01/01_image_classification_digest.ipynb) | [開く](https://kaggle.com/kernels/welcome?src=https://github.com/r-terasawa457/seminar2-2026/blob/main/lesson01/01_image_classification_digest_kaggle.ipynb) |

## 第2回（`lesson02/`）

第2回の教材は `lesson02` ブランチにあります。下のリンクは、このブランチのノートブックを開きます。

| 題材 | 内容 | Colabで開く |
| --- | --- | --- |
| Depth Anything V2 | 写真1枚から奥行きを推定する。手法、例の画像、自分の画像、設定を変える、活用（背景ぼかし、霧、3次元） | [開く](https://colab.research.google.com/github/r-terasawa457/seminar2-2026/blob/lesson02/lesson02/02_depth_anything_v2.ipynb) |
| SAM 2 | 写真の中の物を切り抜く。点・枠・言葉で指定、全自動、活用（背景の差し替え、数えて測る、動画で追う） | [開く](https://colab.research.google.com/github/r-terasawa457/seminar2-2026/blob/lesson02/lesson02/02_sam2_segmentation.ipynb) |
| CLIP | 画像と文章の類似度で分類・検索する。ゼロショット分類、第1回の37品種を学習なしで解く、活用（文章で画像を検索、似た画像を検索） | [開く](https://colab.research.google.com/github/r-terasawa457/seminar2-2026/blob/lesson02/lesson02/02_clip.ipynb) |
| DINOv2 | ViT のパッチ特徴で画像の部分を比べる。主成分分析で特徴を色で見る、似たパッチ・対応点を探す、前景の切り出し、少ない枚数で分類 | [開く](https://colab.research.google.com/github/r-terasawa457/seminar2-2026/blob/lesson02/lesson02/02_dinov2.ipynb) |
| VLM（Qwen3-VL-2B） | 画像を見て質問に答える。画像トークン、次のトークンの確率と温度、もっともらしい誤り、活用（位置の枠、まとめて表にする） | [開く](https://colab.research.google.com/github/r-terasawa457/seminar2-2026/blob/lesson02/lesson02/02_vlm.ipynb) |
| ViT-B/16（おまけ） | ViT で分類して、第1回の ResNet-18 と比べる（同じデータ・同じ学習の進め方で、モデルだけを変える） | [開く](https://colab.research.google.com/github/r-terasawa457/seminar2-2026/blob/lesson02/lesson02/02_vit_classification.ipynb) |
| GitHub のコード（発展） | GitHub で公開されたコードを Colab で動かす（記入例：Depth Anything V2） | [開く](https://colab.research.google.com/github/r-terasawa457/seminar2-2026/blob/lesson02/lesson02/02_run_github_code.ipynb) |

## 使い方

1. リンクからノートブックを開き、「ドライブにコピーを保存」します。
2. メニューの「ランタイム」→「ランタイムのタイプを変更」で GPU を選びます。
3. 上のセルから順に実行します。第1回と、第2回の ViT のノートブックでは、最初のセルで、このリポジトリが Colab に取得されます。第2回のほかのノートブックは、モデルの重みを HF Hub から取得します。ログインやトークンの設定は必要ありません（第2回の発展で、利用の申請が要る重みを選んだときを除く）。

詳しい手順（アカウントの用意、保存、課題の PDF の作り方、困ったとき）は [`lesson01/colab_guide.md`](lesson01/colab_guide.md) にあります。

**Kaggle で動かすとき**（Colab で GPU が使えないときの予備）：「Kaggleで開く」のリンクから Kaggle 版のノートブックを開き、右の欄の「Session options」で Accelerator を **GPU T4 x2**、**Internet** をオンにして、上から順に実行します。Kaggle のアカウントと電話番号の認証が必要です。手順は [`lesson01/kaggle_guide.md`](lesson01/kaggle_guide.md) にあります。

## 第1回のファイル

| ファイル | 中身 |
| --- | --- |
| `01_image_classification.ipynb` | 本編。スライドと対応した節に分けて、学習を1つずつ動かす |
| `01_image_classification_kaggle.ipynb` | Kaggle 版（Colab で GPU が使えないときの予備）。中身は本編と同じで、実行環境の説明と自分の画像を入れるセルだけが違う |
| `01_image_classification_digest.ipynb` | ダイジェスト版。本編の要点（データ、学習の4行を1回だけ動かす、学習と結果、設定を1つ変えた比較）だけを、関数の定義を書かずに動かす |
| `01_image_classification_digest_kaggle.ipynb` | ダイジェスト版の Kaggle 版 |
| `colab_guide.md` | Colab で動かす手順 |
| `kaggle_guide.md` | Kaggle で動かす手順（アカウントの登録から） |
| `data.py` | データの読み込み、train / validation / test への分割、前処理、データ拡張、DataLoader |
| `models.py` | モデルの作成（timm） |
| `engine.py` | 1 epoch の学習、評価、推論 |
| `train.py` | コマンドラインで学習する（`python train.py --help`） |
| `predict.py` | 保存した重みで推論する |
| `viz.py` | ノートブックの表示用 |

## 第2回のファイル

| ファイル | 中身 |
| --- | --- |
| `02_depth_anything_v2.ipynb` | 写真1枚から奥行きを推定する（Depth Anything V2、HF Transformers）。ログインや Google ドライブの許可は不要 |
| `02_sam2_segmentation.ipynb` | 写真の中の物を切り抜く（SAM 2.1 Small ＋ OWLv2、HF Transformers）。ログインや Google ドライブの許可は不要 |
| `02_clip.ipynb` | 画像と文章の類似度で分類・検索する（CLIP ViT-B/32、LAION の OpenCLIP、HF Transformers）。第1回と同じデータセットの test を使う。ログインや Google ドライブの許可は不要 |
| `02_dinov2.ipynb` | ViT のパッチ特徴で画像の部分を比べる（DINOv2-Base、HF Transformers）。第1回と同じデータセットを使う。ログインや Google ドライブの許可は不要 |
| `02_vlm.ipynb` | 画像を見て質問に答える（Qwen3-VL-2B-Instruct、HF Transformers、float16）。ログインや Google ドライブの許可は不要 |
| `02_vit_classification.ipynb` | おまけ。ViT-B/16 で37品種を分類し、ResNet-18 と test の正解率と時間を比べる（timm）。第1回の `data.py`、`engine.py`、`models.py` を使う。ログインや Google ドライブの許可は不要 |
| `02_run_github_code.ipynb` | 発展。GitHub のリポジトリを取得し、README に沿って依存ライブラリと重みをそろえ、公式の例を動かす |
| `guide.md` | リポジトリの選び方、README の読み方、変更の残し方、つまずいたときの対処 |
| `demos/` | 演習 1-b のデモ。言葉で分類（CLIP）、言葉で探す（OWLv2）、写真に質問（Qwen3-VL）、奥行き（Depth Anything V2）、言葉で切り抜く（OWLv2＋SAM 2）。どれもログイン不要 |
| `server/` | 演習 1-a 用。同じモデルを研究用 PC の Web サーバーで動かし、学内 LAN のブラウザから使う（講師用） |

演習 1-b のデモ（どれもログイン不要。GPU を選んで上から実行する）：

| 題材 | 内容 | Colabで開く |
| --- | --- | --- |
| CLIP | 言葉で分類する | [開く](https://colab.research.google.com/github/r-terasawa457/seminar2-2026/blob/lesson02/lesson02/demos/demo1_zero_shot_classification.ipynb) |
| OWLv2 | 言葉で物を探す | [開く](https://colab.research.google.com/github/r-terasawa457/seminar2-2026/blob/lesson02/lesson02/demos/demo2_zero_shot_detection.ipynb) |
| Qwen3-VL | 写真に質問する | [開く](https://colab.research.google.com/github/r-terasawa457/seminar2-2026/blob/lesson02/lesson02/demos/demo3_vision_language.ipynb) |
| Depth Anything V2 | 奥行きを推定する | [開く](https://colab.research.google.com/github/r-terasawa457/seminar2-2026/blob/lesson02/lesson02/demos/demo4_depth.ipynb) |
| OWLv2＋SAM 2 | 言葉で切り抜く | [開く](https://colab.research.google.com/github/r-terasawa457/seminar2-2026/blob/lesson02/lesson02/demos/demo5_text_segmentation.ipynb) |

`02_vit_classification.ipynb` は、HF Hub の重み [`timm/resnet18.a1_in1k`](https://huggingface.co/timm/resnet18.a1_in1k) と [`timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`](https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k)（どちらも Apache-2.0）と、第1回と同じ Oxford-IIIT Pet を使います。

`02_depth_anything_v2.ipynb` は、HF Hub の重み（Small は Apache-2.0、Base と Large は CC BY-NC 4.0。Metric の Small はモデルカードにライセンスの記載がない）と、scikit-image に入っている写真（NASA と SpaceX はパブリックドメイン、coffee は CC0、motorcycle は Middlebury 2014 ステレオデータセット）を使います。

`02_sam2_segmentation.ipynb` は、HF Hub の重み（SAM 2.1、OWLv2 とも Apache-2.0）と、scikit-image に入っている写真（上の4枚と cat（CC0）、coins（Brooklyn Museum。scikit-image の説明では、著作権の制限は知られていない））を使います。

`02_clip.ipynb` は、HF Hub の重み [`laion/CLIP-ViT-B-32-laion2B-s34B-b79K`](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K)（MIT。モデルカードでは研究のための利用を想定）と、第1回と同じ Oxford-IIIT Pet、scikit-image の写真を使います。

`02_dinov2.ipynb` は、HF Hub の重み [`facebook/dinov2-base`](https://huggingface.co/facebook/dinov2-base)（Apache-2.0）と、第1回と同じ Oxford-IIIT Pet、scikit-image の写真を使います。

`02_vlm.ipynb` は、HF Hub の重み [`Qwen/Qwen3-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)（Apache-2.0）と、scikit-image の写真、ノートブックの中で描く表の画像を使います。

`02_run_github_code.ipynb` は、このリポジトリではなく、動かす対象のリポジトリを取得します。記入例は [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)（Apache-2.0）と、HF Hub の重み [`depth-anything/Depth-Anything-V2-Small`](https://huggingface.co/depth-anything/Depth-Anything-V2-Small)（Apache-2.0）です。

## 取得するアセットと権利

| アセット | 取得元 | ライセンス |
| --- | --- | --- |
| Oxford-IIIT Pet | HF Hub [`timm/oxford-iiit-pet`](https://huggingface.co/datasets/timm/oxford-iiit-pet)（原典：[Oxford VGG](https://www.robots.ox.ac.uk/~vgg/data/pets/)） | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)。画像の著作権は元の撮影者にある |
| ResNet-18 学習済み重み | HF Hub [`timm/resnet18.a1_in1k`](https://huggingface.co/timm/resnet18.a1_in1k)（ImageNet-1k で学習） | Apache-2.0 |
| `lesson01/assets/chelsea.png` | scikit-image の `data.chelsea()` を書き出したもの（撮影 Stefan van der Walt） | CC0 |
| `lesson01/assets/coffee.png` | scikit-image の `data.coffee()` を書き出したもの（撮影 Rachel Michetti、Pikolo Espresso Bar の提供） | CC0 |

ライセンスは各配布元の表示による（2026-11-10 確認）。

Oxford-IIIT Pet の出典：O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar, "Cats and Dogs," IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012.
