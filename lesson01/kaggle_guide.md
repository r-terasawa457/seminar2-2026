# Kaggle で第1回のノートブックを動かす

Colab で GPU が使えないとき（GPU の割り当てがない、使える時間を使い切った など）の予備として、Kaggle でも同じ実践ができます。ノートブックは `01_image_classification_kaggle.ipynb` を使います。中身は Colab 版と同じで、実行環境に関わる説明と、自分の画像を入れるセル（8-2）だけが違います。

- Kaggle は Google が運営する機械学習のサイトで、ブラウザから GPU 付きのノートブックを無料で使えます。
- 内容は 2026年11月時点のものです。Kaggle の画面の名前や制限は変わることがあります。違っていたら、近い名前のものを探してください。

## 1. アカウントを登録する

1. <https://www.kaggle.com> を開き、右上の **Sign Up**（または **Register**）を押します。
2. **Register with Google**（Google アカウントで登録）か、**Register with Email**（メールアドレスで登録）を選びます。
   - Google アカウントを使うと、パスワードを新しく作らずに済みます。
   - メールアドレスの場合は、届いたメールのリンクを開いて確認します。
3. 表示名（Display name）とユーザー名（Username）を決めます。ユーザー名はノートブックの URL に入り、あとから変えにくいので、公開されてもよい名前にします。
4. 利用規約（Terms of Use）とプライバシーポリシーを読んで同意します。

## 2. 電話番号を認証する（GPU とインターネットに必要）

認証していないアカウントでは、**GPU** も **インターネット**（リポジトリの取得、データセットのダウンロード、`pip install`）も使えません。このノートブックは両方とも必要なので、最初に済ませます。

1. 右上の自分のアイコン → **Settings** を開きます。
2. **Phone verification**（電話番号の認証）の欄で、携帯電話の番号を入れます（日本の番号は +81 を選び、先頭の 0 を除いて入れます）。
3. SMS で届いた番号を入れて認証します。

## 3. ノートブックを開く

1. Kaggle にログインした状態で、次の URL を開きます（リポジトリの README の「Kaggleで開く」のリンクと同じ）。Kaggle 版のノートブックが、Kaggle の編集画面で開きます。
   `https://kaggle.com/kernels/welcome?src=https://github.com/r-terasawa457/seminar2-2026/blob/main/lesson01/01_image_classification_kaggle.ipynb`
   - ログインしていないと「Sign In or Register to continue editing」と表示され、実行できません。右上の **Sign in** からログインします。
2. 開いたノートブックは、自分のノートブックの下書きとして保存されます（題名の横に「Draft saved」と表示されます）。左上の題名（`notebook` で始まる名前）を「seminar2 lesson01」などに変えておくと、あとで探しやすくなります。
3. 次からは、Kaggle の左のメニューの **Code** → **Your Work** から開けます。URL をもう一度開くと、別の新しいノートブックができます。

URL で開けないときは、**Create** → **New Notebook** で新しいノートブックを作り、**File** → **Import Notebook** で、次の URL（GitHub のノートブック）を貼り付けて取り込みます。
`https://github.com/r-terasawa457/seminar2-2026/blob/main/lesson01/01_image_classification_kaggle.ipynb`

ノートブックは、作った時点では自分だけが見られる設定（Private）です。公開する必要はありません。

## 4. GPU とインターネットを設定する

画面の右側の欄（Settings）にある **Session options** を開きます。右側の欄が隠れているときは、画面の右端のボタンで開きます。

| 項目 | 選ぶもの | 理由 |
| --- | --- | --- |
| **Accelerator** | **GPU T4 x2** | NVIDIA Tesla T4 が2枚。このノートブックは1枚だけ使う |
| **Internet** | オン | リポジトリとデータセットの取得、ライブラリの追加に必要 |

- **GPU P100** は選ばないでください。今の Kaggle の PyTorch は P100 に対応していないため、`no kernel image is available for execution on the device` というエラーになります。
- Accelerator を変えると、セッションが再起動します（数十秒〜数分）。混んでいるときは順番待ちになります。
- **Internet** のスイッチが無い、または押せないときは、2. の電話番号の認証が済んでいません。

## 5. 実行する

上のセルから順に実行します（セルを選んで **Shift + Enter**、またはセルの左の再生ボタン）。メニューの **Run** → **Run All** で全部を実行することもできますが、最初は1つずつ、出力を見ながら進めてください。

1. の最初のセルでリポジトリを取得し、作業フォルダが `/kaggle/working/seminar2-2026/lesson01` になれば成功です。次のセルで次のように表示されれば、GPU で計算しています。

```
device: cuda Tesla T4
```

`device: cpu` と表示されたら、4. の Accelerator を確かめてください。CPU のままでも動きますが、枚数を減らしても時間がかかります。

## 6. 使える時間と、終わったときの操作

| 項目 | 内容 |
| --- | --- |
| GPU の時間 | 1週間に約30時間（週ごとに戻る。混み具合で増えることもある）。残りは Settings の欄や、自分のプロフィールで見られる |
| 1回のセッション | 最長12時間 |
| 操作しないとき | しばらく操作しないと、セッションが止まる |

このノートブックは、本編と 9. の実験を合わせても、GPU で1〜2時間あれば終わります。

**終わったら、セッションを止めてください。** ブラウザのタブを閉じただけでは、しばらくセッションが動き続け、GPU の時間が減ります。画面の左下の **Active Events** を開き、動いているセッションを止めます（ノートブックの画面の上部にある停止のボタンからも止められます）。

セッションが止まると、変数、学習したモデル、取得したファイルが消えます。再開するときは、1. から順に実行し直してください（[確かめよう] は飛ばしてかまいません）。

## 7. 保存とダウンロード

- 書き換えたコードと文章は、下書き（Draft）として自動で保存されます。次に開いたときも残っています。
- 区切りのよいところで、右上の **Save Version** → **Quick Save** を選ぶと、画面に表示されている状態（出力を含む）がバージョンとして残ります。**Save & Run All** は最初から全部を実行し直すので、時間がかかり、GPU の時間も使います。この演習では Quick Save で十分です。
- 手元に保存するときは、**File** → **Download Notebook** で `.ipynb` をダウンロードします。

### 課題（10.）の PDF を作る

1. **Save Version** → **Quick Save** で、出力を含めて保存します。
2. 保存したバージョンを開き（ノートブックの閲覧画面）、ブラウザの印刷（Ctrl + P）で、送信先を「PDF に保存」にして保存します。

表示が崩れるときは、7. の方法で `.ipynb` をダウンロードし、Colab（ファイル → ノートブックをアップロード）や VS Code で開いて PDF にしてもかまいません。

## 8. 自分の画像を使う（8-2 と、10. の発展）

Kaggle では、手元の画像を「データセット」としてアップロードして使います。

1. 右側の欄の **Input** にある **Upload** のボタン（または **Add Input** → **Upload**）を選びます。
2. 画像（または画像をまとめたフォルダ、zip）を選び、データセットの名前を付けて作ります。公開する必要はありません（Private のままでよい）。
3. 作ったデータセットが **Input** に表示されます。その中のフォルダの場所（`/kaggle/input/` で始まる）をコピーします。項目の横のコピーのボタンを押すか、セルで `!ls /kaggle/input` を実行すると分かります。
4. 使い方：
   - **8-2（自分の画像で推論）**：`UPLOAD_DIR = "/kaggle/input/…"` のように、コピーした場所を書いて実行します。フォルダの中の画像（jpg、png、webp）を、1枚ずつ予測します。
   - **10. の発展（自分の画像で学習）**：クラスごとにフォルダを分けた画像（例：`my_images/cat/`、`my_images/dog/`）をアップロードし、2. で `IMAGE_DIR = "/kaggle/input/…/my_images"` のように書いて、最初から実行します。

アップロードした画像は `/kaggle/input` に置かれ、読むことはできますが、書き換えはできません。

## 9. Colab との違い

| | Colab | Kaggle |
| --- | --- | --- |
| ノートブック | `01_image_classification.ipynb` | `01_image_classification_kaggle.ipynb` |
| GPU の選び方 | ランタイム → ランタイムのタイプを変更 | Session options → Accelerator → GPU T4 x2 |
| インターネット | 最初から使える | Session options → Internet をオン（電話番号の認証が必要） |
| 作業フォルダ | `/content` | `/kaggle/working`（書き込める）、`/kaggle/input`（読むだけ） |
| ファイルを見る | 左の「ファイル」 | 右の欄の「Output」（作業フォルダ）、「Input」（アップロードしたもの） |
| 自分の画像 | セルのアップロードのボタン | データセットとしてアップロード（8.） |
| 保存 | ドライブにコピーを保存 | 下書きの自動保存、Save Version |

## 10. 困ったとき

| 症状 | 原因と対処 |
| --- | --- |
| Internet のスイッチが無い、押せない | 電話番号の認証が済んでいない（2.） |
| GPU の選択肢が無い、押せない | 電話番号の認証が済んでいないか、その週の GPU の時間を使い切った（6.） |
| `no kernel image is available for execution on the device` | Accelerator が GPU P100 になっている。GPU T4 x2 に変える（4.） |
| 1. の `git clone` やデータセットの取得で止まる、`Could not resolve host` などと出る | Internet がオフ（4.） |
| `NameError`（名前が定義されていない） | 前のセルを実行していないか、セッションが止まった。1. から順に実行し直す |
| 実行中に止まった、画面に再接続の表示が出た | 操作しない時間が続いたか、12時間を超えた。1. から順に実行し直す |
| `CUDA out of memory` | 9. で大きなモデルや大きな batch サイズを選んだ。`batch_size` を小さくする。それでも出るときは、セッションを止めて開き直し、1. からやり直す |

それでも解決しないときは、エラーの最後の数行と、どのセルで出たかを添えて質問してください。

## 参考

- Kaggle Notebooks のドキュメント：<https://www.kaggle.com/docs/notebooks>（セッションの時間、`/kaggle/working`、Save Version）
- GPU の使い方のヒント：<https://www.kaggle.com/docs/efficient-gpu-usage>（週あたりの GPU の時間）
