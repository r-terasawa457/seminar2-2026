# 演習 1-a 用の Web デモ（研究用 PC で動かす）

HF Spaces などの公開デモは、無料の GPU の枠が小さく（ZeroGPU は1人1日数分）、授業で全員が使うには足りません。そこで、研究用 PC でデモのサーバーを動かし、学内 LAN から受講者のブラウザで使えるようにします。モデルは `../demos/` のノートブックと同じです。

| タブ | モデル | ライセンス |
| --- | --- | --- |
| 言葉で分類 | [laion/CLIP-ViT-B-32-laion2B-s34B-b79K](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K) | MIT |
| 言葉で探す | [google/owlv2-base-patch16-ensemble](https://huggingface.co/google/owlv2-base-patch16-ensemble) | Apache-2.0 |
| 写真に質問 | [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | Apache-2.0 |
| 奥行き | [depth-anything/Depth-Anything-V2-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) | Apache-2.0 |
| 切り抜く（クリック、または言葉で。言葉のときは OWLv2 も使う） | [facebook/sam2.1-hiera-small](https://huggingface.co/facebook/sam2.1-hiera-small) | Apache-2.0 |

## 実測（RTX 4060 8GB、float16、2026-11-10）

- 5つ全部を同時に載せて、GPU のメモリはピークで約 4.9GiB。8GB の GPU に収まる。
- 1件あたりの時間（宇宙飛行士の写真 512×512）：分類 0.7秒、検出 0.4秒、質問（100トークン）3.5秒、奥行き 0.1秒、言葉で切り抜く 0.4秒、点で切り抜く 0.1秒。最初の1件は準備で遅くなる。
- 切り抜きのタブのクリック操作は、ブラウザではまだ試していない（処理の関数だけ確認済み）。
- GPU の処理は全体で1件ずつ順に実行する。受講者が30人で同時に押しても、質問のタブで最後の人の待ち時間は約2分の見込み（3.5秒×30人。未測定）。

## 準備

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130   # GPU と CUDA に合わせる
pip install -r requirements.txt
```

重み（合計 約6.5GB）は、最初に起動したときに HF Hub から取得します。取得が 0 バイトのまま止まるときは、環境変数 `HF_HUB_DISABLE_XET=1` を付けて起動し直します（2026-11-10 に、この PC で起きた）。

## 起動

```bash
python app.py                                  # 5つ全部
python app.py --models clip,owlv2,depth        # 質問（VLM）を外して軽くする
python app.py --auth seminar:パスワード          # 開くときにパスワードを求める
python app.py --host 127.0.0.1                 # この PC だけで確かめる
```

受講者には `http://<研究用 PC の IP アドレス>:7860` を伝えます。IP アドレスは、Windows なら `ipconfig`、Linux なら `ip addr` で確かめます。

## 学内 LAN に公開する前に

- **ファイアウォール**：ポート 7860（TCP）の受信を許可する。Windows では、初回起動時の確認で「プライベート ネットワーク」を許可するか、「Windows Defender ファイアウォール」の受信の規則に追加する。
- **学内の規程**：学内 LAN でサーバーを公開してよいか、情報センターの規程を確かめる。
- **パスワード**：`--auth` を付けると、LAN 内の受講者以外が使うのを防げる。パスワードは授業中に口頭か板書で伝える。
- **アップロードされた画像**：Gradio は、アップロードされた画像を PC の一時フォルダに保存する。授業の後に消す。受講者には、他人の顔や個人情報が写った写真を使わないよう伝える。
- **止める**：起動した端末で Ctrl+C。
