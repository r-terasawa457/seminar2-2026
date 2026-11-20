"""演習 1-a 用の Web デモ（研究用 PC で動かし、学内 LAN に公開する）

デモ用ノートブック（demos/）と同じモデルを、ブラウザから試せるようにする。
GPU のメモリに合わせて、起動時に載せるモデルを選ぶ。GPU を使う処理は全体で1件ずつ順に実行する。

使い方:
    python app.py                          # 全部のモデルを載せる
    python app.py --models clip,depth      # 一部だけ載せる
    python app.py --port 7860 --auth seminar:パスワード
同じ LAN の PC から http://<この PC の IP アドレス>:7860 を開く。
"""
import argparse
import threading

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor, Sam2Model, Sam2Processor, pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32  # T4 などは bfloat16 に対応していない

MODEL_IDS = {
    "clip": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "owlv2": "google/owlv2-base-patch16-ensemble",
    "vlm": "Qwen/Qwen3-VL-2B-Instruct",
    "depth": "depth-anything/Depth-Anything-V2-Small-hf",
    "sam": "facebook/sam2.1-hiera-small",  # 言葉で指定するときは owlv2 も使う
}
models = {}
gpu_lock = threading.Lock()  # 念のため、GPU を使う処理が重ならないようにする


def load(names):
    for name in names:
        print("loading", name, MODEL_IDS[name])
        if name == "clip":
            models[name] = pipeline("zero-shot-image-classification", model=MODEL_IDS[name], device=device, dtype=dtype)
        elif name == "owlv2":
            models[name] = pipeline("zero-shot-object-detection", model=MODEL_IDS[name], device=device, dtype=dtype)
        elif name == "depth":
            models[name] = pipeline("depth-estimation", model=MODEL_IDS[name], device=device, dtype=dtype)
        elif name == "vlm":
            processor = AutoProcessor.from_pretrained(MODEL_IDS[name])
            model = AutoModelForImageTextToText.from_pretrained(MODEL_IDS[name], dtype=dtype).to(device).eval()
            models[name] = (processor, model)
        elif name == "sam":
            processor = Sam2Processor.from_pretrained(MODEL_IDS[name])
            model = Sam2Model.from_pretrained(MODEL_IDS[name], dtype=dtype).to(device).eval()
            models[name] = (processor, model)
    if device == "cuda":
        print(f"GPU memory: {torch.cuda.memory_allocated() / 2**30:.1f} GiB")


def split_labels(text):
    return [s.strip() for s in text.replace("、", ",").split(",") if s.strip()]


def classify(image, labels):
    if image is None or not split_labels(labels):
        return {}
    with gpu_lock:
        results = models["clip"](image, candidate_labels=split_labels(labels))
    return {r["label"]: r["score"] for r in results}


def detect(image, queries, threshold):
    if image is None or not split_labels(queries):
        return None
    with gpu_lock:
        results = models["owlv2"](image, candidate_labels=split_labels(queries), threshold=threshold)
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for r in results:
        b = r["box"]
        draw.rectangle([b["xmin"], b["ymin"], b["xmax"], b["ymax"]], outline="red", width=3)
        draw.text((b["xmin"] + 3, b["ymin"] + 3), f"{r['label']} {r['score']:.2f}", fill="red")
    return out


def ask(image, question, max_new_tokens):
    if image is None or not question.strip():
        return ""
    processor, model = models["vlm"]
    messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": question}]}]
    with gpu_lock:
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False)
    return processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def estimate_depth(image, near_ratio):
    if image is None:
        return None, None
    with gpu_lock:
        depth = np.array(models["depth"](image)["predicted_depth"].float().cpu())
    depth = np.array(Image.fromarray(depth).resize(image.size))
    shown = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)
    mask = depth >= np.quantile(depth, 1 - near_ratio)
    return Image.fromarray(shown), Image.fromarray(np.array(image) * mask[..., None])


COLORS = np.array([[230, 60, 60], [60, 90, 230], [40, 170, 80], [230, 160, 30], [160, 60, 200]])


def sam_masks(image, boxes=None, points=None):
    """枠か点（1つずつ別の物）で指定した物のマスク（物の数×縦×横）"""
    processor, model = models["sam"]
    if boxes is not None:
        inputs = processor(images=image, input_boxes=[boxes], return_tensors="pt").to(device)
    else:
        inputs = processor(images=image, input_points=[[[p] for p in points]],
                           input_labels=[[[1] for _ in points]], return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    return processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0][:, 0].numpy()


def overlay(image, masks):
    out = np.array(image).astype(float)
    for i, mask in enumerate(masks):
        out[mask] = out[mask] * 0.45 + COLORS[i % len(COLORS)] * 0.55
    return Image.fromarray(out.astype(np.uint8))


def segment_text(image, queries, threshold):
    if image is None or not split_labels(queries) or "owlv2" not in models:
        return None, "言葉で切り抜くには、owlv2 も載せて起動してください" if "owlv2" not in models else ""
    with gpu_lock:
        found = models["owlv2"](image, candidate_labels=split_labels(queries), threshold=threshold)
        if not found:
            return image, "見つかりませんでした。しきい値を下げるか、言葉を変えてみてください"
        boxes = [[r["box"]["xmin"], r["box"]["ymin"], r["box"]["xmax"], r["box"]["ymax"]] for r in found]
        masks = sam_masks(image, boxes=boxes)
    return overlay(image, masks), "、".join(f"{r['label']} {r['score']:.2f}" for r in found)


def segment_click(image, points, evt: gr.SelectData):
    """画像をクリックするたびに点を足し、点ごとの物を切り抜く"""
    points = points + [list(evt.index)]
    with gpu_lock:
        masks = sam_masks(image, points=points)
    return overlay(image, masks), points, f"点：{points}"


def build_ui():
    with gr.Blocks(title="セミナーII 第2回 デモ") as demo:
        gr.Markdown("# セミナーII 第2回 デモ\n画像をアップロードするか、カメラで撮って試します。"
                    "混んでいるときは、順番待ちになります。アップロードした画像は、この PC の一時フォルダに残ることがあります。")
        if "clip" in models:
            with gr.Tab("言葉で分類（CLIP）"):
                with gr.Row():
                    image = gr.Image(type="pil", label="画像")
                    with gr.Column():
                        labels = gr.Textbox("a cat, a dog, a person, a cup of coffee", label="クラスの名前（英語、カンマ区切り）")
                        button = gr.Button("分類する", variant="primary")
                        result = gr.Label(label="確率", num_top_classes=5)
                button.click(classify, [image, labels], result)
        if "owlv2" in models:
            with gr.Tab("言葉で探す（OWLv2）"):
                with gr.Row():
                    image = gr.Image(type="pil", label="画像")
                    with gr.Column():
                        queries = gr.Textbox("a face, a cup", label="探す物（英語、カンマ区切り）")
                        threshold = gr.Slider(0.05, 0.6, 0.2, step=0.05, label="しきい値")
                        button = gr.Button("探す", variant="primary")
                result = gr.Image(label="結果")
                button.click(detect, [image, queries, threshold], result)
        if "vlm" in models:
            with gr.Tab("写真に質問（Qwen3-VL）"):
                with gr.Row():
                    image = gr.Image(type="pil", label="画像")
                    with gr.Column():
                        question = gr.Textbox("この画像を日本語で説明してください。", label="質問", lines=2)
                        max_tokens = gr.Slider(32, 512, 200, step=32, label="答えの長さの上限（トークン）")
                        button = gr.Button("質問する", variant="primary")
                        answer = gr.Textbox(label="答え", lines=8)
                button.click(ask, [image, question, max_tokens], answer)
        if "depth" in models:
            with gr.Tab("奥行き（Depth Anything V2）"):
                with gr.Row():
                    image = gr.Image(type="pil", label="画像")
                    with gr.Column():
                        near = gr.Slider(0.1, 0.9, 0.6, step=0.05, label="近いほうから残す割合")
                        button = gr.Button("推定する", variant="primary")
                with gr.Row():
                    depth_view = gr.Image(label="奥行き（明るいほど近い）")
                    cutout = gr.Image(label="近いところだけ")
                button.click(estimate_depth, [image, near], [depth_view, cutout])
        if "sam" in models:
            with gr.Tab("切り抜く（SAM 2）"):
                gr.Markdown("左の画像をクリックすると、その場所の物を切り抜きます（クリックするたびに物を追加）。"
                            "言葉で指定するときは、探す物を書いて「言葉で切り抜く」を押します。")
                points = gr.State([])
                with gr.Row():
                    image = gr.Image(type="pil", label="画像（クリックで指定）")
                    result = gr.Image(label="結果")
                with gr.Row():
                    queries = gr.Textbox("a cup, a spoon", label="探す物（英語、カンマ区切り）")
                    threshold = gr.Slider(0.05, 0.6, 0.2, step=0.05, label="しきい値")
                with gr.Row():
                    button = gr.Button("言葉で切り抜く", variant="primary")
                    clear = gr.Button("点を消す")
                info = gr.Textbox(label="情報")
                button.click(segment_text, [image, queries, threshold], [result, info])
                image.select(segment_click, [image, points], [result, points, info])
                image.upload(lambda: ([], None, ""), None, [points, result, info])
                clear.click(lambda: ([], None, ""), None, [points, result, info])
        gr.Markdown("モデル：" + "、".join(f"`{MODEL_IDS[n]}`" for n in models))
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default=",".join(MODEL_IDS), help=f"載せるモデル（{','.join(MODEL_IDS)}）")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0", help="0.0.0.0 で LAN に公開、127.0.0.1 でこの PC だけ")
    parser.add_argument("--auth", default=None, help="ユーザー名:パスワード（付けると、開くときに求められる）")
    parser.add_argument("--max-queue", type=int, default=50, help="順番待ちの上限")
    args = parser.parse_args()

    load([n.strip() for n in args.models.split(",") if n.strip()])
    demo = build_ui()
    # GPU の処理は全体で1件ずつ（受講者が多くても、メモリが足りなくならない）
    demo.queue(default_concurrency_limit=1, max_size=args.max_queue)
    demo.launch(server_name=args.host, server_port=args.port,
                auth=tuple(args.auth.split(":", 1)) if args.auth else None)
