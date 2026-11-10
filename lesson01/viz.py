"""ノートブックで結果を見るための表示用の関数

学習の仕組みとは関係しないので、ノートブックでは呼び出すだけにしている。
Colab には日本語のフォントが無いので、図の文字は英語にしている。
"""
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from IPython.display import display
from PIL import ImageOps

from data import MEAN, STD


def to_image(tensor, mean=MEAN, std=STD):
    """正規化したテンソル (3, H, W) を、表示できる画像 (H, W, 3) に戻す"""
    x = tensor.cpu() * torch.tensor(std)[:, None, None] + torch.tensor(mean)[:, None, None]
    return x.clamp(0, 1).permute(1, 2, 0).numpy()


def show_samples(data, class_names, names=None, n=8):
    """クラスごとに1枚ずつ、見本の画像を並べる（p90）。names は大文字と小文字を区別しない"""
    lower = [c.lower() for c in class_names]
    labels = [lower.index(name.lower()) for name in names] if names else \
        list(range(0, len(class_names), max(1, len(class_names) // n)))[:n]
    first = {}
    for i, label in enumerate(data["label"]):
        first.setdefault(label, i)
    indices = [first[label] for label in labels if label in first]
    fig, axes = plt.subplots(2, (len(indices) + 1) // 2, figsize=(12, 6))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, indices):
        sample = data[i]
        ax.imshow(sample["image"].convert("RGB"))
        ax.set_title(f"{class_names[sample['label']]} ({sample['label']})", fontsize=10)
    plt.tight_layout()
    plt.show()


def show_example(sample, class_names):
    """データ1件（画像と正解ラベルの組）を表示する（p91）"""
    image = sample["image"]
    plt.figure(figsize=(4, 4))
    plt.imshow(image.convert("RGB"))
    plt.title(f"label: {sample['label']}  ({class_names[sample['label']]})")
    plt.axis("off")
    plt.show()


def show_preprocess(image, transform):
    """前処理の前と後を並べる（p93）"""
    x = transform(image.convert("RGB"))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(image.convert("RGB"))
    axes[0].set_title(f"original {image.width}x{image.height}")
    axes[1].imshow(to_image(x))
    axes[1].set_title(f"after {x.shape[2]}x{x.shape[1]}")
    for ax in axes:
        ax.axis("off")
    plt.show()


def show_augmentations(image, transform, n=6):
    """同じ画像にデータ拡張を何回かかけて並べる（p94）"""
    fig, axes = plt.subplots(1, n + 1, figsize=(2.2 * (n + 1), 2.6))
    axes[0].imshow(image.convert("RGB"))
    axes[0].set_title("original")
    for k, ax in enumerate(axes[1:], 1):
        ax.imshow(to_image(transform(image.convert("RGB"))))
        ax.set_title(f"#{k}")
    for ax in axes:
        ax.axis("off")
    plt.show()


def show_batch(batch, class_names, n=8):
    """mini-batch の中身を、先頭から n 枚表示する（p98）"""
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.4))
    for ax, x, label in zip(axes, batch["image"], batch["label"]):
        ax.imshow(to_image(x))
        ax.set_title(class_names[label], fontsize=8)
        ax.axis("off")
    plt.show()


@torch.no_grad()
def print_feature_shapes(model, device, image_size=224):
    """層を通るごとに、特徴マップの形がどう変わるかを表示する（p72、p78）"""
    model.eval()
    x = torch.zeros(1, 3, image_size, image_size, device=device)
    print(f"{'name':12s} {'output shape (C, H, W)':24s} type")
    print(f"{'input':12s} {str(tuple(x.shape[1:])):24s}")
    for name, module in model.named_children():
        x = module(x)
        print(f"{name:12s} {str(tuple(x.shape[1:])):24s} {type(module).__name__}")


def show_kernels(model, n=64):
    """最初の畳み込み層のカーネルを並べる。1枚ごとに最小が黒、最大が白（p60）"""
    conv = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
    w = conv.weight.detach().cpu()[:n].permute(0, 2, 3, 1).numpy()
    cols = 8
    rows = (len(w) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.1))
    for ax in axes.flat:
        ax.axis("off")
    for ax, k in zip(axes.flat, w):
        ax.imshow((k - k.min()) / (k.max() - k.min() + 1e-8))
    fig.suptitle(f"first conv layer: {conv.weight.shape[0]} kernels of "
                 f"{conv.weight.shape[2]}x{conv.weight.shape[3]}x{conv.weight.shape[1]}")
    plt.show()


def show_probs(probs, class_names, k=10, true_label=None):
    """各クラスの確率を、上位 k 個の棒グラフで表示する（p82、p105）"""
    top = probs.topk(min(k, len(class_names)))
    names = [class_names[i] for i in top.indices]
    colors = ["tab:red" if i == true_label else "tab:blue" for i in top.indices]
    plt.figure(figsize=(6, 0.3 * len(names) + 1))
    plt.barh(names[::-1], top.values.numpy()[::-1], color=colors[::-1])
    plt.axvline(1 / len(class_names), color="gray", linestyle="--", label=f"1/{len(class_names)}")
    plt.xlabel("probability")
    plt.legend()
    plt.show()


def class_examples(reference):
    """クラスの番号ごとに、見本にする1件の位置を返す（各クラスの最初の1件）"""
    first = {}
    for i, label in enumerate(reference["label"]):
        first.setdefault(label, i)
    return first


def show_prediction(image, probs, class_names, k=5, reference=None):
    """画像と、上位 k クラスの確率を並べて表示する（推論。p16）

    reference（データセット）を渡すと、下の段に上位のクラスの見本の画像を並べる。
    """
    top = probs.topk(min(k, len(class_names)))
    names = [class_names[i] for i in top.indices]
    fig = plt.figure(figsize=(12, 3.5 if reference is None else 6.5), layout="constrained")
    # 上の段（入力と確率）と下の段（見本）を別の領域にして、列の幅が互いに影響しないようにする
    top_fig, bottom_fig = fig.subfigures(2, 1, height_ratios=[1.2, 1]) if reference is not None else (fig, None)
    axes = top_fig.subplots(1, 2, width_ratios=[1, 1.4])
    axes[0].imshow(image.convert("RGB"))
    axes[0].set_title("input")
    axes[0].axis("off")
    axes[1].barh(names[::-1], top.values.numpy()[::-1])
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("probability")
    if reference is not None:
        examples = class_examples(reference)
        for ax, p, i in zip(bottom_fig.subplots(1, len(names)), top.values, top.indices):
            # 大きさをそろえるため、中央を正方形に切り抜く
            ax.imshow(ImageOps.fit(reference[examples[int(i)]]["image"].convert("RGB"), (224, 224)))
            ax.set_title(f"{class_names[i]}\n{p.item():.1%}", fontsize=9)
            ax.axis("off")
    plt.show()


def plot_history(history):
    """epoch ごとの Loss と正解率を、train と validation で比べる（p103〜104）"""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, key in zip(axes, ("loss", "acc")):
        ax.plot(epochs, history[f"train_{key}"], "o-", label="train")
        ax.plot(epochs, history[f"val_{key}"], "o-", label="validation")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss" if key == "loss" else "accuracy")
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.legend()
    plt.show()


def print_class_accuracy(preds, labels, class_names, n=5):
    """クラスごとの正解率を計算し、よく当たったクラスと、よく外れたクラスを n 個ずつ表示する"""
    accuracy = {}
    for c in range(len(class_names)):
        mask = labels == c
        if mask.any():
            accuracy[c] = (preds[mask] == c).float().mean().item()
    order = sorted(accuracy, key=accuracy.get, reverse=True)
    print("よく当たったクラス（正解率）")
    for c in order[:n]:
        print(f"  {accuracy[c]:.2f}  {class_names[c]}")
    print("よく外れたクラス（正解率）")
    for c in order[::-1][:n]:
        print(f"  {accuracy[c]:.2f}  {class_names[c]}")


def show_results(results):
    """実験の結果（設定と正解率）を表にし、学習曲線を並べる

    図の見出しは「#番号 変えた設定」。番号は表の番号と同じ（note は日本語なので、図には出さない）。
    """
    display(pd.DataFrame([{k: v for k, v in r.items() if k != "history"} for r in results]))
    fig, axes = plt.subplots(2, len(results), figsize=(3.2 * len(results), 5.5), squeeze=False)
    for i, (r, (ax_loss, ax_acc)) in enumerate(zip(results, axes.T)):
        epochs = range(1, len(r["history"]["train_loss"]) + 1)
        for ax, key in ((ax_loss, "loss"), (ax_acc, "acc")):
            ax.plot(epochs, r["history"][f"train_{key}"], "o-", label="train")
            ax.plot(epochs, r["history"][f"val_{key}"], "o-", label="validation")
            ax.set_xlabel("epoch")
            ax.xaxis.get_major_locator().set_params(integer=True)
        ax_loss.set_title(f"#{i} {r['changes']}", fontsize=9)
        ax_acc.set_ylim(0, 1)  # 正解率はそろえて比べる（Loss は大きさが違うのでそろえない）
    axes[0, 0].set_ylabel("loss")
    axes[1, 0].set_ylabel("accuracy")
    axes[0, 0].legend()
    plt.tight_layout()
    plt.show()

    # 正解率を実験ごとに並べる（横軸は表の番号）
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    numbers = list(range(len(results)))
    for ax, key in zip(axes, ("best_val_acc", "test_acc")):
        values = [r[key] for r in results]
        ax.scatter(numbers, values, s=40)
        for i, v in zip(numbers, values):
            ax.annotate(f"#{i}", (i, v), textcoords="offset points", xytext=(5, 5), fontsize=9)
        ax.axhline(results[0][key], color="gray", linestyle="--")
        ax.text(len(results) - 0.5, results[0][key], "baseline (#0)", color="gray", fontsize=8,
                ha="right", va="bottom")
        ax.set_xticks(numbers)
        ax.set_xlim(-0.5, len(results) - 0.5)
        ax.set_xlabel("experiment #")
        ax.set_ylabel(key)
        ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.show()


def print_confusions(preds, labels, class_names, n=5):
    """間違いが多かった組み合わせ（正解 → 予測）を表示する"""
    pairs = Counter((class_names[t], class_names[p]) for t, p in zip(labels.tolist(), preds.tolist()) if t != p)
    for (true_name, pred_name), count in pairs.most_common(n):
        print(f"  {count:3d}  {true_name} -> {pred_name}")


def show_mistakes(data, preds, labels, class_names, n=4, reference=None):
    """間違いが多かった組み合わせ（正解 → 予測）から1枚ずつ、間違えた画像を表示する

    reference（データセット）を渡すと、右に正解のクラスと予測したクラスの見本の画像を並べる。
    """
    pairs = Counter((t, p) for t, p in zip(labels.tolist(), preds.tolist()) if t != p)
    rows = [int(np.flatnonzero(((labels == t) & (preds == p)).numpy())[0]) for (t, p), _ in pairs.most_common(n)]
    cols = 1 if reference is None else 3
    fig, axes = plt.subplots(len(rows), cols, figsize=(3.3 * cols, 3 * len(rows)), squeeze=False)
    examples = None if reference is None else class_examples(reference)
    for row, i in zip(axes, rows):
        t, p = int(labels[i]), int(preds[i])
        row[0].imshow(data[i]["image"].convert("RGB"))
        row[0].set_title(f"test image\ntrue: {class_names[t]}\npred: {class_names[p]}", fontsize=9)
        if reference is not None:
            row[1].imshow(reference[examples[t]]["image"].convert("RGB"))
            row[1].set_title(f"example of true class\n{class_names[t]}", fontsize=9, color="tab:green")
            row[2].imshow(reference[examples[p]]["image"].convert("RGB"))
            row[2].set_title(f"example of predicted class\n{class_names[p]}", fontsize=9, color="tab:red")
        for ax in row:
            ax.axis("off")
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def show_feature_maps(model, image, transform, device, class_names=None,
                      stages=("act1", "layer1", "layer2", "layer3", "layer4"), n=6):
    """層ごとの特徴マップを、平均的に強く反応したものから n 枚ずつ表示する（p71）

    その下に、分類ヘッドの中身（平均した後の特徴ベクトルと、クラスごとの確率）を表示する（p80〜82）。
    stages は ResNet 系のモデルの層の名前。
    """
    maps = {}
    hooks = [getattr(model, s).register_forward_hook(lambda m, i, o, s=s: maps.__setitem__(s, o[0].cpu()))
             for s in (*stages, "global_pool")]
    model.eval()
    x = transform(image.convert("RGB")).unsqueeze(0).to(device)
    logits = model(x)[0].cpu()
    probs = logits.softmax(dim=0)
    for h in hooks:
        h.remove()

    fig = plt.figure(figsize=((n + 1) * 1.7, (len(stages) + 3) * 1.8))
    grid = fig.add_gridspec(len(stages) + 3, n + 1)
    for r, s in enumerate(stages):
        row = [fig.add_subplot(grid[r, col]) for col in range(n + 1)]
        row[0].imshow(to_image(x[0]))
        top = maps[s].mean(dim=(1, 2)).topk(n).indices
        for ax, c in zip(row[1:], top):
            ax.imshow(maps[s][c], cmap="gray")
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
        c, h, w = maps[s].shape
        row[0].set_ylabel(f"{s}\n{h}x{w}\n{c} maps", fontsize=9)

    # 分類ヘッド：特徴マップをチャンネルごとに平均したベクトル → 全結合でスコア → softmax で確率
    feature = maps["global_pool"]
    rows = [(feature, f"global_pool\n{len(feature)}-dim\nvector", "channel"),
            (logits, f"fc\n{len(logits)} classes\nscore (logits)", "class"),
            (probs, f"softmax\n{len(probs)} classes\nprobability", "class")]
    for r, (values, label, xlabel) in enumerate(rows, start=len(stages)):
        ax_label = fig.add_subplot(grid[r, 0])
        ax_label.axis("off")
        ax_label.text(1.0, 0.5, label, ha="right", va="center", fontsize=9, transform=ax_label.transAxes)
        ax = fig.add_subplot(grid[r, 1:])
        ax.bar(range(len(values)), values.numpy(), width=1.0)
        ax.set_xlim(-0.5, len(values) - 0.5)
        ax.set_xlabel(xlabel)
        if xlabel == "class":
            ax.axhline(0, color="black", linewidth=0.5)
            if class_names is not None:  # 確率（スコア）の高い3クラスに名前を付ける
                for i in probs.topk(3).indices:
                    ax.annotate(class_names[i], (int(i), values[i].item()), textcoords="offset points",
                                xytext=(0, 3), ha="center", fontsize=8)
                low, high = values.min().item(), values.max().item()
                ax.set_ylim(min(0, low * 1.1), high + (high - min(0, low)) * 0.25)
    plt.tight_layout()
    plt.show()
