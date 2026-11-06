"""ノートブックで結果を見るための表示用の関数

学習の仕組みとは関係しないので、ノートブックでは呼び出すだけにしている。
Colab には日本語のフォントが無いので、図の文字は英語にしている。
"""
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from data import MEAN, STD


def to_image(tensor, mean=MEAN, std=STD):
    """正規化したテンソル (3, H, W) を、表示できる画像 (H, W, 3) に戻す"""
    x = tensor.cpu() * torch.tensor(std)[:, None, None] + torch.tensor(mean)[:, None, None]
    return x.clamp(0, 1).permute(1, 2, 0).numpy()


def show_samples(data, class_names, names=None, n=8):
    """クラスごとに1枚ずつ、見本の画像を並べる（p84）。names は大文字と小文字を区別しない"""
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
    """データ1件（画像と正解ラベルの組）を表示する（p85）"""
    image = sample["image"]
    plt.figure(figsize=(4, 4))
    plt.imshow(image.convert("RGB"))
    plt.title(f"label: {sample['label']}  ({class_names[sample['label']]})")
    plt.axis("off")
    plt.show()


def show_preprocess(image, transform):
    """前処理の前と後を並べる（p87）"""
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
    """同じ画像にデータ拡張を何回かかけて並べる（p88）"""
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
    """mini-batch の中身を、先頭から n 枚表示する（p92）"""
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.4))
    for ax, x, label in zip(axes, batch["image"], batch["label"]):
        ax.imshow(to_image(x))
        ax.set_title(class_names[label], fontsize=8)
        ax.axis("off")
    plt.show()


@torch.no_grad()
def print_feature_shapes(model, device, image_size=224):
    """層を通るごとに、特徴マップの形がどう変わるかを表示する（p69〜70）"""
    model.eval()
    x = torch.zeros(1, 3, image_size, image_size, device=device)
    print(f"{'input':12s} {tuple(x.shape[1:])}")
    for name, module in model.named_children():
        x = module(x)
        print(f"{name:12s} {str(tuple(x.shape[1:])):18s} {type(module).__name__}")


def show_kernels(model, n=64):
    """最初の畳み込み層のカーネルを並べる。1枚ごとに最小が黒、最大が白（p65）"""
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
    """各クラスの確率を、上位 k 個の棒グラフで表示する（p74、p99）"""
    top = probs.topk(min(k, len(class_names)))
    names = [class_names[i] for i in top.indices]
    colors = ["tab:red" if i == true_label else "tab:blue" for i in top.indices]
    plt.figure(figsize=(6, 0.3 * len(names) + 1))
    plt.barh(names[::-1], top.values.numpy()[::-1], color=colors[::-1])
    plt.axvline(1 / len(class_names), color="gray", linestyle="--", label=f"1/{len(class_names)}")
    plt.xlabel("probability")
    plt.legend()
    plt.show()


def show_prediction(image, probs, class_names, k=5):
    """画像と、上位 k クラスの確率を並べて表示する（推論。p17）"""
    top = probs.topk(min(k, len(class_names)))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), gridspec_kw={"width_ratios": [1, 1.4]})
    axes[0].imshow(image.convert("RGB"))
    axes[0].axis("off")
    names = [class_names[i] for i in top.indices]
    axes[1].barh(names[::-1], top.values.numpy()[::-1])
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("probability")
    plt.tight_layout()
    plt.show()


def plot_history(history):
    """epoch ごとの Loss と正解率を、train と validation で比べる（p97〜98）"""
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


def print_confusions(preds, labels, class_names, n=5):
    """間違いが多かった組み合わせ（正解 → 予測）を表示する"""
    pairs = Counter((class_names[t], class_names[p]) for t, p in zip(labels.tolist(), preds.tolist()) if t != p)
    for (true_name, pred_name), count in pairs.most_common(n):
        print(f"  {count:3d}  {true_name} -> {pred_name}")


def show_mistakes(data, preds, labels, class_names, n=8):
    """間違えた画像を、正解と予測を付けて並べる"""
    wrong = np.flatnonzero((preds != labels).numpy())[:n]
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(12, 6.5))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, wrong):
        i = int(i)
        ax.imshow(data[i]["image"].convert("RGB"))
        ax.set_title(f"true: {class_names[labels[i]]}\npred: {class_names[preds[i]]}", fontsize=9)
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def show_feature_maps(model, image, transform, device, stages=("act1", "layer1", "layer2", "layer3", "layer4"), n=6):
    """層ごとの特徴マップを、平均的に強く反応したものから n 枚ずつ表示する（p71）

    stages は ResNet 系のモデルの層の名前。
    """
    maps = {}
    hooks = [getattr(model, s).register_forward_hook(lambda m, i, o, s=s: maps.__setitem__(s, o[0].cpu()))
             for s in stages]
    model.eval()
    x = transform(image.convert("RGB")).unsqueeze(0).to(device)
    model(x)
    for h in hooks:
        h.remove()

    fig, axes = plt.subplots(len(stages), n + 1, figsize=((n + 1) * 1.7, len(stages) * 1.8))
    for row, s in zip(axes, stages):
        row[0].imshow(to_image(x[0]))
        top = maps[s].mean(dim=(1, 2)).topk(n).indices
        for ax, c in zip(row[1:], top):
            ax.imshow(maps[s][c], cmap="gray")
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
        c, h, w = maps[s].shape
        row[0].set_ylabel(f"{s}\n{h}x{w}\n{c} maps", fontsize=9)
    plt.tight_layout()
    plt.show()
