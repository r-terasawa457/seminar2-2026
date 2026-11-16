"""学習した重みで、画像のクラスを予測する（推論。p17）

例:
    python predict.py outputs/best.pt assets/chelsea.png assets/coffee.png
"""
import argparse

import torch
from PIL import Image

from data import build_transforms
from engine import predict
from models import build_model


def main():
    parser = argparse.ArgumentParser(description="学習した重みで画像のクラスを予測する")
    parser.add_argument("checkpoint", help="train.py が保存した重み（best.pt）")
    parser.add_argument("images", nargs="+", help="予測する画像のファイル")
    parser.add_argument("--topk", type=int, default=5, help="上位いくつまで表示するか")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint["class_names"]
    # F を作り、θ に学習した値を入れる
    model = build_model(checkpoint["model_name"], len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    _, eval_transform = build_transforms()

    for path in args.images:
        probs = predict(model, Image.open(path), eval_transform, device)
        top = probs.topk(min(args.topk, len(class_names)))
        print(path)
        for p, i in zip(top.values, top.indices):
            print(f"  {p.item():6.1%}  {class_names[i]}")


if __name__ == "__main__":
    main()
