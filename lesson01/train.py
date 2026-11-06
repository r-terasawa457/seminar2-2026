"""画像分類モデルを学習する

ノートブックの「① データを用意する」〜「③ 結果を確かめる」と同じ処理を、
コマンドラインから実行できるようにしたもの。

例:
    python train.py
    python train.py --model resnet50.a1_in1k --epochs 5
    python train.py --no-pretrained
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from data import build_transforms, load_splits, make_loader
from engine import evaluate, fit
from models import build_model


def parse_args():
    # 人間が決める学習の設定（ハイパーパラメータ。p93）
    parser = argparse.ArgumentParser(description="画像分類モデルを学習する")
    parser.add_argument("--model", default="resnet18.a1_in1k", help="timm のモデル名")
    parser.add_argument("--no-pretrained", action="store_true", help="学習済みの重みを使わない")
    parser.add_argument("--epochs", type=int, default=3, help="データ全体を何周するか")
    parser.add_argument("--batch-size", type=int, default=64, help="1回の更新に使う枚数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学習率")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-dir", default=None, help="自分の画像のフォルダ（フォルダ名がクラス名）")
    parser.add_argument("--max-train", type=int, default=None, help="train の枚数の上限")
    parser.add_argument("--max-test", type=int, default=None, help="test の枚数の上限")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out", default="outputs", help="結果を保存するフォルダ")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # ① データを用意する
    train_data, val_data, test_data, class_names = load_splits(
        args.seed, args.image_dir, args.max_train, args.max_test)
    print("train / validation / test:", len(train_data), len(val_data), len(test_data))
    train_transform, eval_transform = build_transforms()
    train_loader = make_loader(train_data, train_transform, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_data, eval_transform, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_data, eval_transform, args.batch_size, False, args.num_workers)

    # モデル、Loss、更新の方法
    model = build_model(args.model, len(class_names), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ② 学習を進める
    history, best_acc = fit(model, train_loader, val_loader, criterion, optimizer, device, args.epochs)

    # ③ 結果を確かめる：一番良い重みで、test を1回だけ測る
    _, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
    print(f"best val acc {best_acc:.3f}  |  test acc {test_acc:.3f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_name": args.model, "class_names": class_names, "state_dict": model.state_dict()},
               out / "best.pt")
    history.update(best_val_acc=best_acc, test_acc=test_acc, args=vars(args))
    (out / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved:", out / "best.pt", out / "history.json")


if __name__ == "__main__":
    main()
