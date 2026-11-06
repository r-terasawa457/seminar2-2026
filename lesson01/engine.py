"""学習、評価、推論

スライド「学習の進め方の全体像」（p90）、「CNN の学習も、同じ繰り返し」（p76）、
「推論」（p17）に対応する。
"""
import torch


def train_one_epoch(model, loader, criterion, optimizer, device):
    """train を1周（1 epoch）学習する。mini-batch ごとに θ を1回更新する（p90〜92）"""
    model.train()
    total_loss, correct, count = 0.0, 0, 0
    for batch in loader:
        images = batch["image"].to(device)   # 入力 X
        labels = batch["label"].to(device)   # 正解 Y

        logits = model(images)               # 予測を計算する：Ŷ = F(X; θ)
        loss = criterion(logits, labels)     # Loss を計算する
        optimizer.zero_grad()                # （前回の勾配を消す）
        loss.backward()                      # 勾配を計算する（誤差逆伝播）
        optimizer.step()                     # θ を更新する

        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += len(labels)
    return total_loss / count, correct / count


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """θ を更新せずに、Loss と正解率を測る（validation、test。p86）"""
    model.eval()
    total_loss, count = 0.0, 0
    all_preds, all_labels = [], []
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * len(labels)
        count += len(labels)
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())
    preds, labels = torch.cat(all_preds), torch.cat(all_labels)
    return total_loss / count, (preds == labels).float().mean().item(), preds, labels


@torch.no_grad()
def predict(model, image, transform, device):
    """1枚の画像（PIL）を入力し、各クラスの確率を返す（推論。p17、p74）"""
    model.eval()
    x = transform(image.convert("RGB")).unsqueeze(0).to(device)  # 1枚だけの mini-batch にする
    return model(x).softmax(dim=1)[0].cpu()
