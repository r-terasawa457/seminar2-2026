"""データを用意する：読み込み、3つへの分割、前処理、データ拡張、mini-batch

スライド「画像分類の実践 ① データを用意する」（p88〜92）と mini-batch（p96）に対応する。
"""
from functools import partial

from datasets import load_dataset
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
from torchvision import transforms

# 授業の途中で内容が変わらないよう、リビジョンを固定している
DATASET_ID = "timm/oxford-iiit-pet"
DATASET_REVISION = "089695c834a7deb60505b7cc506672db1c31a6aa"

IMAGE_SIZE = 224
# 学習済みモデルを ImageNet で学習したときの平均と標準偏差（p91）
MEAN = IMAGENET_DEFAULT_MEAN
STD = IMAGENET_DEFAULT_STD


def load_splits(seed=0, image_dir=None, max_train=None, max_test=None):
    """データセットを読み込み、train / validation / test の3つに分ける（p88、p90）

    image_dir を指定すると、フォルダ名をクラス名として自分の画像を読み込む。
    """
    if image_dir is None:
        dataset = load_dataset(DATASET_ID, revision=DATASET_REVISION)
    else:
        # test が無いので、2割を test として取り分ける
        images = load_dataset("imagefolder", data_dir=image_dir)["train"]
        dataset = images.train_test_split(test_size=0.2, seed=seed, stratify_by_column="label")
    class_names = dataset["train"].features["label"].names

    # 元の train を train と validation に分ける。各クラスの比率がそろうように分割する
    split = dataset["train"].train_test_split(test_size=0.2, seed=seed, stratify_by_column="label")
    train_data, val_data, test_data = split["train"], split["test"], dataset["test"]

    # GPU が無いときなどに、枚数を減らして早く終わらせる
    if max_train is not None:
        train_data = train_data.shuffle(seed=seed).select(range(min(max_train, len(train_data))))
        val_data = val_data.shuffle(seed=seed).select(range(min(max_train // 4, len(val_data))))
    if max_test is not None:
        test_data = test_data.shuffle(seed=seed).select(range(min(max_test, len(test_data))))
    return train_data, val_data, test_data, class_names


def build_transforms(image_size=IMAGE_SIZE, mean=MEAN, std=STD):
    """前処理（p91）とデータ拡張（p92）を作る。学習用と評価用の2つを返す"""
    normalize = transforms.Normalize(mean, std)
    # 評価用：大きさをそろえて（縮小して中央を切り抜く）、値の範囲をそろえる
    eval_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    # 学習用：毎回ランダムに一部を切り抜いて拡大し、左右を反転する
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, eval_transform


def _apply(batch, transform):
    return {
        "image": [transform(img.convert("RGB")) for img in batch["image"]],
        "label": batch["label"],
    }


def make_loader(data, transform, batch_size, shuffle, num_workers=0):
    """画像をテンソルに変換し、batch_size 枚ずつ束ねて取り出す（p96）"""
    return DataLoader(data.with_transform(partial(_apply, transform=transform)),
                      batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
