"""モデル F を作る

スライド「CNNによる画像分類」（p76）、「学習済みモデルから始める」（p98）に対応する。
"""
import timm


def build_model(model_name, num_classes, pretrained=True):
    """F（ネットワークの形）を作り、θ（重み）に学習済みの値を入れる

    num_classes を指定すると、分類ヘッドが num_classes クラス用に付け替えられる（p98）。
    使えるモデルの名前は timm.list_models(pretrained=True) で調べられる。
    """
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
