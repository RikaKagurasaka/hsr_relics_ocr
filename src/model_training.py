import numpy as np
from find_game import find_font_file
import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import yaml
from PIL import Image, ImageFont, ImageDraw

font = ImageFont.truetype(find_font_file(), 18)
config = yaml.load(open("config.yaml", "r", encoding="utf8"), Loader=yaml.FullLoader)


def _get_model(linear_in, linear_out):
    return nn.Sequential(
        Rearrange("b h w -> b () h w"),
        nn.Dropout2d(p=0.5),
        nn.BatchNorm2d(num_features=1),
        nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout2d(p=0.2),
        nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        Rearrange("b c h w -> b (h w c)"),
        nn.Linear(in_features=linear_in * 2, out_features=linear_out),
        nn.Softmax(dim=1),
    )


SuitClassNumber = len(config["suit"])
SuitClassifier = _get_model(768, SuitClassNumber)

AttrClassNumber = len(config["attr"])
AttrClassifier = _get_model(768, AttrClassNumber)

PositionClassNumber = len(config["position"])
PositionClassifier = _get_model(156, PositionClassNumber)

ValueClassNumber = len(config["value"])
ValueClassifier = _get_model(36, ValueClassNumber)


def _generate(text):
    img = Image.new("1", (1024, 1024), 0)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font, fill=1)
    l, t, r, b = img.getbbox()
    img = img.crop((l, t, r, b))
    return img


def _get_text_img(texts, resized):
    results = []
    for text in texts:
        img = _generate(text)
        img = img.resize(resized)
        results.append(img)
    return torch.tensor(np.array(results), dtype=torch.float32)


def _train(model, imgs, name, epoches=1000):
    print("start training %s" % name)
    y_true = torch.arange(imgs.size(0))
    optimizer = Adam(model.parameters(), lr=0.001)
    model.train()
    for _ in range(epoches):
        y_pred = model(imgs)
        loss = F.cross_entropy(y_pred, y_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("loss %.2f" % loss.item())


def main():
    import torch
    from matplotlib.font_manager import fontManager
    from find_game import find_font_file

    fontManager.addfont(find_font_file())

    suit_imgs = _get_text_img(config["suit"], (200, 25))
    attr_imgs = _get_text_img(config["attr"], (200, 25))
    position_imgs = _get_text_img(config["position"], (60, 20))
    value_imgs = _get_text_img(config["value"], (20, 20))

    _train(SuitClassifier, suit_imgs, "SuitClassifier")
    _train(AttrClassifier, attr_imgs, "AttrClassifier")
    _train(PositionClassifier, position_imgs, "PositionClassifier")
    _train(ValueClassifier, value_imgs, "ValueClassifier")

    SuitClassifier.eval()
    AttrClassifier.eval()
    PositionClassifier.eval()
    ValueClassifier.eval()

    print("exporting state dicts")

    torch.save(SuitClassifier.state_dict(), "./model/SuitClassifier.pt")
    torch.save(AttrClassifier.state_dict(), "./model/AttrClassifier.pt")
    torch.save(PositionClassifier.state_dict(), "./model/PositionClassifier.pt")
    torch.save(ValueClassifier.state_dict(), "./model/ValueClassifier.pt")

    dummy_input_suit = torch.randn(1, 25, 200)
    dummy_input_attr = torch.randn(1, 25, 200)
    dummy_input_position = torch.randn(1, 20, 60)
    dummy_input_value = torch.randn(1, 20, 20)

    print("exporting to onnx")

    torch.onnx.export(
        SuitClassifier,
        dummy_input_suit,
        "./model/SuitClassifier.onnx",
        input_names=["input_0"],
        output_names=["output_0"],
        dynamic_axes={"input_0": [0], "output_0": [0]},
    )
    torch.onnx.export(
        AttrClassifier,
        dummy_input_attr,
        "./model/AttrClassifier.onnx",
        input_names=["input_0"],
        output_names=["output_0"],
        dynamic_axes={"input_0": [0], "output_0": [0]},
    )
    torch.onnx.export(
        PositionClassifier,
        dummy_input_position,
        "./model/PositionClassifier.onnx",
        input_names=["input_0"],
        output_names=["output_0"],
        dynamic_axes={"input_0": [0], "output_0": [0]},
    )
    torch.onnx.export(
        ValueClassifier,
        dummy_input_value,
        "./model/ValueClassifier.onnx",
        input_names=["input_0"],
        output_names=["output_0"],
        dynamic_axes={"input_0": [0], "output_0": [0]},
    )


if __name__ == "__main__":
    main()
