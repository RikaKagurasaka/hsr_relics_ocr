import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from io import BytesIO
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from einops.layers.torch import Rearrange
import cv2 as cv
import yaml
from find_game import find_font_file

config = yaml.load(open("config.yaml", "r", encoding="utf8"), Loader=yaml.FullLoader)


def _get_model(linear_in, linear_out):
    return nn.Sequential(
        Rearrange("b h w -> b () h w"),
        nn.Dropout2d(p=0.5),
        nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout2d(p=0.2),
        nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        Rearrange("b c h w -> b (h w c)"),
        nn.Linear(in_features=linear_in, out_features=linear_out),
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




def _generate(text: str):
    plt.figure()
    plt.axis("off")
    plt.text(0, 0, text, {"fontproperties": "SDK_SC_Web", "size": 18})
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close()
    buf.seek(0)
    img_array = np.asarray(bytearray(buf.read()), dtype=np.uint8)
    img = 255 - cv.imdecode(img_array, cv.IMREAD_GRAYSCALE)
    img = np.uint8(img > 180) * 255
    l, t, w, h = cv.boundingRect(img)
    img = img[t : t + h, l : l + w]
    print("generating image %s" % text)
    return img


def _get_text_img(texts, resized):
    results = []
    for text in texts:
        img = _generate(text)
        img = cv.resize(img, resized)
        img = np.uint8(img > 180) * 255
        results.append(img)
    return torch.tensor(np.array(results), dtype=torch.float32) / 255


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


if __name__ == "__main__":
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

    print('exporting state dicts')


    torch.save(SuitClassifier.state_dict(), "./model/SuitClassifier.pt")
    torch.save(AttrClassifier.state_dict(), "./model/AttrClassifier.pt")
    torch.save(PositionClassifier.state_dict(), "./model/PositionClassifier.pt")
    torch.save(ValueClassifier.state_dict(), "./model/ValueClassifier.pt")

    dummy_input_suit = torch.randn(1, 200, 25)
    dummy_input_attr = torch.randn(1, 200, 25)
    dummy_input_position = torch.randn(1, 60, 20)
    dummy_input_value = torch.randn(1, 20, 20)

    print('exporting to onnx')

    torch.onnx.export(SuitClassifier, dummy_input_suit, "./model/SuitClassifier.onnx", input_names=["input_0"], output_names=["output_0"], dynamic_axes={"input_0": [0], "output_0": [0]})
    torch.onnx.export(AttrClassifier, dummy_input_attr, "./model/AttrClassifier.onnx", input_names=["input_0"], output_names=["output_0"], dynamic_axes={"input_0": [0], "output_0": [0]})
    torch.onnx.export(PositionClassifier, dummy_input_position, "./model/PositionClassifier.onnx", input_names=["input_0"], output_names=["output_0"], dynamic_axes={"input_0": [0], "output_0": [0]})
    torch.onnx.export(ValueClassifier, dummy_input_value, "./model/ValueClassifier.onnx", input_names=["input_0"], output_names=["output_0"], dynamic_axes={"input_0": [0], "output_0": [0]})

