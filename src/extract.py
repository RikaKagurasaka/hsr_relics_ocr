from PIL import ImageGrab, Image
import win32gui
import numpy as np
import time
import math
import win32api
import win32con
import yaml
from typing import List, Union
import argparse
import ctypes
import onnxruntime as ort

config = yaml.load(open("config.yaml", "r", encoding="utf8"), Loader=yaml.FullLoader)
value_dict = np.array(config["value"])

parser = argparse.ArgumentParser()
parser.add_argument("--relics-count", "-n", type=int, default=20)


SuitClassifier = ort.InferenceSession("./model/SuitClassifier.onnx")
AttrClassifier = ort.InferenceSession("./model/AttrClassifier.onnx")
PositionClassifier = ort.InferenceSession("./model/PositionClassifier.onnx")
ValueClassifier = ort.InferenceSession("./model/ValueClassifier.onnx")


class RelicsImageData:
    position: Image.Image
    level: Image.Image
    suit: Image.Image
    primary_attr: Image.Image
    primary_val: Image.Image
    secondary_attr: List[Image.Image]
    secondary_val: List[Image.Image]


class RelicsData:
    position: int
    level: int
    suit: int
    primary_attr: int
    primary_val: Union[int, float]
    secondary_attr: List[int]
    secondary_val: List[Union[int, float]]

    def as_dict(self):
        return {
            "position": config["position"][self.position],
            "level": self.level,
            "suit": config["suit"][self.suit],
            "primary_attr": [config["attr"][self.primary_attr], self.primary_val],
            "secondary_attr": [
                [config["attr"][self.secondary_attr[i]], self.secondary_val[i]]
                for i in range(len(self.secondary_attr))
            ],
        }


def crop(idx: int, hwnd: int):
    l, t = win32gui.ClientToScreen(hwnd, (0, 0))
    im = ImageGrab.grab(bbox=(l, t, l + 1600, t + 900), all_screens=True)
    detail_im = im.crop((1165, 108, 1165 + 375, 108 + 430))
    detail_im = np.array(detail_im)
    detail_thresh_primary = Image.fromarray(
        np.logical_and(detail_im[:, :, 0] > 190, detail_im[:, :, 2] < 127)
    )
    detail_thresh_secondary = Image.fromarray(
        np.logical_and(detail_im[:, :, 0] > 190, detail_im[:, :, 2] > 180)
    )

    position = detail_thresh_secondary.crop((13, 129, 13 + 66, 129 + 22))
    level = detail_thresh_secondary.crop((14, 152, 14 + 66, 152 + 28))
    primary_attr = detail_thresh_primary.crop((35, 223, 250, 223 + 33))
    primary_val = detail_thresh_primary.crop((270, 223, 375, 223 + 33))

    secondary_attr = [
        detail_thresh_secondary.crop((35, 258 + i * 33, 250, 258 + (i + 1) * 33))
        for i in range(4)
    ]
    secondary_val = [
        detail_thresh_secondary.crop((270, 258 + i * 33, 375, 258 + (i + 1) * 33))
        for i in range(4)
    ]

    secondary_mask = (
        np.array(list(map(lambda x: np.array(x).sum(), secondary_val))) == 0
    )
    secondary_mask = np.cumsum(secondary_mask, axis=0) == 0

    secondary_attr = [secondary_attr[i] for i in range(4) if secondary_mask[i]]
    secondary_val = [secondary_val[i] for i in range(4) if secondary_mask[i]]

    suit = detail_thresh_primary.crop((0, 254, 374, 254 + 174))

    def cut_off(img: Image.Image):
        return img.crop(img.getbbox())

    def resize(img, resized):
        return cut_off(img).resize(resized)

    suit = resize(suit, (200, 25))
    primary_attr = resize(primary_attr, (200, 25))
    primary_val = cut_off(primary_val)
    position = resize(position, (60, 20))
    level = cut_off(level)
    secondary_attr = [resize(im, (200, 25)) for im in secondary_attr]
    secondary_val = [cut_off(im) for im in secondary_val]

    data = RelicsImageData()
    data.position = position
    data.level = level
    data.suit = suit
    data.primary_attr = primary_attr
    data.primary_val = primary_val
    data.secondary_attr = secondary_attr
    data.secondary_val = secondary_val

    return data


dw = 104
dh = 125
scroll_unit = 25.2
row_count = 9


class Mouse:
    @staticmethod
    def move_to(x, y, hwnd):
        x, y = win32gui.ClientToScreen(hwnd, (x, y))
        win32api.SetCursorPos((x, y))

    @staticmethod
    def click():
        win32api.mouse_event(
            win32con.MOUSEEVENTF_LEFTDOWN | win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0
        )

    @staticmethod
    def scoll_down(num):
        for _ in range(num):
            win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, -120, 0)
            time.sleep(0.05)


def enum_(count):
    hwnd = win32gui.FindWindow("UnityWndClass", None)
    win32gui.SetForegroundWindow(hwnd)
    time.sleep(1)
    idx = 0
    scrolled = 0
    datas = []
    lines = math.ceil(count / row_count)
    for j in range(lines):
        for i in range(row_count):
            assert win32gui.GetWindowRect(hwnd)[3] > 0, RuntimeError(
                "Window is not visible"
            )
            Mouse.move_to(150 + i * dw, 200 + min(4, j) * dh, hwnd)
            Mouse.click()
            datas.append(predict(crop(idx, hwnd)))
            idx += 1
            if j * row_count + i + 1 == count:
                return datas
        total_to_scoll = max(0, j - 3) * dh / scroll_unit
        to_scroll = round(total_to_scoll - scrolled)
        scrolled += to_scroll
        Mouse.scoll_down(to_scroll)
    return datas


def _split_im(im: Image.Image):
    arr = np.array(im)
    hist = np.sum(arr, axis=0)
    split_at = (
        np.argwhere(np.logical_and(hist == 0, np.roll(hist, 1) != 0)).flatten().tolist()
    )
    split_at = [0] + split_at + [hist.size]
    splited = []
    for from_, to in zip(split_at[:-1], split_at[1:]):
        img = im.crop((from_, 0, to, im.height))
        box = img.getbbox()
        img = img.crop(box)
        img = img.resize((20, 20))
        splited.append(img)
    return splited


input_name = "input_0"


def predict(d: RelicsImageData):
    def _p(model, input):
        input = np.float32(input)
        input = input[None, ...]
        return np.argmax(model.run(None, {input_name: input})[0], axis=1)[0]

    def _pv(inputs):
        return "".join(
            value_dict[
                np.argmax(
                    ValueClassifier.run(None, {input_name: np.float32(inputs)})[0],
                    axis=1,
                )
            ]
        )

    result = RelicsData()
    result.position = _p(PositionClassifier, d.position)
    result.suit = _p(SuitClassifier, d.suit)
    result.primary_attr = _p(AttrClassifier, d.primary_attr)
    result.secondary_attr = [_p(AttrClassifier, im) for im in d.secondary_attr]

    result.level = int(_pv(_split_im(d.level))[1:])
    primary_val_str = _pv(_split_im(d.primary_val))
    result.primary_val = (
        float(primary_val_str[:-1]) if "%" in primary_val_str else int(primary_val_str)
    )
    secondary_val_strs = [_pv(_split_im(im)) for im in d.secondary_val]
    result.secondary_val = [
        float(secondary_val_str[:-1])
        if "%" in secondary_val_str
        else int(secondary_val_str)
        for secondary_val_str in secondary_val_strs
    ]

    return result


if __name__ == "__main__":
    if not ctypes.windll.shell32.IsUserAnAdmin():
        raise RuntimeError("Please run as administrator.")
    args = parser.parse_args()
    datas = enum_(args.relics_count)
    result = list(map(lambda v: v.as_dict(), datas))
    result = [v for i, v in enumerate(result) if v != result[i - 1]]
    yaml.dump(
        result,
        open("result.yaml", "w", encoding="utf8"),
        allow_unicode=True,
        sort_keys=False,
    )
