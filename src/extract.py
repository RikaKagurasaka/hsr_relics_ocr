from PIL import ImageGrab
import cv2 as cv
import win32gui
import numpy as np
import time
import math
import win32api
import win32con
import torch
from einops import *
import yaml
from model_training import *
from model_training import config
from typing import List
import argparse
import ctypes

parser = argparse.ArgumentParser()
parser.add_argument("--relics-count", "-n", type=int, required=True)


class RelicsImageData:
    position: np.ndarray
    level: np.ndarray
    suit: np.ndarray
    primary_attr: np.ndarray
    primary_val: np.ndarray
    secondary_attr: List[np.ndarray]
    secondary_val: List[np.ndarray]


def crop(idx: int, hwnd: int):
    l, t = win32gui.ClientToScreen(hwnd, (0, 0))
    im = ImageGrab.grab(bbox=(l, t, l + 1600, t + 900), all_screens=True)
    im = cv.cvtColor(np.array(im), cv.COLOR_RGB2BGR)
    detail_im = im[108 : 108 + 430, 1165 : 1165 + 375]
    detail_thresh_primary = np.logical_and(detail_im[:, :, -1] > 190, detail_im[:, :, 0] < 127)
    detail_thresh_secondary = np.logical_and(detail_im[:, :, -1] > 190, detail_im[:, :, 0] > 180)

    position = np.uint8(detail_thresh_secondary[129 : 129 + 22, 13 : 13 + 66])
    level = np.uint8(detail_thresh_secondary[152 : 152 + 28, 14 : 14 + 66])
    primary_attr = np.uint8(detail_thresh_primary[223 : 223 + 33, 35:250])
    primary_val = np.uint8(detail_thresh_primary[223 : 223 + 33, 270:])

    secondary_attr = np.uint8([detail_thresh_secondary[258 + i * 33 : 258 + 33 + i * 33, 35:250] for i in range(4)])
    secondary_val = np.uint8([detail_thresh_secondary[258 + i * 33 : 258 + 33 + i * 33, 270:] for i in range(4)])

    secondary_mask = np.bool_([(secondary_val[i] > 0).sum() < 5 for i in range(4)])
    secondary_mask = np.cumsum(secondary_mask, axis=0)

    if secondary_mask.any():
        secondary_attr = secondary_attr[secondary_mask == 0, :]
        secondary_val = secondary_val[secondary_mask == 0, :]
    suit = np.uint8(detail_thresh_primary[254 : 254 + 174, :])

    def cut_off(img):
        x, y, w, h = cv.boundingRect(img)
        return img[y : y + h, x : x + w]

    def resize(img, resized):
        return cv.resize(cut_off(img), resized)

    suit = resize(suit, (200, 25))
    primary_attr = resize(primary_attr, (200, 25))
    primary_val = cut_off(primary_val)
    position = resize(position, (60, 20))
    level = cut_off(level)
    secondary_attr = [resize(secondary_attr[i], (200, 25)) for i in range(secondary_attr.shape[0])]
    secondary_val = [cut_off(secondary_val[i]) for i in range(secondary_val.shape[0])]

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
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN | win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    @staticmethod
    def scoll_down(num):
        for _ in range(num):
            win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, -120, 0)
            time.sleep(0.05)


def enum_(count):
    hwnd = win32gui.FindWindow("UnityWndClass", None)
    # gw_wnd = [w for w in gw.getAllWindows() if w._hWnd == hwnd][0]
    # gw_wnd.activate()
    win32gui.SetForegroundWindow(hwnd)
    time.sleep(1)
    idx = 0
    scrolled = 0
    datas = []
    lines = math.ceil(count / row_count)
    for j in range(lines):
        for i in range(row_count):
            assert win32gui.GetWindowRect(hwnd)[3] > 0, RuntimeError("Window is not visible")
            Mouse.move_to(150 + i * dw, 200 + min(4, j) * dh, hwnd)
            Mouse.click()
            datas.append(crop(idx, hwnd))
            idx += 1
            if j * row_count + i + 1 == count:
                return datas
        total_to_scoll = max(0, j - 3) * dh / scroll_unit
        to_scroll = round(total_to_scoll - scrolled)
        scrolled += to_scroll
        Mouse.scoll_down(to_scroll)
    return datas


def _split_im(im):
    hist = np.sum(im, axis=0)
    split_at = np.argwhere(np.logical_and(hist == 0, np.roll(hist, 1) != 0)).flatten().tolist()
    split_at = [0] + split_at + [hist.size]
    splited = []
    for from_, to in zip(split_at[:-1], split_at[1:]):
        img = im[:, from_:to]
        l, t, w, h = cv.boundingRect(img)
        img = img[t : t + h, l : l + w]
        img = cv.resize(img, (20, 20))
        img = torch.from_numpy(img).float()
        splited.append(img)
    return torch.stack(splited)


def recognize(data):
    value_dict = np.array(config["value"])

    SuitClassifier.load_state_dict(torch.load("./model/SuitClassifier.pt"))
    SuitClassifier.eval()
    AttrClassifier.load_state_dict(torch.load("./model/AttrClassifier.pt"))
    AttrClassifier.eval()
    PositionClassifier.load_state_dict(torch.load("./model/PositionClassifier.pt"))
    PositionClassifier.eval()
    ValueClassifier.load_state_dict(torch.load("./model/ValueClassifier.pt"))
    ValueClassifier.eval()

    position, _ = pack([d.position for d in data], "* h w")
    suits, _ = pack([d.suit for d in data], "* h w")
    primary_attr, _ = pack([d.primary_attr for d in data], "* h w")
    secondary_attr, secondary_attr_ps = pack([np.array(d.secondary_attr) for d in data], "* h w")
    position, suits, primary_attr, secondary_attr = map(lambda x: torch.from_numpy(x).float(), [position, suits, primary_attr, secondary_attr])
    level, level_ps = pack([_split_im(d.level) for d in data], "* h w")
    primary_val, primary_value_ps = pack([_split_im(d.primary_val) for d in data], "* h w")
    secondary_val, secondary_val_ps = zip(*[pack([_split_im(im) for im in d.secondary_val], "* h w") for d in data])
    secondary_val, _ = pack(secondary_val, "* h w")

    position_pred = torch.argmax(PositionClassifier(position), dim=1)
    suits_pred = torch.argmax(SuitClassifier(suits), dim=1)
    primary_attr_pred = torch.argmax(AttrClassifier(primary_attr), dim=1)
    secondary_attr_pred = unpack(torch.argmax(AttrClassifier(secondary_attr), dim=1), secondary_attr_ps, "*")
    level_pred = list(map(lambda x: "".join(x), unpack(value_dict[torch.argmax(ValueClassifier(level), dim=1)], level_ps, "*")))
    primary_val_pred = list(map(lambda x: "".join(x), unpack(value_dict[torch.argmax(ValueClassifier(primary_val), dim=1)], primary_value_ps, "*")))

    processed_sec_val_count = 0
    secondary_val_pred_raw = value_dict[torch.argmax(ValueClassifier(secondary_val), dim=1)]
    secondary_val_pred = []
    for relic_attr_ps in secondary_val_ps:
        processing_count = sum([i[0] for i in relic_attr_ps])
        pred = secondary_val_pred_raw[processed_sec_val_count : processed_sec_val_count + processing_count]
        secondary_val_pred.append(list(map(lambda x: "".join(x), unpack(pred, relic_attr_ps, "*"))))
        processed_sec_val_count += processing_count
    result = []
    for i in range(len(data)):
        result.append(
            {
                "suit": config["suit"][suits_pred[i].item()],
                "position": config["position"][position_pred[i].item()],
                "level": int(level_pred[i][1:]),
                "primary_attr": config["attr"][primary_attr_pred[i].item()],
                "primary_val": float(primary_val_pred[i][:-1]) if primary_val_pred[i].endswith("%") else int(primary_val_pred[i]),
                "secondary_attr": [config["attr"][i] for i in secondary_attr_pred[i]],
                "secondary_val": [float(i[:-1]) if i.endswith("%") else int(i) for i in secondary_val_pred[i]],
            }
        )
    result = [v for i, v in enumerate(result) if v != result[i - 1]]
    yaml.dump(result, open("result.yaml", "w", encoding="utf8"), allow_unicode=True, sort_keys=False)


if __name__ == "__main__":
    if not ctypes.windll.shell32.IsUserAnAdmin():
        raise RuntimeError("Please run as administrator.")
    args = parser.parse_args()
    imgs = enum_(args.relics_count)
    recognize(imgs)
