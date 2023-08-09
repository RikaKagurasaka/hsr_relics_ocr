import os
import re
import glob
from pathlib import Path


def find_game():
    log = glob.glob(os.path.join(os.environ["APPDATA"], "../LocalLow/miHoYo/*/Player.log"))
    assert len(log) == 1, "Game log file not found."
    log = open(log[0], "r", encoding="utf-8")
    for i, line in enumerate(log):
        if i > 20:
            raise RuntimeError("Game log file analysis failed.")
        if line.startswith("Loading player data from"):
            return line.replace("Loading player data from ", "").replace("data.unity3d", "").strip()
    raise RuntimeError("Game log file analysis failed.")


def find_font_file():
    gamepath = Path(find_game())
    fontpath = gamepath.joinpath("./StreamingAssets/MiHoYoSDKRes/HttpServerResources/font/zh-cn.ttf")
    assert fontpath.exists(), "Chinese font file not found. May download one somewhere?"
    return str(fontpath)
