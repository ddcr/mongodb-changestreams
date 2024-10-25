__author__ = "Domingos Rodrigues"
__email__ = "domingos.rodrigues@inventvision.com.br"
__copyright__ = "Copyright (C) 2024 Invent Vision"
__license__ = "Strictly proprietary for Invent Vision."

import base64
import io
import json
import os
import shutil
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

DEFAULT_FONT_PATH = Path(__file__).parent / Path("assets/arial.ttf")


class AppError(Exception):
    """
    Class to raise Internal Server Error Messages
    """

    def __init__(self, *args):
        if len(args) == 2:
            self.code = args[0]
            self.message = args[1]
        else:
            self.code = 500
            self.message = "Internal Server Error"


def decode_image(base64_img):
    """
    Decode a base64 string to PIL Image

    Args:
        base64_img (str): Input image in base64 format

    Returns:
        Image: Output image
    """
    msg = base64.b64decode(base64_img)
    buf = io.BytesIO(msg)
    img = Image.open(buf)
    return img


def encode_image(image):
    """
       Encode a PIL Image to base 64 format

       Args:
           image (Image): Input image

    Returns:
           str: Image in base64 format
    """
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    img_str = base64.b64encode(buf.getvalue()).decode()
    return img_str


def is_windows_path(path_str):
    """
    Determines if the given path string is in Windows format.

    Args:
        path_str (str): The path string to check.

    Returns:
        bool: True if the path is in Windows format, False otherwise.
    """
    try:
        # Attempt to parse the path using PureWindowsPath
        windows_path = PureWindowsPath(path_str)
        # Check if the path's drive attribute is not empty or if it contains backslashes
        return bool(windows_path.drive) or "\\" in path_str
    except Exception:
        return False


def assert_same_extensions(*args):
    exts = [os.path.splitext(path)[1] for path in args]
    if not (exts[1:] == exts[:-1]):
        raise OSError(f"Expected {str(args)} to have the same extensions")


def copy_file(inpath, outpath, check_ext=False):
    if check_ext and os.path.splitext(outpath)[1]:
        assert_same_extensions(inpath, outpath)

    basedir = Path(outpath)
    basedir.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy(inpath, outpath)
    except shutil.SameFileError:
        pass


def load_configs(parPath: str = "config.json"):  # Load configurations
    with open(parPath) as config_file:
        configs = json.load(config_file)
    return configs


def load_scrap_rank(filePath: str):
    data = pd.read_csv(filePath, dtype=str)
    data = data.astype({"price_ranking": "int32"})
    return data


def getConfig(configStr):
    return configs[configStr]


def draw_gt_boxes(image, bboxes, label="", font_size=None):
    fsize = font_size or max(round(sum(image.size) / 2 * 0.035), 10)
    font = ImageFont.truetype(str(DEFAULT_FONT_PATH), fsize)

    lw = max(round(sum(image.size) / 2 * 0.003), 2)
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="yellow", width=lw)
        if label:
            p1 = (bbox[0], bbox[1])
            left, top, right, bottom = font.getbbox(label)
            w = right - left
            h = bottom - top
            outside = p1[1] - h >= 0  # label fits outside box
            draw.rectangle(
                (
                    p1[0],
                    p1[1] - h if outside else p1[1],
                    p1[0] + w + 1,
                    p1[1] + 1 if outside else p1[1] + h + 1,
                ),
                fill="yellow",
            )
            draw.text(
                (p1[0], p1[1] - h if outside else p1[1]), label, fill="blue", font=font
            )


def overlay_mask(image, seg_mask):
    """_summary_

    Arguments:
        image -- _description_
        seg_mask -- _description_
    """
    # palette = [[120, 120, 120], [255, 40, 40]]  # background, scrap
    palette = [[0, 0, 0], [255, 0, 0]]  # background, scrap
    palette = np.array(palette)
    color_seg = np.zeros((seg_mask.shape[0], seg_mask.shape[1], 3), dtype=np.uint8)

    for label, color in enumerate(palette):
        color_seg[seg_mask == 255 * label, :] = color
    img = np.array(image) * 0.7 + color_seg * 0.3
    img = img.astype(np.uint8)
    return Image.fromarray(np.uint8(img))


configs = load_configs()
scrapRankDF = load_scrap_rank("classes_info.csv")
scrapRank = scrapRankDF.set_index("code")["name"].to_dict()
authToken = getConfig
