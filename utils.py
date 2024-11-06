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
from logzero import logger
from PIL import Image, ImageDraw, ImageFont
from pymongo import MongoClient

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


def singleton(cls):
    instances = {}

    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper


@singleton
class MongoConnection:
    db = None
    client = None

    def __init__(self, host: str, port: int, database="gerdau_scrap_classification"):
        try:
            self.client = MongoClient(
                host,
                port,
                username=os.getenv("MONGO_INITDB_ROOT_USERNAME", "ivision"),
                password=os.getenv("MONGO_INITDB_ROOT_PASSWORD", "ivSN"),
                maxPoolSize=20,
                minPoolSize=5,
            )
            self.db = self.client[database]
            logger.info(
                f"Connected to MongoDB '{host}:{port}'; database = '{database}'"
            )
        except Exception as e:
            logger.exception(f"Failed to connect to MongoDB: {e}")
            raise

    def get_db(self):
        return self.db


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


def count_files_in_subfolders_efficient(folder_paths):
    """Obs: consider using this with multiprocessing if necessary

    Arguments:
        folder_paths -- _description_

    Returns:
        _description_
    """
    file_counts = {}  # Dictionary to store the count of files in each subfolder

    for folder_path in folder_paths:
        count = 0
        # Using os.scandir() for better performance
        with os.scandir(folder_path) as entries:
            for entry in entries:
                if entry.is_file():  # Check if the entry is a file
                    count += 1
        file_counts[folder_path] = count  # Store the count in the dictionary

    return file_counts


def count_files_in_subfolders(folder_paths):
    file_counts = {}  # Dictionary to store the count of files in each subfolder

    for folder_path in folder_paths:
        folder = Path(folder_path)

        # Count files in the current subfolder
        file_count = sum(1 for item in folder.glob("*") if item.is_file())
        file_counts[folder_path] = file_count  # Store the count in the dictionary

    return file_counts


def copy_file(inpath, outpath, check_ext=False):
    if check_ext and os.path.splitext(outpath)[1]:
        assert_same_extensions(inpath, outpath)

    basedir = Path(outpath)
    basedir.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy(inpath, outpath)
    except shutil.SameFileError:
        pass


def add_annotation(bboxes, filetxt, img_shape, gt_id):
    img_width, img_height = img_shape
    with open(filetxt, "w") as annotation_file:
        for bbox in bboxes:
            x_center = (bbox[0] + (bbox[2] - bbox[0]) / 2) / img_width
            y_center = (bbox[1] + (bbox[3] - bbox[1]) / 2) / img_height
            width = (bbox[2] - bbox[0]) / img_width
            height = (bbox[3] - bbox[1]) / img_height
            if gt_id != -1:
                annotation_file.write(
                    f"{gt_id} {x_center} {y_center} {width} {height}\n"
                )


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


def connect_to_mongo(host: str, port: int, database="gerdau_scrap_classification"):
    """
    Establishes a connection to the MongoDB instance.
    Returns the connected database client.
    """
    try:
        mongo_client = MongoClient(
            host,
            port,
            username=os.getenv("MONGO_INITDB_ROOT_USERNAME", "ivision"),
            password=os.getenv("MONGO_INITDB_ROOT_PASSWORD", "ivSN"),
            maxPoolSize=20,
            minPoolSize=5,
        )
        db = mongo_client[database]
        logger.info(f"Connected to MongoDB '{host}:{port}'; database = '{database}'")
        return db
    except Exception as e:
        logger.exception(f"Failed to connect to MongoDB: {e}")
        raise


configs = load_configs()
scrapRankDF = load_scrap_rank("classes_info.csv")
# scrapRank = scrapRankDF.set_index("code")["name"].to_dict()
scrapRank = {row["code"]: (index, row["name"]) for index, row in scrapRankDF.iterrows()}


if __name__ == "__main__":
    print(scrapRankDF)
    print(scrapRank)
