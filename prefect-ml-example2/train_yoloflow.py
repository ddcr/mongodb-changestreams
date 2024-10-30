import os
from pathlib import Path
from typing import List

import pandas as pd
from PIL import Image
from prefect import flow, task
from prefect.logging import get_run_logger
from sklearn.model_selection import train_test_split


def get_bbox_image_slices(label_path: str):
    bbox = None
    bboxes = []
    try:
        with open(label_path, "r") as fd:
            lines = [lb.strip() for lb in fd.readlines()]
        if len(lines) == 0:
            raise Exception("Empty bbox file")

        for line in lines:
            cls, *bbox = line.split(" ")
            bbox = [float(b) for b in bbox]
            bboxes.append(bbox)

    except Exception as e:
        print(f"Exception: {e}")
        return None
    return bboxes


def xywhn_to_bbox(yolo_bb, im_w, im_h):
    # re-scale and uncenter
    bbox_pix = [
        yolo_bb[0] * im_w,
        yolo_bb[1] * im_h,
        yolo_bb[2] * im_w,
        yolo_bb[3] * im_h,
    ]
    bbox_pix = [
        bbox_pix[0] - bbox_pix[2] / 2,
        bbox_pix[1] - bbox_pix[3] / 2,
        bbox_pix[2],
        bbox_pix[3],
    ]
    # Round values
    return [int(x) for x in bbox_pix]


def move_images(image_list: List[str], crop_destdir: Path, split_name: str):
    for crop_src_relpath in image_list:
        crop_dst_relpath = crop_src_relpath.replace("images", split_name)
        crop_src_abspath = crop_destdir / crop_src_relpath
        crop_dst_abspath = crop_destdir / crop_dst_relpath

        # Ensure the destination directory exists
        crop_dst_abspath.parent.mkdir(parents=True, exist_ok=True)
        os.rename(crop_src_abspath, crop_dst_abspath)


@task(log_prints=True)
def convert_dataset_to_yolo(dataset_inputdir: str, only_matched_classes=True):
    logger = get_run_logger()
    logger.info(f"Reading data the dataset folder: '{dataset_inputdir}'")

    csvfile = Path(dataset_inputdir) / "images.csv"
    df = pd.read_csv(str(csvfile))

    # only consider files with absolute aggreement
    if only_matched_classes:
        rows_to_drop = df[df["ai_class"] != df["human_class"]].index
        df.drop(index=rows_to_drop, inplace=True)

    file_counts_per_class = df.groupby("ai_class").size().reset_index(name="file_count")
    file_total = file_counts_per_class["file_count"].sum()
    logger.info(file_counts_per_class)
    logger.info(f"total number of files = {file_total}")

    logger.info("Create DATASET for YOLO CLASSIFICATION [cropping bounding boxes]")
    yolo_dataset_dir = Path(dataset_inputdir).parent / "gerdau_yolo_dataset"
    yolo_dataset_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"yolo_dataset_dir = {yolo_dataset_dir}")

    # yolo_dataset_cropdir = yolo_dataset_dir / "images"
    # yolo_dataset_cropdir.mkdir(parents=True, exist_ok=True)

    # get column of labels
    crop_paths = []
    for row in df.itertuples():
        img_src_relpath = str(row.path)
        img_src_abspath = Path(dataset_inputdir) / img_src_relpath
        image = Image.open(str(img_src_abspath)).convert("RGB")
        im_w, im_h = image.size

        # crop if ROI is found
        label_src_relpath = img_src_relpath.replace("images", "labels")
        label_src_relpath = Path(label_src_relpath).with_suffix(".txt")
        label_src_abspath = Path(dataset_inputdir) / label_src_relpath

        # logger.info(img_src_abspath)
        # logger.info(label_src_abspath)

        img_dst_abspath = Path(yolo_dataset_dir) / img_src_relpath
        bboxes = get_bbox_image_slices(str(label_src_abspath))

        # Skip images with no labels
        if bboxes is not None:
            for ib, yolo_bbox in enumerate(bboxes):
                x, y, w, h = xywhn_to_bbox(yolo_bbox, im_w, im_h)
                image_crop = image.crop((x, y, x + w, y + h))

                img_crop_name = f"{img_dst_abspath.stem}_bb{ib}{img_dst_abspath.suffix}"
                img_crop_dst_abspath = img_dst_abspath.with_name(img_crop_name)
                # logger.info(f"{img_src_abspath} => {img_crop_dst_abspath}")
                img_crop_dst_abspath.parent.mkdir(parents=True, exist_ok=True)
                image_crop.save(str(img_crop_dst_abspath))

                img_crop_dst_relpath = img_crop_dst_abspath.relative_to(
                    yolo_dataset_dir
                )
                crop_paths.append(str(img_crop_dst_relpath))

    return yolo_dataset_dir, crop_paths


@task(log_prints=True)
def data_split(
    crop_destdir: Path,
    crop_images_list: List,
    train_split_sz: float = 0.3,
    seed: int = 42,
):
    train_list, tmp_list = train_test_split(
        crop_images_list, test_size=train_split_sz, random_state=seed
    )
    val_list, test_list = train_test_split(tmp_list, test_size=0.5, random_state=42)

    print(f"crop_images_list: {len(crop_images_list)}")
    print(f"train_list: {len(train_list)}")
    print(f"val_list: {len(val_list)}")
    print(f"test_list: {len(test_list)}")

    move_images(train_list, crop_destdir, "train")
    move_images(val_list, crop_destdir, "val")
    move_images(test_list, crop_destdir, "test")


@task
def train_model(X_train, X_test, y_train):
    pass


@task
def get_prediction(X_test, model):
    pass


@task
def evaluate_model(y_test, prediction: pd.DataFrame):
    pass


@task
def save_model(model):
    pass


@flow(log_prints=True)
def yolo_workflow(dset_inputdir: str = ""):
    crop_destdir, crop_images_list = convert_dataset_to_yolo(dset_inputdir)
    data_split(crop_destdir, crop_images_list)

    # prep_data = preprocessing(data)
    # X_train, X_test, y_train, y_test = data_split(prep_data)
    # model = train_model(X_train, X_test, y_train)
    # predictions = get_prediction(X_test, model)
    # evaluate_model(y_test, predictions)
    # save_model(model)


if __name__ == "__main__":
    yolo_workflow()
