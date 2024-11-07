import datetime
import os
import shutil
from pathlib import Path
from typing import List

import pandas as pd
import prefect
import torch
from PIL import Image
from prefect import flow, task

# from prefect.client import Client
from prefect.logging import get_run_logger
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

script_path = os.path.abspath(__file__)
WORKING_DIR = os.path.dirname(script_path)


def generate_flow_run_name():
    date = datetime.datetime.now(datetime.timezone.utc)
    return f"yolo_train-{date:%Y-%m-%dT%H%M%S}"


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


def move_images(image_list: List[str], yolo_datadir: Path, split_name: str):
    for image_src_relpath in image_list:
        image_src_abspath = yolo_datadir / image_src_relpath

        image_dst_relpath = image_src_relpath.replace("images", split_name)
        image_dst_abspath = yolo_datadir / image_dst_relpath

        # Ensure the destination directory exists
        image_dst_abspath.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.rename(image_src_abspath, image_dst_abspath)
        except IsADirectoryError as e:
            print(f"{image_dst_abspath} is a directory: {e}")
        except NotADirectoryError as e:
            print(f"{image_src_abspath} is a directory: {e}")
        except PermissionError as e:
            print("Operation not permitted")
        except OSError as e:
            print(e)


@task(name="setup_folder_hierarchy", log_prints=True)
def setup_folder_hierarchy(dataset_inputdir):
    timestamp = datetime.datetime.now()

    # TODO: refactor this bit ... using config.json?
    yolo_data_basedir = (
        Path(dataset_inputdir).parent
        / "automl_data"
        / "prepared_datasets"
        / "yolo"
        / timestamp.strftime("%Y/%m%d")
    )
    yolo_data_basedir.mkdir(parents=True, exist_ok=True)

    yolo_run_basedir = (
        Path(dataset_inputdir).parent
        / "automl_runs"
        / "yolo"
        / timestamp.strftime("%Y/%m%d")
    )
    yolo_run_basedir.mkdir(parents=True, exist_ok=True)

    yolo_model_basedir = Path(dataset_inputdir).parent / "automl_models" / "yolo"
    yolo_model_basedir.mkdir(parents=True, exist_ok=True)

    return yolo_data_basedir, yolo_run_basedir, yolo_model_basedir


@task(name="convert_raw_dataset_to_yolo_format", log_prints=True)
def convert_raw_dataset_to_yolo_format(
    source_basedir: str, images_path: str, yolo_basedir: Path, only_matched_classes=True
):
    """Generates a YOLO-formatted dataset from a time-accumulated
        working dataset of images.

    Arguments:
        source_basedir -- source folder of raw images and annotations
        images_path -- list of images selected for the training session
        yolo_basedir -- base directory holding the final YOLO tree structure

    Keyword Arguments:
        only_matched_classes -- Builds a YOLO dataset of images with matching
                                AI and human labels. (default: {True})

    Returns:
        Path to the YOLO dataset directory and a list of bounding box coordinates.
    """
    logger = get_run_logger()

    df = pd.read_csv(images_path)

    if only_matched_classes:
        rows_to_drop = df[df["ai_class"] != df["human_class"]].index
        df.drop(index=rows_to_drop, inplace=True)

    file_counts_per_class = df.groupby("ai_class").size().reset_index(name="file_count")
    file_total = file_counts_per_class["file_count"].sum()

    logger.info(file_counts_per_class)
    logger.info(f"Total number of images = {file_total}")

    logger.info(
        "Initiating export: Converting staging directory contents to YOLO format..."
    )

    yolo_imagedir = yolo_basedir / "images"
    yolo_imagedir.mkdir(parents=True, exist_ok=True)

    # get bbox crop images
    crop_paths = []
    for row in df.itertuples():
        img_src_relpath = str(row.path)
        class_label = str(row.human_class)
        img_src_abspath = Path(source_basedir) / img_src_relpath
        image = Image.open(str(img_src_abspath)).convert("RGB")
        im_w, im_h = image.size
        label_src_relpath = img_src_relpath.replace("images", "labels")
        label_src_relpath = Path(label_src_relpath).with_suffix(".txt")
        label_src_abspath = Path(source_basedir) / label_src_relpath

        bboxes = get_bbox_image_slices(str(label_src_abspath))

        # flatten
        img_dst_abspath = yolo_imagedir / class_label / Path(img_src_relpath).name

        if bboxes is not None:
            for ib, yolo_bbox in enumerate(bboxes):
                x, y, w, h = xywhn_to_bbox(yolo_bbox, im_w, im_h)
                image_crop = image.crop((x, y, x + w, y + h))

                img_crop_name = f"{img_dst_abspath.stem}_bb{ib}{img_dst_abspath.suffix}"
                img_crop_dst_abspath = img_dst_abspath.with_name(img_crop_name)
                # logger.info(f"{img_src_abspath} => {img_crop_dst_abspath}")
                img_crop_dst_abspath.parent.mkdir(parents=True, exist_ok=True)
                image_crop.save(str(img_crop_dst_abspath))

                img_crop_dst_relpath = img_crop_dst_abspath.relative_to(yolo_basedir)
                crop_paths.append(str(img_crop_dst_relpath))

    return crop_paths


@task(log_prints=True)
def data_split(
    yolo_datadir: Path,
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

    move_images(train_list, yolo_datadir, "train")
    move_images(val_list, yolo_datadir, "val")
    move_images(test_list, yolo_datadir, "test")

    # create dataset.yaml to order the scrap classes during training


@task(log_prints=True)
def train_classification_model(
    yolo_datadir,
    yolo_rundir,
    task_name,
    imgsz=224,
    epochs=5,
    project="automl-yolo-runs",
    model_variant="yolo11m-cls",
    amp=False,
    val=True,
):
    """_summary_

    Arguments:
        crop_dsetdir -- _description_

    Keyword Arguments:
        imgsz -- _description_ (default: {224})
        epochs -- _description_ (default: {5})
        project -- _description_ (default: {"automl-yolo-runs"})
        model_variant -- _description_ (default: {"yolo11m-cls"})
        amp -- _description_ (default: {False})
        val -- _description_ (default: {True})

    Returns:
        _description_
    """
    try:
        import clearml
        from clearml import Task
    except ImportError:
        clearml = None

    # creating a ClearML Task
    if clearml:
        task = Task.init(
            project_name=project, task_name=task_name, tags=[model_variant, "AutoML"]
        )
        task.set_parameter("model_variant", model_variant)

    # load a pretrained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_obj = YOLO(f"{model_variant}.pt").to(device)

    args = dict(
        data=yolo_datadir,
        imgsz=imgsz,
        epochs=epochs,
        project=str(yolo_rundir),
        name=task_name,
        amp=amp,
        val=val,
        close_mosaic=0,
        label_smoothing=1,
    )

    if clearml:
        task.connect(args)

    # Start training
    print("Training started!")
    model_results = model_obj.train(**args)

    return model_results, model_obj


@task
def get_prediction(X_test, model):
    pass


@task
def evaluate_model(y_test, prediction: pd.DataFrame):
    pass


@task(log_prints=True)
def save_model(model_obj, model_fpath):
    """_summary_

    Arguments:
        model_obj -- _description_
        model_basedir -- _description_
    """
    print(f"model_info: {model_obj.info()}")
    # Save final model
    model_obj.save(model_fpath)
    print(f"Model saved: '{model_fpath}'")


@flow(
    name="YOLOv11 classification training",
    log_prints=True,
    flow_run_name=generate_flow_run_name,
)
def yolo_workflow(dset_inputdir, images_path):
    """
    YOLOv11 Classification Training Pipeline

    Prepares the dataset, trains a YOLOv11 classification model, evaluates its performance,
    and saves the trained model.

    Arguments:
        dset_inputdir (str): Directory with original images for training.
        images_path (str): Path to CSV file containing image metadata for training.

    Returns:
        None
    """
    logger = get_run_logger()

    task_id = prefect.runtime.flow_run.id[:8]
    logger.info(f"Started YOLO classification training pipeline ID: {task_id}")

    # === 1 ===
    yolo_datadir, yolo_rundir, yolo_modeldir = setup_folder_hierarchy(dset_inputdir)
    yolo_datadir /= Path(f"dataset_{task_id}")
    yolo_datadir.mkdir(parents=True, exist_ok=True)

    # copy source file list of images
    shutil.copy(images_path, yolo_datadir / Path(images_path).name)

    # === 2 ===
    crop_images_list = convert_raw_dataset_to_yolo_format(
        dset_inputdir, images_path, yolo_datadir
    )

    # === 3 ===
    data_split(yolo_datadir, crop_images_list)

    task_name = f"run_{task_id}"
    task_res, model_obj = train_classification_model(
        yolo_datadir, yolo_rundir, task_name, model_variant="yolo11n-cls"
    )
    print(f"task_res: {task_res}")

    # predictions = get_prediction(X_test, model)

    # evaluate_model(y_test, predictions)

    timestamp_save_model = datetime.datetime.now()
    model_fpath = yolo_modeldir / f"model_{task_id}.pt"
    save_model(model_obj, model_fpath)


if __name__ == "__main__":
    yolo_workflow()
