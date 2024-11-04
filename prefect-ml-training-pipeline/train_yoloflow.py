import datetime
import os
from pathlib import Path
from typing import List

import pandas as pd
import torch
from PIL import Image
import prefect
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


def move_images(image_list: List[str], crop_destdir: Path, split_name: str):
    for crop_src_relpath in image_list:
        crop_dst_relpath_str = crop_src_relpath.replace("images", split_name)
        crop_src_abspath = crop_destdir / crop_src_relpath

        # flatten the folder tree structure under each class folder
        crop_dst_relpath_parts = Path(crop_dst_relpath_str).parts
        crop_dst_relpath_name = Path(crop_dst_relpath_str).name
        crop_dst_relpath = Path(
            crop_dst_relpath_parts[0], crop_dst_relpath_parts[1], crop_dst_relpath_name
        )
        # print(f"{crop_src_relpath} => {crop_dst_relpath}")
        crop_dst_abspath = crop_destdir / crop_dst_relpath

        # Ensure the destination directory exists
        crop_dst_abspath.parent.mkdir(parents=True, exist_ok=True)
        os.rename(crop_src_abspath, crop_dst_abspath)


@task(name="create_yolo_dataset", log_prints=True)
def convert_dataset_to_yolo(dataset_inputdir: str, only_matched_classes=True):
    """Generates a YOLO-formatted dataset from a time-accumulated
        working dataset of images.

    Arguments:
        dataset_inputdir -- origin working directory of images and labels

    Keyword Arguments:
        only_matched_classes -- Builds a dataset of images with matching AI and human labels. (default: {True})

    Returns:
        Path to the YOLO dataset directory and a list of bounding box coordinates.
    """

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
    logger.info(f"Total number of truck images = {file_total}")

    logger.info("Create DATASET for YOLO CLASSIFICATION [cropping bounding boxes]")
    yolo_dataset_dir = Path(dataset_inputdir).parent / "gerdau_yolo_dataset"
    # yolo_dataset_dir = Path(WORKING_DIR) / "gerdau_yolo_dataset"
    yolo_dataset_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"YOLO dataset location: {yolo_dataset_dir}")

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
    yolo_dsetdir: Path,
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

    move_images(train_list, yolo_dsetdir, "train")
    move_images(val_list, yolo_dsetdir, "val")
    move_images(test_list, yolo_dsetdir, "test")

    # create dataset.yaml to order the scrap classes during training


@task(log_prints=True)
def train_classification_model(
    crop_dsetdir,
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

    flow_run_id = prefect.runtime.flow_run.id

    # create run folders
    base_path = crop_dsetdir.parent / project
    timestamp = datetime.datetime.now()
    task_name = f"prefect_run_{flow_run_id[:8]}"

    yolo_rundir = base_path / timestamp.strftime("%Y/%m%d")
    if not yolo_rundir.exists():
        yolo_rundir.mkdir(parents=True, exist_ok=True)

    print(f"Flow run ID: {flow_run_id}")
    print(f"Local storage for this run: {str(yolo_rundir)}")

    # creating a ClearML Task
    if clearml:
        task = Task.init(
            project_name = project,
            task_name = task_name,
            tags=[model_variant, "AutoML"]
        )
        task.set_parameter("model_variant", model_variant)

    # load a pretrained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(f"{model_variant}.pt").to(device)

    args = dict(
        data=crop_dsetdir,
        imgsz=imgsz,
        epochs=epochs,
        project=str(yolo_rundir),
        name=task_name,
        amp=amp,
        val=val,
        close_mosaic=0,
        label_smoothing=1
    )

    if clearml:
        task.connect(args)

    # Start training
    print("Training started!")
    model_results = model.train(**args)

    return model_results, model


@task
def get_prediction(X_test, model):
    pass


@task
def evaluate_model(y_test, prediction: pd.DataFrame):
    pass


@task(log_prints=True)
def save_model(model_results, model):
    """_summary_

    Arguments:
        model_results -- _description_
        model -- _description_
    """
    print(f"model_results: {model_results}")

    save_dir = getattr(model_results, "save_dir", None)
    print(f"save_dir from model_results: {save_dir}")

    print(f"model_info: {model.info()}")
    # Save final model
    model.save("auto_ml.pt")
    print("Model saved as 'auto_ml.pt'")


@flow(
    name="YOLOv11 classification training",
    log_prints=True,
    flow_run_name=generate_flow_run_name,
)
def yolo_workflow(dset_inputdir: str = ""):
    """_summary_

    Keyword Arguments:
        dset_inputdir -- _description_ (default: {""})
    """
    #######################################################################
    logger = get_run_logger()
    logger.info("Started YOLO classification training pipeline")
    #######################################################################

    yolo_dsetdir, crop_images_list = convert_dataset_to_yolo(dset_inputdir)

    data_split(yolo_dsetdir, crop_images_list)

    # this model fits on 'knuth' GPU
    res, model = train_classification_model(yolo_dsetdir, model_variant="yolo11n-cls")

    # predictions = get_prediction(X_test, model)

    # evaluate_model(y_test, predictions)

    save_model(res, model)


if __name__ == "__main__":
    yolo_workflow()
