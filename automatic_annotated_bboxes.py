__author__ = "Domingos Rodrigues"
__email__ = "domingos.rodrigues@inventvision.com.br"
__copyright__ = "Copyright (C) 2024 Invent Vision"
__license__ = "Strictly proprietary for Invent Vision."

import csv
import datetime
import sys
from pathlib import Path

import cv2
import numpy as np
import requests
from logzero import logger
from PIL import Image
from shapely import Polygon

import lir.largestinteriorrectangle as lir
from utils import (
    AppError,
    add_annotation,
    configs,
    copy_file,
    decode_image,
    draw_gt_boxes,
    encode_image,
    overlay_mask,
    scrapRank,
)


def xywh_to_shapely_polygon(tl: tuple, br: tuple, buffer: float = 1.0):
    """Converts the rectangle to Shapely format

    Arguments:
        tl -- top left absolute coordinates
        br -- bottom right coordinates
        buffer -- increase the size of the rectangle with a buffer zone

    Returns:
        Rectangle in Shapely format (Polygon)
    """
    (x, y) = tl
    (x2, y2) = br
    return Polygon([(x, y), (x2, y), (x2, y2), (x, y2)]).buffer(buffer)


def get_inscribed_box(patch):
    """Find the largest rectangle that can be inscribed
        within a convex region defined by the `patch`.

    Arguments:
        patch -- list of vertices of the convex region

    Returns:
        rect_np -- the list of coordinates of the inscribed rectangle
        sub_patches -- the region(s) outside rectangle in the Shapely
                       format (MultiPolygon)
    """
    # get largest inscribed rectangle
    patch_lir = patch.transpose((1, 0, 2))
    rect_lir = lir.lir(patch_lir)

    # convert rectangle to a shapely Polygon
    tl = lir.pt1(rect_lir)
    br = lir.pt2(rect_lir)
    rect_shapely = xywh_to_shapely_polygon(tl, br)

    patch_shapely = Polygon(np.squeeze(patch))

    # get patches outside rectangle
    sub_patches = patch_shapely.difference(rect_shapely, grid_size=0)

    rect_np = list(tl + br)
    return rect_np, sub_patches


def get_inscribed_rect_recursive(region, min_area: int):
    """Recursively cover the region with inscribed rectangles.

    Arguments:
        region_np -- list of vertices of the convex region
        min_area -- minimum threshold area required for inscribing rectangles

    Returns:
        list of rectangles covering the region
    """
    rectangle, outer_regions = get_inscribed_box(region)

    if isinstance(outer_regions, Polygon):
        print("[WARNING]: increase buffer zone of rectangle")
        return []

    inscribed_rectangles = [rectangle]
    for patch in outer_regions.geoms:
        if patch.area > min_area:
            patch_np = (
                np.array(patch.exterior.coords).reshape(-1, 1, 2).astype(np.int32)
            )
            inscribed_rectangles.extend(
                get_inscribed_rect_recursive(patch_np, min_area)
            )
    return inscribed_rectangles


def polygon_to_inscribed_boxes(mask_np, threshold=0.3):
    """_summary_

    Arguments:
        mask_np -- _description_

    Keyword Arguments:
        threshold -- _description_ (default: {0.3})

    Returns:
        _description_
    """

    # Find contours
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) < 1:
        return [], []

    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]

    # approximate contour by the convex polygon
    hull = cv2.convexHull(cnt)

    # minimum area for fitting an inscribed rectangle
    min_area = threshold * Polygon(np.squeeze(hull)).area
    bboxes = get_inscribed_rect_recursive(hull, min_area)  # type: ignore
    return contours, bboxes


def build_bboxes(
    image_path,
    human_label=None,
    ai_label=None,
    dbg_outdir=None,
    box_cover_heuristic="grid",
):
    """_summary_

    Arguments:
        imagepath -- _description_
    """
    host, port, api_version = (
        configs["host"],
        configs["listening_port"],
        configs["api_version"],
    )
    endpoint = f"/ivision/segmentation/scrap/{api_version}/predict"
    endpoint_url = f"{host}:{port}{endpoint}"
    authToken = configs["authToken"]

    logger.debug(f"endpoint_url = {endpoint_url}, {authToken}")
    pil_image = Image.open(image_path).convert("RGB")
    b64_image = encode_image(pil_image)
    data = {"b64Image": b64_image, "authToken": authToken}
    r = requests.post(endpoint_url, json=data)

    if r.status_code == 200:
        rjson = r.json()

        b64_mask = rjson["b64_mask"]
        pil_mask = decode_image(b64_mask)
        mask_np = np.array(pil_mask, dtype=np.uint8)

        # heuristic
        if box_cover_heuristic == "grid":
            bboxes = polygon_to_boxes(mask_np, roi_sz=200)
        else:
            contour_list, bboxes = polygon_to_inscribed_boxes(mask_np)

        if dbg_outdir is not None:
            if not dbg_outdir.exists():
                dbg_outdir.mkdir(parents=True, exist_ok=True)
            pil_image_dbg = pil_image.copy()
            pil_image_dbg = overlay_mask(pil_image_dbg, mask_np)
            if len(bboxes) > 0:
                draw_gt_boxes(
                    pil_image_dbg, bboxes, human_label=human_label, ai_label=ai_label
                )
            dbg_path = Path(dbg_outdir) / Path(image_path).stem
            pil_image_dbg.save(f"{dbg_path}.debug.jpg")

        return pil_image.size, bboxes, pil_mask

    raise AppError(r.status_code, "Failed to process image on segmentation service!")


def polygon_to_boxes(mask: np.ndarray, roi_sz: int = 100, threshold: float = 0.8):
    """
    Split polygon area into multiples squares

    Args:
        mask (np.ndarray): Binarized image mask
        roi_sz (int, optional): Desired ROI size. Defaults to 100.
        threshold (float, optional): Minimum threshold acceptable to be inside of the mask limits or intersect. Defaults to 0.8.

    Returns:
        list: bounding boxes of proposed polygons
    """

    # Definitions
    m_h, m_w = mask.shape
    instance = np.where(mask == 255)
    if len(instance[0]) == 0:
        return []

    bboxes = []
    y_start = instance[0].min()
    h = instance[0].max() - y_start

    d = h % roi_sz
    v_div = h // roi_sz
    if v_div == 0 and d >= (roi_sz * threshold):
        v_div = 1
    elif d >= (roi_sz * threshold):
        v_div += 1

    # Updates initial top point
    y_diff = h - (v_div * roi_sz)
    if y_diff <= 0 and y_diff < -1 * (roi_sz / 2.0):
        return []
    y_step = 0 if (y_diff == 0 or v_div == 0) else int(y_diff / v_div / 2.0)
    if y_step < 0:
        y_step = 0.0
        y_start += y_diff / 2
    if y_start < 0:
        y_start = 0

    # Iterates vertically (top-down)
    for dd in range(v_div):
        y1 = int(y_start + dd * roi_sz + y_step + dd * 2 * y_step)
        y2 = int(y1 + roi_sz)
        if y2 > m_h:
            y2 = m_h

        # Get horizontal space
        y_center = int((y2 + y1) / 2)
        x_pos = np.where(instance[0] == y_center)
        x_vals = instance[1][x_pos]
        w = x_vals.max() - x_vals.min()
        x_start = x_vals.min()

        # Obtain boxes along horizontal axis
        d = w % roi_sz
        h_div = w // roi_sz
        if h_div == 0:
            h_div = 1
        elif d >= threshold * roi_sz:
            h_div += 1

        # Updates initial left point
        x_diff = w - (h_div * roi_sz)
        x_step = 0 if (x_diff == 0 or h_div == 0) else int(x_diff / h_div / 2.0)
        if x_step < 0:
            x_step = 0.0
            x_start += int(x_diff / 2)

        # Fill the boxes horizontally (left-right)
        for xx in range(h_div):
            x1 = int(x_start + xx * roi_sz + x_step + xx * 2 * x_step)
            if x1 < 0:
                x1 = 0
            x2 = x1 + roi_sz
            if x2 > m_w:
                x2 = m_w

            bboxes.append([x1, y1, x2, y2])

    # Cast array type to uint32
    bboxes = np.array(bboxes, dtype=np.uint32)

    return bboxes


def append_image_path_to_csv(line, csv_header=None, csv_fd=None):
    if csv_fd:
        try:
            if csv_header:
                fieldnames = csv_header.split(",")
                writer = csv.DictWriter(csv_fd, fieldnames=fieldnames)
                writer.writerow(line)
                csv_fd.flush()
            else:
                csv_fd.write(f"{line}\n")
                csv_fd.flush()
        except Exception as e:
            logger.exception(f"Failed to add image info: {e}")


def add_image_to_dataset(
    full_document,
    mongo_db,
    dataset_basedir="staging_dataset",
    csv_header=None,
    csv_fd=None,
):
    """_summary_

    Arguments:
        full_document -- _description_
    """

    try:
        inspection_id = full_document.get("_id")
        gscs_id = full_document.get("gscs_id")

        ai_classcode = full_document.get("result").get("detection").get("classCode")

        if ai_classcode:
            ai_class_index, ai_class = scrapRank[ai_classcode]
            gscs_info = full_document.get("gscs_classification")
            human_classcode = gscs_info.get("classCode")
            human_class_index, human_class = scrapRank[human_classcode]

            logger.info(f"Inspection {inspection_id}: extract content")

            # extract image paths for the inspection points
            for i in full_document.get("inspections", []):
                camera, inpath_str = i["inspectionPoint"], i["imagePath"]

                outdir = Path(dataset_basedir) / "images" / human_class / camera
                annot_dir = Path(dataset_basedir) / "labels" / human_class / camera
                masks_dir = Path(dataset_basedir) / "masks" / human_class / camera
                dbg_outdir = Path(dataset_basedir) / "debug" / human_class / camera

                # The MongoDB instance is installed and running on a Windows-based machine
                if sys.platform.startswith("linux"):
                    inpath_str = inpath_str.replace("\\", "/")
                    inpath_str = inpath_str.replace("D:", "/media/ddcr/sahagun")
                    outdir = str(outdir).replace("\\", "/")

                logger.info("Cover truck cargo with bounding boxes")

                inpath = Path(inpath_str)
                if not (inpath.exists() and inpath.is_file()):
                    raise Exception(f"Missing image file: {inpath_str}")

                img_shape, bboxes_list, mask_pil = build_bboxes(
                    inpath_str,
                    human_label=human_class,
                    ai_label=ai_class,
                    dbg_outdir=dbg_outdir,
                )

                if len(bboxes_list) > 0:
                    # Add segmentation mask to folder
                    masks_dir.mkdir(parents=True, exist_ok=True)
                    mask_file = masks_dir / Path(inpath_str).name
                    mask_file = mask_file.with_suffix(".png")
                    mask_pil.save(mask_file)

                    # copy image file to staging directory
                    copy_file(inpath_str, outdir)

                    # Add annotations to folder
                    annot_dir.mkdir(parents=True, exist_ok=True)
                    annot_file = annot_dir / Path(inpath_str).name
                    annot_file = annot_file.with_suffix(".txt")
                    add_annotation(
                        bboxes_list, annot_file, img_shape, human_class_index
                    )

                    # Log image info to CSV file
                    dataset_relative_dir = Path(outdir).relative_to(dataset_basedir)
                    relpath = dataset_relative_dir / Path(inpath_str).name
                    created_at = full_document.get("date")
                    added_at = datetime.datetime.now()

                    image_lineinfo = {
                        "insp_id": inspection_id,
                        "gscs_id": gscs_id,
                        "path": str(relpath),
                        "camera": camera,
                        "created_at": created_at,
                        "added_at": added_at,
                        "ai_class": ai_class,
                        "human_class": human_class,
                    }
                    append_image_path_to_csv(
                        image_lineinfo, csv_header=csv_header, csv_fd=csv_fd
                    )

                else:
                    logger.warning(f"No bounding boxes found for image {inpath_str}")
                    failed_outdir = Path(
                        str(outdir).replace("/images/", "/images_no_bboxes/")
                    )
                    failed_outdir.mkdir(parents=True, exist_ok=True)
                    # Add segmentation mask to folder
                    mask_file = failed_outdir / Path(inpath_str).name
                    mask_file = mask_file.with_suffix(".mask.png")
                    mask_pil.save(mask_file)

                    copy_file(inpath_str, failed_outdir)

            logger.info(f"Inspection {inspection_id} incorporated into the dataset.")
        else:
            logger.error(f"Inspection '{inspection_id}' has no AI classification.")
    except AppError as e:
        httpReturnCode = e.code
        responseErrorMessage = e.message
        logger.exception(f"AppError [{httpReturnCode}]: {responseErrorMessage}")
    except Exception as e:
        logger.exception(f"Failure during mongo document processing: {e}")
