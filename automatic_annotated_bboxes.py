__author__ = "Domingos Rodrigues"
__email__ = "domingos.rodrigues@inventvision.com.br"
__copyright__ = "Copyright (C) 2024 Invent Vision"
__license__ = "Strictly proprietary for Invent Vision."

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


def get_train_bboxes(mask_np, threshold=0.3):
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

    min_area = threshold * Polygon(np.squeeze(hull)).area
    bboxes = get_inscribed_rect_recursive(hull, min_area)
    return contours, bboxes


def build_bboxes(image_path, label="", dbg_outdir=None):
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

    logger.info(f"endpoint_url = {endpoint_url}, {authToken}")
    pil_image = Image.open(image_path).convert("RGB")
    b64_image = encode_image(pil_image)
    data = {"b64Image": b64_image, "authToken": authToken}
    r = requests.post(endpoint_url, json=data)

    if r.status_code == 200:
        rjson = r.json()

        b64_mask = rjson["b64_mask"]
        pil_mask = decode_image(b64_mask)
        mask_np = np.array(pil_mask, dtype=np.uint8)

        contour_list, bboxes = get_train_bboxes(mask_np)

        if dbg_outdir is not None:
            pil_image_dbg = pil_image.copy()
            pil_image_dbg = overlay_mask(pil_image_dbg, mask_np)
            # draw_gt_boxes(image_dbg, bboxes, label=str(ground_truth))
            draw_gt_boxes(pil_image_dbg, bboxes, label=label)
            dbg_path = Path(dbg_outdir) / Path(image_path).stem
            pil_image_dbg.save(f"{dbg_path}.debug.jpg")

        return pil_image.size, bboxes, pil_mask

    raise AppError(r.status_code, "Failed to process image on segmentation service!")


def append_image_path_to_csv(line, csv_fd=None):
    if csv_fd:
        try:
            csv_fd.write(f"{line}\n")
            csv_fd.flush()
            logger.info(f"Append image info: {line}")
        except Exception as e:
            logger.exception(f"Failed to add image info: {e}")


def add_image_to_dataset(
    full_document, mongo_db, dataset_basedir="staging_dataset", csv_fd=None
):
    """_summary_

    Arguments:
        full_document -- _description_
    """

    try:
        gscs_id = full_document.get("gscs_id")
        manual_classcode = full_document.get("grade")

        logger.info(f"Processing document with GSCS ID '{gscs_id}'")

        # retriev AI inspection with gscs_id
        ai_inspection = mongo_db.inspections.find_one({"gscs_id": gscs_id})
        if ai_inspection:
            inspection_id = ai_inspection.get("_id")
            ai_label = ai_inspection.get("result", {}).get("detection", {}).get("class")

            # Accumulate all images regardless of classification aggrement between AI
            # and human classifications
            logger.debug(f"[AI inspection] - {inspection_id} | Action: add to DataSet")

            ground_truth_index, ground_truth_name = scrapRank[manual_classcode]

            # extract image paths for the inspection points
            for i in ai_inspection.get("inspections", []):
                camera, inpath_str = i["inspectionPoint"], i["imagePath"]

                outdir = Path(dataset_basedir) / "images" / ground_truth_name / camera
                annot_dir = Path(str(outdir).replace("/images/", "/labels/"))
                masks_dir = Path(str(outdir).replace("/images/", "/masks/"))
                dbg_outdir = Path(str(outdir).replace("/images/", "/debug/"))

                # The MongoDB instance is installed and running on a Windows-based machine
                if sys.platform.startswith("linux"):
                    inpath_str = inpath_str.replace("\\", "/")
                    inpath_str = inpath_str.replace("D:", "/media/ddcr/sahagun")
                    outdir = str(outdir).replace("\\", "/")

                # add this image and its associated annotated bounding boxes
                logger.info(f"{camera}: {inpath_str} -> {outdir}")

                inpath = Path(inpath_str)
                if not (inpath.exists() and inpath.is_file()):
                    raise Exception(f"Missing image file: {inpath_str}")

                logger.info("Segmenting image and fitting bounding boxes")

                if not dbg_outdir.exists():
                    dbg_outdir.mkdir(parents=True, exist_ok=True)

                img_shape, bboxes_list, mask_pil = build_bboxes(
                    inpath_str,
                    label=ground_truth_name,
                    dbg_outdir=dbg_outdir,
                )

                if len(bboxes_list) > 0:
                    logger.debug(
                        f"Segmentation and bounding box application successful: \n"
                        f"Found {len(bboxes_list)} bounding boxes for image {inpath_str}"
                    )
                    # Add segmentation mask to folder
                    masks_dir.mkdir(parents=True, exist_ok=True)
                    mask_file = masks_dir / Path(inpath_str).name
                    mask_file = mask_file.with_suffix(".png")
                    mask_pil.save(mask_file)

                    # copy image file
                    copy_file(inpath_str, outdir)

                    # Add annotations to folder
                    annot_dir.mkdir(parents=True, exist_ok=True)
                    annot_file = annot_dir / Path(inpath_str).name
                    annot_file = annot_file.with_suffix(".txt")
                    add_annotation(
                        bboxes_list, annot_file, img_shape, ground_truth_index
                    )

                    # Log image info to CSV file
                    dataset_relative_dir = Path(outdir).relative_to(dataset_basedir)
                    relpath = dataset_relative_dir / Path(inpath_str).name
                    created_at = ai_inspection.get("date")
                    image_lineinfo = f"{str(relpath)},{created_at},{camera},{ai_label},{ground_truth_name}"
                    append_image_path_to_csv(image_lineinfo, csv_fd=csv_fd)
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

            logger.info(
                "Inspection successfully completed and incorporated into the dataset."
            )
        else:
            logger.warning(f"No AI inspection found for GSCS ID: {gscs_id}")
    except AppError as e:
        httpReturnCode = e.code
        responseErrorMessage = e.message
        logger.exception(f"AppError [{httpReturnCode}]: {responseErrorMessage}")
    except Exception as e:
        logger.exception(f"Failure during mongo document processing: {e}")
