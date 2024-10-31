#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" """

__author__ = "Domingos Rodrigues"
__email__ = "domingos.rodrigues@inventvision.com.br"
__copyright__ = "Copyright (C) 2024 Invent Vision"
__license__ = "Strictly proprietary for Invent Vision."

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import logzero
import tornado.ioloop
import tornado.websocket
from bson import json_util
from logzero import logger

from automatic_annotated_bboxes import build_bboxes
from utils import (
    AppError,
    add_annotation,
    connect_to_mongo,
    copy_file,
    count_files_in_subfolders,
    scrapRank,
)

mongo_db = None

if sys.platform.startswith("linux"):
    DATASET_BASEDIR = os.getenv("DATASET_DIR", r"dataset_in_preparation")
else:
    DATASET_BASEDIR = os.getenv(
        "DATASET_DIR", r"D:\ivision\automatic_retraining\dataset_in_preparation"
    )


class WebSocketClient:
    def __init__(
        self,
        io_loop,
        url="ws://127.0.0.1:8000/socket",
        max_retries=10,
        retry_interval=3,
        file_path="images.csv",
    ):
        self.connection = None
        self.io_loop = io_loop
        self.url = url
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self.retries = 0
        self.file_path = Path(DATASET_BASEDIR) / file_path
        self.file = None

    def start(self):
        # make sure DATASET_BASEDIR exists
        Path(DATASET_BASEDIR).mkdir(parents=True, exist_ok=True)
        self.open_file()
        self.connect_and_read()

    def stop(self):
        self.close_file()
        self.io_loop.stop()

    def open_file(self):
        try:
            self.file = open(self.file_path, "a")
            if self.file.tell() == 0:
                header = "path,created_at,camera,ai_class,human_class"
                self.file.write(f"{header}\n")
            self.file.flush()
            logger.info(f"File opened for appending data: {self.file_path}")
        except Exception as e:
            logger.exception(f"Failed to open file {self.file_path}: {e}")

    def close_file(self):
        if self.file:
            try:
                self.file.close()
                logger.info(f"File closed: {self.file_path}")
            except Exception as e:
                logger.exception(f"Failed to close file {self.file_path}: {e}")

    def connect_and_read(self):
        logger.info("Reading ...")
        tornado.websocket.websocket_connect(
            url=self.url,
            callback=self.maybe_retry_connection,
            on_message_callback=self.on_message,
            ping_interval=10,
            ping_timeout=30,
        )

    def maybe_retry_connection(self, future):
        try:
            self.connection = future.result()
            self.retries = 0
        except Exception:
            logger.exception(f"Failed to connect to {self.url} ...")

            if self.retries < self.max_retries:
                self.retries += 1
                logger.info(f"Retrying ... {self.retries}/{self.max_retries}")
                self.io_loop.call_later(self.retry_interval, self.connect_and_read)
            else:
                logger.info(
                    f"Max attempts reached. Could not connect to server {self.url}, exiting."
                )
                self.stop()

    def on_message(self, message):
        """Receives Change Stream from MongoDB

        Arguments:
            message -- BSON document with keys ['_id', 'operationType',
                       'clusterTime', 'fullDocument', 'documentKey']
        """
        if message is None:
            logger.info("Disconnected, reconnecting ...")
            self.connect_and_read()
        else:
            try:
                message_json = json_util.loads(message)
                parse_change_stream_event(message_json, fd=self.file)

                # TODO: After writing to the CSV, read it into a DataFrame
                # TODO: df = pd.read_csv(self.file_path)
                # TODO: logger.info(f"Loaded DataFrame with {len(df)} records")

            except Exception as e:
                logger.exception(
                    f"An error occurred while parsing the change stream event: {e}"
                )
            finally:
                # check if dataset is ready for training
                if not message_json.get("trigger"):  # bypass ML
                    res = self.is_dataset_ready()
                    if res:
                        self.signal_to_ml_workflow({"trigger": "train_ml"})
                    else:
                        logger.warning("Dataset is not yet ready for training")

    def is_dataset_ready(self):
        """Count images per class"""
        return False

    def signal_to_ml_workflow(self, message):
        if self.connection:
            try:
                self.connection.write_message(json_util.dumps(message))
            except Exception as e:
                logger.exception(f"Failed to trigger the ML workflow: {e}")


def parse_change_stream_event(change_event, fd=None):
    operation_type = change_event.get("operationType")
    event_id = change_event.get("_id")
    event_docuid = change_event.get("documentKey")
    if event_id:
        logger.debug(
            f"Worker: received Change Stream event:\n"
            f"{'ID:':<10} {event_id}\n"
            f"{'OP:':<10} {operation_type}\n"
            f"{'Doc Mongo ID:':<10} {event_docuid}"
        )

    # verify type of document
    full_document = change_event.get("fullDocument")

    # handle cases with fullDocument = None
    if full_document is None:
        if operation_type == "delete":
            logger.debug(f"Document with ID {event_docuid} was deleted.")
        else:
            logger.warning(
                f"This change stream event involves no 'fullDocument' [OP: {operation_type}]."
            )
        return

    # proceed with handling based on the content of 'fullDocument'
    document_origin = full_document.get("origin")

    if document_origin is None:
        if "need_sync" in full_document:
            # GSCS entry inserted
            gscs_id = full_document.get("gscs_id")

            if operation_type == "insert":
                logger.debug(
                    f"[GSCS insertion] - {gscs_id} | Action: Add GSCS initial info"
                )
            elif operation_type == "update":
                logger.debug(
                    f"[GSCI update] - {gscs_id} | Action: Relay to function 'add_image_to_dataset'"
                )
                #########################
                #  Add image to dataset
                #########################
                add_image_to_dataset(full_document, fd=fd)
            else:
                logger.debug(
                    f"[GSCS {operation_type}] - {gscs_id} | Action: 'operationType' not handled"
                )

    elif document_origin == "/gerdau/scrap_detection/inspect":
        # inspection classified by AI

        inspection_id = full_document.get("_id")
        if operation_type == "insert":
            logger.debug(
                f"[AI inspection {operation_type}] - {inspection_id} | Action: Relay to another endpoint"
            )
        elif operation_type in ["update", "replace"]:
            logger.debug(
                f"[AI inspection {operation_type}] - {inspection_id} | Action: 'operationType' not handled"
            )
        elif operation_type == "delete":
            logger.debug(
                f"[AI inspection {operation_type}] - {inspection_id} | Document deleted."
            )
        else:
            logger.warning(
                f"[AI inspection {operation_type}] - {inspection_id} | Unexpected operation type"
            )

    else:
        logger.error(
            "Change Stream Error: Unknown document type detected."
            "Unable to process the associated document."
        )


def append_image_path_to_csv(line, fd=None):
    if fd:
        try:
            fd.write(f"{line}\n")
            fd.flush()
            logger.info(f"Append image info: {line}")
        except Exception as e:
            logger.exception(f"Failed to add image info: {e}")


def add_image_to_dataset(full_document, fd=None):
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

            # TODO: Accumulate all images regardless of classification aggrement
            # TODO: between AI and human classifications
            # ai_classcode = (
            #     ai_inspection.get("result", {}).get("detection", {}).get("classCode")
            # )
            # if manual_classcode == ai_classcode:
            logger.debug(f"[AI inspection] - {inspection_id} | Action: add to DataSet")

            ground_truth_index, ground_truth_name = scrapRank[manual_classcode]

            # extract image paths for the inspection points
            for i in ai_inspection.get("inspections", []):
                camera, inpath_str = i["inspectionPoint"], i["imagePath"]
                outdir = Path(DATASET_BASEDIR) / "images" / ground_truth_name / camera
                annot_dir = Path(str(outdir).replace("/images/", "/labels/"))
                masks_dir = Path(str(outdir).replace("/images/", "/masks/"))

                # The MongoDB instance is installed and running on a Windows-based machine
                if sys.platform.startswith("linux"):
                    inpath_str = inpath_str.replace("\\", "/")
                    inpath_str = inpath_str.replace("D:", "/media/ddcr/sahagun")
                    outdir = str(outdir).replace("\\", "/")

                ###################################
                #
                #    Standard Folder structure
                #
                ###################################
                # add this image and its associated annotated bounding boxes
                logger.info(f"{camera}: {inpath_str} -> {outdir}")

                inpath = Path(inpath_str)
                if not (inpath.exists() and inpath.is_file()):
                    raise Exception(f"Missing image file: {inpath_str}")

                logger.info("Segment image and automatically fit bounding boxes")

                dbg_outdir = Path(str(outdir).replace("/images/", "/debug/"))
                dbg_outdir.mkdir(parents=True, exist_ok=True)

                img_shape, bboxes_list, mask_pil = build_bboxes(
                    inpath_str,
                    label=ground_truth_name,
                    dbg_outdir=dbg_outdir,
                )


                if len(bboxes_list) > 0:
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
                    add_annotation(bboxes_list, annot_file, img_shape, ground_truth_index)

                    # Add image info to external file
                    dataset_relative_dir = Path(outdir).relative_to(DATASET_BASEDIR)
                    relpath = dataset_relative_dir / Path(inpath_str).name
                    created_at = ai_inspection.get("date")
                    image_lineinfo = f"{str(relpath)},{created_at},{camera},{ai_label},{ground_truth_name}"
                    append_image_path_to_csv(image_lineinfo, fd=fd)
                else:
                    failed_outdir = Path(str(outdir).replace("/images/", "/images_no_bboxes/"))
                    failed_outdir.mkdir(parents=True, exist_ok=True)
                    # Add segmentation mask to folder
                    mask_file = failed_outdir / Path(inpath_str).name
                    mask_file = mask_file.with_suffix(".mask.png")
                    mask_pil.save(mask_file)

                    copy_file(inpath_str, failed_outdir)

            # else:
            #     logger.warning(
            #         f"[AI inspection] - {inspection_id} | AI and human classifications differ"
            #     )

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


def main():
    """ """
    global mongo_db

    parser = argparse.ArgumentParser(
        description="Establishes connections to both a MongoDB database and a WebSocket server.",
        add_help=True,
    )
    parser.add_argument(
        "--host",
        type=str,
        required=False,
        default="127.0.0.1",
        help="Mongo hostname (e.g., localhost) [default: %(default)s]",
    )

    parser.add_argument(
        "--port",
        type=int,
        required=False,
        default=27017,
        help="MongoDB access port (e.g. 27017) [default: %(default)s]",
    )

    args = parser.parse_args()
    logzero.logfile(
        "logs/webclient_preprocess_dataset.log", maxBytes=1000000, backupCount=5
    )

    mongo_db = connect_to_mongo(args.host, args.port)

    io_loop = tornado.ioloop.IOLoop.current()
    client = WebSocketClient(io_loop)
    io_loop.add_callback(client.start)

    io_loop.start()


if __name__ == "__main__":
    main()
