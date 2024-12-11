#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" """

__author__ = "Domingos Rodrigues"
__email__ = "domingos.rodrigues@inventvision.com.br"
__copyright__ = "Copyright (C) 2024 Invent Vision"
__license__ = "Strictly proprietary for Invent Vision."

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import logzero
import pandas as pd
import tornado.ioloop
import tornado.websocket
from bson import json_util
from filelock import FileLock
from logzero import logger

from automatic_annotated_bboxes import add_image_to_dataset
from utils import connect_to_mongo, count_files_in_subfolders_efficient

mongo_db = None

if sys.platform.startswith("linux"):
    DATASET_BASEDIR = os.getenv("DATASET_DIR", r"staging_dataset")
else:
    DATASET_BASEDIR = os.getenv(
        "DATASET_DIR", r"D:\ivision\automatic_retraining\staging_dataset"
    )


class DatasetChecker:
    def __init__(
        self, initial_threshold=50, initial_class_thresholds=None, increment=50
    ):
        self.threshold = initial_threshold
        self.class_thresholds = initial_class_thresholds or {}
        self.increment = increment

    def is_dataset_ready(self, method="total"):
        """ """
        topdir = Path(DATASET_BASEDIR) / "images"
        if not topdir.exists():
            return False

        class_folders = [
            subfolder for subfolder in topdir.iterdir() if subfolder.is_dir()
        ]
        class_counts = count_files_in_subfolders_efficient(class_folders)
        total_counts = sum(class_counts.values())

        logger.info(f"[{total_counts}] {class_counts}")

        # Perform check
        is_ready = False
        if method == "total":
            is_ready = total_counts >= self.threshold
        elif method == "per_class":
            is_ready = all(
                class_counts.get(cls, 0) >= thresh
                for cls, thresh in self.class_thresholds.items()
            )
        # elif method == "both":
        #     is_ready = total_counts > self.threshold and all(
        #         class_counts.get(cls, 0) >= thresh for cls, thresh in self.class_thresholds.items()
        #     )

        # Adjust thresholds if dataset is ready, as to delay the next training trigger
        if is_ready:
            self.threshold += self.increment
            for cls in self.class_thresholds:
                self.class_thresholds[cls] += self.increment
            logger.info(
                f"Thresholds increased to delay next trigger: {self.threshold}, {self.class_thresholds}"
            )

        return is_ready


class WebSocketClient:
    def __init__(
        self,
        io_loop,
        url="ws://127.0.0.1:8000/socket",
        dataset_dir="staging_dataset",
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
        self.dataset_dir = dataset_dir
        self.file_path = Path(self.dataset_dir) / file_path
        self.file = None
        self.csv_header = (
            "insp_id,gscs_id,path,camera,created_at,added_at,ai_class,human_class"
        )
        # TODO: change default parameters, maybe by reading a config.json
        self.checker = DatasetChecker()

    def start(self):
        if not Path(self.dataset_dir).exists():
            Path(self.dataset_dir).mkdir(parents=True, exist_ok=True)

        self.open_file()
        self.connect_and_read()

    def stop(self):
        self.close_file()
        self.io_loop.stop()

    def open_file(self):
        try:
            self.file = open(self.file_path, "a")
            if self.file.tell() == 0:
                self.file.write(f"{self.csv_header}\n")
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
        """Receives Change Stream message from MongoDB

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
                if not message_json.get("trigger"):  # bypass ML trigger
                    self.parse_change_stream_event(message_json)

                # read images.csv into a dataframe e provide some
                # statistics of the folder content
                try:
                    df = pd.read_csv(self.file_path)
                    logger.info(
                        f"Loaded dataframe of directory content with {len(df)} records"
                    )
                except pd.errors.EmptyDataError:
                    logger.info("The CSV file is empty")
                except pd.errors.ParserError as e:
                    logger.exception(f"Error parsing the CSV file: {e}")
                except Exception as e:
                    logger.exception(
                        f"An error occurred while reading the CSV file: {e}"
                    )

            except Exception as e:
                logger.exception(
                    f"An error occurred while parsing the change stream event: {e}"
                )
            finally:
                # check if dataset is ready for training
                if not message_json.get("trigger"):  # bypass ML trigger
                    res = self.checker.is_dataset_ready(method='total')
                    if res:
                        logger.warning(
                            "Staging dataset ready for ML training. Trigger Prefect server"
                        )
                        self.signal_to_ml_workflow({"trigger": "train_ml"})
                        # self.rotate_dataset_directory()

    def is_dataset_ready(self, threshold=50, class_thresholds=None, method="total"):
        """Count images per class"""
        topdir = Path(DATASET_BASEDIR) / "images"
        if not topdir.exists():
            return False

        class_folders = [
            subfolder for subfolder in topdir.iterdir() if subfolder.is_dir()
        ]
        class_counts = count_files_in_subfolders_efficient(class_folders)
        total_counts = sum(class_counts.values())

        logger.info(f"[{total_counts}] {class_counts}")

        if method == "total":
            return total_counts > threshold

        if method == "per_class" and class_thresholds:
            for cls_name, cls_thresh in class_thresholds.items():
                if class_counts.get(cls_name, 0) < cls_thresh:
                    return False
            return True

        return False

    def snapshot_csv_file(self):
        """Take a snapshot of the images.csv."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        snapshot_file_path = self.file_path.with_stem(
            f"{self.file_path.stem}_{timestamp}"
        )
        lock_file_path = self.file_path.with_suffix(".lock")
        lock = FileLock(lock_file_path)

        try:
            with lock:
                shutil.copy2(self.file_path, snapshot_file_path)
                logger.info(f"Snapshot of 'images.csv' created: {snapshot_file_path}")
        except Exception as e:
            logger.exception(f"Failed to create a snapshot of 'images.csv': {e}")

    def signal_to_ml_workflow(self, message):
        if self.connection:
            try:
                self.snapshot_csv_file()
                self.connection.write_message(json_util.dumps(message))
            except Exception as e:
                logger.exception(f"Failed to trigger the ML workflow: {e}")

    def parse_change_stream_event(self, change_event):
        operation_type = change_event.get("operationType")
        event_id = change_event.get("_id").get("_data")
        inspection_id = change_event.get("documentKey").get("_id")
        if event_id:
            logger.info(
                f"Worker: received Change Stream event:\n"
                f"{'Resume token:':<20} {event_id[:20]}\n"
                f"{'Change Stream OP:':<20} {operation_type}\n"
                f"{'Inspection ID:':<20} {inspection_id}"
            )

        if operation_type == "delete":
            logger.debug(f"Inspection {inspection_id} was deleted.")
            return

        full_document = change_event.get("fullDocument")
        if full_document:
            # Mongo change streams with operation type 'insert/update/replace'
            human_classcode = full_document.get("gscs_classification").get("classCode")
            if human_classcode:
                # Inspection with GSCS classification: add to dataset
                add_image_to_dataset(
                    full_document,
                    mongo_db,
                    dataset_basedir=self.dataset_dir,
                    csv_header=self.csv_header,
                    csv_fd=self.file,
                )
        else:
            logger.warning("Change Stream with no 'fullDocument'")


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
