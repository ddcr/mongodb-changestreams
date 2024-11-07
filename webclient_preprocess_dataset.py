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
from utils import connect_to_mongo

mongo_db = None

if sys.platform.startswith("linux"):
    DATASET_BASEDIR = os.getenv("DATASET_DIR", r"staging_dataset")
else:
    DATASET_BASEDIR = os.getenv(
        "DATASET_DIR", r"D:\ivision\automatic_retraining\staging_dataset"
    )


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
        self.csv_header = "path,camera,created_at,added_at,ai_class,human_class"

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
                if not message_json.get("trigger"):  # bypass ML
                    res = self.is_dataset_ready()
                    if res:
                        self.signal_to_ml_workflow({"trigger": "train_ml"})
                        # self.rotate_dataset_directory()
                    else:
                        logger.warning(
                            "Threshold not met: [current_count] images in staging directory."
                        )
                        logger.warning(
                            "Awaiting additional images to trigger ML training workflow."
                        )

    def is_dataset_ready(self):
        """Count images per class"""
        # TODO
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
                    f"Change stream event missing 'fullDocument' for operation type '{operation_type}'.\n"
                    f"Event details: operationType={operation_type}, documentKey={event_docuid}"
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
                    add_image_to_dataset(
                        full_document,
                        mongo_db,
                        dataset_basedir=self.dataset_dir,
                        csv_fd=self.file,
                    )
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
