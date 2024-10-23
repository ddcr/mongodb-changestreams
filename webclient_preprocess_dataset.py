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
from pathlib import Path

import logzero
import tornado.ioloop
import tornado.websocket
from bson import json_util
from logzero import logger
from pymongo import MongoClient

from utils import scrapRank, copy_file

mongo_db = None
DATASET_BASEDIR = os.getenv("DATASET_DIR", r"D:\ivision\automatic_retraining\dataset")


class WebSocketClient:
    def __init__(
        self,
        io_loop,
        url="ws://127.0.0.1:8000/socket",
        max_retries=10,
        retry_interval=3,
    ):
        self.connection = None
        self.io_loop = io_loop
        self.url = url
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self.retries = 0

    def start(self):
        self.connect_and_read()

    def stop(self):
        self.io_loop.stop()

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
            logger.info(f"Failed to connect to {self.url} ...")

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
            message_json = json_util.loads(message)
            parse_change_stream_event(message_json)


def parse_change_stream_event(change_event):
    operation_type = change_event.get("operationType")
    event_id = change_event.get("_id")
    event_docuid = change_event.get("documentKey")
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
            logger.error(
                "This change stream event involves no 'fullDocument'.\n"
                f"Operation: {operation_type}"
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
                add_image_to_dataset(full_document)
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


def add_image_to_dataset(full_document):
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
            ai_classcode = (
                ai_inspection.get("result", {}).get("detection", {}).get("classCode")
            )

            if manual_classcode == ai_classcode:
                logger.debug(
                    f"[AI inspection] - {inspection_id} | Action: add to DataSet"
                )

                # extract image paths for the inspection points
                for i in ai_inspection.get("inspections", []):
                    camera, inpath = i["inspectionPoint"], i["imagePath"]
                    outpath = Path(DATASET_BASEDIR) / 'images' / camera / scrapRank[ai_classcode]

                    # The MongoDB instance is installed and running on a Windows-based machine
                    if sys.platform.startswith('linux'):
                        inpath = inpath.replace('\\', '/')
                        outpath = str(outpath).replace('\\', '/')

                        inpath = inpath.replace("D:", "/media/ddcr/sahagun")
                        outpath = outpath.replace("D:", "/media/ddcr/sahagun")

                    logger.info(f"{camera}: {inpath} -> {outpath}")
                    # add_image_to_dataset(ins)

            else:
                logger.warning(
                    f"[AI inspection] - {inspection_id} | AI and human classifications differ"
                )
        else:
            logger.warning(f"No AI inspection found for GSCS ID: {gscs_id}")
    except Exception as e:
        logger.error(f"Failed to process document: {e}")


def connect_to_mongo(host: str, port: int):
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
        db = mongo_client.gerdau_scrap_classification
        logger.info(f"Connected to MongoDB '{host}:{port}'")
        return db
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise


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
    logzero.logfile("webclient_preprocess_dataset.log")

    mongo_db = connect_to_mongo(args.host, args.port)

    io_loop = tornado.ioloop.IOLoop.current()
    client = WebSocketClient(io_loop)
    io_loop.add_callback(client.start)

    io_loop.start()


if __name__ == "__main__":
    main()
