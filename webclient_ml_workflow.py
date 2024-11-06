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
import uuid
from pathlib import Path

import httpx
import tornado.ioloop
import tornado.websocket
from bson import json_util
from logzero import logger

script_path = os.path.abspath(__file__)
WORK_DIR = os.path.dirname(script_path)

if sys.platform.startswith("linux"):
    DATASET_BASEDIR = os.getenv(
        "DATASET_DIR", os.path.join(WORK_DIR, "staging_dataset")
    )
else:
    DATASET_BASEDIR = os.getenv(
        "DATASET_DIR", r"D:\ivision\automatic_retraining\staging_dataset"
    )


class WebSocketClient:
    def __init__(
        self,
        io_loop,
        deployment_ids,
        staging_dataset_dir=DATASET_BASEDIR,
        url="ws://127.0.0.1:8000/socket",
        max_retries=10,
        retry_interval=3,
    ):
        self.connection = None
        self.io_loop = io_loop
        self.deployment_ids = deployment_ids
        self.staging_dataset_dir = staging_dataset_dir
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

    def original_maybe_retry_connection(self, future):
        try:
            self.connection = future.result()
        except Exception as e:
            logger.exception("Could not reconnect, retrying ...")
            self.io_loop.call_later(self.retry_interval, self.connect_and_read)

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
        if message is None:
            logger.info("Disconnected, reconnecting ...")
            self.connect_and_read()
        else:
            message_json = json_util.loads(message)
            if message_json.get("trigger") == "start_ml":
                logger.info("Trigger the ML workflow ...")
                trigger_prefect_flow(self.deployment_ids, self.staging_dataset_dir)


def trigger_prefect_flow(deployment_ids, dataset_dir):
    headers = {"Authorization": "Bearer of PREFECT_API_KEY"}
    payload = {
        "name": "ivision-automl",  # not required
        "parameters": {"dset_inputdir": dataset_dir},
    }

    for deployment_id in deployment_ids:
        try:
            with httpx.Client() as client:
                response = client.post(
                    f"http://localhost:4200/api/deployments/{deployment_id}/create_flow_run",
                    json=payload,
                )
                response.raise_for_status()

                flow_run_info = response.json()
                logger.debug(
                    f"Triggered the ML flow run for deployment {deployment_id}: {flow_run_info}"
                )
        except httpx.HTTPStatusError as e:
            logger.exception(
                f"HTTP error occurred for deployment {deployment_id}: {e.response.status_code} - {e.response.text}"
            )
        except Exception as e:
            logger.exception(f"An error occurred for deployment {deployment_id}: {e}")


def valid_uuid(uuid_string):
    try:
        # Attempt to create a UUID object to validate the format
        uuid.UUID(uuid_string)
        return uuid_string
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid UUID: '{uuid_string}'")


def main():
    global deployment_id

    parser = argparse.ArgumentParser(
        description="WebClient that triggers a Prefect flow via API call", add_help=True
    )

    parser.add_argument(
        "--folder",
        type=lambda d: Path(d).absolute() if d else None,
        default=None,
        help="Directory for staging images prior to processing and training.",
    )

    parser.add_argument(
        "--id",
        type=valid_uuid,
        nargs="+",
        required=True,
        help="ID(s) of Prefect deployment workflow. Accepts a single UUID or a list separated by spaces",
    )

    args = parser.parse_args()

    io_loop = tornado.ioloop.IOLoop.current()
    if args.folder is None:
        client = WebSocketClient(io_loop, args.id)
    else:
        client = WebSocketClient(io_loop, args.id, staging_dataset_dir=args.folder)

    io_loop.add_callback(client.start)

    io_loop.start()


if __name__ == "__main__":
    main()
