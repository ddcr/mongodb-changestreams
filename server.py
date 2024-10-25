#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This server focus on just watching and relaying change stream events via WebSockets
Uses Motor
"""

__author__ = "Domingos Rodrigues"
__email__ = "domingos.rodrigues@inventvision.com.br"
__copyright__ = "Copyright (C) 2024 Invent Vision"
__license__ = "Strictly proprietary for Invent Vision."


import argparse

import logzero
import tornado.httpserver
import tornado.ioloop
import tornado.web
import tornado.websocket
from bson import json_util
from logzero import logger
from motor.motor_asyncio import AsyncIOMotorClient

# from motor.motor_tornado import MotorClient


# Just a page web template
class WebpageHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("templates/index.html")


class ChangesHandler(tornado.websocket.WebSocketHandler):
    connected_clients = set()

    def check_origin(self, origin):
        return True

    def open(self):
        ChangesHandler.connected_clients.add(self)

    def on_close(self):
        ChangesHandler.connected_clients.remove(self)

    @classmethod
    def send_updates(cls, message):
        for connected_client in cls.connected_clients:
            try:
                connected_client.write_message(message)
            except Exception as e:
                logger.exception(e)

    @classmethod
    def on_change(cls, change):
        logger.debug(change)

        change_json = json_util.dumps(change)
        ChangesHandler.send_updates(change_json)


change_stream = None


async def watch(collection):
    global change_stream
    resume_token = None

    try:
        while True:
            """
            This loop is used to resume watching if some 'invalidate' operation is detected.
            An invalidate event is emitted when:
            - a collection is  dropped
            - a collection is renamed
            - a database is dropped
            """
            resume_options = {}
            if resume_token is not None:
                resume_options["start_after"] = resume_token

            async with collection.watch(
                full_document="updateLookup", **resume_options
            ) as change_stream:
                async for change in change_stream:
                    resume_token = change.get("_id")
                    if change.get("operationType") == "invalidate":
                        logger.warning(
                            f"[{collection}] An 'invalidate' operation was detected. Resuming watching ..."
                        )
                        break  # break loop and restart watch process
                    ChangesHandler.on_change(change)

    except Exception as e:
        logger.exception(f"Error watching change stream: {e}")


def main():
    logzero.logfile("logs/server.log", maxBytes=1000000, backupCount=5)

    parser = argparse.ArgumentParser(
        description="WebSocket Server that proxies any new data from the Mongo change stream to connected clients",
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

    # client = MotorClient(args.host, args.port, username="ivision", password="ivSN")
    client = AsyncIOMotorClient(
        args.host,
        args.port,
        username="ivision",
        password="ivSN",
        maxPoolSize=20,
        minPoolSize=5,
    )

    # create a web app whose only endpoint is a WebSocket, and start the
    # web app on port 8000
    app = tornado.web.Application(
        [(r"/socket", ChangesHandler), (r"/", WebpageHandler)],
    )
    app.listen(8000)

    loop = tornado.ioloop.IOLoop.current()

    inspections_collection = client.gerdau_scrap_classification.inspections
    loop.add_callback(watch, inspections_collection)

    gscs_classifications_collection = (
        client.gerdau_scrap_classification.gscs_classifications
    )
    loop.add_callback(watch, gscs_classifications_collection)

    logger.info("Started listening ...")

    try:
        loop.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shuting down server ...")
        pass
    except Exception as e:
        logger.exception(e)
    finally:
        if change_stream is not None:
            change_stream.close()
        loop.stop()
        logger.info("Server stopped.")


if __name__ == "__main__":
    main()
