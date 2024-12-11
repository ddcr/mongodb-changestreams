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
import asyncio
import sys

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
        """Allows all origins (security risk)"""
        return True

    def open(self):
        ChangesHandler.connected_clients.add(self)

    def on_close(self):
        ChangesHandler.connected_clients.remove(self)

    def on_message(self, message):
        """Broadcast the message to other clients"""
        client_message = json_util.loads(message)
        if client_message.get("trigger") == "start_ml":
            self.relay_trigger_to_clients(client_message)
            logger.warning("A trigger warning 'start_ml' was relayed")

    @classmethod
    def send_updates(cls, message):
        for connected_client in cls.connected_clients:
            try:
                connected_client.write_message(message)
            except Exception as e:
                logger.exception(e)

    @classmethod
    def on_change(cls, change):
        if 'fullDocument' in change: #  insert/update/delete
            change_log = change.copy()
            change_log.pop('fullDocument')
            logger.info(change_log)
        else:
            logger.info(change)

        operationType = change.get('operationType')
        if operationType:
            if  operationType == 'drop':
                logger.warning(f"Collection '{change['ns']['db']}.{change['ns']['coll']}' dropped")
            elif operationType == 'dropDatabase':
                logger.warning(f"Database '{change['ns']['db']}' dropped")
            else:
                if 'updateDescription' in change:
                    change_json = json_util.dumps(change)
                    ChangesHandler.send_updates(change_json)
        else:
            logger.error("'operationType' not present!")

    def relay_trigger_to_clients(self, message):
        relayed_message = json_util.dumps(message)
        # relay 'message' to all connected clients except the sender
        for client in self.connected_clients:
            if client != self:
                try:
                    client.write_message(relayed_message)
                except Exception as e:
                    logger.exception(f"Failed to relay message to client: {e}")


change_stream = None


async def watch(collection):
    global change_stream
    resume_token = None

    while True:
        try:
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

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.exception(f"Error in change stream watch: {e}")
            await asyncio.sleep(5)


def main():
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

    parser.add_argument(
        "--disable-stderr-logger",
        action="store_true",
        help="Disable the stderr logger for logzero.",
    )

    args = parser.parse_args()

    logzero.logfile(
        "logs/server.log",
        maxBytes=1000000,
        backupCount=5,
        disableStderrLogger=args.disable_stderr_logger,
    )

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

    logger.warning("Started listening ...")

    try:
        loop.start()
    except (KeyboardInterrupt, SystemExit):
        logger.warning("Shuting down server ...")
        pass
    except Exception as e:
        logger.exception(e)
    finally:
        if change_stream is not None:
            loop.run_in_executor(None, change_stream.close)
        loop.stop()
        logger.warning("Server stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
