import argparse

import logzero
import tornado.httpserver
import tornado.ioloop
import tornado.web
import tornado.websocket
from bson import json_util
from logzero import logger
from motor.motor_tornado import MotorClient


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
                print(e)

    @classmethod
    def on_change(cls, change):
        logger.debug(change)

        # operationType = change['operationType']
        # if operationType in ['update', 'insert', 'replace']:
        #     message = f"{operationType}: {change['fullDocument']}"
        # else:
        #     message = f"{change['operationType']}: {change['documentKey']}"

        change_json = json_util.dumps(change)
        ChangesHandler.send_updates(change_json)


change_stream = None


async def watch(collection):
    global change_stream

    try:
        while True:
            """
            This loop is used to resume watching if some 'invalidate' operation is detected.
            An invalidate event is emitted when:
            - a collection is  dropped
            - a collection is renamed
            - a database is dropped
            """

            async with collection.watch(full_document="updateLookup") as change_stream:
                async for change in change_stream:
                    if change.get("operationType") == "invalidate":
                        logger.warning("An 'invalidate' operation was detected. Resuming watching ...")
                        break  # break loop and restart watch process

                    ChangesHandler.on_change(change)
    except Exception as e:
        logger.error(f"Error watching change stream: {e}")


def main():
    logzero.logfile("server.log")

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

    client = MotorClient(args.host, args.port, username="ivision", password="ivSN")

    # create a web app whose only endpoint is a WebSocket, and start the
    # web app on port 8000
    app = tornado.web.Application(
        [(r"/socket", ChangesHandler), (r"/", WebpageHandler)],
    )
    app.listen(8000)


    loop = tornado.ioloop.IOLoop.current()

    inspections_collection = client.gerdau_scrap_classification.inspections
    loop.add_callback(watch, inspections_collection)

    gscs_classifications_collection = client.gerdau_scrap_classification.gscs_classifications
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
