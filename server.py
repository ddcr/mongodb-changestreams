import os
import tornado.httpserver
import tornado.websocket
import tornado.ioloop
import tornado.web
from bson import json_util
from motor.motor_tornado import MotorClient
from logzero import logger


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
        operationType = change['operationType']

        if operationType in ['update', 'insert', 'replace']:
            message = f"{operationType}: {change['fullDocument']}"
        else:
            message = f"{change['operationType']}: {change['documentKey']}"
        change_json = json_util.dumps(change)
        ChangesHandler.send_updates(change_json)


change_stream = None


async def watch(collection):
    global change_stream

    async with collection.watch(full_document='updateLookup') as change_stream:
        async for change in change_stream:
            ChangesHandler.on_change(change)


def main():
    client = MotorClient('127.0.0.1', 27017, username='ivision', password='ivSN')
    collection = client.gerdau_scrap_classification.inspections

    # create a web app whose only endpoint is a WebSocket, and start the
    # web app on port 8000
    app = tornado.web.Application(
        [(r"/socket", ChangesHandler), (r"/", WebpageHandler)],
    )

    app.listen(8000)

    loop = tornado.ioloop.IOLoop.current()
    loop.add_callback(watch, collection)
    try:
        loop.start()
    except KeyboardInterrupt:
        pass
    finally:
        if change_stream is not None:
            change_stream.close()


if __name__ == "__main__":
    main()
