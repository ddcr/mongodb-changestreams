import tornado.ioloop
import tornado.websocket
from bson import json_util


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

    async def start(self):
        await self.connect_and_read()

    def stop(self):
        self.io_loop.stop()

    async def connect_and_read(self):
        print("Connecting and reading ...")
        try:
            self.connection = await tornado.websocket.websocket_connect(
                url=self.url,
                ping_interval=10,
                ping_timeout=30,
            )
            self.retries = 0  # Reset retries on successful connection
            self.connection.read_message(self.on_message)
        except Exception as e:
            print(f"Failed to connect to {self.url}: {e}")
            if self.retries < self.max_retries:
                self.retries += 1
                print(f"Retrying ... {self.retries}/{self.max_retries}")
                self.io_loop.call_later(self.retry_interval, lambda: self.io_loop.add_callback(self.connect_and_read))
            else:
                print(f"Max attempts reached. Could not connect to server {self.url}")
                self.stop()

    def on_message(self, message):
        if message is None:
            print("Disconnected, reconnecting ...")
            self.io_loop.add_callback(self.connect_and_read)
        else:
            print(f"Received: {message}")
            print("Determine image class distribution")
            print("=" * 82)

    def signal_to_ml_workflow(self, message):
        if self.connection:
            try:
                self.connection.write_message(json_util.dumps(message))
            except Exception as e:
                print(f"Failed to trigger the ML workflow: {e}")


async def main():
    io_loop = tornado.ioloop.IOLoop.current()
    client = WebSocketClient(io_loop)
    await client.start()

    # Send a message to trigger the ML workflow once connected
    client.signal_to_ml_workflow({"trigger": "start_ml"})


if __name__ == "__main__":
    tornado.ioloop.IOLoop.current().run_sync(main)
