import tornado.ioloop
import tornado.websocket



class WebSocketClient:
    def __init__(self, io_loop, url="ws://127.0.0.1:8000/socket", max_retries=10, retry_interval=3):
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
        print("Reading ...")
        tornado.websocket.websocket_connect(
            url=self.url,
            callback = self.maybe_retry_connection,
            on_message_callback = self.on_message,
            ping_interval = 10,
            ping_timeout = 30
        )

    def original_maybe_retry_connection(self, future):
        try:
            self.connection = future.result()
        except:
            print("Could not reconnect, retrying ...")
            self.io_loop.call_later(self.retry_interval, self.connect_and_read)

    def maybe_retry_connection(self, future):
        try:
            self.connection = future.result()
            self.retries = 0
        except Exception as e:
            print(f"Failed to connect to {self.url} ...")

            if self.retries < self.max_retries:
                self.retries += 1
                print(f"Retrying ... {self.retries}/{self.max_retries}")
                self.io_loop.call_later(self.retry_interval, self.connect_and_read)
            else:
                print(f"Max attempts reached. Could not connect to server {self.url}")
                self.stop()

    def on_message(self, message):
        if message is None:
            print("Disconnected, reconnecting ...")
            self.connect_and_read()
        else:
            print(f"Received: {message}")
            print("Determine image class distribution")
            print("="*82)


def main():
    io_loop = tornado.ioloop.IOLoop.current()
    client = WebSocketClient(io_loop)
    io_loop.add_callback(client.start)

    io_loop.start()


if __name__ == "__main__":
    main()