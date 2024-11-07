import shutil
from datetime import datetime
from pathlib import Path

import tornado.ioloop
import tornado.websocket
from bson import json_util
from filelock import FileLock


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

    async def start(self):
        self.open_file()
        await self.connect_and_read()

    def stop(self):
        self.io_loop.stop()

    def open_file(self):
        try:
            self.file = open(self.file_path, "a")
            print(f"File opened for appending data: {self.file_path}")
        except Exception as e:
            print(f"Failed to open file {self.file_path}: {e}")

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
                self.io_loop.call_later(
                    self.retry_interval,
                    lambda: self.io_loop.add_callback(self.connect_and_read),
                )
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

    def snapshot_csv_file(self):
        """Take a snapshot of the images.csv."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_file_path = self.file_path.with_stem(
            f"{self.file_path.stem}_{timestamp}"
        )
        lock_file_path = self.file_path.with_suffix(".lock")
        lock = FileLock(lock_file_path)

        try:
            with lock:
                shutil.copy2(self.file_path, snapshot_file_path)
                print(f"Snapshot of 'images.csv' created: {snapshot_file_path}")
                return snapshot_file_path
        except Exception as e:
            print(f"Failed to create a snapshot of 'images.csv': {e}")
            return None

    def signal_to_ml_workflow(self, message):
        if self.connection:
            try:
                snapshot_file = self.snapshot_csv_file()
                # append this filename to message
                if snapshot_file:
                    # snapshot_file: get only the final path component
                    message["images_file_path"] = str(snapshot_file.name)

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
