import logzero
import tornado.ioloop
import tornado.websocket
from bson import json_util
from logzero import logger


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
        f"Worker: receive Change Stream Event with ID:\n"
        f"{'ID:':<20} {event_id}"
        f"{'OP:':<20} {operation_type}\n"
        f"{'Document Mongo ID:':<20} {event_docuid}"
    )

    # verify type of document
    full_document = change_event.get('fullDocument')

    # handle cases with fullDocument = None
    if full_document is None:
        if operation_type == 'delete':
            logger.debug(f"Document with ID {event_docuid} was deleted.")
        else:
            logger.error("No full document available for this event.")
        return

    # proceed with handling based on the contnt of 'fullDocument'
    document_origin = full_document.get("origin")

    if document_origin is None:
        if 'need_sync' in full_document:
            # GSCS entry inserted
            gscs_id = full_document.get("gscs_id")

            if operation_type == 'insert':
                logger.debug(f"Inserted GSCS info with ID: {gscs_id}")
            elif operation_type == 'update':
                logger.debug(f"GSCI ID {gscs_id} has been synchronized with Gerdau DB")
                # A human classification has been added. Let us process this inspection
                add_image_to_dataset(full_document)
            else:
                logger.debug(f"GSCS {gscs_id}: operationType = '{operation_type}'")

    elif document_origin == "/gerdau/scrap_detection/inspect":
        # inspection classified by AI
        inspection_id = full_document.get("_id")
        if operation_type == 'insert':
            logger.debug(f"Inspection ID {inspection_id}: {operation_type}")
            logger.warning(f"Send Inspection {inspection_id} to another endpoint")
        elif operation_type in ['update', 'replace']:
            logger.debug(f"Inspection ID {inspection_id}: {operation_type}")
        elif operation_type == 'delete':
            logger.debug(f"Inspection document with ID {event_docuid} was deleted.")
        else:
            logger.warning(f"Unexpected operation type for inspection document: {operation_type}")
    else:
        logger.error("Unknown document type associated with change stream")


def add_image_to_dataset(full_document):
    logger.debug(full_document)
    logger.warning("Check if this image can go to the dataset")


def main():
    """
    """
    logzero.logfile("webclient_preprocess_dataset.log")
    io_loop = tornado.ioloop.IOLoop.current()
    client = WebSocketClient(io_loop)
    io_loop.add_callback(client.start)

    io_loop.start()


if __name__ == "__main__":
    main()
