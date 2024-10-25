Forked repository


# Subscribe to MongoDB Change Streams via WebSockets using Python and Tornado

This is the example code for a [MongoDB Developer Hub](https://developer.mongodb.com/) tutorial.

## Getting started

For a complete guide please see the tutorial ["Subscribe to MongoDB Change Streams via WebSockets using Python and Tornado"](https://developer.mongodb.com/how-to/subscribing-changes-browser-websockets)

    git clone git@github.com:aaronbassett/mongodb-changestreams-tornado-example.git
    cd mongodb-changestreams-tornado-example
    pip install -r requirements.txt
    export MONGO_SRV=
    python server.py


Additional note:
- For inscribing boxes we need the following packages:
  - numba
  - shapely
  - opencv
  - third-party source package: lir-0.2.1.zip

- Installing opencv in a conda environment with python=3.10.14:
	mamba install opencv=4.10.0=headless_py310hc53ca14_10

