FROM tf-model-server

MAINTAINER Christian Fröhlingsdorf <chris@5cf.de>

RUN pip install tensorflow zerorpc msgpack_numpy json_format
RUN mkdir /pyserver_app
COPY . /pyserver_app
RUN ls -l /pyserver_app

WORKDIR /pyserver_app
CMD ["python", "/pyserver_app/serving_rpc.py"]
