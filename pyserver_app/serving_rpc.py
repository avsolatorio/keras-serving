#!/usr/bin/env python2.7
from __future__ import print_function
import zerorpc
import msgpack_numpy
import argparse

from grpc.beta import implementations
import tensorflow as tf

# Proto taken from here: https://github.com/tobegit3hub/deep_recommend_system/tree/master/python_predict_client
# Found in issue: https://github.com/tensorflow/serving/issues/237#issuecomment-269313683
import predict_pb2
import prediction_service_pb2
# requires pip install json_format
from google.protobuf.json_format import MessageToJson

msgpack_numpy.patch()

parser = argparse.ArgumentParser(description="Model Server")
parser.add_argument(
    "--port", type=int, default=19000,
    help="Model port to serve"
)
parser.add_argument(
    "--server-host", type=str, default="model-server",
    help="Faiss port to serve"
)

args = parser.parse_args()


# TODO: finalize the folder for faiss
def _getfile(name, version):
    return 1


class ModelRPC(object):
    # Note: for some reason, when creating a new index, you need to make
    # sure you have a reference to the base index, i.e. we're using the
    # index IndexIDMap over the base index IndexFlatIP, but we still need
    # to keep the reference to IndexFlatIP
    def __init__(self):
        self.server_host = 'model-server'
        self.server_port = 9000

    def infer(self, data, shape):
        channel = implementations.insecure_channel(
            self.server_host, self.server_port
        )
        stub = prediction_service_pb2.beta_create_PredictionService_stub(
            channel
        )
        # Send request
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'main_model'
        request.model_spec.signature_name = 'predict'

        request.inputs['inputs'].CopyFrom(
            tf.contrib.util.make_tensor_proto(
                data, shape=shape
            )
        )

        result = stub.Predict(request, 10.0)  # 10 secs timeout
        # print("Type:", type(result))
        # print(result)
        result = MessageToJson(result)
        result = result.get('outputs' {}).get('outputs', {}).get('floatVal')

        return result

server = zerorpc.Server(ModelRPC())
server.bind("tcp://0.0.0.0:{}".format(args.port))
server.run()

