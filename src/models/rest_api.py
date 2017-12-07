import bottle
import base64

from grpc.beta import implementations
import json
import tensorflow as tf

import predict_pb2
import prediction_service_pb2

from random import *
import os

tf.app.flags.DEFINE_string("host", "127.0.0.1", "gRPC server host")
tf.app.flags.DEFINE_integer("port", 9000, "gRPC server port")
tf.app.flags.DEFINE_string("model_name", "ip5wke", "TensorFlow model name")
tf.app.flags.DEFINE_integer("model_version", 1, "TensorFlow model version")
tf.app.flags.DEFINE_float("request_timeout", 100.0, "Timeout of gRPC request")
FLAGS = tf.app.flags.FLAGS




class Inference:
    def __init__(self, host, port):
        serv_host = FLAGS.host
        serv_port = FLAGS.port
        model_name = FLAGS.model_name
        model_version = FLAGS.model_version
        self.request_timeout = FLAGS.request_timeout

        # Create gRPC client and request
        channel = implementations.insecure_channel(serv_host, serv_port)
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(
            channel)
        self.request = predict_pb2.PredictRequest()
        self.request.model_spec.name = model_name
        self.request.model_spec.signature_name = 'predict_images'

        if model_version > 0:
            self.request.model_spec.version.value = model_version

        self._host = host
        self._port = port
        self._app = bottle.Bottle()
        self._route()

    def _route(self):
        self._app.route('/', method="POST", callback=self._POST)
        self._app.route('/new_workpiece_id', method="GET", callback=self.new_workpiece_id)
        self._app.route('/add_workpiece_image', method="POST", callback=self.add_workpiece_image)

    def start(self):
        self._app.run(host=self._host, port=self._port)

    def _POST(self): #TODO better name...
        # REST endpoint for prediction, takes base64 encoded jpg and returns the top 3 predicted classes with class probabilities
        file_data = base64.b64decode(bottle.request.json['image'])

        with open('current.jpg', 'wb') as f:
            f.write(file_data)
            f.close()

        self.request.inputs['images'].CopyFrom(
            tf.contrib.util.make_tensor_proto(file_data, shape=[1]))

        # Send request
        answer = self.stub.Predict(self.request, self.request_timeout)
        scores = [answer.outputs['scores'].float_val[0],
                             answer.outputs['scores'].float_val[1],
                             answer.outputs['scores'].float_val[2]]
        classes = [answer.outputs['classes'].string_val[0].decode("utf-8"),
                             answer.outputs['classes'].string_val[1].decode("utf-8"),
                             answer.outputs['classes'].string_val[2].decode("utf-8")]
	images = []

	for i in range(0, 3):
	    f = open('classimages/' + str(classes[i]) + '.PNG', 'rb')
	    images.append(base64.b64encode(f.read()))
	    f.close()

        return {"scores": scores, "classes": classes, "images": images}

    def new_workpiece_id(self):
        # REST endpoint for getting a new workpiece id
	print("new workpiece id requested")
	new_id = randint(1000, 1000000) #TODO: get next free id instead
	print("new workpiece id = " + str(new_id))

        return {"workpieceId": new_id}

    def add_workpiece_image(self):
        # REST endpoint for adding an image for a new workpiece
	print("adding image for new workpiece")
	workpiece_id = bottle.request.json['workpieceId']
	print(bottle.request)
	print(bottle.request.json)
	
        image_number = bottle.request.json['imageNumber']
	image = bottle.request.json['image']
    	directory = 'new_workpieces/' + str(workpiece_id) + '/' #TODO configure these folders at the top as constants
    	if not os.path.exists(directory):
	    os.makedirs(directory)
   	f = open(directory + str(image_number) + '.jpg', 'wb')
    	f.write(base64.b64decode(image))
    	f.close()

        return {"workpieceId": workpiece_id}



if __name__ == '__main__':
    # start server
    server = Inference(host='0.0.0.0', port=8888)
    server.start()