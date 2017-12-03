"""Cache the bottleneck values of the given image files and the SavedModel.

The SavedModel is used to generate the bottleneck values for the new images with
the new classification. 
Bottleneck is an inofficial term for the values right before the softmax
function.
These are used for retraining the model so we have to cache them. Otherwise
we would have to recalculate it every iteration.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf
import os
import sys
import numpy as np
import ip5wke_input
from google.protobuf import text_format

FLAGS = tf.app.flags.FLAGS

# TODO Replace with flags
BOTTLENECK_TENSOR_NAME = 'local5/local5/local5/sparsity:0'
INPUT_TENSOR_NAME = 'map/TensorArrayStack/TensorArrayGatherV3:0'

FLAGS = tf.app.flags.FLAGS
                           
tf.app.flags.DEFINE_string('retrain_data_dir', os.path.join(os.path.dirname(__file__),
                                                    os.pardir, os.pardir,
                                                    'data', 'retrain', 
                                                    'processed'), 
                            """Data Directory containing the category folders""")

def assemble_example(value, label):
    return_example = tf.train.Example(features=tf.train.Features(feature={
        "bottleneck_tensor_value": tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tostring()])),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }))
    return return_example

def create_file_list():
    filelist = {}
    for category in ['train', 'test', 'validation']:
        with open(os.path.join(FLAGS.retrain_data_dir, category, 'files.txt')) as f:
            filelist[category] = [l.strip().split(" ") for l in f]
    return filelist
    
def read_png(path):
    file_contents = tf.read_file(path)
    image = tf.image.decode_png(file_contents, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    return image

def convert_bottlenecks_to_tfrecords():
    with tf.Session(graph=tf.Graph()) as sess:
        
        meta = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], "../../models/1")
#        sess.run(tf.global_variables_initializer()) 
        
        saver = tf.train.Saver()
        saver.restore(sess, '../../models/1/variables/variables.index')
        
        bottleneck_tensor, input_tensor = tf.import_graph_def(graph_def=meta.graph_def, name="", return_elements=[BOTTLENECK_TENSOR_NAME, INPUT_TENSOR_NAME])
        
        filelist = create_file_list()
        
        for category in ['train', 'test', 'validation']:
            if not os.path.exists("../../data/retrain/tfrecords/" + category + "/"):
                os.makedirs("../../data/retrain/tfrecords/" + category + "/")
            writer = tf.python_io.TFRecordWriter("../../data/retrain/tfrecords/" + category + "/tfrecords")
            
            for entry in filelist[category]:
                sample = sess.run(read_png(os.path.join(os.pardir, os.pardir, entry[0])))
                bottleneck_tensor_value = sess.run(bottleneck_tensor, {input_tensor: [sample]})
                bottleneck_tensor_value = np.squeeze(bottleneck_tensor_value)
                example = assemble_example(bottleneck_tensor_value, label_id)
                writer.write(example.SerializeToString())
            writer.close()


if __name__ == "__main__":
    convert_bottlenecks_to_tfrecords()