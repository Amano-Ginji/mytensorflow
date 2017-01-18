#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
from six.moves import urllib

import pandas as pd
import tensorflow as tf

import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", "model", "Base directory for output models.")
flags.DEFINE_string("model_type", "wide_n_deep", "Valid model types: {'wide', 'deep', 'wide_n_deep'}.")
flags.DEFINE_integer("train_steps", 200, "Number of training steps.")
flags.DEFINE_string("train_data", "", "Path to the training data.")
flags.DEFINE_string("test_data", "", "Path to the test data.")
flags.DEFINE_string("predict_data", "", "Path to the predict data.")
flags.DEFINE_float("learning_rate", 0.001, "learning rate")
flags.DEFINE_float("dropout", 0.01, "dropout")


def maybe_download():
    """Maybe downloads training data and returns train and test file names."""
    if FLAGS.train_data:
        train_file_name = FLAGS.train_data
    else:
        train_file = tempfile.NamedTemporaryFile(delete=False)
        urllib.request.urlretrieve("http://download.tensorflow.org/data/abalone_train.csv", train_file.name)
        train_file_name = train_file.name
        train_file.close()
        print("Training data is downloaded to %s" % train_file_name)

    if FLAGS.test_data:
        test_file_name = FLAGS.test_data
    else:
        test_file = tempfile.NamedTemporaryFile(delete=False)
        urllib.request.urlretrieve("http://download.tensorflow.org/data/abalone_test.csv", test_file.name)
        test_file_name = test_file.name
        test_file.close()
        print("Test data is downloaded to %s" % test_file_name)

    if FLAGS.predict_data:
        predict_file_name = FLAGS.predict_data
    else:
        predict_file = tempfile.NamedTemporaryFile(delete=False)
        urllib.request.urlretrieve('http://download.tensorflow.org/data/abalone_predict.csv', predict_file.name)
        predict_file_name = predict_file.name
        predict_file.close()
        print("Predict data is download to %s" % predict_file_name)

    return train_file_name, test_file_name, predict_file_name


def model_fn(features, targets, mode, params):
    # Model definition
    h1 = tf.contrib.layers.relu(features, 10)
    h2 = tf.contrib.layers.relu(h1, 10)
    output_layer = tf.contrib.layers.linear(h2, 1)
    
    predictions = tf.reshape(output_layer, [-1])
    predictions_dict = {"age": predictions}

    # Loss
    loss = tf.contrib.losses.mean_squared_error(predictions, targets)

    # Optimizer
    train_op = tf.contrib.layers.optimize_loss(
        loss = loss,
        global_step = tf.contrib.framework.get_global_step(),
        learning_rate = params["learning_rate"],
        optimizer = "SGD"
    )
    
    return predictions_dict, loss, train_op


def main(_):
    # Download data
    train_file, test_file, predict_file = maybe_download()
    
    # Load dataset
    train_data = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename = train_file,
        target_dtype = np.int,
        features_dtype = np.float64
    )

    test_data = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename = test_file,
        target_dtype = np.int,
        features_dtype = np.float64
    )

    predict_data = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename = predict_file,
        target_dtype = np.int,
        features_dtype = np.float64
    )

    # Build model
    model_params = {
        "learning_rate": FLAGS.learning_rate,
        "dropout": FLAGS.dropout
    }

    model = tf.contrib.learn.Estimator(
        model_fn = model_fn,
        params = model_params
    )

    # Train
    model.fit(
        x = train_data.data,
        y = train_data.target,
        steps = 5000
    )

    # Evaluation
    ev = model.evaluate(
        x = test_data.data,
        y = test_data.target,
        steps = 1
    )
    loss_score = ev["loss"]
    print("loss: %s" % loss_score)

    # Predict
    predictions = model.predict(
        x = predict_data.data,
        as_iterable = True
    )
    for i, p in enumerate(predictions):
        print("Prediction %s: %s" % (i+1, p["age"]))


if __name__ == "__main__":
  tf.app.run()

