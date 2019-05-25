from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import traceback


class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)


class Reader(object):
    """Wrapper for dataset generation given a read function"""

    def __init__(self, dtypes):
        """Constructs a Reader instance

        Args:
            read_fn: Input function returning features which is a dictionary of
                string feature name to `Tensor` or `SparseTensor`. If it
                returns a tuple, first item is extracted as features.
                Prediction continues until `input_fn` raises an end-of-input
                exception (`OutOfRangeError` or `StopIteration`).
            dtypes:  A nested structure of tf.DType objects corresponding to
                each component of an element yielded by generator.

        """
        self.dtypes = dtypes


    def serving_input_receiver_fn(self, placeholder_shapes):
        """Build the serving inputs.

        Args:
            placeholder_shapes: A nested structure of lists or tuples
                corresponding to the shape of each component of the feature
                elements yieled by the read_fn.

        Returns:
            function: A function to be passed to the tf.estimator.Estimator
            instance when exporting a saved model with estimator.export_savedmodel.
        """

        def f():
            inputs = {k: tf.placeholder(
                shape=[None] + list(placeholder_shapes['features'][k]),
                dtype=self.dtypes['features'][k]) for k in list(self.dtypes['features'].keys())}

            return tf.estimator.export.ServingInputReceiver(inputs, inputs)
        return f
