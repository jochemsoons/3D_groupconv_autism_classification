from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import SimpleITK as sitk
import tensorflow as tf
import os
import numpy as np

from dltk.io.augmentation import extract_random_example_array, flip
from dltk.io.preprocessing import whitening

def read_fn(df, mode, params=None):
    """A custom python read function for interfacing with nii image files.

    Args:
        file_references (list): A list of lists containing file references,
            such as [['id_0', 'image_filename_0', target_value_0], ...,
            ['id_N', 'image_filename_N', target_value_N]].
        mode (str): One of the tf.estimator.ModeKeys strings: TRAIN, EVAL or
            PREDICT.
        params (dict, optional): A dictionary to parametrise read_fn outputs
            (e.g. reader_params = {'n_examples': 10, 'example_size':
            [64, 64, 64], 'extract_examples': True}, etc.).

    Yields:
        dict: A dictionary of reader outputs for dltk.io.abstract_reader.
    """
    summary = 'entropy'

    for i in df.index:
        subject_id =  df['SUB_ID'].iloc[i]
        # Read the image nii with sitk

        t1_fn = df[summary].iloc[i]

        t1 = sitk.GetArrayFromImage(sitk.ReadImage(str(t1_fn)))

        # Normalise volume image
        t1 = whitening(t1)

        images = np.expand_dims(t1, axis=-1).astype(np.float32)

        if mode == tf.estimator.ModeKeys.PREDICT:
            yield {'features': {'x': images}, 'img_id': subject_id}

        # Parse the regression targets from the file_references
        y = np.float(df['DX_GROUP'].iloc[i])
        y = np.expand_dims(y, axis=-1).astype(np.float32)


        # Check if the reader is supposed to return training examples or full
        # images
        if params['extract_examples']:
            images = extract_random_example_array(
                image_list=images,
                example_size=params['example_size'],
                n_examples=params['n_examples'])

            for e in range(params['n_examples']):
                yield {'features': {'x': images[e].astype(np.float32)},
                       'labels': {'y': y.astype(np.float32)},
                       'img_id': subject_id}

        else:
            yield {'features': {'x': images},
                   'labels': {'y': y.astype(np.float32)},
                   'img_id': subject_id}

    return
