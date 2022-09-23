# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python create_xml_tf_record.py --data_dir=D:\workspace\dev\objects\datasets\augmented-set/ --output_path=D:\workspace\dev\objects\datasets\tfrecord/new_augmentedx9_add_self_no_toy_train_320.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import glob

from lxml import etree
import PIL.Image
import tensorflow.compat.v1 as tf

from object_detection.utils import dataset_util
# from object_detection.utils import label_map_util


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'merged set.')
flags.DEFINE_string('annotations_dir', 'xml-augmented',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
FLAGS = flags.FLAGS

def class_text_to_int(row_label):
    if row_label == 'pet':
        return 1
    elif row_label == 'slipper':
        return 2
    elif row_label == 'stationary':
        return 3
    elif row_label == 'poo pad':
        return 4
    elif row_label == 'remote control':
        return 5
    elif row_label == 'poo':
        return 6
    elif row_label == 'laundry':
        return 7
    elif row_label == 'wire bundle':
        return 8
    elif row_label == 'liquid':
        return 9
    elif row_label == 'cloth':
        return 10
    elif row_label == 'curtain':
        return 11
    elif row_label == 'drying rack':
        return 12
    elif row_label == 'wire':
        return 13
    elif row_label == 'band':
        return 14
    elif row_label == 'support':
        return 15
    elif row_label == 'chair':
        return 16
    elif row_label == 'munti outlet':
        return 17
    elif row_label == 'phone':
        return 18
    elif row_label == 'mask':
        return 19
    else:
        None


def dict_to_tf_example(data,
                       dataset_directory,
                       ignore_difficult_instances=False,
                       image_subdirectory='JPEGImages'):
  img_path = os.path.join(r'D:\workspace\dev\objects\datasets\augmented-set\resized-augmented-320', data['filename'])
  full_path = os.path.join(dataset_directory, img_path)
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  if 'object' in data:
    for obj in data['object']:
      if obj['name'] == 'toy':
        continue
      difficult = bool(int(obj['difficult']))
      if ignore_difficult_instances and difficult:
        continue
      if obj['name'] == 'poo pad' or obj['name'] == 'poo' or obj['name'] == 'curtain' or obj['name'] == 'laundry' or obj['name'] == 'mask':
        for _ in range(2):
          difficult_obj.append(int(difficult))

          xmin.append(float(obj['bndbox']['xmin']) / width)
          ymin.append(float(obj['bndbox']['ymin']) / height)
          xmax.append(float(obj['bndbox']['xmax']) / width)
          ymax.append(float(obj['bndbox']['ymax']) / height)
          classes_text.append(obj['name'].encode('utf8'))
          classes.append(class_text_to_int(obj['name']))
          # print(classes)
          # print(obj['name'])
          truncated.append(int(obj['truncated']))
          poses.append(obj['pose'].encode('utf8'))
      else:
          difficult_obj.append(int(difficult))

          xmin.append(float(obj['bndbox']['xmin']) / width)
          ymin.append(float(obj['bndbox']['ymin']) / height)
          xmax.append(float(obj['bndbox']['xmax']) / width)
          ymax.append(float(obj['bndbox']['ymax']) / height)
          classes_text.append(obj['name'].encode('utf8'))
          classes.append(class_text_to_int(obj['name']))
          # print(classes)
          # print(obj['name'])
          truncated.append(int(obj['truncated']))
          poses.append(obj['pose'].encode('utf8'))

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(320),
      'image/width': dataset_util.int64_feature(320),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example


def main(_):
  data_dir = FLAGS.data_dir
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
  annotations_dir = os.path.join(data_dir, FLAGS.annotations_dir)

  count = 1
  for xml_file in glob.glob(annotations_dir + '/*.xml'):
    print(count, xml_file)
    if (count % 100) != 0:
        with tf.gfile.GFile(xml_file, 'r') as fid:
          xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

        tf_example = dict_to_tf_example(data, FLAGS.data_dir,
                                          FLAGS.ignore_difficult_instances)
        writer.write(tf_example.SerializeToString())
    count += 1

  writer.close()


if __name__ == '__main__':
  tf.app.run()
