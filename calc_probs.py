"""Segmentation results visualization and export of predicted probabilities on a given set of images.

derived from deeplab/vis.py
"""
import math
import os
import os.path
import sys
import time
import numpy as np
import tensorflow as tf

from metaseg_io import probs_gt_save, probs_gt_load, \
                       metaseg

sys.path.append(metaseg.get("DEEPLAB_PARENT_DIR"))

from deeplab import common
from deeplab import model
from deeplab.datasets import segmentation_dataset
from deeplab.utils import input_generator
from deeplab.utils import save_annotation

from shutil import copyfile
import datasets as ds
from PIL import Image
import h5py

import non_dataset

slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

# Settings for log directories.

flags.DEFINE_string('vis_logdir', None, 'Where to write the event logs.')

flags.DEFINE_string('checkpoint_dir', None, 'Directory of model checkpoints.')

# Settings for visualizing the model.

flags.DEFINE_integer('vis_batch_size', 1,
                     'The number of images in each batch during evaluation.')

flags.DEFINE_multi_integer('vis_crop_size', [513, 513],
                           'Crop size [height, width] for visualization.')

flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                     'How often (in seconds) to run evaluation.')

flags.DEFINE_integer('max_num_batches', 99999,
                     'Max number of batches to process')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

# Change to [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] for multi-scale test.
flags.DEFINE_multi_float('eval_scales', [1.0],
                         'The scales to resize images for evaluation.')

# Change to True for adding flipped images during test.
flags.DEFINE_bool('add_flipped_images', False,
                  'Add flipped images for evaluation or not.')

# Dataset settings.

flags.DEFINE_string('dataset', None,
                    'Name of the segmentation dataset.')

flags.DEFINE_string('vis_split', 'val',
                    'Which split of the dataset used for visualizing results')

flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')

flags.DEFINE_enum('colormap_type', None, ['pascal', 'cityscapes'],
                  'Visualization colormap type.')

flags.DEFINE_boolean('also_save_raw_predictions', False,
                     'Also save raw predictions.')

flags.DEFINE_integer('max_number_of_iterations', 0,
                     'Maximum number of visualization iterations. Will loop '
                     'indefinitely upon nonpositive values.')

flags.DEFINE_boolean('create_images', False, 'create predition-related images')

flags.DEFINE_boolean('save_probs', True, 'save probabilities')

flags.DEFINE_boolean('copy_ground_truth', None,
                     'Copy ground truth to predictions')

flags.DEFINE_string('gt_dir', None, 'Ground truth directory')

flags.DEFINE_boolean('make_overlays', False, 'make overlays of prediction and gt')

flags.DEFINE_string('model_variant_sub', 'fine',
                    'outputstride=8 (+ matching atrous rates), multi-scale eval / ' + \
                    'coarse: outputstride=16 (+ matching atrous rates), single-scale eval / ' + \
                    'selection details coded in main()')

flags.DEFINE_boolean('non_dataset', False, 'mode for running on non dataset images')

flags.DEFINE_integer('num_samples', 0, 'number of non dataset images, only needed if non_dataset=1')

# The folder where semantic segmentation predictions are saved.
_SEMANTIC_PREDICTION_SAVE_FOLDER = 'segmentation_results'

# The folder where raw semantic segmentation predictions are saved.
_RAW_SEMANTIC_PREDICTION_SAVE_FOLDER = 'raw_segmentation_results'

# The format to save image.
_IMAGE_FORMAT = '%06d_image'

# The format to save prediction
_PREDICTION_FORMAT = '%06d_prediction'

# To evaluate Cityscapes results on the evaluation server, the labels used
# during training should be mapped to the labels for evaluation.
_CITYSCAPES_TRAIN_ID_TO_EVAL_ID = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                                   23, 24, 25, 26, 27, 28, 31, 32, 33]

def _output_logits(images, model_options, image_pyramid=None): 
  """output segmentation logits. derived from model.py / predict_labels

  Args:
    images: A tensor of size [batch, height, width, channels].
    model_options: A ModelOptions instance to configure models.
    image_pyramid: Input image scales for multi-scale feature extraction.

  Returns:
    A dictionary with keys specifying the output_type (e.g., semantic
      prediction) and values storing Tensors representing logits for each channel.
      Each output has size [batch, height, width, class_channels].
  """
  outputs_to_scales_to_logits = model.multi_scale_logits(
      images,
      model_options=model_options,
      image_pyramid=image_pyramid,
      is_training=False,
      fine_tune_batch_norm=False)

  predictions = {}
  for output in sorted(outputs_to_scales_to_logits):
    scales_to_logits = outputs_to_scales_to_logits[output]
    logits = tf.image.resize_bilinear(
        scales_to_logits[model.MERGED_LOGITS_SCOPE],
        tf.shape(images)[1:3],
        align_corners=True)
    predictions[output] = logits

  return predictions

def _output_logits_multi_scale(images,
                               model_options,
                               eval_scales=(1.0,),
                               add_flipped_images=False):
  """output segmentation logits. derived from model.py / predict_labels_multi_scale

  Args:
    images: A tensor of size [batch, height, width, channels].
    model_options: A ModelOptions instance to configure models.
    eval_scales: The scales to resize images for evaluation.
    add_flipped_images: Add flipped images for evaluation or not.

  Returns:
    A dictionary with keys specifying the output_type (e.g., semantic
      prediction) and values storing Tensors representing logits for each channel.
      Each output has size [batch, height, width, class_channels].
  """
  outputs_to_predictions = {
      output: []
      for output in model_options.outputs_to_num_classes
  }

  for i, image_scale in enumerate(eval_scales):
    with tf.variable_scope(tf.get_variable_scope(), reuse=True if i else None):
      outputs_to_scales_to_logits = model.multi_scale_logits(
          images,
          model_options=model_options,
          image_pyramid=[image_scale],
          is_training=False,
          fine_tune_batch_norm=False)

    if add_flipped_images:
      with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        outputs_to_scales_to_logits_reversed = model.multi_scale_logits(
            tf.reverse_v2(images, [2]),
            model_options=model_options,
            image_pyramid=[image_scale],
            is_training=False,
            fine_tune_batch_norm=False)

    for output in sorted(outputs_to_scales_to_logits):
      scales_to_logits = outputs_to_scales_to_logits[output]
      logits = tf.image.resize_bilinear(
          scales_to_logits[model.MERGED_LOGITS_SCOPE],
          tf.shape(images)[1:3],
          align_corners=True)
      outputs_to_predictions[output].append(
          tf.expand_dims(tf.nn.softmax(logits), 4))

      if add_flipped_images:
        scales_to_logits_reversed = (
            outputs_to_scales_to_logits_reversed[output])
        logits_reversed = tf.image.resize_bilinear(
            tf.reverse_v2(scales_to_logits_reversed[model.MERGED_LOGITS_SCOPE], [2]),
            tf.shape(images)[1:3],
            align_corners=True)
        outputs_to_predictions[output].append(
            tf.expand_dims(tf.nn.softmax(logits_reversed), 4))

  for output in sorted(outputs_to_predictions):
    predictions = outputs_to_predictions[output]
    # Compute average prediction across different scales and flipped images.
    predictions = tf.reduce_mean(tf.concat(predictions, 4), axis=4)
    outputs_to_predictions[output] = predictions

  return outputs_to_predictions


def _convert_train_id_to_eval_id(prediction, train_id_to_eval_id):
  """Converts the predicted label for evaluation.

  There are cases where the training labels are not equal to the evaluation
  labels. This function is used to perform the conversion so that we could
  evaluate the results on the evaluation server.

  Args:
    prediction: Semantic segmentation prediction.
    train_id_to_eval_id: A list mapping from train id to evaluation id.

  Returns:
    Semantic segmentation prediction whose labels have been changed.
  """
  converted_prediction = prediction.copy()
  for train_id, eval_id in enumerate(train_id_to_eval_id):
    converted_prediction[prediction == train_id] = eval_id

  return converted_prediction



def _save_overlay(img_back, img_front, savepath):
  """ overlays img_front on img_back with 0.5 alpha and saves it to savepath
  Args:
    img_back, img_front, savepath: image paths
  """
  img_back = Image.open(img_back).convert('RGB')
  img_front = Image.open(img_front).convert('RGB')
  Image.blend(img_back, img_front, .5).save(savepath, 'PNG')


def softmax(X, axis = -1):
    y = np.atleast_2d(X) # make at least 2d
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    y = np.exp(y)
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    p = y / ax_sum
    if len(X.shape) == 1: p = p.flatten() # flatten if X was 1D
    return p


def _process_batch(sess, original_images, semantic_predictions, logits, labels, image_names,
                   image_heights, image_widths, save_dir,
                   raw_save_dir, train_id_to_eval_id=None, batch_i=0, num_batches=0):
  """Evaluates one single batch qualitatively.

  Args:
    sess: TensorFlow session.
    original_images: One batch of original images.
    semantic_predictions: One batch of semantic segmentation predictions.
    image_names: Image names.
    image_heights: Image heights.
    image_widths: Image widths.
    image_id_offset: Image id offset for indexing images.
    save_dir: The directory where the predictions will be saved.
    raw_save_dir: The directory where the raw predictions will be saved.
    train_id_to_eval_id: A list mapping from train id to eval id.
    batch_i: batch index
    num_batches: total number of batches to process

  Returns:
    probabilities, image names, ground truths, each as list
  """
  
  # save probs per batch and not at end, avoid recalculation in case of memory errors
  save_path = os.path.join(save_dir,'probs_{}.hdf5'.format(batch_i))

  if os.path.isfile(save_path):
    # skip this batch if probs already saved = batch already processed, only advance sess pointer
    sys.stdout.write('skipping batch {} / {}\r'.format(batch_i + 1, num_batches))
    (original_images) = sess.run([original_images])
    return

  # mark processing for parallel processes
  f = h5py.File(save_path, "w")
  f.close()

  start = time.time()
  sys.stdout.write('processing batch {} / {}\n'.format(batch_i + 1, num_batches))
  
  
  
  (original_images,
   semantic_predictions,
   logits,
   labels,
   image_names,
   image_heights,
   image_widths) = sess.run([original_images, semantic_predictions, logits, labels, 
                             image_names, image_heights, image_widths])

  num_image = semantic_predictions.shape[0]

  probabilities = []
  ground_truths = []
  file_names = []
  for i in range(num_image):
    image_height = np.squeeze(image_heights[i])
    image_width = np.squeeze(image_widths[i])
    original_image = np.squeeze(original_images[i])
    semantic_prediction = np.squeeze(semantic_predictions[i])
    crop_semantic_prediction = semantic_prediction[:image_height, :image_width]

    image_filename = os.path.basename(image_names[i]).decode('UTF-8').replace('.png','')
    sys.stdout.write('processing image {}\n'.format(image_filename))

    if np.min(logits[i]) < 0 or np.max(logits[i]) > 1:
      logits[i] = softmax(logits[i]) # only apply softmax (e.g. for mobilenet) if not already applied (e.g. for xception)
    probabilities.append(logits[i])
    ground_truths.append(labels[i])
    file_names.append(image_filename.encode('utf8'))

    if FLAGS.create_images:
      sys.stdout.write('creating images\r')
      # Save image 
      save_annotation.save_annotation(
          original_image, save_dir, image_filename, add_colormap=False)

      if FLAGS.copy_ground_truth:
        # Copy ground truth TODO: generalise to pascal
        src = os.path.join(FLAGS.gt_dir,FLAGS.vis_split,
                           image_filename.split('_')[0],image_filename+'_gtFine_color.png')
        dst = os.path.join(save_dir,image_filename+'_gt.png')
        if os.path.isfile(src):
          copyfile(src, dst)
          if FLAGS.make_overlays:
            _save_overlay(os.path.join(save_dir,image_filename+'.png'), dst,
                          os.path.join(save_dir,image_filename+'_gt_overlay.png'))
        else:
          sys.stdout.write('gt file {} not found\n'.format(src))
          

      save_annotation.save_annotation(
          crop_semantic_prediction, save_dir,
          image_filename+'_prediction', add_colormap=True,
          colormap_type=FLAGS.colormap_type)

      if FLAGS.make_overlays:
            _save_overlay(os.path.join(save_dir,image_filename+'.png'),
                          os.path.join(save_dir,image_filename+'_prediction.png'),
                          os.path.join(save_dir,image_filename+'_prediction_overlay.png'))

      if FLAGS.also_save_raw_predictions:
        if train_id_to_eval_id is not None:
          crop_semantic_prediction = _convert_train_id_to_eval_id(
              crop_semantic_prediction,
              train_id_to_eval_id)
        save_annotation.save_annotation(
            crop_semantic_prediction, raw_save_dir, image_filename,
            add_colormap=False)

      if FLAGS.non_dataset:
        # create entropy + probdist image
        #save_entropy_img(logits[i],
                         #os.path.join(save_dir,image_filename+'_entropy.png'),
                         #colour_max=(255,0,0))
        #save_prob_distance_img(logits[i],
                         #os.path.join(save_dir,image_filename+'_probdist.png'),
                         #colour_max=(255,0,0))
##        _save_overlay(os.path.join(save_dir,image_filename+'.png'),
##              os.path.join(save_dir,image_filename+'_entropy.png'),
##              os.path.join(save_dir,image_filename+'_entropy_overlay.png'))
        non_dataset.save_composite_image(
          [os.path.join(save_dir,image_filename+'.png'), # top left
            os.path.join(save_dir,image_filename+'_prediction_overlay.png'), # top right
            os.path.join(save_dir,image_filename+'_entropy.png'), # bottom left
            os.path.join(save_dir,image_filename+'_probdist.png')], # bottom right
          os.path.join(save_dir,image_filename+'_composite.png') # target file
          )
        
  if FLAGS.save_probs: # only save probs if needed
    f = h5py.File(save_path, "w")
    f.create_dataset("probabilities", data=probabilities)
    f.create_dataset("ground_truths", data=ground_truths)
    f.create_dataset("file_names", data=file_names)
    sys.stdout.write('saved batch probabilities to {}\n'.format(save_path))
    f.close()
  sys.stdout.write('batch processed in {}s\n'.format(round(time.time()-start)))
  sys.stdout.flush()


def main(unused_argv):
  
  tf.logging.set_verbosity(tf.logging.INFO)
  # Get dataset-dependent information.
  if FLAGS.non_dataset:
    dataset = non_dataset.get_non_dataset(
      FLAGS.dataset, dataset_dir=FLAGS.dataset_dir, num_samples=FLAGS.num_samples)
  else:
    dataset = segmentation_dataset.get_dataset(
        FLAGS.dataset, FLAGS.vis_split, dataset_dir=FLAGS.dataset_dir)
  train_id_to_eval_id = None
  if dataset.name == segmentation_dataset.get_cityscapes_dataset_name():
    tf.logging.info('Cityscapes requires converting train_id to eval_id.')
    train_id_to_eval_id = _CITYSCAPES_TRAIN_ID_TO_EVAL_ID

  # set flags based on model and model_sub
  if FLAGS.model_variant == "xception_65": FLAGS.decoder_output_stride = 4
  if FLAGS.model_variant_sub == "fine":
      FLAGS.output_stride = 8
      FLAGS.eval_scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
      if FLAGS.model_variant == "xception_65": FLAGS.atrous_rates = [12, 24, 36]
  else:
      FLAGS.output_stride = 16
      FLAGS.eval_scales = [1.0]
      if FLAGS.model_variant == "xception_65": FLAGS.atrous_rates = [6, 12, 18]

   # construct model id string
  model_id_string = "mn" if FLAGS.model_variant == "mobilenet_v2" else "xc"
  model_id_string += ".sscl" if tuple(FLAGS.eval_scales) == (1.0,) else ".mscl"
  model_id_string += ".os" + str(FLAGS.output_stride)    

  FLAGS.vis_logdir = os.path.join(FLAGS.vis_logdir,model_id_string)

  # Prepare for visualization.
  tf.gfile.MakeDirs(FLAGS.vis_logdir)

  raw_save_dir = os.path.join(FLAGS.vis_logdir, _RAW_SEMANTIC_PREDICTION_SAVE_FOLDER)
  if FLAGS.also_save_raw_predictions:
    save_dir = os.path.join(FLAGS.vis_logdir, _SEMANTIC_PREDICTION_SAVE_FOLDER)
    tf.gfile.MakeDirs(raw_save_dir)
  else:
    save_dir = FLAGS.vis_logdir  
  tf.gfile.MakeDirs(save_dir)

  tf.logging.info('Selected set: {}'.format(FLAGS.vis_split))
  tf.logging.info('Selected model: {}'.format(model_id_string))

  g = tf.Graph()
  with g.as_default():
    samples = input_generator.get(dataset,
                                  FLAGS.vis_crop_size,
                                  FLAGS.vis_batch_size,
                                  min_resize_value=FLAGS.min_resize_value,
                                  max_resize_value=FLAGS.max_resize_value,
                                  resize_factor=FLAGS.resize_factor,
                                  dataset_split=FLAGS.vis_split,
                                  is_training=False,
                                  model_variant=FLAGS.model_variant)

    model_options = common.ModelOptions(
        outputs_to_num_classes={common.OUTPUT_TYPE: dataset.num_classes},
        crop_size=FLAGS.vis_crop_size,
        atrous_rates=FLAGS.atrous_rates,
        output_stride=FLAGS.output_stride)

    if tuple(FLAGS.eval_scales) == (1.0,):
      tf.logging.info('Performing single-scale test.')
      logits = _output_logits(
          samples[common.IMAGE],
          model_options=model_options,
          image_pyramid=FLAGS.image_pyramid)
    else:
      tf.logging.info('Performing multi-scale test.')
      logits = _output_logits_multi_scale(
          samples[common.IMAGE],
          model_options=model_options,
          eval_scales=FLAGS.eval_scales,
          add_flipped_images=FLAGS.add_flipped_images)

    logits = logits[common.OUTPUT_TYPE]
    predictions = tf.argmax(logits, 3)

    if FLAGS.min_resize_value and FLAGS.max_resize_value:
      tf.logging.info('using FLAGS.min/max_resize_value')
      # Only support batch_size = 1, since we assume the dimensions of original
      # image after tf.squeeze is [height, width, 3].
      assert FLAGS.vis_batch_size == 1

      # Reverse the resizing and padding operations performed in preprocessing.
      # First, we slice the valid regions (i.e., remove padded region) and then
      # we reisze the predictions back.
      original_image = tf.squeeze(samples[common.ORIGINAL_IMAGE])
      original_image_shape = tf.shape(original_image)
      predictions = tf.slice(
          predictions,
          [0, 0, 0],
          [1, original_image_shape[0], original_image_shape[1]])
      resized_shape = tf.to_int32([tf.squeeze(samples[common.HEIGHT]),
                                   tf.squeeze(samples[common.WIDTH])])
      predictions = tf.squeeze(
          tf.image.resize_images(tf.expand_dims(predictions, 3),
                                 resized_shape,
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                 align_corners=True), 3)

    tf.train.get_or_create_global_step()
    saver = tf.train.Saver(slim.get_variables_to_restore())
    sv = tf.train.Supervisor(graph=g,
                             logdir=FLAGS.vis_logdir,
                             init_op=tf.global_variables_initializer(),
                             summary_op=None,
                             summary_writer=None,
                             global_step=None,
                             saver=saver)
    num_batches = min(FLAGS.max_num_batches,int(math.ceil(dataset.num_samples / float(FLAGS.vis_batch_size))))
    last_checkpoint = None

    # Loop to visualize the results when new checkpoint is created.
    num_iters = 0
    while (FLAGS.max_number_of_iterations <= 0 or
           num_iters < FLAGS.max_number_of_iterations):
      num_iters += 1
      last_checkpoint = slim.evaluation.wait_for_new_checkpoint(
          FLAGS.checkpoint_dir, last_checkpoint)
      start = time.time()
      tf.logging.info(
          'Starting processing at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                       time.gmtime()))
      tf.logging.info('Processing with model %s', last_checkpoint)

      # try to decrease out of memory errors
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      config.gpu_options.per_process_gpu_memory_fraction = 0.90

      with sv.managed_session(FLAGS.master, config=config, 
                              start_standard_services=False) as sess:
        sv.start_queue_runners(sess)
        sv.saver.restore(sess, last_checkpoint)
        tf.logging.info('checkpoint restored')

        for batch_i in range(num_batches):    
          _process_batch(sess=sess,
                             original_images=samples[common.ORIGINAL_IMAGE],
                             semantic_predictions=predictions,
                             logits=logits,
                             labels=samples[common.LABEL],
                             image_names=samples[common.IMAGE_NAME],
                             image_heights=samples[common.HEIGHT],
                             image_widths=samples[common.WIDTH],
                             save_dir=save_dir,
                             raw_save_dir=raw_save_dir,
                             train_id_to_eval_id=train_id_to_eval_id,
                             batch_i=batch_i, num_batches=num_batches)          

      tf.logging.info(
          'Finished processing at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                       time.gmtime()))
      
      # delete probs marker for missing images, combination results from crash, so that batch is reprocessed in next run
      # only works if one image per batch!!!
      #video_name = FLAGS.dataset_dir.split(os.sep)[-2]
      #missing = 0
      #i = 0
      #while missing < 10:
          #im_test_path1 = os.path.join(FLAGS.vis_logdir,video_name + str(i+1).zfill(5) + '_composite.png')
          #im_test_path2 = os.path.join(FLAGS.vis_logdir,video_name + '_' + str(i+1).zfill(5) + '_composite.png')
          #probs_path = os.path.join(FLAGS.vis_logdir,'probs_{}.hdf5'.format(i))
          #if os.path.isfile(probs_path):
              #if (not os.path.isfile(im_test_path1)) and (not os.path.isfile(im_test_path2)):
                  #os.unlink(probs_path)
                  #print("removed " + probs_path)
          #else:
              #missing += 1
          #i += 1

      
      time_to_next_eval = start + FLAGS.eval_interval_secs - time.time()
      if time_to_next_eval > 0:
        time.sleep(time_to_next_eval)


if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint_dir')
  flags.mark_flag_as_required('vis_logdir')
  flags.mark_flag_as_required('dataset_dir')
  flags.mark_flag_as_required('dataset')
  flags.mark_flag_as_required('colormap_type')
  flags.mark_flag_as_required('copy_ground_truth')
  flags.mark_flag_as_required('gt_dir')
  tf.app.run()
