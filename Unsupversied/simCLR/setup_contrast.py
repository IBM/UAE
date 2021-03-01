import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
import tensorflow_hub as hub
import os
import functools
import data_util as data_util

def pad_to_batch(dataset, batch_size):
  """Pad Tensors to specified batch size.

  Args:
    dataset: An instance of tf.data.Dataset.
    batch_size: The number of samples per batch of input requested.

  Returns:
    An instance of tf.data.Dataset that yields the same Tensors with the same
    structure as the original padded to batch_size along the leading
    dimension.

  Raises:
    ValueError: If the dataset does not comprise any tensors; if a tensor
      yielded by the dataset has an unknown number of dimensions or is a
      scalar; or if it can be statically determined that tensors comprising
      a single dataset element will have different leading dimensions.
  """
  def _pad_to_batch(*args):
    """Given Tensors yielded by a Dataset, pads all to the batch size."""
    flat_args = tf.nest.flatten(args)

    for tensor in flat_args:
      if tensor.shape.ndims is None:
        raise ValueError(
            'Unknown number of dimensions for tensor %s.' % tensor.name)
      if tensor.shape.ndims == 0:
        raise ValueError('Tensor %s is a scalar.' % tensor.name)

    # This will throw if flat_args is empty. However, as of this writing,
    # tf.data.Dataset.map will throw first with an internal error, so we do
    # not check this case explicitly.
    first_tensor = flat_args[0]
    first_tensor_shape = tf.shape(first_tensor)
    first_tensor_batch_size = first_tensor_shape[0]
    difference = batch_size - first_tensor_batch_size

    for i, tensor in enumerate(flat_args):
      control_deps = []
      if i != 0:
        # Check that leading dimensions of this tensor matches the first,
        # either statically or dynamically. (If the first dimensions of both
        # tensors are statically known, the we have to check the static
        # shapes at graph construction time or else we will never get to the
        # dynamic assertion.)
        if (first_tensor.shape[:1].is_fully_defined() and
            tensor.shape[:1].is_fully_defined()):
          if first_tensor.shape[0] != tensor.shape[0]:
            raise ValueError(
                'Batch size of dataset tensors does not match. %s '
                'has shape %s, but %s has shape %s' % (
                    first_tensor.name, first_tensor.shape,
                    tensor.name, tensor.shape))
        else:
          curr_shape = tf.shape(tensor)
          control_deps = [tf.Assert(
              tf.equal(curr_shape[0], first_tensor_batch_size),
              ['Batch size of dataset tensors %s and %s do not match. '
               'Shapes are' % (tensor.name, first_tensor.name), curr_shape,
               first_tensor_shape])]

      with tf.control_dependencies(control_deps):
        # Pad to batch_size along leading dimension.
        flat_args[i] = tf.pad(
            tensor, [[0, difference]] + [[0, 0]] * (tensor.shape.ndims - 1))
      flat_args[i].set_shape([batch_size] + tensor.shape.as_list()[1:])

    return tf.nest.pack_sequence_as(args, flat_args)

  return dataset.map(_pad_to_batch)

def get_preprocess_fn(is_training, is_pretrain):
  """Get function that accepts an image and returns a preprocessed image."""
  # Disable test cropping for small images (e.g. CIFAR)
  return functools.partial(
      data_util.preprocess_image,
      height=32,
      width=32,
      is_training=is_training,
      color_distort=is_pretrain,
      test_crop=False)

def add_contrastive_loss(hidden,
                         hidden_norm=True,
                         temperature=1.0,
                         tpu_context=None,
                         weights=1.0):
  """Compute loss for model.

  Args:
    hidden: hidden vector (`Tensor`) of shape (bsz, dim).
    hidden_norm: whether or not to use normalization on the hidden vector.
    temperature: a `floating` number for temperature scaling.
    tpu_context: context information for tpu.
    weights: a weighting number or vector.

  Returns:
    A loss scalar.
    The logits for contrastive prediction task.
    The labels for contrastive prediction task.
  """
  LARGE_NUM = 1e9
  # Get (normalized) hidden1 and hidden2.
  if hidden_norm:
    hidden = tf.math.l2_normalize(hidden, -1)
  hidden1, hidden2 = tf.split(hidden, 2, 0)
  batch_size = tf.shape(hidden1)[0]

  # Gather hidden1/hidden2 across replicas and create local labels.
  if tpu_context is not None:
    hidden1_large = tpu_cross_replica_concat(hidden1, tpu_context)
    hidden2_large = tpu_cross_replica_concat(hidden2, tpu_context)
    enlarged_batch_size = tf.shape(hidden1_large)[0]
    # TODO(iamtingchen): more elegant way to convert u32 to s32 for replica_id.
    replica_id = tf.cast(tf.cast(xla.replica_id(), tf.uint32), tf.int32)
    labels_idx = tf.range(batch_size) + replica_id * batch_size
    labels = tf.one_hot(labels_idx, enlarged_batch_size * 2)
    masks = tf.one_hot(labels_idx, enlarged_batch_size)

  else:
    hidden1_large = hidden1
    hidden2_large = hidden2
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(tf.range(batch_size), batch_size)

  logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
  logits_aa = logits_aa - masks * LARGE_NUM
  logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
  logits_bb = logits_bb - masks * LARGE_NUM
  logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
  logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature

  loss_a = tf.losses.softmax_cross_entropy(
      labels, tf.concat([logits_ab, logits_aa], 1), weights=weights)
  loss_b = tf.losses.softmax_cross_entropy(
      labels, tf.concat([logits_ba, logits_bb], 1), weights=weights)
  loss = loss_a + loss_b

  return loss, logits_ab, labels

class ContrastModel:
    def __init__(self, restore, session=None):
        self.num_channels = 3
        self.image_size = 32
        self.num_labels = 10
        self.sess = session
        self.hub_module = hub.load(restore)

    def predict(self, data, labels):
        is_training = False
        data = tf.tile(data,[50,1,1,1])
        preprocess_fn_pretrain = get_preprocess_fn(is_training, is_pretrain=True)
        def map_fn(image):
            """Produces multiple transformations of the same batch."""
            xs = []
            for _ in range(2):  # Two transformations
                xs.append(preprocess_fn_pretrain(image))
            image = tf.concat(xs, -1)
            return image
        map_f = tf.map_fn(lambda a: map_fn(a), data)
        images = tf.pad(map_f, [[0, 0]] + [[0, 0]] * (map_f.shape.ndims - 1))
        features_list = tf.split(images, num_or_size_splits=2, axis=-1)
        features = tf.concat(features_list, 0)
        a = self.hub_module.signatures['default'](features)
        c_loss,_,_=add_contrastive_loss(hidden=a['proj_head_output'],temperature=0.5)
        return c_loss
    def conv1(self, data):
    	conv = self.hub_module.signatures['default'](tf.convert_to_tensor(data))
    	return conv['initial_conv']
   