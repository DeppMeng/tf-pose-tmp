import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.contrib.layers.python.layers import batch_norm
# from config import cfg
# from hrnet.model import HRNet
from .engine import ModelDesc


class Model(ModelDesc):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg

    def head_net(self, blocks, trainable=True):

        msra_initializer = tf.contrib.layers.variance_scaling_initializer()

        with slim.arg_scope([slim.conv2d],  # NOTE(NHWC)
                            weights_regularizer=slim.l2_regularizer(1e-4)):
            out = slim.conv2d(blocks[0], self.cfg.num_kps, [1, 1],
                              trainable=trainable, weights_initializer=msra_initializer,
                              padding='SAME', normalizer_fn=None, activation_fn=None,
                              scope='out')
        return out

    def concat_124_head_net(self, blocks, trainable=True):

        msra_initializer = tf.contrib.layers.variance_scaling_initializer()

        with slim.arg_scope([slim.conv2d],  # NOTE(NHWC)
                            weights_regularizer=slim.l2_regularizer(1e-4)):
            x0 = blocks[0]
            shape = x0.get_shape().as_list()
            x1 = tf.image.resize_images(blocks[1], [shape[1], shape[2]])
            x2 = tf.image.resize_images(blocks[2], [shape[1], shape[2]])
            x = tf.concat([x0, x1, x2], 3)
            x = slim.conv2d(x, 7 * shape[3], [1, 1],
                              trainable=trainable, weights_initializer=msra_initializer,
                              padding='SAME', normalizer_fn=None, activation_fn=None,
                              biases_initializer=tf.constant_initializer(0.), biases_regularizer=None, scope='extra_1x1')
            x = batch_norm(x, activation_fn=tf.nn.relu, scope='extra_bn_relu')
            out = slim.conv2d(x, self.cfg.num_kps, [1, 1],
                              trainable=trainable, weights_initializer=msra_initializer,
                              padding='SAME', normalizer_fn=None, activation_fn=None,
                              scope='out')
        return out

    def render_gaussian_heatmap(self, coord, output_shape, sigma):

        x = [i for i in range(output_shape[1])]
        y = [i for i in range(output_shape[0])]
        xx, yy = tf.meshgrid(x, y)
        xx = tf.reshape(tf.to_float(xx), (1, *output_shape, 1))
        yy = tf.reshape(tf.to_float(yy), (1, *output_shape, 1))

        x = tf.floor(tf.reshape(coord[:, :, 0], [-1, 1, 1, self.cfg.num_kps]) / self.cfg.input_shape[1] * output_shape[1] + 0.5)
        y = tf.floor(tf.reshape(coord[:, :, 1], [-1, 1, 1, self.cfg.num_kps]) / self.cfg.input_shape[0] * output_shape[0] + 0.5)

        heatmap = tf.exp(-(((xx - x) / tf.to_float(sigma)) ** 2) / tf.to_float(2) - (
                ((yy - y) / tf.to_float(sigma)) ** 2) / tf.to_float(2))

        return heatmap * 255.

    def make_network(self, is_train):

        if is_train:
            image = tf.placeholder(tf.float32, shape=[self.cfg.batch_size, *self.cfg.input_shape, 3])
            target_coord = tf.placeholder(tf.float32, shape=[self.cfg.batch_size, self.cfg.num_kps, 2])
            valid = tf.placeholder(tf.float32, shape=[self.cfg.batch_size, self.cfg.num_kps])
            self.set_inputs(image, target_coord, valid)
        else:
            image = tf.placeholder(tf.float32, shape=[None, *self.cfg.input_shape, 3])
            self.set_inputs(image)
        
        if self.cfg.model == 'hrnet':
            from hrnet.model import HRNet
        elif self.cfg.model == 'hr_rnet':
            from hr_rnet.model import HRNet
        elif 'full_rnet' in self.cfg.model:
            from full_rnet_concat_124.model import HRNet

        with tf.variable_scope('HRNET'):
            hrnet_fms = HRNet(self.cfg.hrnet_config, image, is_train)
            if 'full_rnet' in self.cfg.model:
                heatmap_outs = self.concat_124_head_net(hrnet_fms)
            else:
                heatmap_outs = self.head_net(hrnet_fms)

        if is_train:
            gt_heatmap = tf.stop_gradient(self.render_gaussian_heatmap(target_coord, self.cfg.output_shape, self.cfg.sigma))
            valid_mask = tf.reshape(valid, [self.cfg.batch_size, 1, 1, self.cfg.num_kps])
            loss = tf.reduce_mean(tf.square(heatmap_outs - gt_heatmap) * valid_mask)
            self.add_tower_summary('loss', loss)
            self.set_loss(loss)
        else:
            self.set_outputs(heatmap_outs)
