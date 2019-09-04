from .layers import *

class ExtraTransition():
    def __init__(self,
                 stage_id,
                 num_channels,
                 num_branches,
                 num_out_channels):
        self.scope = str(stage_id) + 'ExtraTransition'
        self.num_channels = num_channels
        self.num_branches = num_branches
        self.num_out_channels = num_out_channels

    def forward(self, input):
        with tf.variable_scope(self.scope):
            _out = []
            for i in range(self.num_branches):
                if tf.shape(input[i])[3] != self.num_out_channels:
                    print(1)
                    _tmp_out = slim.conv2d(input[i], num_outputs=self.num_out_channels * pow(2, i),
                                                kernel_size=[3,3], stride=1, activation_fn=tf.nn.relu,
                                                normalizer_fn=batch_norm)
                else:
                    _tmp_out = input[i]
                _out.append(_tmp_out)
        return _out