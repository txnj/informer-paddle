import paddle
import paddle.nn as nn


# 这是一个PaddlePaddle的DecoderLayer类，继承自nn.Layer。
# 在初始化方法中，它定义了self_attention、cross_attention、d_model、d_ff（默认为4*d_model）、dropout（默认为0.1）和activation（默认为"relu"）等参数。
# self_attention和cross_attention分别是自注意力机制和交叉注意力机制的实例。
# d_ff是FFN（前馈神经网络）的输入和输出通道数，默认为4*d_model。conv1和conv2是1D卷积层实例，用于将输入张量转换为d_model通道数。
# norm1、norm2和norm3是规范化层实例，用于对输入张量进行归一化。dropout是dropout层实例，用于防止过拟合。activation是激活函数实例，默认为ReLU激活函数，但也可以设置为GELU激活函数。
# 在forward方法中，它接收一个输入张量x、一个可选的注意力掩码attn_mask和多个可选参数，并返回输出张量x、self_attn_output、cross_attn_output和Norm后处理结果。
# 注意，这里有两个self_attn_output和cross_attn_output输出，分别表示自注意力机制和交叉注意力机制的输出。
# 这些输出还需要通过Norm层进行归一化处理，并使用dropout层进行dropout处理，最后通过激活函数进行激活。
class DecoderLayer(nn.Layer):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1D(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1D(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = paddle.nn.functional.relu if activation == "relu" else paddle.nn.functional.gelu

    # 这是一个PaddlePaddle的DecoderLayer类的forward方法，用于执行Decoder的每一层计算。
    # 该方法接受两个输入张量x和cross，以及可选参数x_mask和cross_mask，表示输入张量的注意力掩码。
    # 首先，它使用self.self_attention计算自注意力机制的输出，并使用dropout、规范化层self.norm1和dropout对输出进行处理。
    # 然后，它使用self.cross_attention计算交叉注意力机制的输出，并再次使用dropout、规范化层self.norm2和dropout对输出进行处理。
    # 接下来，它使用1D卷积层self.conv1和self.conv2对输出进行处理，并使用dropout进行dropout处理。
    # 最后，它将所有输出张量相加，并使用规范化层self.norm3对结果进行归一化处理。
    # 需要注意的是，该方法返回的是经过所有处理后的最终输出张量。
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


# 这是一个PaddlePaddle的Decoder类，继承自nn.Layer。
# 在初始化方法中，它接受一个layers参数，这是一个包含多个DecoderLayer实例的列表，以及一个可选的norm_layer参数，表示使用的规范化层类型。
# 在forward方法中，它遍历self.layers中的每个DecoderLayer实例，并传递输入张量x、交叉注意力机制的输出cross、注意力掩码x_mask和cross_mask。
# 最后，它使用规范化层self.norm对输出进行处理，并返回处理后的输出张量x。
class Decoder(nn.Layer):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.LayerList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x
