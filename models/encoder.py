import paddle
import paddle.nn as nn


# 这个ConvLayer类的初始化方法中，
# 创建了一个nn.Conv1d对象downConv，一个nn.BatchNorm1d对象norm，一个nn.ELU对象activation和一个nn.MaxPool1d对象maxPool。
# 在forward方法中，先将输入x进行转置，然后经过卷积层downConv、批归一化层norm、激活函数层activation和最大池化层maxPool的处理，最后将结果转置回去并返回。
class ConvLayer(nn.Layer):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1
        self.downConv = nn.Conv1D(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1D(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1D(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # ✏️x = self.downConv(x.permute(0, 2, 1))
        x = self.downConv(paddle.transpose(x, perm=[0, 2, 1]))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = paddle.transpose(x, perm=[0, 2, 1])
        return x


class EncoderLayer(nn.Layer):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1D(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1D(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = paddle.nn.functional.relu if activation == "relu" else paddle.nn.functional.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(paddle.transpose(y, perm=[0, 2, 1]))))
        y = self.dropout(paddle.transpose(self.conv2(y), perm=[0, 2, 1]))
        return self.norm2(x + y), attn


# 这是一个PaddlePaddle的Encoder类，继承自nn.Layer。在初始化方法中，它定义了一个nn.ModuleList类型的attn_layers成员变量，一个可选的nn.ModuleList类型的conv_layers成员变量和一个可选的norm_layer成员变量。
# 在forward方法中，它接收一个输入张量x和一个可选参数attn_mask（一个注意力掩码张量）。
# 首先，它遍历self.attn_layers中的每个自注意力机制实例和可选的卷积层实例，并使用zip函数将它们一一配对。对于每个配对，它使用attn_layer处理x，并将结果与attn_mask一起传递给attn_layer。
# 然后，它使用conv_layer处理x，并将结果与attn一起添加到attns列表中。
# 接着，它将x和最后一个自注意力机制的注意力张量添加到attns列表中。
# 如果conv_layers为None，则跳过卷积层处理。否则，在遍历完所有自注意力机制后，它将x和最后一个自注意力机制的注意力张量添加到attns列表中。
# 最后，如果self.norm不为None，则使用self.norm处理x。最终输出是处理后的x和每个自注意力机制的注意力张量列表attns。
class Encoder(nn.Layer):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.LayerList(attn_layers)
        self.conv_layers = nn.LayerList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)  # polling后再减半,提升速度
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class EncoderStack(nn.Layer):
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.LayerList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        x_stack = []
        attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1] // (2 ** i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s)
            attns.append(attn)
        x_stack = paddle.concat(x_stack, axis=-2)

        return x_stack, attns
