import paddle
import paddle.nn as nn
import math


# 创建一个长度为 max_len 的零张量 pe，其维度为 d_model。
# 将 pe.require_grad 设置为 False，以确保反向传播时不会计算 pe 的梯度。
# 创建一个与 pe 相同维度的张量 position，其中包含从 0 到 max_len-1 的数字。
# 创建一个与 pe 相同维度的张量 div_term，其中包含 -(math.log(10000.0) / d_model) 的幂次。
# 将 pe 中每两个相邻的元素设置为 position * div_term 的正弦和余弦值。
# 将 pe 张量展开为一个大小为 (1, max_len, d_model) 的张量，并将其传递到前向传递中。在前向传递中，该类将输入张量 x 的长度与位置编码矩阵的长度进行比较，并返回相应长度的位置编码张量。
# 这个 PositionalEmbedding 类可以用于为输入序列中的每个位置提供位置编码，这些位置编码可用于各种任务。
class PositionalEmbedding(nn.Layer):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = paddle.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = paddle.arange(0, max_len).float().unsqueeze(1)
        div_term = (paddle.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


# 这是一个PaddlePaddle中的TokenEmbedding类，继承自nn.Layer。它包括一个卷积层，用于从输入张量x中提取嵌入特征。
# 在初始化方法中，它定义了一个名为tokenConv的卷积层，输入通道数为c_in，输出通道数为d_model，卷积核大小为3，填充大小为1（如果使用的是PaddlePaddle 1.5.0及以上的版本）或2，填充方式为循环模式。
# 然后，它使用nn.init.kaiming_normal_函数初始化所有模块的权重，包括卷积层，使用fan_in模式和leaky_relu非线性激活函数。
# 在前向传递方法中，它首先将输入张量x的维度转置为(0, 2, 1)，然后通过卷积层tokenConv进行处理。处理后，它将输出张量的维度转置回(0, 1, 2)，并返回结果。
# 这个TokenEmbedding类可以用于为输入序列中的每个token提供嵌入特征，这些嵌入特征可以用于各种NLP任务，例如文本分类、命名实体识别等。
class TokenEmbedding(nn.Layer):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if paddle.__version__ >= '1.5.0' else 2
        # ✏️self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
        #                            kernel_size=3, padding=padding, padding_mode='circular')
        self.tokenConv = nn.Conv1D(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias_attr=True,
                                   weight_attr=nn.initializer.KaimingNormal())

        # ✏️for m in self.modules():
        #     if isinstance(m, nn.Conv1D):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


# 这是一个名为FixedEmbedding的PaddlePaddle层类，它继承自nn.Layer。
# 在初始化方法中，它定义了一个名为w的张量，其大小为c_in x d_model，并将其梯度设置为False。
# 接下来，它创建了一个位置张量position，其大小为c_in x 1，并将其梯度设置为False。然后，它计算了div_term，即(d_model / 2) * math.log(10000)的幂次方。
# 最后，它将w张量设置为正弦和余弦函数的组合，这些函数是位置张量乘以div_term的结果。
# 在forward方法中，它创建了一个名为emb的nn.Embedding对象，并将权重设置为初始化方法中计算出的w张量。
# 这个FixedEmbedding类可以在固定的位置编码上使用标准的nn.Embedding。因此，它的实现不需要在运行时进行昂贵的计算。
# 通常，它适用于需要高效计算的任务，例如大规模文本分类。
class FixedEmbedding(nn.Layer):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = paddle.zeros(c_in, d_model).float()
        w.require_grad = False

        position = paddle.arange(0, c_in).float().unsqueeze(1)
        div_term = (paddle.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = paddle.sin(position * div_term)
        w[:, 1::2] = paddle.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        # ✏️self.emb.weight = nn.Parameter(w, requires_grad=False)
        self.emb.weight = paddle.create_parameter(shape=w.shape, dtype=str(w.numpy().dtype))

    def forward(self, x):
        return self.emb(x).detach()


# 这是一个名为TemporalEmbedding的PaddlePaddle层类，它继承自nn.Layer。
# 在初始化方法中，它定义了一些大小，具体取决于要嵌入的时间特征的类型和频率。
# 然后，根据embed_type参数，它创建了一个FixedEmbedding对象（如果embed_type=='fixed'）或nn.Embedding对象。
# 最后，它根据freq参数创建了一个时间特征的嵌入对象。
# 在forward方法中，如果freq='t'，则根据分钟大小创建一个嵌入对象minute_embed，并根据小时大小创建一个嵌入对象hour_embed。
# 然后，根据星期大小创建一个嵌入对象weekday_embed，根据天大小创建一个嵌入对象day_embed，根据月大小创建一个嵌入对象month_embed。
# 最后，它将所有嵌入对象的输出连接在一起，并返回结果。
# 这个TemporalEmbedding类可以用于为时间序列数据中的不同时间特征（例如分钟、小时、星期、天和月）添加位置编码。
class TemporalEmbedding(nn.Layer):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = embed(minute_size, d_model)
        self.hour_embed = embed(hour_size, d_model)
        self.weekday_embed = embed(weekday_size, d_model)
        self.day_embed = embed(day_size, d_model)
        self.month_embed = embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


# 这是一个名为TimeFeatureEmbedding的PaddlePaddle层类，它继承自nn.Layer。
# 在初始化方法中，它定义了一个名为embed的nn.Linear对象，输入维度为freq_map[freq]，输出维度为d_model。
# 其中，freq_map是一个字典，用于将频率（'h'、't'、's'、'm'、'a'、'w'、'd'和'b'分别代表小时、分钟、秒、分钟内的刻钟、天、周、月和周内的小时）映射到相应的输入维度。
# 在前向传递方法中，它通过调用nn.Linear对象的forward方法来计算嵌入特征。
# 这个TimeFeatureEmbedding类可以用于将时间序列数据中的不同时间特征（例如小时、分钟、秒、分钟内的刻钟、天、周、月和周内的小时）映射到一个低维向量空间中。
class TimeFeatureEmbedding(nn.Layer):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):
        return self.embed(x)


# 这是一个名为TimeFeatureEmbedding的PaddlePaddle层类，它继承自nn.Layer。
# 在初始化方法中，它定义了三个嵌入对象：value_embedding（用于嵌入输入序列中的值）、position_embedding（用于嵌入输入序列中的位置）和temporal_embedding（用于嵌入时间序列数据中的时间特征）。
# 其中，如果embed_type不为'timeF'，则使用TemporalEmbedding对象；否则，使用TimeFeatureEmbedding对象。
# 在前向传递方法中，它首先将输入序列x传递给value_embedding对象，并将位置编码添加到结果中。
# 然后，如果embed_type不为'timeF'，则将时间序列数据中的时间特征x_mark传递给temporal_embedding对象，并将结果添加到嵌入特征中。
# 最后，它通过调用nn.Dropout对象的forward方法来应用dropout正则化，并返回结果。
class DataEmbedding(nn.Layer):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # Conv1d是卷积神经网络中的一维卷积层的缩写,Conv: 代表"卷积"（Convolution）,1d: 表示一维操作
        # Conv1d主要用于处理一维序列数据，如时间序列或音频信号。它在输入数据上滑动一个固定大小的窗口（卷积核），执行卷积操作，从而提取局部特征
        # value_embedding(x):12列映射到512个特征,position_embedding:96行->512特征,temporal_embedding:时间特征->512特征 然后做个加法
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)

        return self.dropout(x)
