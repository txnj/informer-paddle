import paddle
import paddle.nn as nn
import math


class PositionalEmbedding(nn.Layer):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = paddle.zeros([max_len, d_model], dtype='float32')
        # 用于控制反向传播过程中梯度的计算。当设置为 True 时，它会阻止梯度继续向前传播
        pe.stop_gradient = True

        # unsqueeze(1):在索引为 1 的位置(即第二个维度)添加一个新的维度,将一维张量转换为二维张量
        position = paddle.arange(0, max_len, dtype='float32').unsqueeze(1)
        div_term = (paddle.arange(0, d_model, 2, dtype='float32') * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term)

        pe = pe.unsqueeze(0)
        # 最终得到pe:Tensor(1,max_len,d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.shape[1]]


class TokenEmbedding(nn.Layer):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1
        # in_channels 就是输入特征个数 这里就是列数 测试案例为值为12
        # out_channels 输出特征个数 默认值:512
        # kernel_size=3：卷积核大小为3
        # padding=padding：填充大小，由变量padding指定
        # padding_mode='circular'：使用循环填充模式，即在边界处进行循环填充
        # bias_attr=True：启用偏置项
        # weight_attr=nn.initializer.KaimingNormal()：使用He正态分布初始化权重。这种初始化方法适用于使用ReLU激活函数的网络。
        self.tokenConv = nn.Conv1D(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias_attr=True,
                                   weight_attr=nn.initializer.KaimingNormal())  # ❓torch中没有weight_attr

        for m in self.sublayers():
            if isinstance(m, nn.Conv1D):
                paddle.nn.initializer.KaimingNormal(nonlinearity='leaky_relu')(m.weight)

    def forward(self, x):
        # ✏️x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        x = paddle.transpose(self.tokenConv(paddle.transpose(x, perm=[0, 2, 1])), perm=[0, 2, 1])
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

        w = paddle.zeros([c_in, d_model])
        w.stop_gradient = True

        position = paddle.arange(0, c_in, dtype=paddle.float32).unsqueeze(1)
        div_term = (paddle.arange(0, d_model, 2, dtype=paddle.float32) * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = paddle.sin(position * div_term)
        w[:, 1::2] = paddle.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        # ✏️self.emb.weight = nn.Parameter(w, requires_grad=False)
        self.emb.weight = paddle.create_parameter(shape=w.shape, dtype=str(w.numpy().dtype))

    def forward(self, x):
        return self.emb(x).detach()


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


class TimeFeatureEmbedding(nn.Layer):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Layer):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()
        # 自定义的嵌入层类，用于将输入标记转换为密集向量表示
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        # 用于创建位置嵌入。位置嵌入是在Transformer模型中常用的一种技术，用于为序列中的每个位置提供位置信息
        # Tensor(1,max_len,d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        # 时间嵌入层的作用是将时间信息转换为向量表示,使模型能够捕捉时间序列数据中的时间模式和周期性特征。这在时间序列预测任务中非常重要,因为它能帮助模型理解数据的时间结构
        # freq:时间频率(如'h'表示小时,'d'表示天等)
        self.temporal_embedding = TemporalEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # Conv1d是卷积神经网络中的一维卷积层的缩写,Conv: 代表"卷积"（Convolution）,1d: 表示一维操作
        # Conv1d主要用于处理一维序列数据，如时间序列或音频信号。它在输入数据上滑动一个固定大小的窗口（卷积核），执行卷积操作，从而提取局部特征
        # value_embedding(x):12列映射到512个特征,position_embedding:96行->512特征,temporal_embedding:时间特征->512特征 然后做个加法
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)

        return self.dropout(x)
