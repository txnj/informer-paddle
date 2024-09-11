import paddle
import paddle.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask


# mask_flag：是否对输入的 queries 进行掩码处理，如果为 True，则使用 TriangularCausalMask 对 queries 进行掩码处理。
# factor：计算注意力权重的缩放因子。
# scale：缩放因子，可以替代 factor。
# attention_dropout：注意力机制中的dropout概率。
# output_attention：是否输出注意力权重。
# 在 forward 函数中，输入的 queries、keys 和 values 分别具有形状 (B, L, H, E)、(B, S, H, E) 和 (B, L, D)。
# 其中，B 表示批次大小，L 表示 queries 的长度，H 表示 queries 的通道数，E 表示 queries 的维度。paddle.einsum函数用于执行多维张量的线性操作，这里用于计算注意力分数。
# 在计算分数之后，根据 self.mask_flag 是否为 True，以及是否传递了 attn_mask 参数来判断是否进行掩码处理。
# 然后通过 self.dropout 函数对注意力权重进行 dropout 处理，并使用 paddle.softmax 函数计算注意力权重。
# 最后，将注意力权重和 values 相乘得到输出结果。
# 如果 self.output_attention 为 True，则返回输出结果和注意力权重；否则只返回输出结果。
class FullAttention(nn.Layer):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        b, l, h, e = queries.shape
        _, s, _, d = values.shape
        scale = self.scale or 1. / sqrt(e)

        scores = paddle.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(b, l, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        # ✏️A = self.dropout(torch.softmax(scale * scores, dim=-1))
        a = self.dropout(paddle.nn.functional.softmax(scale * scores, axis=-1))
        v = paddle.einsum("bhls,bshd->blhd", a, values)

        if self.output_attention:
            return v, a
        else:
            return v, None


class ProbAttention(nn.Layer):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    # 这个方法实现了概率注意力的核心逻辑。它随机采样键值对，计算查询和键之间的相似度，并选择最相关的top-k个查询
    @staticmethod
    def _prob_qk(_q, _k, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        b, h, l_k, e = _k.shape
        _, _, l_q, _ = _q.shape

        # 在-3维度上加上一个维度,复制l_q
        k_expand = paddle.expand(_k.unsqueeze(-3), shape=[b, h, l_q, l_k, e])
        # 随机采样sample_k个,示例中是25个
        index_sample = paddle.randint(low=0, high=l_k, shape=[l_q, sample_k])
        # 想当于96个Q去随机选取index_sample(25)个K
        k_sample = k_expand[:, :, paddle.arange(l_q).unsqueeze(1), index_sample, :]
        # 得到采样96个Q和25个K的值(32,8,96,25)
        q_k_sample = paddle.matmul(_q.unsqueeze(-2), paddle.transpose(k_sample, perm=[0, 1, 2, 4, 3])).squeeze(-2)
        # 使用稀疏度量查找Top_k
        # ✏️M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        m = q_k_sample.max(-1)[0] - q_k_sample.sum(-1) / l_k  # 最大值-均值=差异值
        m_top = m.topk(n_top, sorted=False)[1]  # 取差异值最大的前top个

        # 使用简化后的m_top(m_top里面是索引)来获取完整的q_reduce
        # ✏️Q_reduce = _q[paddle.arange(b)[:, None, None], paddle.arange(h)[None, :, None], m_top, :]  # factor*ln(L_q)
        q_reduce = _q[paddle.arange(b).unsqueeze([1, 2]), paddle.arange(h).unsqueeze([0, 2]), m_top, :]
        # ✏️q_k = paddle.matmul(q_reduce, _k.transpose(-2, -1))  # factor*ln(L_q)*L_k
        q_k = paddle.matmul(q_reduce, paddle.transpose(_k, perm=[0, 1, 3, 2]))
        return q_k, m_top

    def _get_initial_context(self, _v, l_q):
        b, h, l_v, d = _v.shape
        if not self.mask_flag:
            # 对v,倒数第二个维度(Q:96)求均值,对应特征给予均值
            v_sum = _v.mean(axis=-2)
            # ✏️contex = v_sum.unsqueeze(-2).expand(b, h, l_q, v_sum.shape[-1]).clone()
            contex = paddle.tile(v_sum.unsqueeze(-2), repeat_times=[1, 1, l_q, 1])
        else:  # use mask
            assert (l_q == l_v)  # 要求L_Q == L_V，即仅用于自注意力机制
            contex = paddle.cumsum(_v, axis=2)
        return contex

    # 更新上下文向量。它应用softmax函数计算注意力权重，然后用这些权重更新上下文
    def _update_context(self, context_in, _v, scores, index, l_q, attn_mask):
        b, h, l_v, d = _v.shape

        if self.mask_flag:
            attn_mask = ProbMask(b, h, l_q, index, scores)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = paddle.nn.functional.softmax(scores, axis=-1)
        # 更新context,把top值填充到均值中
        context_in[paddle.arange(b)[:, None, None], paddle.arange(h)[None, :, None], index, :] = (
            paddle.matmul(attn, _v).astype(context_in.dtype))

        if self.output_attention:
            attns = (paddle.ones([b, h, l_v, l_v]) / l_v).astype(attn.dtype)
            attns[paddle.arange(b)[:, None, None], paddle.arange(h)[None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    # 前向传播函数,实现了一种稀疏注意力机制
    def forward(self, queries, keys, values, attn_mask):
        # b: 批次大小
        # l_q: 查询序列长度
        # h: 注意力头数
        # d: 每个头的维度
        b, l_q, h, d = queries.shape
        _, l_k, _, _ = keys.shape
        # 对查询、键和值进行转置操作，将注意力头维度移到第二维。
        queries = paddle.transpose(queries, perm=[0, 2, 1])
        keys = paddle.transpose(keys, perm=[0, 2, 1])
        values = paddle.transpose(values, perm=[0, 2, 1])
        # 用于键的采样数
        u_part = self.factor * np.ceil(np.log(l_k)).astype('int').item()
        # 用于查询的采样数
        u = self.factor * np.ceil(np.log(l_q)).astype('int').item()

        u_part = u_part if u_part < l_k else l_k
        u = u if u < l_q else l_q

        scores_top, index = self._prob_qk(queries, keys, sample_k=u_part, n_top=u)

        # 设置缩放因子
        scale = self.scale or 1. / sqrt(d)
        if scale is not None:
            scores_top = scores_top * scale
        # 求得context中k都是均值
        context = self._get_initial_context(values, l_q)
        # 更新context,替换scores_top对应的值.
        context, attn = self._update_context(context, values, scores_top, index, l_q, attn_mask)
        return paddle.transpose(context, perm=[0, 2, 1]), attn


# 注意力层类
class AttentionLayer(nn.Layer):
    # attention:注意力机制
    # d_model:隐层特征数
    # n_heads:多头注意力数
    # d_keys和d_values:键和值的维度
    # mix:是否混合
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        # 传入的注意力机制
        self.inner_attention = attention
        # query_projection、key_projection、value_projection：线性投影层，用于将输入转换为查询、键和值
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        # out_projection输出投影层，用于将注意力的输出转换回原始维度
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    # 多头注意力机制的前向传播函数
    def forward(self, queries, keys, values, attn_mask):
        # b: 批次大小
        # l: 查询序列长度
        b, l, _ = queries.shape
        # s: 键/值序列长度
        _, s, _ = keys.shape
        # h: 注意力头数
        h = self.n_heads

        # 将查询、键、值通过线性层投影到新的空间
        # 重塑张量以适应多头结构
        queries = self.query_projection(queries).reshape([b, l, h, -1])  # 这里-1表示最后一个维度自动计算
        keys = self.key_projection(keys).reshape([b, s, h, -1])
        values = self.value_projection(values).reshape([b, s, h, -1])
        # 调用inner_attention函数计算注意力
        # 返回注意力输出和注意力权重
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = paddle.transpose(out, perm=[0, 2, 1])
        out = paddle.reshape(out, [b, l, -1])

        return self.out_projection(out), attn
