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

    @staticmethod
    def _prob_qk(_q, _k, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        b, h, l_k, e = _k.shape
        _, _, l_q, _ = _q.shape

        # 计算采样后的Q_K
        k_expand = _k.unsqueeze(-3).expand(b, h, l_q, l_k, e)
        index_sample = paddle.randint(l_k, (l_q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        k_sample = k_expand[:, :, paddle.arange(l_q).unsqueeze(1), index_sample, :]
        q_k_sample = paddle.matmul(_q.unsqueeze(-2), k_sample.transpose(-2, -1)).squeeze(-2)
        # 使用稀疏度量查找Top_k
        # ✏️M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        m = q_k_sample.max(-1)[0] - q_k_sample.sum(-1) / l_k  # 96个Q和25个K之间的关系
        m_top = m.topk(n_top, sorted=False)[1]

        # 使用简化后的Q来计算Q_K
        # q_reduce = _q[paddle.arange(_b)[:, None, None],paddle.arange(_h)[None, :, None],m_top, :]  # factor*ln(L_q)
        q_reduce = paddle.take_along_axis(_q, m_top[:, :, :, None], axis=-2)
        q_k = paddle.matmul(q_reduce, _k.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return q_k, m_top

    def _get_initial_context(self, _v, l_q):
        b, h, l_v, d = _v.shape
        if not self.mask_flag:
            v_sum = _v.mean(dim=-2)
            contex = v_sum.unsqueeze(-2).expand(b, h, l_q, v_sum.shape[-1]).clone()
        else:  # use mask
            assert (l_q == l_v)  # 要求L_Q == L_V，即仅用于自注意力机制
            contex = _v.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, _v, scores, index, l_q, attn_mask):
        b, h, l_v, d = _v.shape

        if self.mask_flag:
            attn_mask = ProbMask(b, h, l_q, index, scores, device=_v.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = paddle.nn.functional.softmax(scores, axis=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[paddle.arange(b)[:, None, None], paddle.arange(h)[None, :, None], index, :] = (
            paddle.matmul(attn, _v).type_as(context_in))

        if self.output_attention:
            attns = (paddle.ones([b, h, l_v, l_v]) / l_v).type_as(attn).to(attn.device)
            attns[paddle.arange(b)[:, None, None], paddle.arange(h)[None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask):
        b, l_q, h, d = queries.shape
        _, l_k, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        u_part = self.factor * np.ceil(np.log(l_k)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(l_q)).astype('int').item()  # c*ln(L_q)

        u_part = u_part if u_part < l_k else l_k
        u = u if u < l_q else l_q

        scores_top, index = self._prob_qk(queries, keys, sample_k=u_part, n_top=u)

        # 加比例因子
        scale = self.scale or 1. / sqrt(d)
        if scale is not None:
            scores_top = scores_top * scale
        # 获取上下文
        context = self._get_initial_context(values, l_q)
        # 用选定的top_k个查询更新上下文
        context, attn = self._update_context(context, values, scores_top, index, l_q, attn_mask)
        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Layer):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        b, l, _ = queries.shape
        _, s, _ = keys.shape
        h = self.n_heads

        queries = self.query_projection(queries).view(b, l, h, -1)
        keys = self.key_projection(keys).view(b, s, h, -1)
        values = self.value_projection(values).view(b, s, h, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(b, l, -1)

        return self.out_projection(out), attn
