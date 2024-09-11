import paddle


class TriangularCausalMask:
    def __init__(self, _b, _l, device="cpu"):
        mask_shape = [_b, 1, _l, _l]
        with paddle.no_grad():
            self._mask = paddle.triu(paddle.ones(mask_shape, dtype='bool'), diagonal=1)

    @property
    def mask(self):
        return self._mask


class ProbMask:
    def __init__(self, _b, _h, _l, index, scores):
        # ✏️_mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask = paddle.ones((_l, scores.shape[-1]), dtype='bool').triu(1)
        # ✏️_mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        _mask_ex = _mask.unsqueeze([0, 1]).expand([_b, _h, _l, scores.shape[-1]])
        # ✏️indicator = _mask_ex[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :].to(device)
        indicator = _mask_ex[paddle.arange(_b).unsqueeze([1, 2]), paddle.arange(_h).unsqueeze([0, 2]), index, :]
        # ✏️self._mask = indicator.view(scores.shape).to(device)
        self._mask = paddle.reshape(indicator, shape=scores.shape)

    @property
    def mask(self):
        return self._mask
