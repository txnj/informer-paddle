import paddle


class TriangularCausalMask:
    def __init__(self, _b, _l, device="cpu"):
        mask_shape = [_b, 1, _l, _l]
        with paddle.no_grad():
            self._mask = paddle.triu(paddle.ones(mask_shape, dtype=paddle.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask:
    def __init__(self, _b, _h, _l, index, scores, device="cpu"):
        # ✏️_mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask = paddle.triu(paddle.ones([_l, scores.shape[-1]]), diagonal=1)
        _mask_ex = _mask[None, None, :].expand(_b, _h, _l, scores.shape[-1])
        indicator = _mask_ex[paddle.arange(_b)[:, None, None], paddle.arange(_h)[None, :, None], index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask
