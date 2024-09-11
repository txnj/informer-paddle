import os
import paddle


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu and paddle.is_compiled_with_cuda():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = paddle.device.set_device('gpu:{}'.format(self.args.gpu))
            print('🖥️Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = paddle.device.set_device("cpu")
            print(f'🏿Use CPU:{device}')
        return device

    def _get_data(self, flag: str):
        pass

    def vali(self, vali_data, vali_loader, criterion):
        pass

    def train(self, setting: str):
        pass

    def test(self, setting: str):
        pass
