from data_loader.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import DataLoader

import os
import time

import warnings

warnings.filterwarnings('ignore')


class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'informer': Informer,
            'informerstack': InformerStack,
        }
        if self.args.model == 'informer' or self.args.model == 'informerstack':
            e_layers = self.args.e_layers if self.args.model == 'informer' else self.args.s_layers
            # 构建model
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in,
                self.args.c_out,
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.factor,
                self.args.d_model,
                self.args.n_heads,
                e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.dropout,
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
            )
            if self.args.use_multi_gpu and self.args.use_gpu:
                model = paddle.DataParallel(model)
            return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom,
            'WTH_small': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'custom': Dataset_Custom,
            'my_data': Dataset_Custom,
        }
        dataset_class = data_dict[self.args.data]
        timeenc = 0 if args.embed != 'timeF' else 1

        if flag == 'test':  # 测试
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        elif flag == 'pred':  # 预测
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = args.detail_freq
            dataset_class = Dataset_Pred
        else:
            shuffle_flag = True  # 训练时,是否洗牌
            drop_last = True  # 丢掉最后不足一个batch的数据,比如321条数据,batch_size:32,就会丢调最后一条数据
            batch_size = args.batch_size
            freq = args.freq
        data_set = dataset_class(  # 定义data_set
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(f'📑_get_data处理结束,flag:【{flag}】,data_set.length:【{len(data_set)}】')
        # 构建 torch DataLoader
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = paddle.optimizer.Adam(parameters=self.model.parameters(), learning_rate=self.args.learning_rate)
        return model_optim

    @staticmethod
    def _select_criterion():
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        # 获得data_set, data_loader
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        # 保存模型的路径path
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)  # 执行多少个steps
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)  # 体验停止策略

        model_optim = self._select_optimizer()  # 指定优化器:Adam
        criterion = self._select_criterion()  # 损失函数:MSELoss

        # if self.args.use_amp:  # ✏️mf-windows中不需要加这个，容易崩，Linus可以加
        #     scaler = torch.cuda.amp.GradScaler()  # mf-这里暂时没有改
        # 🏷️循环指定的epochs次数,默认值为6
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()  # 开始训练
            epoch_time = time.time()
            # 遍历data_loader,循环次数为batch_size,🏷️默认值32次,会遍历自定义的:__getitem__,见data_loader.py
            # x(32,96,12),y(32,72,12),x_mark(32,96,4),y_mark(32,72,4)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                model_optim.clear_grad()  # ✏️mf-梯度清零，防止梯度累加
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred, true)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # ✏️if self.args.use_amp:
                #     scaler.scale(loss).backward()
                #     scaler.step(model_optim)
                #     scaler.update()
                # else:
                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.set_state_dict(paddle.load(best_model_path))

        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()

        preds = []
        trues = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape([-1, preds.shape[-2], preds.shape[-1]])
        trues = trues.reshape([-1, trues.shape[-2], trues.shape[-1]])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints.removeprefix('.'), setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.set_state_dict(paddle.load(best_model_path))

        self.model.eval()

        preds = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape([-1, preds.shape[-2], preds.shape[-1]])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.astype(paddle.float32)
        batch_y = batch_y.astype(paddle.float32)

        # 进行encode 和 decode会需要x_mark和y_mark,所以都要to(device)
        batch_x_mark = batch_x_mark.astype(paddle.float32)
        batch_y_mark = batch_y_mark.astype(paddle.float32)

        if self.args.padding == 0:
            # decoder输入,以torch.zeros为初始化进行构建(32,24,12),默认padding为0
            # ✏️dec_inp = paddle.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
            dec_inp = paddle.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]], dtype='float32')
        elif self.args.padding == 1:
            # ✏️dec_inp = paddle.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
            dec_inp = paddle.ones(shape=[batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]], dtype='float32')
        else:
            raise ValueError('padding must be 0 or 1')
        # 拼接decoder输入,torch.cat在维度1(数据条数)上拼接预测值dep_inp(全0值)
        # ✏️dec_inp = paddle.concat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        dec_inp = paddle.concat([batch_y[:, :self.args.label_len, :], dec_inp], axis=1).astype('float32')
        # dec_inp长度为72，其中前48为真实值，后面24个是要预测的值（用0初始化）
        if self.args.use_amp:
            with paddle.amp.auto_cast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

        return outputs, batch_y
