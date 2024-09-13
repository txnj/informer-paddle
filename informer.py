import argparse
from datetime import datetime

import paddle
from exp.exp_informer import Exp_Informer
from paddle import utils

utils.run_check()
print('🛶🛶🛶')

parser = argparse.ArgumentParser(description='[Informer] 长序列预测')
parser.add_argument('--model', type=str, required=True, default='informer',
                    help='🏷️model of experiment, options: [informer, informerstack, informerlight(TBD)]')

parser.add_argument('--data', type=str, required=True, default='WTH_small', help='data')
parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file,传入相对路径')
parser.add_argument('--data_path', type=str, default='WTH_small.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                         'S:univariate predict univariate, MS:multivariate predict univariate')

parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                    help='location of model checkpoints,🏷模型保存位置')

parser.add_argument('--seq_len', type=int, default=96,
                    help='input sequence length of Informer encoder,🏷️输入序列长度,根据项目自定义')
parser.add_argument('--label_len', type=int, default=48,
                    help='start token length of Informer decoder,🏷️起始已知值')
parser.add_argument('--pred_len', type=int, default=24,
                    help='prediction sequence length,🏷️预测序列长度,根据项目自定义')

# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

parser.add_argument('--enc_in', type=int, default=7, help='encoder input size,🏷️指有多少列')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model,隐层特征数')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads,多头注意力数')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor,🏷️采样因子数')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling,'
                         '类似polling操作',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob',
                    help='attention used in encoder, options:[prob, full],🏷️注意力机制计算方法,这个是informer核心')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation,激活函数')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
# --cols column1 column2 column3
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times,实验次数')
parser.add_argument('--train_epochs', type=int, default=6, help='train epochs,🏷️跑几次epoch,就是循环跑多少次')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=4, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function,损失函数')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu,是否使用gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
# 从命令行中解析参数
args = parser.parse_args()

args.use_gpu = True if paddle.device.is_compiled_with_cuda() and args.use_gpu else False

# 多gpu配置
if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

args.s_layers = [int(s_l) for s_l in str(args.s_layers).replace(' ', '').split(',')]
args.detail_freq = args.freq
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)

Exp = Exp_Informer

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_{}_{}_{}_{}_{}'.format(
        datetime.now().strftime("%Y-%m-%d_%H_%M"),
        args.model,
        args.data,
        args.features,
        args.attn,
        args.embed,
        ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>🚀start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)

    print('>>>>>>>🧪testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)

    if args.do_predict:
        print('>>>>>>>🎲predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, True)

    paddle.device.cuda.empty_cache()
    print('🚩执行结束🚩')
