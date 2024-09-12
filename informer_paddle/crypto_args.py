import paddle

from utils.tools import dotdict


def get_crypto_args(
        root_prefix='d:',
        model='informer',
        data='BTC',
        filename='binance_btc_usdt_2020.csv',
        freq='h',
        train_epochs=6,
) -> dotdict:
    args = dotdict()
    args.model = model
    args.data = data
    args.root_path = root_prefix + '/data/'
    args.data_path = filename
    args.features = 'M'
    args.target = 'OT'  # target feature in S or MS task
    args.freq = freq
    args.checkpoints = root_prefix + '/checkpoints/'
    args.seq_len = 96
    args.label_len = 48
    args.pred_len = 24
    # ─── ⋆⋅☆⋅⋆ ──
    args.enc_in = 7  # encoder input size
    args.dec_in = 7  # decoder input size
    args.c_out = 7  # output size
    args.d_model = 512
    args.n_heads = 8
    args.e_layers = 2  # num of encoder layers
    args.d_layers = 1  # num of decoder layers
    args.s_layers = '3,2,1'  # num of stack encoder layers
    args.d_ff = 2048
    args.factor = 5
    args.padding = 0
    args.distil = True
    args.dropout = 0.05
    args.attn = 'prob'
    args.embed = 'timeF'
    args.activation = 'gelu'
    args.output_attention = True  # whether to output attention in ecoder
    args.do_predict = False  # whether to predict unseen future data
    args.mix = True
    args.num_workers = 0
    args.itr = 1  # 要训练的次数
    args.train_epochs = train_epochs  # 跑几次epoch,就是循环跑多少次
    args.batch_size = 32
    args.patience = 3  # early stopping patience
    args.learning_rate = 0.0001
    args.loss = 'mse'
    args.lradj = 'type1'  # adjust learning rate
    args.use_amp = False
    args.inverse = False
    args.use_gpu = True if paddle.device.is_compiled_with_cuda() else False
    args.gpu = 0
    args.use_multi_gpu = False
    args.devices = '0,1,2,3'

    if args.use_gpu and args.use_multi_gpu:  # 多gpu设置
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    args.s_layers = [int(s_l) for s_l in str(args.s_layers).replace(' ', '').split(',')]
    args.detail_freq = args.freq
    args.freq = args.freq[-1:]
    print(f'🔣 args:{args}')
    return args
