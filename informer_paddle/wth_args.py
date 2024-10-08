import paddle

from utils.tools import dotdict


def say():
    print("say hello")


def get_wth_args() -> dotdict:
    path_prefix = 'd:'  # '.'
    args = dotdict()
    args.model = 'informer'
    args.data = 'WTH_small'
    args.root_path = path_prefix + '/data/'
    args.data_path = 'WTH_small.csv'
    args.features = 'M'
    args.target = 'OT'
    args.freq = 'h'
    args.checkpoints = path_prefix + '/checkpoints/'
    args.seq_len = 96
    args.label_len = 48
    args.pred_len = 24
    # ─── ⋆⋅☆⋅⋆ ──
    args.enc_in = 12
    args.dec_in = 12
    args.c_out = 12
    args.d_model = 512
    args.n_heads = 8
    args.e_layers = 2
    args.d_layers = 1
    args.s_layers = '3,2,1'
    args.d_ff = 2048
    args.factor = 5
    args.padding = 0
    args.distil = True
    args.dropout = 0.05
    args.attn = 'prob'
    args.embed = 'timeF'
    args.activation = 'gelu'
    args.output_attention = True
    args.mix = True
    args.num_workers = 0
    args.itr = 1
    args.train_epochs = 6
    args.batch_size = 32
    args.patience = 4  # early stopping patience
    args.learning_rate = 0.0001
    args.des = 'test'
    args.loss = 'mse'
    args.lradj = 'type1'
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
