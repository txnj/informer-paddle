import paddle
from exp.exp_informer import Exp_Informer
from informer_paddle.args import get_setting
from paddle import utils
from informer_paddle.crypto_args import get_crypto_args

if __name__ == '__main__':
    utils.run_check()
    args = get_crypto_args()
    Exp = Exp_Informer

    for ii in range(args.itr):
        setting = get_setting(args, ii)

        exp = Exp(args)
        print('>>>>>>>🚀start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>🧪testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        if args.do_predict:
            print('>>>>>>>🎲predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, True)

        paddle.device.cuda.empty_cache()
        print('🚩执行结束🚩')
