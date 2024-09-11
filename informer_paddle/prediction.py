import os
import paddle
import numpy as np


def predict(exp, setting, load=False):
    pred_data, pred_loader = exp._get_data(flag='pred')

    if load:
        path = os.path.join(exp.args.checkpoints, setting)
        best_model_path = path + '/' + 'checkpoint.pth'
        exp.model.set_state_dict(paddle.load(best_model_path))

    exp.model.eval()

    preds = []

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
        batch_x = batch_x.astype('float32')
        batch_y = batch_y.astype('float32')
        batch_x_mark = batch_x_mark.astype('float32')
        batch_y_mark = batch_y_mark.astype('float32')

        # decoder input
        if exp.args.padding == 0:
            dec_inp = paddle.zeros([batch_y.shape[0], exp.args.pred_len, batch_y.shape[-1]])
        elif exp.args.padding == 1:
            dec_inp = paddle.ones([batch_y.shape[0], exp.args.pred_len, batch_y.shape[-1]])
        else:
            dec_inp = paddle.zeros([batch_y.shape[0], exp.args.pred_len, batch_y.shape[-1]])
        dec_inp = paddle.concat([batch_y[:, :exp.args.label_len, :], dec_inp], axis=1)
        # encoder - decoder
        if exp.args.output_attention:
            outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        f_dim = -1 if exp.args.features == 'MS' else 0
        batch_y = batch_y[:, -exp.args.pred_len:, f_dim:]

        pred = outputs.detach().cpu().numpy()  # .squeeze()

        preds.append(pred)

    preds = np.array(preds)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

    # result save
    folder_path = './results/' + setting + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    np.save(folder_path + 'real_prediction.npy', preds)

    return preds
