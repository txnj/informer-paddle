from utils.tools import dotdict
from datetime import datetime


def get_setting(args: dotdict, ii) -> str:
    setting = '{}_{}_{}_{}_{}_{}_{}'.format(
        datetime.now().strftime("%Y-%m-%d_%H_%M"),
        args.model,
        args.data,
        args.features,
        args.attn,
        args.embed,
        ii)
    print(f'⚙️setting:{setting}')
    return setting

