import json
import time
import pytz
import requests
import pandas as pd
from datetime import datetime, timedelta
import os


def btc_history_candles(_file_path: str, _symbol: str, _interval: str, limit: int):
    start_time = '2020-01-01 00:00:00'
    csv_header = True
    # 读取现有的 CSV 文件
    if os.path.exists(_file_path):
        existing_df = pd.read_csv(_file_path)
        start_time = existing_df.iloc[-1, 0]
        csv_header = False
        print(f'🕒csv中最后一个时间:{start_time}')

    start_time_ms = int(datetime.strptime(start_time, "%Y-%m-%d %H:%M").timestamp() * 1000)
    url = ('https://api.binance.com/api/v3/klines?symbol={}&interval={}&startTime={}&timeZone=8&limit={}'
           .format(_symbol, _interval, start_time_ms, limit))

    # 从URL获取JSON数据
    response = requests.get(url)
    data = json.loads(response.text)
    print(data)
    if isinstance(data, list) and len(data) > 0:
        pass
    else:
        print('🚫missing data...')
        return
    # 将JSON数据转换为DataFrame
    df = pd.DataFrame(data, columns=[
        'date',  # k线开盘时间
        'OpeningPx',  # 开盘价格
        'HighestPx',  # 最高价格
        'LowestPx',  # 最低价格
        'ClosingPx',  # 收盘价格
        'BaseVolume',  # 基础币成交量
        'CloseTime',  # k线收盘时间
        'QuotedVolume',  # 计价币成交量
        'Count',  # 成交笔数
        'ActiveVolume',  # 主动买入成交量
        'ActiveQuotedVolume',  # 主动买入成交额
        "_",  # 请忽略该参数
    ])
    df = df.iloc[:, :9]
    df = df.drop(['CloseTime'], axis=1)
    print(df.shape)
    tz = pytz.timezone('Asia/Shanghai')
    df['date'] = pd.to_datetime(df['date'], unit='ms', utc=True).dt.tz_convert('Asia/Shanghai').dt.strftime(
        '%Y-%m-%d %H:%M')
    # 创建一个不包含 'Time' 的列名列表
    columns_to_float = [col for col in df.columns if col != 'date' and col != 'Count']
    # 将特定列转换为 float 类型
    df[columns_to_float] = df[columns_to_float].astype(float)

    # 保存为CSV文件
    # index=False:这个参数设置为False表示不将DataFrame的索引写入CSV文件
    # model=a:追加模式
    df.to_csv(_file_path, mode='a', index=False, encoding='utf-8-sig', header=csv_header)
    print(f"💾start_time:{start_time},数据已保存到 {_file_path}")


def deduplicated(_file_path: str, _column_name: str):
    if os.path.exists(_file_path):
        existing_df = pd.read_csv(_file_path)
        existing_df.drop_duplicates(subset=[_column_name], keep='first', inplace=True)
        existing_df.to_csv(_file_path, index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    file_path = '../data/binance_btc_usdt_2020.csv'
    symbol = 'BTCUSDT'
    interval = '1h'
    deduplicated_column_name = 'date'
    count = 0
    while count < 2:
        btc_history_candles(file_path, symbol, interval, limit=1000)
        time.sleep(1)
        count += 1

    deduplicated(file_path, deduplicated_column_name)
