import investpy
import numpy as np

import pandas as pd
from datetime import date


def rand_date(start="2000-01-01", end=None):
    min_date = np.datetime64(start)

    today = np.datetime64(date.today())
    if (end == None):
        end = today
    else:
        end = np.datetime64(end)

    return min_date + np.random.randint(0,
                                        end - np.datetime64(min_date))


def get_forex_data(start, end, forex, country="United States"):
    df = investpy.get_currency_cross_historical_data(currency_cross=forex, from_date=start, to_date=end,
                                                     country=country)

    del df['Currency']
    df = df.reset_index()
    del df['Date']
    df = df.rename_axis(None)

    data_x = np.array([])

    for (i, j) in df.iterrows():
        if np.array_equal(data_x, []):
            data_x = [df.iloc[[i]].to_numpy()[0]]
        else:
            data_x = np.vstack((data_x, df.iloc[[i]].to_numpy()[0]))
    return data_x


def get_stock_data(start, end, stock, country="United States"):
    df = investpy.get_stock_historical_data(stock=stock, from_date=start, to_date=end, country=country)

    del df['Currency']
    df = df.reset_index()
    del df['Date']
    df = df.rename_axis(None)

    data_x = np.array([])

    for (i, j) in df.iterrows():
        if np.array_equal(data_x, []):
            data_x = [df.iloc[[i]].to_numpy()[0]]
        else:
            data_x = np.vstack((data_x, df.iloc[[i]].to_numpy()[0]))
    return data_x


def get_index_data(start, end, stock, country="United States"):
    df = investpy.get_index_historical_data(index=stock, from_date=start, to_date=end, country=country)

    del df['Currency']
    df = df.reset_index()
    del df['Date']
    df = df.rename_axis(None)

    data_x = np.array([])

    for (i, j) in df.iterrows():
        if np.array_equal(data_x, []):
            data_x = [df.iloc[[i]].to_numpy()[0]]
        else:
            data_x = np.vstack((data_x, df.iloc[[i]].to_numpy()[0]))
    return data_x


def get_etf_data(start, end, stock, country="United States"):
    df = investpy.get_etf_historical_data(etf=stock, from_date=start, to_date=end, country=country)

    del df['Currency']
    df = df.reset_index()
    del df['Date']
    df = df.rename_axis(None)

    data_x = np.array([])

    for (i, j) in df.iterrows():
        if np.array_equal(data_x, []):
            data_x = [df.iloc[[i]].to_numpy()[0]]
        else:
            data_x = np.vstack((data_x, df.iloc[[i]].to_numpy()[0]))
    return data_x


def get_fund_data(start, end, stock, country="United States"):
    df = investpy.get_fund_historical_data(fund=stock, from_date=start, to_date=end, country=country)

    del df['Currency']
    df = df.reset_index()
    del df['Date']
    df = df.rename_axis(None)

    data_x = np.array([])

    for (i, j) in df.iterrows():
        if np.array_equal(data_x, []):
            data_x = [df.iloc[[i]].to_numpy()[0]]
        else:
            data_x = np.vstack((data_x, df.iloc[[i]].to_numpy()[0]))
    return data_x


def bachify(array, res=0):
    output = []
    for i in range(len(array[0])):
        a = []
        for j in range(len(array)):
            a += array[j][i]
        if res == 1:
            a = [a]
            output.append(a)
        if res == 0:
            output.append(a)
    return output


def timeseries_from_array(array):
    output = np.zeros(shape=(len(array) - 1, 2, 5))
    for i in range(len(array) - 1):
        output[i] = [array[i], array[i + 1]]
    return output
