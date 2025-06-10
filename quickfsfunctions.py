import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.metrics import mean_absolute_error
import yfinance as yf
import os
from datetime import datetime, timedelta
import requests

import pickle
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import shap

def add_days(date, days):
    return str(datetime(int(date[:4]), int(date[5:7]), int(date[8:10])) + timedelta(days=days))[:10]

def get_tickers():
    return pd.read_csv("data/tickers.csv", index_col=0)

def preprocess(file, startdate, years, metric):

    # Access the compiled data for a ticker and drop misc columns
    data = pd.read_csv(f'data/quickfs/{file}', index_col=0).drop(['shares_eop', 'dividends', 'period_end_price', 'period_end_date', "fiscal_quarter_number", "fiscal_quarter_key"], axis=1)
    
    # Possible metric types: all, income_statement, balance_sheet, cash_flow_statement, computed
    if metric != "all":
        metric_type = metric
        metric_data = pd.read_csv("data/metric_metadata.csv", index_col=0)
        missing_cols = ['long_term_debt_and_capital_lease_obligation', 'original_filing_date', 'restated_filing_date', 'preliminary', "fiscal_quarter_number", "fiscal_quarter_key"]

        new_df = pd.DataFrame()
        new_df.index = data.index

        for i in range(len(metric_data)):
            metric = metric_data['metric'].iloc[i]
            ctype = metric_data['company_types'].iloc[i]
            stype = metric_data['statement_type'].iloc[i]
            if 'normal' in ctype and metric not in missing_cols and stype == metric_type:
                new_df[metric] = data[metric]
        data = new_df
    
    # Get the first earnings data before startdate and assign the training data to x
    enddate = add_days(startdate, years * 365)
    # x = data[enddate:startdate].iloc[-1]
    x = data[:startdate].iloc[-1]

    # Reassign the startdate to be the actual first day we can invest in the stock.
    # This should always be startdate since we choose x before startdate,
    # but this makes it so the code can work with later dates if needed
    startdate = max(startdate, add_days(x.name, 1))
    
    # Grab price data from yfinance, so we can determine whether the price increased and assign a label
    hist = pd.read_csv(f'data/yfinance/{file}', index_col=0)['Open']
    
    # Get the intial and final values of the price
    startval = hist.loc[startdate:].iloc[0]
    endval = hist.loc[enddate:].iloc[0]
    # Assign the label to represent the stocks percent increase/decrease in value
    if startval == 0:
        raise Exception("Divide by zero: startval == 0")
    y = (endval - startval) / startval

    return x, y, startdate

def train_ranking(model, startdate, num_stocks, years, metric):
    # Compile the data
    train_start = startdate
    xlist, ylist, tickerlist = list(), list(), list()
    tickers = get_tickers()
    for i in range(num_stocks):
        ticker = tickers['ticker'].iloc[i]
        try:
            x, y, date = preprocess(f"{ticker}.csv", train_start, years, metric)
            xlist.append(x)
            ylist.append(y)
            tickerlist.append(ticker)
        except:
            continue
    
    # Sort and rank the stocks by return
    for i in range(len(ylist)):
        maxval = np.inf
        j = 0
        for k in range(i, len(ylist)):
            if ylist[k] < maxval:
                maxval = ylist[k]
                j = k

        temp = ylist[i]
        ylist[i] = ylist[j]
        ylist[j] = temp

        temp = xlist[i]
        xlist[i] = xlist[j]
        xlist[j] = temp
    
    for i in range(len(ylist)):
        ylist[i] = len(ylist) - i - 1
    
    # Train and save model to data
    if type(model) == LGBMRegressor:
        model.fit(xlist, ylist, eval_metric="MAE")
    else:
        model.fit(xlist, ylist, verbose=False)
    # pickle.dump(model, open("data/xgbmodel.pkl", "wb"))
    return model, xlist

def train(model, startdate, num_stocks, years, metric):
    # Compile the training data
    train_start = startdate
    xlist, ylist, tickerlist = list(), list(), list()
    tickers = get_tickers()
    for i in range(num_stocks):
        ticker = tickers['ticker'].iloc[i]
        try:
            x, y, date = preprocess(f"{ticker}.csv", train_start, years, metric)
            xlist.append(x)
            ylist.append(y)
            tickerlist.append(ticker)
        except:
            continue
    
    # Train and save model to data
    model.fit(xlist, ylist)
    # pickle.dump(model, open("data/xgbmodel.pkl", "wb"))
    return model, xlist

def train_and_predict(model, startdate, num_stocks, years, metric):
    model = train(model, startdate, num_stocks, years, metric)[0]
    
    # Compile the testing data
    test_start = add_days(startdate, years * 365)
    xlist, ylist, tickerlist = list(), list(), list()
    tickers = get_tickers()
    for i in range(num_stocks):
        ticker = tickers['ticker'].iloc[i]
        try:
            x, y, date = preprocess(f"{ticker}.csv", test_start, years, metric)
            xlist.append(x)
            ylist.append(y)
            tickerlist.append(ticker)
        except:
            continue
    
    # Return predictions and observed data
    predictions = model.predict(xlist)
    return model, predictions, ylist

def train_ten_years(model, startdate, num_stocks, years, metric):
    tickers = get_tickers()
    for i in range(10):
        xlist, ylist = list(), list()
        for i in range(num_stocks):
            ticker = tickers['ticker'].iloc[i]
            try:
                x, y, date = preprocess(f"{ticker}.csv", startdate, years, metric)
                xlist.append(x)
                ylist.append(y)
                tickerlist.append(ticker)
            except:
                continue
        model.fit(xlist, ylist)
        startdate = add_days(startdate, years * 365)
    return model

def train_ten_years_ranking(model, startdate, num_stocks, years, metric):
    for i in range(10):
        model = train_ranking(model, startdate, num_stocks, years, metric)[0]
        startdate = add_days(startdate, years * 365)
    return model

def predict(model, startdate, num_stocks, years, metric):
    xlist, ylist, tickerlist = list(), list(), list()
    tickers = get_tickers()
    for i in range(num_stocks):
        ticker = tickers['ticker'].iloc[i]
        try:
            x, y, date = preprocess(f"{ticker}.csv", startdate, years, metric)
            xlist.append(x)
            ylist.append(y)
            tickerlist.append(ticker)
        except:
            continue
    
    # Return predictions and observed data
    predictions = model.predict(xlist)
    return predictions, ylist

def predict_df(model, startdate, num_stocks, years, metric, pct=0.99):
    xlist, ylist, tickerlist, datelist = list(), list(), list(), list()
    tickers = get_tickers()
    return_df = pd.DataFrame(columns=["Stock Ticker", 'Date', 'Return'])
    for i in range(num_stocks):
        ticker = tickers['ticker'].iloc[i]
        try:
            x, y, date = preprocess(f"{ticker}.csv", startdate, years, metric)
            xlist.append(x)
            ylist.append(y)
            datelist.append(date)
            tickerlist.append(ticker)
        except:
            continue

    predicted = model.predict(xlist)
    for i in pct_indices(predicted, pct):
        return_df.loc[len(return_df)] = [tickerlist[i], datelist[i], ylist[i]]

    return return_df

def pct_indices(a, pct):
    # Sort and return indices of bottom pct of array a
    array = list(a)
    indices = list(range(len(array)))
    for i in range(len(array)):
        maxval = np.inf
        j = 0
        for k in range(i, len(array)):
            if array[k] < maxval:
                maxval = array[k]
                j = k

        temp = array[i]
        array[i] = array[j]
        array[j] = temp

        temp = indices[i]
        indices[i] = indices[j]
        indices[j] = temp

    return indices[:int(pct * len(array))]

def return_pct(predicted, test, pct=0.95):
    returns_list, ticker_list = list(), list()
    for i in pct_indices(predicted, pct):
        returns_list.append(test[i])
        # ticker_list.append(tickertest[i])
    return returns_list, ticker_list


def return_std(predicted, test, stds=1.96):
    threshold = np.mean(predicted) + np.std(predicted) * stds
    returns_list, ticker_list = list(), list()
    for i in range(len(predicted)):
        if predicted[i] >= threshold:
            returns_list.append(test[i])
            # ticker_list.append(tickertest[i])
    return returns_list, ticker_list

def predict_final(model, num_stocks, startdate, metric, pct=0.99):
    # Compile the testing data
    xlist, tickerlist = list(), list()
    return_df = pd.DataFrame(columns=["Stock Ticker", "Date"])
    tickers = get_tickers()
    for i in range(num_stocks):
        ticker = tickers['ticker'].iloc[i]
        try:
            # Access the compiled data for a ticker and drop misc columns
            data = pd.read_csv(f'data/quickfs/{file}', index_col=0).drop(['shares_eop', 'dividends', 'period_end_price', 'period_end_date', "fiscal_quarter_number", "fiscal_quarter_key"], axis=1)
    
            # Possible metric types: all, income_statement, balance_sheet, cash_flow_statement, computed
            if metric != "all":
                metric_type = metric
                metric_data = pd.read_csv("data/metric_metadata.csv", index_col=0)
                missing_cols = ['long_term_debt_and_capital_lease_obligation', 'original_filing_date', 'restated_filing_date', 'preliminary', "fiscal_quarter_number", "fiscal_quarter_key"]

                new_df = pd.DataFrame()
                new_df.index = data.index

                for i in range(len(metric_data)):
                    metric = metric_data['metric'].iloc[i]
                    ctype = metric_data['company_types'].iloc[i]
                    stype = metric_data['statement_type'].iloc[i]
                    if 'normal' in ctype and metric not in missing_cols and stype == metric_type:
                        new_df[metric] = data[metric]
                data = new_df

            # Get the first earnings data before startdate and assign the training data to x
            # enddate = add_days(startdate, years * 365)
            # x = data[enddate:startdate].iloc[-1]
            x = data[:startdate].iloc[-1]
            xlist.append(x)
            tickerlist.append(ticker)
        except:
            continue
    
    # Return predictions and observed data
    predictions = model.predict(xlist)
    for i in pct_indices(predictions, pct):
        return_df.loc[len(return_df)] = [tickerlist[i], startdate]

    return return_df

def backtest(stock_df, startdate, years, enddate = ""):
    ''' stock_df is a df as returned by the predict_df() function. '''
    if enddate == "":
        enddate = add_days(startdate, years * 365)
    return_hist = pd.DataFrame(columns=['Return', 'Date'])
    data_list = [pd.read_csv(f'data/yfinance/{ticker}.csv', index_col=0)['Open'] for ticker in stock_df['Stock Ticker']]
    num_stocks = len(stock_df)
    invested_list, invested_prices = [0] * num_stocks, [0] * num_stocks
    date = startdate

    while date <= enddate:
        net_return = 1
        for i in range(num_stocks):
            if date == stock_df.loc[i, 'Date']:
                invested_list[i] = 1
                invested_prices[i] = data_list[i].loc[date:].iloc[0]
            elif stock_df.loc[i, 'Date'] < date and invested_list[i] == 0:
                invested_list[i] = 1
                invested_prices[i] = data_list[i].loc[date:].iloc[0]
            if invested_list[i] == 1:
                net_return += (data_list[i].loc[date:].iloc[0] - invested_prices[i]) / (invested_prices[i] * num_stocks)
        return_hist.loc[len(return_hist)] = [net_return, date]
        date = add_days(date, 1)
    return return_hist

def daily_sharpe(returns):
    return_list = []
    for i in range(1, len(returns['Return'])):
        return_list.append(returns['Return'].iloc[i] - returns['Return'].iloc[i - 1])
    return np.mean(return_list) / np.std(return_list)

def one_year_sharpe(returns):
    return daily_sharpe(returns) * np.sqrt(252)

def yearly_sharpe(returns):
    # The risk free rate varies over time, but was about 3% on average over the period of the data
    risk_free = 0.03
    return (np.mean(returns) - risk_free) / np.std(returns)

def feature_importance(model, xlist):
    explainer = shap.Explainer(model)
    shap_values = explainer(xlist)

    num_values = len(shap_values[0])
    values = [0] * num_values
    for i in range(num_values):
        for j in range(len(shap_values)):
            values[i] += np.abs(shap_values.values[j][i] / len(shap_values))
    
    keys = list(shap_values.data[0].index)
    return pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)