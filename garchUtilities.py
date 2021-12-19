import pandas as pd
import numpy as np


def preprocessSPXdata(filePath, startDate, endDate):
    ## Data is obtained from https://www.wsj.com/market-data/quotes/index/SPX/historical-prices
    ## startDate and endData are string with the format "2021-10-12"
    ## Datefram is sliced from startDate + 1 to endDate
    spx = pd.read_csv(filePath)
    spx["Date"] = pd.to_datetime(spx["Date"], format = "%m/%d/%y")
    spx = spx.set_index("Date")
    return spx.loc[startDate:endDate]

def preProcessVIXData(filePath, startDate, endDate):
    ## Data is obtained from https://www.cboe.com/tradable_products/vix/vix_historical_data/
    ## startDate and endData are string with the format "2021-10-12"
    ## Datefram is sliced from startDate + 1 to endDate
    vix = pd.read_csv(filePath)
    vix["DATE"] = pd.to_datetime(vix["DATE"], format = "%m/%d/%Y")
    vix = vix.set_index("DATE")
    return vix.loc[startDate:endDate].iloc[::-1]

def combineVixSPX(spxdf, vixdf):
    if (spxdf.size != vixdf.size):
        raise Exception("combineVixSPX: Unequal size of spx and vix dataframe")
    resultdf = pd.merge(spxdf, vixdf, left_index=True, right_index=True)
    if (resultdf.isna().any().any()):
        raise Exception("combineVixSPX: the combined dataframe has nan value")
    return resultdf