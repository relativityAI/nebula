"""
I envision a UI
where you create a super finetuned profile 
with specific data sources
specific areas to look at
everything to the point
nothing left to subjectivity
Voyager can allow that level of sophistication
"""

import requests
from pprint import pprint
import pandas as pd
from datetime import datetime
import math


BASE_URL = "http://0.0.0.0:8000"


def call(url, params):
    return requests.get(url, params=params).json()


def nse_announcements_search(
    symbol:str, 
    keywords: str,
    cutoff_date : str = datetime.utcnow().strftime("%Y-%m-%d")
    ):

    return call(
        f"{BASE_URL}/nse-announcements-search",
        params = {
            "symbol" : symbol,
            "keywords" : keywords,
            'cutoff_date' : cutoff_date
        }
        )


def nse_announcements_download(symbol : str):
    return call(
        f"{BASE_URL}/nse-announcements-download",
        params = {
            "symbol" : symbol,
        }
        )


def nse_announcement_extract(path_or_url : str):
    return call(
        f"{BASE_URL}/nse-announcement-extract",
        params = {
            "path_or_url" : path_or_url,
        }
        )

def nse_annual_reports_list(symbol: str):
    return call(
        f"{BASE_URL}/nse-annual-reports-list",
        params = {
            'symbol': symbol
        }
    )

def nse_list_annual_report_sections(path_or_url : str):
    return call(
        f"{BASE_URL}/nse-list-annual-report-sections",
        params = {
            'path_or_url': path_or_url
        }
    )
    

def nse_annual_report_section_download(
    path_or_url: str,
    keywords : str = "management discussion analysis",
    lag: int = 0
    ):

    return call(
        f"{BASE_URL}/nse-annual-report-section-download",
        params = {
            "path_or_url" : path_or_url,
            "keywords" : keywords,
            "lag" : lag
        }
        )


def nse_financials(
        symbol:str, 
        start:str = None, 
        end:str = None, 
        consolidated=True
    ):

    data = call(
        f"{BASE_URL}/nse-financials",
        params = {
            "symbol" : symbol,
            "start" : start,
            "end" : end,
            "consolidated" : consolidated 
        }
        )

    data = list(data)
    df = pd.json_normalize(data, record_path = ['financials'], meta=['date'])
    df['month'] = df['date'].apply(lambda x : datetime.strptime(x, "%Y-%m-%d").month )
    return df


def nse_shareholdings(
        symbol:str, 
        start:str = None, 
        end:str = None, 
    ):

    data = call(
        f"{BASE_URL}/nse-shareholdings",
        params = {
            "symbol" : symbol,
            "start" : start,
            "end" : end,
        }
        )

    data = list(data)
    df = pd.json_normalize(data, record_path = ['financials'], meta=['date'])
    df['month'] = df['date'].apply(lambda x : datetime.strptime(x, "%Y-%m-%d").month )
    return df

def nse_preprocess_financials(pt : pd.DataFrame):


    if 'RevenueFromOperations' in pt.columns:
        pt['revenue'] = pt['RevenueFromOperations'].astype('float')
    elif 'InterestEarned' in pt.columns:
        pt['revenue'] = pt['InterestEarned'].astype('float')
    else:
        pt['revenue'] = pd.NA  

    pt['operating_expenses'] = pt['Expenses'].astype('float') - pt['FinanceCosts'].astype('float') - pt['DepreciationDepletionAndAmortisationExpense'].astype('float')
    pt['operating_expenses_percentage'] = 100 * pt['operating_expenses'] / pt['revenue']
    pt['operating_profit'] = pt['revenue'] -  pt['operating_expenses']
    pt['pbt'] = pt['ProfitBeforeTax'].astype('float')
    pt['pat'] = pt['ProfitLossForPeriod'].astype('float')
    pt['tax'] = pt['TaxExpense'].astype('float')
    pt['tax_percentage'] = 100*pt['tax']/ pt['pbt']
    pt['material_cost'] = pt['CostOfMaterialsConsumed'].astype('float')
    pt['material_cost_percentage'] = 100 * pt['material_cost'] / pt['revenue']
    pt['employee_cost'] = pt['EmployeeBenefitExpense'].astype('float')
    pt['employee_cost_percentage'] = 100 * pt['employee_cost'] / pt['revenue']
    pt['interest'] = pt['FinanceCosts'].astype('float')
    pt['depreciation'] = pt['DepreciationDepletionAndAmortisationExpense'].astype('float')
    # pt['assets'] = pt['NetSegmentAssets'].astype(float) 
    # pt['liabilities'] = pt['NetSegmentLiabilities'].astype('float')
    pt['eps'] = pt['DilutedEarningsLossPerShareFromContinuingAndDiscontinuedOperations'].astype('float')

    if 'NetSegmentAssets' in pt.columns:
        pt['assets'] = pd.to_numeric(pt['NetSegmentAssets'], errors='coerce')

    elif 'Assets' in pt.columns:
        pt['assets'] = pd.to_numeric(pt['Assets'], errors='coerce')

    elif {'CurrentAssets', 'NoncurrentAssets'}.issubset(pt.columns):
        pt['assets'] = (
            pd.to_numeric(pt['CurrentAssets'], errors='coerce') +
            pd.to_numeric(pt['NoncurrentAssets'], errors='coerce')
        )

    else:
        pt['assets'] = pd.NA  # or raise/log if you prefer

    
    if 'NetSegmentLiabilities' in pt.columns:
        pt['liabilities'] = pd.to_numeric(pt['NetSegmentLiabilities'], errors='coerce')

    elif 'Liabilities' in pt.columns:
        pt['liabilities'] = pd.to_numeric(pt['Liabilities'], errors='coerce')

    elif {'CurrentLiabilities', 'NoncurrentLiabilities'}.issubset(pt.columns):
        pt['liabilities'] = (
            pd.to_numeric(pt['CurrentLiabilities'], errors='coerce') +
            pd.to_numeric(pt['NoncurrentLiabilities'], errors='coerce')
        )

    else:
        pt['liabilities'] = pd.NA  # or raise/log if you prefer


    pt['equity'] = pt['assets'] = pt['liabilities']

    return pt


def nse_profitability_quarterly(df: pd.DataFrame):

    filtered_df = df[ 
        # (df['tag'].isin(filter_columns)) 
        # &
        (df['contextRef']=='OneD')
        |
        (df['contextRef'] == 'OneI')

        # (df['contextRef'].isin(['FourD']))
        # &
        # (df['month']==12 )

        ]

    # pprint(filtered_df['tag'].to_list())

    pt = filtered_df.pivot_table(
            index="date",                   # rows
            columns="tag",                  # new columns
            values="value",                 # cell values
            aggfunc="first"                 # in case of duplicates
        )


    # prep

    pt = nse_preprocess_financials(pt)

    # ratios

    pt['opm'] = pt['operating_profit'] / pt['revenue']
    pt['pat_margin'] = pt['pat'] / pt['revenue']
    pt['employee_cost_margin'] = pt['employee_cost'] / pt['revenue']
    pt['material_cost_margin'] = pt['material_cost'] / pt['revenue']
    pt['tax_percentage'] = pt['tax'] / pt['pbt']
    pt['roa'] = pt['pat'] / pt['assets']

    # pt['return_on_equity'] = pt['ProfitLossForPeriod'].astype('float') / (pt['EquityShareCapital'].astype('float') + pt['OtherEquity'].astype('float') )


    final_columns = [
        'opm',
        'pat_margin',
        'employee_cost_margin',
        'roa',
    ]

    return pt[final_columns].round(2)


def nse_profitability_annual(df: pd.DataFrame):

    filtered_df = df[ 
        (
        (df['contextRef'] == 'OneI')
        |
        (df['contextRef'] == 'FourD')
        )
        &
        (
        (df['month']==3 )
        |
        (df['month']==9 )
        )
        ]


    # pprint(filtered_df['tag'].to_list())

    pt = filtered_df.pivot_table(
            index="date" ,   # rows
            columns="tag",                  # new columns
            values="value",                 # cell values
            aggfunc="first"                 # in case of duplicates
        )

    # prep

    pt = nse_preprocess_financials(pt)


    # ratios

    pt['opm'] = pt['operating_profit'] / pt['revenue']
    pt['pat_margin'] = pt['pat'] / pt['revenue']
    pt['employee_cost_margin'] = pt['employee_cost'] / pt['revenue']
    pt['material_cost_margin'] = pt['material_cost'] / pt['revenue']
    pt['tax_percentage'] = pt['tax'] / pt['pbt']
    pt['roa'] = pt['pat'] / pt['assets']
    pt['roe'] = pt['pat'] / pt['equity']


    final_columns = [
        'opm',
        'pat_margin',
        'employee_cost_margin',
        'roa',
        'roe',
    ]


    return pt[final_columns].round(2)

def calc_cagr(df, col_name : str, years : int = 1):
    df[f'{col_name}_cagr_{years}y'] = ( df[col_name] / df[col_name].shift( 4*years ) ) ** (1/years) - 1

def nse_growth_quarterly(df : pd.DataFrame):

    filtered_df = df[ 
        # (df['tag'].isin(filter_columns)) 
        # &
        (df['contextRef']=='OneD')
        |
        (df['contextRef'] == 'OneI')

        # (df['contextRef'].isin(['FourD']))
        # &
        # (df['month']==12 )

        ]

    # pprint(filtered_df['tag'].to_list())

    pt = filtered_df.pivot_table(
            index="date",                   # rows
            columns="tag",                  # new columns
            values="value",                 # cell values
            aggfunc="first"                 # in case of duplicates
        )

    df = nse_preprocess_financials(pt)

    df.index = pd.to_datetime(df.index)

    df['revenue_qoq'] = df['revenue'].pct_change(periods = 1)
    df['revenue_ttm'] = df['revenue'].rolling(4).sum()
    df['revenue_yoy'] = df['revenue'].pct_change(periods = 4)

    df['pat_qoq'] = df['pat'].pct_change(periods = 1)
    df['pat_ttm'] = df['pat'].rolling(4).sum()
    df['pat_yoy'] = df['pat'].pct_change(periods = 4)

    df['eps_qoq'] = df['eps'].pct_change(periods = 1)
    df['eps_ttm'] = df['eps'].rolling(4).sum()
    df['eps_yoy'] = df['eps'].pct_change(periods = 4)

    

    calc_cagr(df, 'revenue_ttm', 1)
    calc_cagr(df, 'revenue_ttm', 3)
    calc_cagr(df, 'revenue_ttm', 5)
    calc_cagr(df, 'revenue_ttm', 10)

    calc_cagr(df, 'pat_ttm', 1)
    calc_cagr(df, 'pat_ttm', 3)
    calc_cagr(df, 'pat_ttm', 5)
    calc_cagr(df, 'pat_ttm', 10)

    calc_cagr(df, 'eps_ttm', 1)
    calc_cagr(df, 'eps_ttm', 3)
    calc_cagr(df, 'eps_ttm', 5)
    calc_cagr(df, 'eps_ttm', 10)


    final_columns = [
        'revenue_qoq',
        'revenue_yoy',
        'pat_qoq',
        'pat_yoy',
        'eps_ttm',
        'eps_qoq',
        'eps_yoy',
        'revenue_ttm_cagr_1y',
        'revenue_ttm_cagr_3y',
        'revenue_ttm_cagr_5y',
        'revenue_ttm_cagr_10y',
        'pat_ttm_cagr_1y',
        'pat_ttm_cagr_3y',
        'pat_ttm_cagr_5y',
        'pat_ttm_cagr_10y',
        'eps_ttm_cagr_1y',
        'eps_ttm_cagr_3y',
        'eps_ttm_cagr_5y',
        'eps_ttm_cagr_10y',
    ]

    return df[final_columns].round(2)
    
    # df['revenue_cagr_1y'] = 
    # df['revenue_cagr_3y'] = 
    # df['revenue_cagr_5y'] = 
    # df['revenue_cagr_10y'] = 

    # pat
    # eps
    # margin growth
    
def nse_valuations(financials_df : pd.DataFrame, prices_df: pd.DataFrame):

    symbol = prices_df.columns[0][-1]
    prices_df.columns = prices_df.columns.get_level_values(0)
    prices_df.index = pd.to_datetime(prices_df.index)
    financials_df.index = pd.to_datetime(financials_df.index)

    filtered_df = financials_df[ 
        # (df['tag'].isin(filter_columns)) 
        # &
        (financials_df['contextRef']=='OneD')
        |
        (financials_df['contextRef'] == 'OneI')

        # (df['contextRef'].isin(['FourD']))
        # &
        # (df['month']==12 )

        ]

    pt = filtered_df.pivot_table(
            index="date",                   # rows
            columns="tag",                  # new columns
            values="value",                 # cell values
            aggfunc="first"                 # in case of duplicates
        )


    # # prep
    pt = nse_preprocess_financials(pt)
    pt.index = pd.to_datetime(pt.index)


    pt['eps_ttm'] = pt['eps'].rolling(4).sum()
    calc_cagr(pt, 'eps_ttm', 3)
    calc_cagr(pt, 'eps_ttm', 5)
    pt['shares_outstanding'] = pt['pat'] / pt['eps']
    pt['revenue_ttm'] = pt['revenue'].rolling(4).sum()


    df = pd.merge_asof(
        prices_df.reset_index(),
        pt.reset_index(),
        left_on="Date",
        right_on="date",
        direction="backward"
    ).set_index("Date")

    df.rename(columns={'Close': 'price',}, inplace=True)

    df['pe'] = df['price']/ df['eps_ttm']
    df['peg_3y'] = df['pe'] / (df['eps_ttm_cagr_3y'] *100)
    df['peg_5y'] = df['pe'] / (df['eps_ttm_cagr_5y']*100)
    df['mcap'] = df['price'] * df['shares_outstanding'] 
    df['ps'] = df['mcap'] / df['revenue_ttm']

    final_columns = [
        'mcap',
        'pe',
        'ps',
        'peg_3y',
        'peg_5y',
    ]

    final_df = df[final_columns].round(2)
    return final_df

THRESHOLDS = {

                "opm":{"type":"higher","midpoint":0.15,"weight":1.2},
                "pat_margin":{"type":"higher","midpoint":0.12,"weight":1.1},
                "roa":{"type":"higher","midpoint":0.10,"weight":1.0},
                "roe":{"type":"higher","midpoint":0.18,"weight":1.2},
                "employee_cost_margin":{"type":"lower","midpoint":0.06,"weight":0.8},

                "pe":{"type":"lower","midpoint":20,"weight":0.9},
                "ps":{"type":"lower","midpoint":4,"weight":0.8},
                "peg_3y":{"type":"lower","center":1.0,"width":0.6,"weight":0.9},
                "peg_5y":{"type":"lower","center":1.0,"width":0.6,"weight":0.9},

                "revenue_qoq":{"type":"higher","midpoint":0.08,"weight":0.7},
                "revenue_yoy":{"type":"higher","midpoint":0.15,"weight":1.0},
                "revenue_ttm_cagr_1y":{"type":"higher","midpoint":0.20,"weight":1.1},
                "revenue_ttm_cagr_3y":{"type":"higher","midpoint":0.20,"weight":1.1},
                "revenue_ttm_cagr_5y":{"type":"higher","midpoint":0.20,"weight":1.1},

                "eps_qoq": {"type":"higher","midpoint":0.08,"weight":0.7},
                "eps_yoy": {"type":"higher","midpoint":0.15,"weight":1.0},
                "eps_ttm_cagr_1y": {"type":"higher","midpoint":0.25,"weight":1.0},
                "eps_ttm_cagr_3y": {"type":"higher","midpoint":0.25,"weight":1.0},
                "eps_ttm_cagr_5y": {"type":"higher","midpoint":0.25,"weight":1.0},

                "pat_qoq": {"type":"higher","midpoint":0.08,"weight":0.7},
                "pat_yoy": {"type":"higher","midpoint":0.15,"weight":1.0},
                "pat_ttm_cagr_1y": {"type":"higher","midpoint":0.25,"weight":1.0},
                "pat_ttm_cagr_3y": {"type":"higher","midpoint":0.25,"weight":1.0},
                "pat_ttm_cagr_5y": {"type":"higher","midpoint":0.25,"weight":1.0}

            }

from src.technicals import get_price_data
from nebula.utils.score import *

def nse_financials_analysis(symbol: str, thresholds=THRESHOLDS):
    df = nse_financials(symbol)

    profitability_q = nse_profitability_quarterly(df)
    profitability_a = nse_profitability_annual(df)
    growth_q = nse_growth_quarterly(df)
    valuations_full = nse_valuations(df, get_price_data(f"{symbol}.NS", period='3y'))

    prof_q = profitability_q.iloc[-1].dropna().to_dict()
    prof_a = profitability_a.iloc[-1].dropna().to_dict()
    growth = growth_q.iloc[-1].dropna().to_dict()
    valuations = valuations_full.iloc[-1].dropna().to_dict()

    all_metrics = {
        **prof_q,
        **prof_a,
        **growth,
        **valuations
    }

    score = score_metrics(all_metrics, thresholds=thresholds)

    return {
        "symbol": symbol,
        "metrics": all_metrics,
        "score": score
    }



if __name__ == "__main__":
    # https://www.geeksforgeeks.org/python/converting-nested-json-structures-to-pandas-dataframes/
    # https://www.geeksforgeeks.org/pandas/ways-to-filter-pandas-dataframe-by-column-values/

    # symbol = "KEI"
    # df = financials_df = nse_financials(symbol)
    # profitability = nse_profitability_annual(df)
    # profitability = nse_profitability_quarterly(df)
    # growth = nse_growth_quarterly(df)
    # print(profitability)
    # print(growth)


    from technicals import get_price_data
    # prices_df = get_price_data(f"{symbol}.NS", period='3y')
    # valuations = nse_valuations(financials_df, prices_df )
    # pprint(valuations)

    # nse_financials_analysis('POLYCAB')


    symbols = [
        'ABCAPITAL',
        'BALUFORGE',
        'BHARTIARTL',
        'BSE',
        'COFORGE',
        'CUMMINSIND',
        'DIXON',
        'ETERNAL',
        'FORTIS',
        'KALYANKJIL',
        'KEI',
        'MAXHEALTH',
        'PAYTM',
        'PERSISTENT',
        'POLYCAB',
        'SUZLON',
        'TRENT',
        'UNOMINDA'
        ]

    for symbol in symbols:
        score = nse_financials_analysis(symbol)['score']['composite_score']
        print(f'{int(score*100)} : {symbol}')