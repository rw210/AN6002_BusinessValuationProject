"""
AN6002 - ANALYTICS & MACHINE LEARNING IN BUSINESS
Group Project - Team 3

"""

import numpy as np
import pandas as pd
import yfinance as yf
import os
from dash import Dash, html, dcc, dash_table, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_daq as daq
from datetime import datetime
import webbrowser
import threading
import socket
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import Lasso
# nltk.download('vader_lexicon')


# ========== Function 1: User Company Financial Summary and Valuation ==========
# Helper function to ensure valid float input
def get_non_negative_float(prompt):
    while True:
        try:
            value = float(input(prompt))
            if value < 0:
                raise ValueError("The value cannot be negative.")
            return value
        except ValueError as e:
            print(f"Invalid input: {e}. Please enter a valid non-negative number.")


# Function to fetch company info from Yahoo Finance
def get_financial_data(ticker, df):
    company = yf.Ticker(ticker)
    info = company.info
    return {
        'ticker': ticker,
        'name': info.get('longName', 'Unknown'),
        'sector': df.loc[df['Symbol'] == ticker, 'GICS Sector'].values[0],
        'market_cap': info.get('marketCap', 0),
        'revenue': info.get('totalRevenue', 0),
        'pe_ratio': info.get('forwardPE', 0),
        'debt_to_equity': info.get('debtToEquity', 0),
        'revenue_growth': info.get('revenueGrowth', 0),
        'earnings_growth': info.get('earningsGrowth', 0),
        'free_cash_flow_growth': info.get('freeCashflowGrowth', 0)
    }


# Function to find comparable companies based on user-defined data
def find_comparable_companies(user_company_data):
    df = pd.read_csv("SnP_tickers_sector.csv")
    company_tickers = df['Symbol'].tolist()

    data = [get_financial_data(ticker, df) for ticker in company_tickers]
    df = pd.DataFrame(data)

    df.fillna(0, inplace = True)

    features = ['market_cap', 'revenue', 'pe_ratio', 'revenue_growth']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(scaled_features)

    target_sector = user_company_data['industry']  # Use industry from user input

    comparable_companies = df[df['sector'] == target_sector]
    comparable_companies = comparable_companies.sort_values(by='market_cap', ascending=False).head(3)

    return comparable_companies


# Function to calculate average growth rate
def calculate_average_growth_rate(comparable_companies):
    avg_growth_rate = comparable_companies['revenue_growth'].mean()
    return avg_growth_rate


# Function to gather user input for company information
def get_company_info():
    try:
        name = input("Enter your company's name: ")
        industry = input("Enter your company's industry: ")
        marketcap = get_non_negative_float("What is your company's market capital: ")

        revenue = get_non_negative_float("Enter your company's revenue: ")
        cos = get_non_negative_float("Enter your company's cost of sale: ")
        opex = get_non_negative_float("Enter your company's operating expense: ")
        non_operating_income = get_non_negative_float("Enter your company's non-operating income: ")
        non_operating_loss = get_non_negative_float("Enter your company's non-operating loss: ")
        tax_expense = get_non_negative_float("Enter your company's tax expense: ")

        current_assets = get_non_negative_float("Enter your company's current assets: ")
        non_current_assets = get_non_negative_float("Enter your company's non-current assets: ")
        current_liabilities = get_non_negative_float("Enter your company's current liabilities: ")
        non_current_liabilities = get_non_negative_float("Enter your company's non-current liabilities: ")

        while True:
            debt = get_non_negative_float("Enter your company's debt: ")
            if debt > (current_liabilities + non_current_liabilities):
                print("Please enter a valid amount (debt should not exceed total liabilities).")
            else:
                break

        outstanding_share = get_non_negative_float("Enter your company's outstanding shares: ")

        while True:
            cash = get_non_negative_float("Enter your company's cash: ")
            if cash > current_assets:
                print("Please enter a valid amount (cash should not exceed current assets).")
            else:
                break

        total_assets = current_assets + non_current_assets
        total_liabilities = current_liabilities + non_current_liabilities
        total_equity = total_assets - total_liabilities
        total_equity_liab = total_equity + total_liabilities

        gross_profit = revenue - cos
        operating_income = gross_profit - opex
        EBT = operating_income + non_operating_income - non_operating_loss
        net_income = EBT - tax_expense

        # Financial ratios
        gross_margin = (gross_profit / revenue) * \
            100 if revenue > 0 else 0  # Avoid division by zero
        operating_margin = (operating_income / revenue) * \
            100 if revenue > 0 else 0
        EPS = net_income / outstanding_share if outstanding_share > 0 else 0
        cos_percent = cos / revenue if revenue > 0 else 0
        opex_percent = opex / revenue if revenue > 0 else 0
        net_income_percent = net_income / revenue if revenue > 0 else 0

        current_ratio = current_assets / \
            current_liabilities if current_liabilities > 0 else 0
        debt_to_equity = (current_liabilities + non_current_liabilities) / \
            total_equity if total_equity > 0 else 0
        ROE = net_income / total_equity if total_equity > 0 else 0

        # Market Ratio
        enterprise_value = debt + total_equity - cash

        # Collect all data into a dictionary
        company_data = {
            'company_name': name,
            'industry': industry,
            'marketcap': marketcap,
            'enterprise_value': enterprise_value,
            'current_ratio': current_ratio,
            'debt_to_equity': debt_to_equity,
            'ROE_percent': ROE * 100,
            'outstanding_share': outstanding_share,
            'EPS': EPS,
            'current_assets': current_assets,
            'non_current_assets': non_current_assets,
            'total_assets': total_assets,
            'current_liabilities': current_liabilities,
            'non_current_liabilities': non_current_liabilities,
            'total_liabilities': total_liabilities,
            'debt': debt,
            'cash': cash,
            'total_equity': total_equity,
            'total_equity_liab': total_equity_liab,
            'revenue': revenue,
            'cos': cos,
            'gross_profit': gross_profit,
            'gross_margin': gross_margin,
            'opex': opex,
            'operating_income': operating_income,
            'operating_margin': operating_margin,
            'non_operating_income': non_operating_income,
            'non_operating_loss': non_operating_loss,
            'EBT': EBT,
            'tax_expense': tax_expense,
            'net_income': net_income,
            'net_income_percent': (net_income / revenue) * 100 if revenue > 0 else 0
        }

        return company_data

    except Exception as e:
        print(f"An error occurred: {e}")


# DCF calculation function
def investor_dcf(user_company_data, time):
    # Step 1: Get FCF - using user input for revenue instead
    try:
        fcf = user_company_data['net_income'] + user_company_data['debt']  # Assuming FCF as net income + debt
    except (KeyError, IndexError):
        print("Free Cash Flow data not available for this company.")
        return

    # Step 2: Forecasting Free Cash Flow
    revenue_growth_rate = 0.05  # Default growth rate if not found

    projected_fcf = []
    for year in range(1, time + 1):
        fcf_next_year = fcf * (1 + revenue_growth_rate)**year
        projected_fcf.append(fcf_next_year)

    # Step 3: Discount Rate - required rate of return (use WACC)
    treasury_yield = yf.Ticker('^TNX')
    yield_data = treasury_yield.history(period='1d')
    risk_free_rate = yield_data['Close'].iloc[-1] / 100  # Most recent yield
    beta = 1  # Default beta if not provided
    market_return = 0.1  # Average market return

    total_debt = user_company_data['debt']
    market_cap = user_company_data['marketcap']
    cost_of_debt = 0.05  # Default cost of debt
    cost_of_equity = risk_free_rate + beta * (market_return - risk_free_rate)
    corporate_tax_rate = 0.21
    total_value = market_cap + total_debt
    wacc = (market_cap / total_value * cost_of_equity) + (total_debt / total_value * cost_of_debt * (1 - corporate_tax_rate))

    # Step 4: Calculate Terminal Value and discounted TV
    perpetual_growth = 0.025  # Use 2.5%
    terminal_value = projected_fcf[-1] * (1 + perpetual_growth) / (wacc - perpetual_growth)
    discounted_tv = terminal_value / (1 + wacc)**time

    # Step 5: Calculate Intrinsic Value
    discounted_fcf = []
    for year, fcf_year in enumerate(projected_fcf, start=1):
        discounted_value = fcf_year / (1 + wacc)**year
        discounted_fcf.append(discounted_value)

    # Step 6: Current Value
    current_val = discounted_tv + sum(discounted_fcf)
    # outstanding_shares = user_company_data['outstanding_share']
    # final_value = current_val / outstanding_shares if outstanding_shares > 0 else 0

    print(f"DCF Value (Total): ${current_val:,.2f}")
    return current_val


def user_company_valuation():
    try:
        user_company_data = get_company_info()
        comparable_companies = find_comparable_companies(user_company_data)

        if not comparable_companies.empty:
            print("\nTop 3 comparable companies in the same sector:")

            for index, row in comparable_companies.iterrows():
                print(f"{row['ticker']}: Market Cap: ${row['market_cap'] / 1e9:,.2f} billion, Revenue: ${row['revenue'] / 1e9:,.2f} billion")

            avg_growth_rate = calculate_average_growth_rate(comparable_companies)
            print(f"\nAverage Growth Rate of Comparable Companies: {avg_growth_rate:.2%}")

        time = int(input("Enter the forecast period (in years): "))
        dcf_value = investor_dcf(user_company_data, time)

        # HTML output
        html_output = f"""
        <!DOCTYPE html>
        <html lang = "en">
        <head>
            <meta charset = "UTF-8">
            <meta name = "viewport" content = "width = device-width, initial-scale = 1.0">
            <title>Company Financial Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                }}
                .container {{
                    max-width: 800px;
                    margin: 0 auto;
                }}
                h1, h2 {{
                    text-align: center;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                table, th, td {{
                    border: 1px solid black;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                .right-align {{
                    text-align: right;
                }}
                td:first-child {{
                text-align: left;
            }}
            .summary, .income, .balance-sheet {{
                margin-top: 20px;
            }}

            </style>

        </head>
        <body>
            <div class = "container">
                <h1>Company Financial Report</h1>
                <h2>{user_company_data['company_name']} ({user_company_data['industry']})</h2>

                <section class = "summary">
                    <h2>Basic Information</h2>
                    <p><strong>Market Capital:</strong> ${user_company_data['marketcap']:,.2f}</p>
                    <p><strong>Enterprise Value:</strong> ${user_company_data['enterprise_value']:,.2f}</p>
                    <p><strong>Current Ratio:</strong> {user_company_data['current_ratio']:,.2f}</p>
                    <p><strong>Debt-to-Equity:</strong> {user_company_data['debt_to_equity']:,.2f}</p>
                    <p><strong>ROE:</strong> {user_company_data['ROE_percent']:,.2f}%</p>
                    <p><strong>Outstanding Shares:</strong> {user_company_data['outstanding_share']:,.0f}</p>
                    <p><strong>Earnings Per Share:</strong> ${user_company_data['EPS']:,.2f}</p>
                </section>

                <section class = "balance-sheet">
                    <h2>Balance Sheet</h2>
                    <table>
                        <tr><th>Item</th><th class = "right-align">Amount</th></tr>
                        <tr><td>Current Assets</td><td class = "right-align"> ${user_company_data['current_assets']:,.2f}</td></tr>
                        <tr><td>Non-Current Assets</td><td class = "right-align"> ${user_company_data['non_current_assets']:,.2f}</td></tr>
                        <tr><td><b>Total Assets</td></b><td class = "right-align"><b> ${user_company_data['total_assets']:,.2f}</td></tr></b>
                        <tr><td>Current Liabilities</td><td class = "right-align"> ${user_company_data['current_liabilities']:,.2f}</td></tr>
                        <tr><td>Non-Current Liabilities</td><td class = "right-align"> ${user_company_data['non_current_liabilities']:,.2f}</td></tr>
                        <tr><td><b>Total Liabilities</td></b><td class = "right-align"><b> ${user_company_data['total_liabilities']:,.2f}</td></tr></b>
                        <tr><td><b>Total Equity</td></b><td class = "right-align"><b> ${user_company_data['total_equity']:,.2f}</td></tr></b>
                        <tr><td><b>Total Liabilities and Equity</td></b><td class = "right-align"><b> ${user_company_data['total_equity_liab']:,.2f}</td></tr></b>
                    </table>
                </section>

                <section class = "income">
                    <h2>Income Statement</h2>
                    <table>
                        <tr><th>Item</th><th class = "right-align">Amount</th></tr>
                        <tr><td>Revenue</td><td class = "right-align"> ${user_company_data['revenue']:,.2f}</td></tr>
                        <tr><td>Cost of Sale</td><td class = "right-align"> ${user_company_data['cos']:,.2f}</td></tr>
                        <tr><td>Gross Profit</td><td class = "right-align"> ${user_company_data['gross_profit']:,.2f}</td></tr>
                        <tr><td><i>Gross Margin</td></i><td class = "right-align"><i> {user_company_data['gross_margin']:,.2f}%</td></tr></i>
                        <tr><td>Operating Expense</td><td class = "right-align"> ${user_company_data['opex']:,.2f}</td></tr>
                        <tr><td>Operating Income</td><td class = "right-align"> ${user_company_data['operating_income']:,.2f}</td></tr>
                        <tr><td><i>Operating Margin</td></i><td class = "right-align"><i> {user_company_data['operating_margin']:,.2f}%</td></tr></i>
                        <tr><td>Non-Operating Income</td><td class = "right-align"> ${user_company_data['non_operating_income']:,.2f}</td></tr>
                        <tr><td>Non-Operating Loss</td><td class = "right-align"> ${user_company_data['non_operating_loss']:,.2f}</td></tr>
                        <tr><td>Earnings Before Tax</td><td class = "right-align"> ${user_company_data['EBT']:,.2f}</td></tr>
                        <tr><td>Tax Expense</td><td class = "right-align"> ${user_company_data['tax_expense']:,.2f}</td></tr>
                        <tr><td>Net Income</td><td class = "right-align"> ${user_company_data['net_income']:,.2f}</td></tr>
                        <tr><td><i>Net Profit Margin</td></i><td class = "right-align"><i> {user_company_data['net_income_percent']:,.2f}%</td></tr></i>
                    </table>
                </section>

                <section class="comparable">
                    <h2>Comparable Companies</h2>
                    <table>
                        <tr><th>Ticker</th><th class="right-align">Market Cap (Billion)</th><th class="right-align">Revenue (Billion)</th></tr>
                        {''.join([f'<tr><td>{row["ticker"]}</td><td class="right-align">${row["market_cap"] / 1e9:,.2f}</td><td class="right-align">${row["revenue"] / 1e9:,.2f}</td></tr>' for index, row in comparable_companies.iterrows()])}
                    </table>
                    <p><strong>Average Growth Rate of Comparable Companies:</strong> {avg_growth_rate:.2%}</p>
                </section>

                <section class="dcf-value">
                <h2>Discounted Cash Flow (DCF) Valuation</h2>
                <table>
                    <tr><th>Period</th><th class = "right-align">DCF Value</th></tr>
                    <tr><td>{time}</td><td class = "right-align"> ${dcf_value:,.2f}</td></tr>
                <table>
            </section>
            </div>
        </body>
        </html>
        """

        temp_file_path = "temp_financial_report.html"
        with open(temp_file_path, "w") as file:
            file.write(html_output)

        # Open the temporary file in the default web browser
        webbrowser.open(f'file://{os.path.realpath(temp_file_path)}')

    except Exception as e:
        print(f"An error occurred: {e}")
# ========== Function 1 End ==========


# ========== Function 2: S&P 500 Company Analysis and Valution ==========
# Get basic imformation of the company
def get_company_information(stock, df, ticker):
    try:
        info = stock.info
        company_info = {
            'Company Name': info.get('shortName', 'N/A'),
            'Address': info.get('address1', 'N/A'),
            'City': info.get('city', 'N/A'),
            'State': info.get('state', 'N/A'),
            'Zip': info.get('zip', 'N/A'),
            'Phone': info.get('phone', 'N/A'),
            'Website': info.get('website', 'N/A'),
            'GICS Sector': df.loc[ticker, 'GICS Sector'],
            'Enterprise Value': info.get('enterpriseValue', 'N/A'),
            'Market Cap': info.get('marketCap', 'N/A'),
            'Profit Margins': info.get('profitMargins', 'N/A'),
            'Shares Outstanding': info.get('sharesOutstanding', 'N/A'),
            'Current Ratio': info.get('currentRatio', 'N/A'),
            'Debt to Equity (D/E)': info.get('debtToEquity', 'N/A'),
            'Return on Equity (ROE)': info.get('returnOnEquity', 'N/A'),
            'Five-Year Average Dividend Yields': info.get('fiveYearAvgDividendYield', 'N/A'),
            'Summary': info.get('longBusinessSummary', 'N/A')
        }
        company_info = [{'Metric': key, 'Value': value}
                        for key, value in company_info.items()]
    except Exception as e:
        print(f"Error retrieving company info for {ticker}: {e}")
        company_info = [{}]

    return company_info


# Get the latest company news
def get_recent_news(stock):
    try:
        news = stock.get_news()
        recent_news = [
            {
                'title': article['title'],
                'url': article['link'],
                'publisher': article['publisher'],
                'publishTime': datetime.fromtimestamp(article['providerPublishTime']).strftime('%Y-%m-%d %H:%M:%S')
            }
            for article in news
        ]
    except Exception as e:
        print(f"Error retrieving news: {e}")
        recent_news = []

    return recent_news


# Get company financial data
def get_financials(stock):
    try:
        financials = stock.financials
        company_financials = financials.loc[[
            'EBITDA', 'Total Revenue', 'Cost Of Revenue', 'Net Income', 'Gross Profit', 'Basic EPS'], financials.columns[:4]]
        company_financials.columns = company_financials.columns.astype(str)
    except Exception as e:
        print(f"Error retrieving financials: {e}")
        company_financials = pd.DataFrame()

    return company_financials


# Get company balance sheet
def get_balance_sheet(stock):
    try:
        balance_sheet = stock.balance_sheet
        company_balance_sheet = balance_sheet.loc[[
            'Total Assets', 'Total Liabilities Net Minority Interest', 'Stockholders Equity'], balance_sheet.columns[:4]]
        company_balance_sheet.columns = company_balance_sheet.columns.astype(
            str)
    except Exception as e:
        print(f"Error retrieving balance sheet: {e}")
        company_balance_sheet = pd.DataFrame()

    return company_balance_sheet


# Get company cashfolw data
def get_cashflow(stock):
    try:
        cashflow = stock.cashflow
        company_cashflow = cashflow.loc[['Free Cash Flow', 'Investing Cash Flow',
                                         'Financing Cash Flow', 'Operating Cash Flow'], cashflow.columns[:4]]
        company_cashflow.columns = company_cashflow.columns.astype(str)
    except Exception as e:
        print(f"Error retrieving cashflow statement: {e}")
        company_cashflow = pd.DataFrame()

    return company_cashflow


# Get company historical stock prices
def get_historical_data(stock, period):
    try:
        historical_data = stock.history(period = period)
        # Reset index to get date as a column
        historical_data.reset_index(inplace = True)
        historical_data['Date'] = historical_data['Date'].dt.strftime(
            '%Y-%m-%d')  # Format date
        historical_data = round(historical_data, 3)
    except Exception as e:
        print(f"Error retrieving historical data: {e}")
        historical_data = pd.DataFrame()  # Return empty DataFrame

    return historical_data


# Get company recommendations data
def get_recommendations(stock):
    try:
        recommendations = stock.recommendations
        recommendations.set_index('period', inplace = True)
    except Exception as e:
        print(f"Error retrieving recommendations: {e}")
        recommendations = pd.DataFrame()

    return recommendations


# Get company sustainability data
def get_sustainability(stock):
    try:
        sustainability = stock.sustainability
        sustainability = sustainability.loc[[
            'environmentScore', 'governanceScore', 'socialScore', 'totalEsg']]
    except Exception as e:
        print(f"Error retrieving sustainability data: {e}")
        sustainability = pd.DataFrame()

    return sustainability


# Draw line charts for financials, balance sheet and cashflow
def create_line_chart(data, title):
    fig = go.Figure()
    for metric in data.index:  # Iterate through metrics
        fig.add_trace(go.Scatter(
            x = data.columns,  # Years are the column names
            y = data.loc[metric],  # Get the values for each metric
            mode = 'lines+markers',
            name = metric
        ))

    fig.update_layout(
        title = title,
        title_x = 0.5,
        xaxis_title = 'Year',
        yaxis_title = 'Value',
        template = 'plotly_white',
        legend = dict(
            y = 0.5,
            yanchor = 'middle',
            x = 1.05,
            xanchor = 'left'
        )
    )
    return fig


# Function to calculate compound sentiment score
def calculate_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']


# Function to analyze sentiment for a given ticker's news
def company_sentiment_of_2024(ticker):
    def getnews(ticker):
        stock = yf.Ticker(ticker)
        news = stock.get_news()
        return news

    sentiment_by_year = {}
    news = getnews(ticker)
    for article in news:
        publish_time = datetime.fromtimestamp(article['providerPublishTime'])
        year = publish_time.year
        text = article['title']

        if text:
            sentiment_score = calculate_sentiment(text)
            if year not in sentiment_by_year:
                sentiment_by_year[year] = []
            sentiment_by_year[year].append(sentiment_score)

    # Calculate average sentiment score by year
    average_sentiment_by_year = {year: sum(scores) / len(scores) for year, scores in sentiment_by_year.items()}
    return average_sentiment_by_year


def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    return {
        'current_price': stock.info.get('currentPrice'),
        'target_mean_price': stock.info.get('targetMeanPrice'),
        'target_low_price': stock.info.get('targetLowPrice'),
        'target_high_price': stock.info.get('targetHighPrice'),
        '52_week_low': stock.info.get('fiftyTwoWeekLow'),
        '52_week_high': stock.info.get('fiftyTwoWeekHigh')
    }


def plot_stock_valuation(ticker, y_pred, y_pred_adjusted, stock_pred_upper, stock_pred_lower):
    data = get_stock_data(ticker)

    # Calculate the minimum and maximum values for the prediction bar graph
    min_pred = min(np.min(y_pred), np.min(y_pred_adjusted))
    max_pred = max(np.max(y_pred), np.max(y_pred_adjusted))

    # Calculate min/max values for DCF prediction range
    min_dcf_pred = stock_pred_lower
    max_dcf_pred = stock_pred_upper

    # Define the ranges for each row, including the sentiment adjustment bar
    ranges = [
        ("52 Week Range", data['52_week_low'], data['52_week_high']),
        ("Target Price Range", data['target_low_price'], data['target_high_price']),
        ("Lasso Regression with Sentiment Adjustment", min_pred, max_pred),  # Sentiment-adjusted prediction
        ("DCF Valuation Range", min_dcf_pred, max_dcf_pred)  # Upper and Lower DCF predictions
    ]

    colors = ['#5DADE2', '#48C9B0', '#F39C12', '#E74C3C']  # Colors for the bars

    fig = go.Figure()

    for i, (label, low, high) in enumerate(ranges):
        fig.add_trace(go.Bar(
            x=[high - low],
            y=[label],
            orientation='h',
            base=low,
            marker=dict(color=colors[i]),
            text=[f'${low:.2f} to ${high:.2f}'],
            textposition='outside'
        ))

    target_mean_price = data['target_mean_price']
    current_price = data['current_price']

    fig.add_shape(type='line', x0=target_mean_price, x1=target_mean_price, y0=-0.5, y1=len(ranges) - 0.5,
                  line=dict(color='black', dash='dash'), name='Target Mean Price')
    fig.add_shape(type='line', x0=current_price, x1=current_price, y0=-0.5, y1=len(ranges) - 0.5,
                  line=dict(color='red', dash='dash'), name='Current Price')

    fig.update_layout(
        xaxis=dict(title='Price ($)', range=[0, max(data['52_week_high'], data['target_high_price'], max_pred, max_dcf_pred) * 1.1]),
        yaxis=dict(title='', tickvals=np.arange(len(ranges)), ticktext=[r[0] for r in ranges]),
        showlegend=False
    )

    return fig


# Modify the combined_investor_analysis function to return necessary values and call the plotting function
def combined_investor_analysis(ticker):
    time = 1
    stock = yf.Ticker(ticker)

    # 1. Predict stock price using Lasso regression

    # Get financial data for the past years
    financials = stock.financials.T
    balance_sheet = stock.balance_sheet.T
    cash_flow = stock.cashflow.T

    # Set year as index
    financials.index = financials.index.year
    balance_sheet.index = balance_sheet.index.year
    cash_flow.index = cash_flow.index.year

    # Combine all financial data
    all_data = pd.concat([financials, balance_sheet, cash_flow], axis=1)

    # Remove duplicate columns
    all_data = all_data.loc[:, ~all_data.columns.duplicated()]

    # Fill NaN values with 0 instead of dropping columns and remove 2019 data
    all_data = all_data.fillna(0)
    all_data = all_data[all_data.index != 2019]

    # Get historical stock prices
    stock_data = stock.history(start="2019-01-01", end="2023-12-31")
    yearly_price = stock_data['Close'].resample('Y').mean()
    yearly_price.index = yearly_price.index.year

    # Merge financial data with stock prices
    merged_data = pd.merge(all_data, yearly_price, left_index=True, right_index=True, how='inner')
    merged_data = merged_data.rename(columns={'Close': 'Average_Stock_Price'})

    # Train set, test set
    X = merged_data.drop('Average_Stock_Price', axis=1)
    y = merged_data['Average_Stock_Price']

    # Handle NaN values
    X = X.fillna(X.mean())

    # Split the data into training and testing sets
    X_train = X.loc[X.index < 2023]
    y_train = y.loc[y.index < 2023]
    X_test = X.loc[X.index == 2023]
    y_test = y.loc[y.index == 2023]

    # Lasso Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lasso = Lasso(alpha=0.1, random_state=42)
    lasso.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = lasso.predict(X_test_scaled)[0]
    print(f"Non-Adjusted stock price prediction: {y_pred}")

    # 2. Sentiment analysis for adjusting the stock price prediction
    sentiment_dict = company_sentiment_of_2024(ticker)

    # Check if sentiment for 2024 is available
    if 2024 in sentiment_dict:
        sentiment_value = sentiment_dict[2024]
        # Adjust predictions based on sentiment
        y_pred_adjusted = y_pred * (1 + sentiment_value)
        print(f"Sentiment-adjusted stock price prediction: {y_pred_adjusted}")
    else:
        print("No sentiment data available for 2024, using original prediction.")
        y_pred_adjusted = y_pred

    # 3. DCF Calculation
    cashflow = stock.cashflow
    try:
        fcf = cashflow.loc['Free Cash Flow'].iloc[0]
    except (KeyError, IndexError):
        print("Free Cash Flow data not available for this company.")
        return

    # Get beta, market cap, debt info for WACC calculation
    treasury_yield = yf.Ticker('^TNX')
    yield_data = treasury_yield.history(period='1d')
    risk_free_rate = yield_data['Close'].iloc[-1] / 100  # Most recent risk-free rate

    beta = stock.info.get('beta', 1)  # Default to 1 if not available
    market_return = 0.1
    market_cap = stock.info.get('marketCap', 0)
    total_debt = stock.info.get('totalDebt', 0)
    interest_expense = stock.info.get('interestExpense', 0)

    if total_debt > 0 and interest_expense > 0:
        cost_of_debt = interest_expense / total_debt
    else:
        cost_of_debt = 0.05  # Default value

    cost_of_equity = risk_free_rate + beta * (market_return - risk_free_rate)
    corporate_tax_rate = 0.21
    total_value = market_cap + total_debt
    wacc = (market_cap / total_value * cost_of_equity) + (total_debt / total_value * cost_of_debt * (1 - corporate_tax_rate))

    # Forecasting Free Cash Flow and Terminal Value
    CI_Lower, CI_Upper = 0.05, 0.07  # Example confidence intervals
    projected_fcf_upper = [fcf * (1 + CI_Upper) ** year for year in range(1, time + 1)]
    projected_fcf_lower = [fcf * (1 + CI_Lower) ** year for year in range(1, time + 1)]

    perpetual_growth = 0.025  # Perpetual growth rate of 2.5%
    terminal_value_upper = projected_fcf_upper[-1] * (1 + perpetual_growth) / (wacc - perpetual_growth)
    terminal_value_lower = projected_fcf_lower[-1] * (1 + perpetual_growth) / (wacc - perpetual_growth)

    discounted_tv_upper = terminal_value_upper / (1 + wacc) ** time
    discounted_tv_lower = terminal_value_lower / (1 + wacc) ** time

    # Discount future cash flows
    discounted_fcf_upper = [fcf_year / (1 + wacc) ** year for year, fcf_year in enumerate(projected_fcf_upper, start=1)]
    discounted_fcf_lower = [fcf_year / (1 + wacc) ** year for year, fcf_year in enumerate(projected_fcf_lower, start=1)]

    # Calculate intrinsic value
    intrinsic_value_upper = discounted_tv_upper + sum(discounted_fcf_upper)
    intrinsic_value_lower = discounted_tv_lower + sum(discounted_fcf_lower)

    # Enterprise Value Calculation
    balance_sheet = stock.balance_sheet
    outstanding_shares = stock.info.get('sharesOutstanding', 1)

    try:
        cash = balance_sheet.loc['Cash And Cash Equivalents'].iloc[0]
    except KeyError:
        cash = 0

    try:
        short_term_debt = balance_sheet.loc['Short Long Term Debt'].iloc[0]
    except KeyError:
        short_term_debt = 0

    try:
        long_term_debt = balance_sheet.loc['Long Term Debt'].iloc[0]
    except KeyError:
        long_term_debt = 0

    total_debt = short_term_debt + long_term_debt

    try:
        minority_interest = balance_sheet.loc['Minority Interest'].iloc[0]
    except KeyError:
        minority_interest = 0

    stock_pred_upper = (intrinsic_value_upper + cash - total_debt - minority_interest) / outstanding_shares
    stock_pred_lower = (intrinsic_value_lower + cash - total_debt - minority_interest) / outstanding_shares

    # Final results
    results = {
        "Non-Adjusted Stock Price Prediction": y_pred,
        "Sentiment-Adjusted Stock Price Prediction": y_pred_adjusted,
        "DCF Value (Upper)": f"${intrinsic_value_upper:,.2f}",
        "DCF Value (Lower)": f"${intrinsic_value_lower:,.2f}",
        "Stock Value (Upper)": f"${stock_pred_upper:,.2f}",
        "Stock Value (Lower)": f"${stock_pred_lower:,.2f}"
    }

    results = [{"Metric": key, "Value": value}
               for key, value in results.items()]

    # At the end of the function, add:
    fig = plot_stock_valuation(ticker, y_pred, y_pred_adjusted, stock_pred_upper, stock_pred_lower)

    return results, fig


# Common table styles
style_table = {
    'overflowX': 'auto',
    'margin': 'auto',
    'width': '90%',
}

style_cell = {
    'textAlign': 'center',
    'padding': '5px',
}

style_header = {
    'backgroundColor': 'rgba(0, 0, 0, 0.1)',
    'fontWeight': 'bold',
}


# Generate company report
def generate_company_report():
    # Creat a dashboard
    app = Dash(__name__, suppress_callback_exceptions = True)

    app.layout = html.Div(style = {'textAlign': 'center', 'width': '80%', 'margin': '0 auto'}, children = [
        dcc.Location(id = 'url', refresh = False),
        dcc.Input(id = 'ticker-input', type = 'text',
                  placeholder = 'Enter ticker symbol'),
        html.Button('Submit', id = 'submit-button', n_clicks = 0),
        daq.Indicator(id = 'loading-indicator', value = False),
        html.Div(id = 'report-output')
    ])

    @app.callback(
        Output('loading-indicator', 'value'),
        Output('report-output', 'children'),
        Input('submit-button', 'n_clicks'),
        Input('url', 'pathname'),
        State('ticker-input', 'value')
    )
    def update_report(n_clicks, pathname, ticker):
        if pathname and pathname.strip('/') != '':
            ticker = pathname.strip('/')

        if n_clicks > 0 and ticker:

            # Show loading indicator
            loading = True

            try:
                df = pd.read_csv('SnP_tickers_sector.csv', index_col = 0)

                ticker = ticker.strip().upper().replace(" ", "")
                if ticker not in df.index:
                    valid_tickers = ', '.join(df.index)
                    return False, html.Div([
                        f"Invalid ticker symbol '{ticker}'. Please enter the ticker from S&P 500:",
                        html.Br(),
                        html.Br(),
                        valid_tickers
                    ])

                stock = yf.Ticker(ticker)

                company_info = get_company_information(stock, df, ticker)
                recent_news = get_recent_news(stock)
                # sentiment = company_sentiment_of_2024(ticker)
                company_financials = get_financials(stock)
                company_balance_sheet = get_balance_sheet(stock)
                company_cashflow = get_cashflow(stock)
                historical_data = get_historical_data(stock, '6mo')
                recommendations = get_recommendations(stock)
                sustainability = get_sustainability(stock)
                value_prediction, fig_valuation = combined_investor_analysis(ticker)

                logo_url = f"https://logo.clearbit.com/{stock.info.get('website', 'placeholder.com')}"

                if not historical_data.empty:
                    # Create line charts
                    financials_fig = create_line_chart(
                        company_financials, 'Financials Over Years')
                    balance_sheet_fig = create_line_chart(
                        company_balance_sheet, 'Balance Sheet Over Years')
                    cashflow_fig = create_line_chart(
                        company_cashflow, 'Cashflow Statement Over Years')

                    # Calculate moving averages
                    historical_data['5avg'] = historical_data['Close'].rolling(
                        window = 5).mean()
                    historical_data['10avg'] = historical_data['Close'].rolling(
                        window = 10).mean()
                    historical_data['30avg'] = historical_data['Close'].rolling(
                        window = 30).mean()

                    # Create subplots: one row for candlestick, another for volume
                    fig_historical_data = make_subplots(rows = 2, cols = 1, shared_xaxes = True,
                                                        vertical_spacing = 0,
                                                        row_heights = [0.7, 0.3])

                    # Create the candlestick chart
                    candlestick = go.Candlestick(
                        x = historical_data['Date'],
                        open = historical_data['Open'],
                        high = historical_data['High'],
                        low = historical_data['Low'],
                        close = historical_data['Close'],
                        name = 'Candlestick',
                        increasing = dict(line = dict(color = 'green'),
                                          fillcolor = 'rgba(0, 255, 0, 0.5)'),
                        decreasing = dict(line = dict(color = 'red'),
                                          fillcolor = 'rgba(255, 0, 0, 0.5)'),
                    )

                    moving_avg_5 = go.Scatter(
                        x = historical_data['Date'],
                        y = historical_data['5avg'],
                        mode = 'lines',
                        name = '5-Day Avg',
                        line = dict(color = 'blue', width = 1.5)
                    )

                    moving_avg_10 = go.Scatter(
                        x = historical_data['Date'],
                        y = historical_data['10avg'],
                        mode = 'lines',
                        name = '10-Day Avg',
                        line = dict(color = 'purple', width = 1.5)
                    )

                    moving_avg_30 = go.Scatter(
                        x = historical_data['Date'],
                        y = historical_data['30avg'],
                        mode = 'lines',
                        name = '30-Day Avg',
                        line = dict(color = 'orange', width = 1.5)
                    )

                    historical_data['Volume Color'] = np.where(
                        historical_data['Close'] >= historical_data['Open'], 'green', 'red')

                    volume = go.Bar(
                        x = historical_data['Date'],
                        y = historical_data['Volume'],
                        name = 'Volume',
                        marker = dict(color = historical_data['Volume Color']),
                        showlegend = False
                    )

                    # Add traces to the figure
                    fig_historical_data.add_trace(candlestick, row = 1, col = 1)
                    fig_historical_data.add_trace(moving_avg_5, row = 1, col = 1)
                    fig_historical_data.add_trace(moving_avg_10, row = 1, col = 1)
                    fig_historical_data.add_trace(moving_avg_30, row = 1, col = 1)
                    fig_historical_data.add_trace(volume, row = 2, col = 1)

                    fig_historical_data.update_layout(
                        yaxis_title = 'Price',
                        xaxis2_title = 'Date',
                        yaxis2_title = 'Volume',
                        xaxis_rangeslider_visible = False,
                        template = 'plotly_white',
                        margin = dict(l = 40, r = 40, t = 40, b = 40),
                        height = 600,
                        legend = dict(yanchor = "top", y = 0.99,
                                      xanchor = "left", x = 0.01)
                    )

                # Stacked bar chart of recommendations
                if not recommendations.empty:
                    recommendations = recommendations.reset_index()
                    recommendations_melted = recommendations.melt(
                        id_vars = 'period',
                        value_vars = ['strongBuy', 'buy',
                                      'hold', 'sell', 'strongSell'],
                        var_name = 'Recommendation',
                        value_name = 'Count'
                    )

                    # Create the stacked bar chart
                    fig_recommendations = go.Figure()

                    # Define color mapping
                    color_map = {
                        'strongBuy': 'darkgreen',
                        'buy': 'green',
                        'hold': 'yellow',
                        'sell': 'orange',
                        'strongSell': 'red'
                    }

                    # Sort recommendations in desired order
                    order = ['strongSell', 'sell', 'hold', 'buy', 'strongBuy']

                    for recommendation in order:
                        filtered_data = recommendations_melted[
                            recommendations_melted['Recommendation'] == recommendation]
                        fig_recommendations.add_trace(go.Bar(
                            x = filtered_data['period'],
                            y = filtered_data['Count'],
                            name = recommendation,
                            marker_color = color_map[recommendation]
                        ))

                    fig_recommendations.update_layout(
                        barmode = 'stack',
                        xaxis_title = 'Period',
                        yaxis_title = 'Count',
                        template = 'plotly_white',
                        legend = dict(
                            y = 0.5,
                            yanchor = 'middle',
                            x = 1.05,
                            xanchor = 'left'
                        )
                    )

                # Radar chart of sustainability
                if not sustainability.empty:
                    # Define the categories and values
                    categories = ['Environment', 'Governance', 'Social']
                    values = sustainability.loc[[
                        'environmentScore', 'governanceScore', 'socialScore']].values.flatten()

                    # Create the radar chart
                    fig_sustainability = go.Figure(data = go.Scatterpolar(
                        # Closing the radar chart
                        r = values.tolist() + [values[0]],
                        theta = categories + [categories[0]],
                        fill = 'toself'
                    ))

                    # Update layout
                    fig_sustainability.update_layout(
                        polar = dict(radialaxis = dict(
                            visible = True, range = [0, max(values)])),
                        showlegend = False
                    )

                # Get similar companies
                similar_companies = df[(df.index != ticker) & (
                    df["GICS Sector"] == df.loc[ticker, 'GICS Sector'])].index

                # Prepare report content
                report_content = html.Div([
                    html.H1(f"Company Data Report for {ticker}", style = {
                            'textAlign': 'center'}),
                    html.H2("Company Information", style = {
                            'textAlign': 'center'}),
                    html.Div(
                        style = {
                            'display': 'flex',
                            'width': '100%',
                            'boxSizing': 'border-box',
                        },
                        children = [
                            html.Div(
                                style = {
                                    'flex': '0 0 68%',
                                    'border': '1px solid #ccc',
                                    'borderRadius': '5px',
                                    'padding': '10px',
                                    'height': 'auto',
                                    'maxHeight': '1000px',
                                    'overflowY': 'auto'
                                },
                                children = [
                                    dash_table.DataTable(
                                        data = company_info,
                                        columns = [
                                            {"name": "Metric", "id": "Metric"},
                                            {"name": "Value", "id": "Value"}
                                        ],
                                        style_table = {'overflowX': 'auto'},
                                        style_cell = {
                                            'padding': '5px',
                                            'textAlign': 'center',
                                            'whiteSpace': 'normal',
                                            'height': 'auto',
                                        },
                                        style_header = {
                                            'display': 'none',
                                        },
                                        style_cell_conditional = [
                                            {
                                                'if': {'column_id': 'Metric'},
                                                'fontWeight': 'bold',
                                                'backgroundColor': 'rgba(0, 0, 0, 0.1)',
                                            },
                                            {
                                                'if': {'column_id': 'Value'},
                                                'textAlign': 'left',
                                            }
                                        ],
                                        cell_selectable = False,
                                    )
                                ]
                            ),
                            html.Div(
                                style = {
                                    'border': '1px solid #ccc',
                                    'padding': '10px',
                                    'borderRadius': '5px',
                                    'maxHeight': '1000px',
                                    'overflowY': 'auto',
                                    'display': 'flex',
                                    'flexDirection': 'column',
                                },
                                children = [
                                    # Logo section
                                    html.Div(
                                        style = {'textAlign': 'center',
                                                 'marginBottom': '10px'},
                                        children = [
                                            html.Img(src = logo_url, style = {'maxHeight': '100px'}) if logo_url else html.Div(
                                                "Logo not available.")
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            html.H3("Recent News", style = {
                                                    'textAlign': 'center'}),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        style = {
                                                            'border': '1px solid #ddd',
                                                            'borderRadius': '5px',
                                                            'padding': '10px',
                                                            'margin': '10px 0',
                                                            'boxShadow': '2px 2px 5px rgba(0, 0, 0, 0.1)',
                                                        },
                                                        children = [
                                                            html.A(
                                                                article['title'],
                                                                href = article['url'],
                                                                target = "_blank",
                                                                style = {
                                                                    'textDecoration': 'none',
                                                                    'color': '#007BFF',
                                                                    'fontWeight': 'bold',
                                                                    'fontSize': '16px'
                                                                }
                                                            ),
                                                            html.P(
                                                                article['publisher'],
                                                                style = {
                                                                    'marginTop': '5px',
                                                                    'fontSize': '14px',
                                                                    'color': '#555'
                                                                }
                                                            ),
                                                            html.P(
                                                                f"Published on: {article['publishTime']}",
                                                                style = {
                                                                    'fontSize': '12px',
                                                                    'color': '#999'
                                                                }
                                                            )
                                                        ]
                                                    )
                                                    for article in recent_news
                                                ] if recent_news else [  # If recent_news is empty, show a message
                                                    html.P(
                                                        "No recent news available.",
                                                        style = {
                                                            'textAlign': 'center',
                                                            'fontSize': '14px',
                                                            'color': '#999'
                                                        }
                                                    )
                                                ],
                                                style = {'textAlign': 'left'}
                                            )
                                        ],
                                        style = {'flex': '1'}
                                    )
                                ]
                            )
                        ]
                    ),

                    html.H2("Financials", style = {'textAlign': 'center'}),
                    html.Div([
                        dash_table.DataTable(
                            data = company_financials.reset_index(drop = False)
                            .rename(columns = {'index': 'Metrics'})
                            .apply(lambda row: row.map(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) and abs(x) >= 1000 else str(x)), axis = 1)
                            .to_dict('records'),
                            columns = [{"name": str(i), "id": str(i)} for i in company_financials.reset_index(
                                drop = False).rename(columns = {'index': 'Metrics'}).columns],
                            style_table = style_table,
                            style_cell = style_cell,
                            style_header = style_header,
                            style_data_conditional = [
                                {
                                    'if': {'column_id': 'Metrics'},
                                    'fontWeight': 'bold'
                                }
                            ],
                        ) if not company_financials.empty else
                        html.Div("No financial data available.", style = {
                                 'textAlign': 'center', 'color': 'red'})
                    ]),
                    dcc.Graph(id = 'financials-chart',
                              figure = financials_fig) if not company_financials.empty else None,

                    html.H2("Balance Sheet", style = {'textAlign': 'center'}),
                    html.Div([
                        dash_table.DataTable(
                            data = company_balance_sheet.reset_index(drop = False)
                            .rename(columns = {'index': 'Metrics'})
                            .apply(lambda row: row.map(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) and abs(x) >= 1000 else str(x)), axis = 1)
                            .to_dict('records'),
                            columns = [{"name": str(i), "id": str(i)} for i in company_balance_sheet.reset_index(
                                drop = False).rename(columns = {'index': 'Metrics'}).columns],
                            style_table = style_table,
                            style_cell = style_cell,
                            style_header = style_header,
                            style_data_conditional = [
                                {
                                    'if': {'column_id': 'Metrics'},
                                    'fontWeight': 'bold'
                                }
                            ],
                        ) if not company_balance_sheet.empty else
                        html.Div("No balance sheet available.", style = {
                                 'textAlign': 'center', 'color': 'red'})
                    ]),
                    dcc.Graph(id = 'balance-sheet-chart',
                              figure = balance_sheet_fig) if not company_balance_sheet.empty else None,

                    html.H2("Cashflow Statement", style = {
                            'textAlign': 'center'}),
                    html.Div([
                        dash_table.DataTable(
                            data = company_cashflow.reset_index(drop = False)
                            .rename(columns = {'index': 'Metrics'})
                            .apply(lambda row: row.map(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) and abs(x) >= 1000 else str(x)), axis = 1)
                            .to_dict('records'),
                            columns = [{"name": str(i), "id": str(i)} for i in company_cashflow.reset_index(
                                drop = False).rename(columns = {'index': 'Metrics'}).columns],
                            style_table = style_table,
                            style_cell = style_cell,
                            style_header = style_header,
                            style_data_conditional = [
                                {
                                    'if': {'column_id': 'Metrics'},
                                    'fontWeight': 'bold'
                                }
                            ],
                        ) if not company_cashflow.empty else
                        html.Div("No cashflow statement available.", style = {
                                 'textAlign': 'center', 'color': 'red'})
                    ]),
                    dcc.Graph(
                        id = 'cashflow-chart', figure = cashflow_fig) if not company_cashflow.empty else None,

                    html.H2("Historical Data", style = {'textAlign': 'center'}),
                    html.Div(style = {'textAlign': 'center', 'margin': '20px 0'}, children = [
                        html.Div(style = {'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}, children = [
                            html.Label("Select Time Period:", style = {
                                       'marginRight': '10px'}),
                            dcc.Dropdown(
                                id = 'period-dropdown',
                                 options = [
                                     {'label': '1 Month', 'value': '1mo'},
                                     {'label': '3 Months', 'value': '3mo'},
                                     {'label': '6 Months', 'value': '6mo'},
                                     {'label': '1 Year', 'value': '1y'},
                                     {'label': '2 Years', 'value': '2y'},
                                     {'label': '5 Years', 'value': '5y'},
                                 ],
                                 value = '6mo',  # Default value
                                 placeholder = "Select a period",
                                 clearable = False,
                                 style = {'width': '200px',
                                          'display': 'inline-block'}
                                 )
                        ])
                    ]),
                    dash_table.DataTable(
                        id = 'historical-data-table',
                        columns = [
                            {"name": "Date", "id": "Date"},
                            {"name": "Open", "id": "Open"},
                            {"name": "High", "id": "High"},
                            {"name": "Low", "id": "Low"},
                            {"name": "Close", "id": "Close"},
                        ],
                        data = historical_data.to_dict('records'),
                        page_size = 10,
                        style_table = style_table,
                        style_cell = style_cell,
                        style_header = style_header,
                        style_data_conditional = [
                            {
                                'if': {'column_id': 'Date'},
                                'fontWeight': 'bold'
                            }
                        ],
                    ),
                    dcc.Graph(id = 'candlestick-chart',
                              figure = fig_historical_data),
                    html.Div(id = 'historical-data-output'),

                    html.H2("Recommendations", style = {'textAlign': 'center'}),
                    html.Div(
                        style = {
                            'display': 'flex',
                            'justifyContent': 'space-between',
                            'alignItems': 'center',
                            'width': '100%'
                        },
                        children = [
                            html.Div([
                                dash_table.DataTable(
                                    data = recommendations.to_dict('records'),
                                    columns = [{"name": str(i), "id": str(i)}
                                               for i in recommendations.columns],
                                    style_table = style_table,
                                    style_cell = style_cell,
                                    style_header = style_header,
                                ) if not recommendations.empty else
                                html.Div("No recommendations data available.", style = {
                                         'textAlign': 'center', 'color': 'red'})
                            ],
                                style = {'flex': '1', 'marginLeft': '10px', 'marginRight': '10px'}),
                            dcc.Graph(
                                id = 'recommendations-chart',
                                figure = fig_recommendations,
                                style = {'flex': '1', 'height': '400px'}
                            ) if not recommendations.empty else None
                        ]
                    ),

                    html.H2("Sustainability Data", style = {
                            'textAlign': 'center'}),
                    html.Div(
                        style = {
                            'display': 'flex',
                            'justifyContent': 'space-between',
                            'alignItems': 'center',
                            'width': '100%'
                        },
                        children = [
                            html.Div([
                                dash_table.DataTable(
                                    data = sustainability.reset_index(drop = False).rename(
                                        columns = {'index': 'Metrics'}).to_dict('records'),
                                    columns = [{"name": str(i), "id": str(i)} for i in sustainability.reset_index(
                                        drop = False).rename(columns = {'index': 'Metrics'}).columns],
                                    style_table = style_table,
                                    style_cell = style_cell,
                                    style_header = style_header,
                                    style_data_conditional = [
                                        {
                                            'if': {'column_id': 'Metrics'},
                                            'fontWeight': 'bold'
                                        }
                                    ],
                                ) if not sustainability.empty else
                                html.Div("No sustainability data available.", style = {
                                         'textAlign': 'center', 'color': 'red'})
                            ],
                                style = {'flex': '1', 'marginLeft': '10px', 'marginRight': '10px'}),
                            dcc.Graph(
                                id = 'sustainability-chart',
                                figure = fig_sustainability,
                                style = {'flex': '1', 'height': '400px'}
                            ) if not sustainability.empty else None
                        ]
                    ),

                    html.H2("Value Prediction", style = {'textAlign': 'center'}),
                    html.Div(
                        style = {
                            'width': '100%',
                            'margin': '0 auto',
                        },
                        children = [
                            dash_table.DataTable(
                                data = value_prediction,
                                columns = [
                                    {"name": "Metric", "id": "Metric"},
                                    {"name": "Value", "id": "Value"}
                                ],
                                style_table = {'overflowX': 'auto'},
                                style_cell = {
                                    'padding': '5px',
                                    'textAlign': 'center',
                                    'whiteSpace': 'normal',
                                    'height': 'auto',
                                },
                                style_header = {
                                    'display': 'none',
                                },
                                style_cell_conditional = [
                                    {
                                        'if': {'column_id': 'Metric'},
                                        'fontWeight': 'bold',
                                        'width': '40%',
                                        'backgroundColor': 'rgba(0, 0, 0, 0.1)',
                                    },
                                    {
                                        'if': {'column_id': 'Value'},
                                        'textAlign': 'center',
                                    }
                                ],
                                cell_selectable = False,
                            )
                        ]
                    ),
                    dcc.Graph(
                        id = 'value_prediction',
                        figure = fig_valuation,
                        style = {'height': '400px'}
                    ),

                    html.H2("Discover more...", style = {
                            'textAlign': 'center', 'marginTop': '100px'}),
                    html.Div(
                        id = 'similar-companies',
                        style = {
                            'display': 'flex',
                            'flexWrap': 'wrap',
                            'justifyContent': 'center',
                            'gap': '20px',
                            'margin': '20px 0'
                        },
                        children = [
                            html.Div(
                                style = {
                                    'border': '1px solid #ccc',
                                    'borderRadius': '8px',
                                    'padding': '15px',
                                    'width': '150px',
                                    'textAlign': 'center',
                                    'boxShadow': '2px 2px 10px rgba(0,0,0,0.1)',
                                    'transition': '0.3s',
                                    'cursor': 'pointer'
                                },
                                children = [
                                    dcc.Link(
                                        company,
                                        href = f"/{company}",
                                        style = {'textDecoration': 'none',
                                                 'color': '#333'}
                                    )
                                ]
                            ) for company in similar_companies
                        ]
                    ),
                ])

                loading = False

                return loading, report_content
            except Exception as e:
                loading = False
                return loading, html.Div(f"Error generating report: {e}")

        return False, html.Div("Enter a ticker symbol and click submit to generate the report.")

    @app.callback(
        [Output('historical-data-table', 'data'),
         Output('candlestick-chart', 'figure')],
        Input('period-dropdown', 'value'),
        State('ticker-input', 'value')
    )
    # Update historical data and candlestick chart when time period is changed
    def update_historical_data(selected_period, ticker):
        if ticker:
            ticker = ticker.upper()
            stock = yf.Ticker(ticker)
            historical_data = get_historical_data(stock, selected_period)

            # Calculate moving averages
            historical_data['5avg'] = historical_data['Close'].rolling(
                window = 5).mean()
            historical_data['10avg'] = historical_data['Close'].rolling(
                window = 10).mean()
            historical_data['30avg'] = historical_data['Close'].rolling(
                window = 30).mean()

            # Create subplots: one row for candlestick, another for volume
            fig = make_subplots(rows = 2, cols = 1, shared_xaxes = True,
                                vertical_spacing = 0,
                                row_heights = [0.7, 0.3])

            # Create the candlestick chart
            candlestick = go.Candlestick(
                x = historical_data['Date'],
                open = historical_data['Open'],
                high = historical_data['High'],
                low = historical_data['Low'],
                close = historical_data['Close'],
                name = 'Candlestick',
                increasing = dict(line = dict(color = 'green'),
                                  fillcolor = 'rgba(0, 255, 0, 0.5)'),
                decreasing = dict(line = dict(color = 'red'),
                                  fillcolor = 'rgba(255, 0, 0, 0.5)'),
            )

            moving_avg_5 = go.Scatter(
                x = historical_data['Date'],
                y = historical_data['5avg'],
                mode = 'lines',
                name = '5-Day Avg',
                line = dict(color = 'blue', width = 1.5)
            )

            moving_avg_10 = go.Scatter(
                x = historical_data['Date'],
                y = historical_data['10avg'],
                mode = 'lines',
                name = '10-Day Avg',
                line = dict(color = 'purple', width = 1.5)
            )

            moving_avg_30 = go.Scatter(
                x = historical_data['Date'],
                y = historical_data['30avg'],
                mode = 'lines',
                name = '30-Day Avg',
                line = dict(color = 'orange', width = 1.5)
            )

            historical_data['Volume Color'] = np.where(
                historical_data['Close'] >= historical_data['Open'], 'green', 'red')

            volume = go.Bar(
                x = historical_data['Date'],
                y = historical_data['Volume'],
                name = 'Volume',
                marker = dict(color = historical_data['Volume Color']),
                showlegend = False
            )

            # Add traces to the figure
            fig.add_trace(candlestick, row = 1, col = 1)
            fig.add_trace(moving_avg_5, row = 1, col = 1)
            fig.add_trace(moving_avg_10, row = 1, col = 1)
            fig.add_trace(moving_avg_30, row = 1, col = 1)
            fig.add_trace(volume, row = 2, col = 1)

            fig.update_layout(
                yaxis_title = 'Price',
                xaxis2_title = 'Date',
                yaxis2_title = 'Volume',
                xaxis_rangeslider_visible = False,
                template = 'plotly_white',
                margin = dict(l = 40, r = 40, t = 40, b = 40),
                height = 600,
                legend = dict(yanchor = "top", y = 0.99, xanchor = "left", x = 0.01)
            )

            return historical_data.to_dict('records'), fig

        return [], go.Figure()

    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    def open_browser(port):
        webbrowser.open_new(f"http://127.0.0.1:{port}/")

    port = find_free_port()
    # Start the timer
    threading.Timer(1, open_browser(port)).start()

    # Run the server
    app.run_server(debug = True, port = port)
# ========== Function 2 End ==========


# ========== Function 3: S&P 500 Industry Sector Analysis ==========
# Load the data
df = pd.read_csv('SnP_tickers_sector.csv')


def get_market_cap(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.info.get('marketCap', 0)
    except KeyError:
        return None


# Apply market cap function
df['Market Cap'] = df['Symbol'].apply(get_market_cap)

# Define the sector tickers
sector_tickers = {
    'Industrials': 'XLI',
    'Health Care': 'XLV',
    'Information Technology': 'XLK',
    'Utilities': 'XLU',
    'Financials': 'XLF',
    'Materials': 'XLB',
    'Consumer Discretionary': 'XLY',
    'Real Estate': 'XLRE',
    'Communication Services': 'XLC',
    'Consumer Staples': 'XLP',
    'Energy': 'XLE'
}

# Summary of sector data
sector_summary = df.groupby('GICS Sector').agg({
    'Security': 'count',
    'Market Cap': 'sum'
}).reset_index()
sector_summary.columns = ['Sector', 'Number of Companies', 'Total Market Cap']
sector_summary['Total Market Cap'] = sector_summary['Total Market Cap'].map('${:,.0f}'.format)


def get_companies_in_the_sector(sector):
    sector_companies = df[df['GICS Sector'] == sector].sort_values(by='Market Cap', ascending=False)
    tickers = sector_companies['Symbol'].tolist()

    companies_data = []
    for ticker in tickers:
        info = yf.Ticker(ticker).info
        if isinstance(info, dict):
            total_market_cap = float(sector_summary.loc[sector_summary['Sector'] == sector, 'Total Market Cap'].values[0].replace('$', '').replace(',', ''))
            companies_data.append({
                'Name': info.get('longName', 'N/A'),
                'Market Weight': f"{info.get('marketCap', 0) / total_market_cap:.2%}",
                'Market Cap': f"${info.get('marketCap', 0):,.0f}",
                'Recommendation': info.get('recommendationKey', 'N/A').capitalize()
            })
        else:
            print(f"Unexpected info format for {ticker}: {info}")

    return pd.DataFrame(companies_data)


# Generate sector report
def generate_sector_report():
    app = Dash(__name__, suppress_callback_exceptions=True)

    # Define the layout
    app.layout = html.Div(style={'textAlign': 'center', 'width': '80%', 'margin': '0 auto'}, children=[
        dcc.Location(id='url', refresh=False),
        html.Div(style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'margin': '20px 0'}, children=[
            html.Label("Select Sector:", style={'marginBottom': '10px'}),
            dcc.Dropdown(id='name-dropdown', style={'width': '300px', 'textAlign': 'left'}),
        ]),
        html.Div(id='report-output')
    ])

    # Populate dropdown with sector options
    @app.callback(
        Output('name-dropdown', 'options'),
        Input('url', 'pathname')
    )
    def populate_dropdown(pathname):
        return [{'label': name, 'value': name} for name in sector_tickers.keys()]

    # Update report output based on selected sector
    @app.callback(
        Output('report-output', 'children'),
        Input('name-dropdown', 'value'),
        Input('url', 'pathname')
    )
    def update_report(selected_name, url):
        if selected_name is None:
            return html.Div("Please select a sector from the dropdown.")

        ticker = sector_tickers[selected_name]
        tickers = [ticker, "^GSPC"]

        stock_data = pd.DataFrame()
        for ticker in tickers:
            stock_data[ticker] = yf.Ticker(ticker).history(period="6mo")['Close']

        stock_pct = stock_data.pct_change().cumsum() * 100
        summary = sector_summary[sector_summary['Sector'] == selected_name]
        largest_companies = get_companies_in_the_sector(selected_name)

        # Create stock chart
        fig = go.Figure()
        for column in stock_pct.columns:
            fig.add_trace(go.Scatter(x=stock_pct.index, y=stock_pct[column], mode='lines', name=column))

        # Create market cap heatmap
        sector_companies = df[df['GICS Sector'] == selected_name].sort_values(by='Market Cap', ascending=False)
        fig_heatmap = px.treemap(sector_companies,
                                 path=[px.Constant(selected_name), 'Symbol'],
                                 values='Market Cap',
                                 color='Market Cap',
                                 color_continuous_scale='Reds')
        fig_heatmap.update_traces(textinfo='label+value', marker=dict(line=dict(width=0)))
        fig_heatmap.update_layout(margin=dict(t=50, l=25, r=25, b=25))

        # Prepare report content
        report_content = html.Div([
            html.H1(selected_name, style={'textAlign': 'center'}),
            html.Div([
                dash_table.DataTable(
                    data=summary.to_dict('records'),
                    columns=[{"name": str(i), "id": str(i)} for i in summary.columns],
                    style_table=style_table,
                    style_cell=style_cell,
                    style_header=style_header
                ) if not summary.empty else
                html.Div("No summary data available.", style={'textAlign': 'center', 'color': 'red'})
            ]),
            html.H2(f"{selected_name} Stock Data", style={'textAlign': 'center'}),
            html.Div(style={'textAlign': 'center', 'margin': '20px 0'}, children=[
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}, children=[
                    html.Label("Select Time Period:", style={'marginRight': '10px'}),
                    dcc.Dropdown(
                        id='period-dropdown',
                        options=[
                            {'label': '1 Month', 'value': '1mo'},
                            {'label': '3 Months', 'value': '3mo'},
                            {'label': '6 Months', 'value': '6mo'},
                            {'label': '1 Year', 'value': '1y'},
                            {'label': '2 Years', 'value': '2y'},
                            {'label': '5 Years', 'value': '5y'},
                        ],
                        value='6mo',
                        placeholder="Select a period",
                        clearable=False,
                        style={'width': '200px', 'display': 'inline-block'}
                    ),
                    html.Label("Select Company:", style={'marginRight': '10px', 'marginLeft': '20px'}),
                    dcc.Dropdown(
                        id='company-dropdown',
                        options=[{'label': company, 'value': ticker} for company, ticker in zip(largest_companies['Name'], df[df['GICS Sector'] == selected_name]['Symbol'])],
                        placeholder='Select a company',
                        style={'width': '300px', 'display': 'inline-block'}
                    )
                ])
            ]),
            dcc.Graph(id='stock-chart', figure=fig),
            html.Div(id='stock-data-output'),
            html.H2("Largest Companies in the Sector", style={'textAlign': 'center'}),
            html.Div([
                dash_table.DataTable(
                    data=largest_companies.to_dict('records'),
                    columns=[{"name": str(i), "id": str(i)} for i in largest_companies.columns],
                    page_size=10,
                    style_table=style_table,
                    style_cell=style_cell,
                    style_header=style_header,
                ) if not largest_companies.empty else
                html.Div("No data available.", style={'textAlign': 'center', 'color': 'red'})
            ]),
            html.H2("Market Cap Heatmap", style={'textAlign': 'center'}),
            dcc.Graph(figure=fig_heatmap)
        ])

        return report_content

    # Update stock chart based on dropdown selections
    @app.callback(
        Output('stock-chart', 'figure'),
        Input('period-dropdown', 'value'),
        Input('name-dropdown', 'value'),
        Input('company-dropdown', 'value')
    )
    def update_chart(selected_period, selected_name, selected_company):
        if selected_name is None:
            return go.Figure()

        ticker = sector_tickers[selected_name]
        tickers = [ticker, "^GSPC"]

        if selected_company:
            tickers.append(selected_company)

        stock_data = pd.DataFrame()
        for ticker in tickers:
            stock_data[ticker] = yf.Ticker(ticker).history(period=selected_period)['Close']
        stock_pct = stock_data.pct_change().cumsum() * 100

        fig = go.Figure()
        for column in stock_pct.columns:
            fig.add_trace(go.Scatter(x=stock_pct.index, y=stock_pct[column], mode='lines', name=column))

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Percentage Change",
            legend_title="Stocks"
        )

        return fig

    def open_browser():
        webbrowser.open_new("http://127.0.0.1:8060/")

    # Start the timer
    threading.Timer(1, open_browser).start()

    # Run the server
    app.run_server(debug=True, port=8060)
# ========== Function 3 End ==========


# ========== Main Function ==========
menu = '''
    Please select an option:
    1. User Company Financial Summary and Valuation
    2. S&P 500 Company Analysis and Valution
    3. S&P 500 Industry Sector Analysis
    4. Exit
'''


def main():
    while True:
        option = input(f"{menu} \nPlease select an option: ").strip().lower()
        if option == '1':
            user_company_valuation()
        elif option == '2':
            generate_company_report()
        elif option == '3':
            generate_sector_report()
        elif option == 'x':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice, please try again.")


if __name__ == "__main__":
    main()
