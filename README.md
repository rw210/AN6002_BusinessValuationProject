# AN6002 Business Valuation Project
An AI-powered app to generate financial reports and predict stock prices for both public and private companies.

## Project Overview
This project aims to develop an AI-based Python program for business valuation. It addresses the limitations of traditional valuation models by incorporating AI-enabled forecasting, financial ratios, and news sentiment analysis to provide a more dynamic and convenient valuation method.

## Program Overview
1. **Private Company Valuation**: Predict the market cap using K-Means clustering based on features such as market cap, revenue, PE ratio, and revenue growth. Select the top three comparable companies, calculate the average revenue growth rate of these companies, and use this average growth rate as an input to the DCF model.

2. **S&P 500 Public Company Valuation**: Predict stock prices using two methods
   - Adjust the yearly revenue growth rate in the DCF model using sentiment scores.
   - Apply Lasso regression to select the top 10 predictive variables from financial data, macroeconomic indices, and market indices, then refine the predictions using sentiment scores.

3. **S&P 500 Industry Sector Analysis**: Evaluate sector-specific trends, historical stock performance, and market cap distributions through heatmaps, providing insights into overall market dynamics.

## Project Workflow
1. **Brainstorming**: Select the market and determine the structure and main features of the program.

2. **Data Collection**:  
   - **Private Companies**: Collect user inputs for company profiling.  
   - **Public Companies**: Use datasets from Hugging Face and Yahoo Finance to train sentiment classifiers and gather financial metrics from major market indices such as the S&P 500, NASDAQ, and DJI, as well as macroeconomic indicators like GDP growth rates, inflation rates, and the Federal Funds rate.

3. **Model Building**:  
   - DCF Value Prediction.  
   - Sentiment Analysis.  
   - K-Means Clustering for private companies.  
   - Lasso Regression for public companies.

4. **Dashboard Design**: Develop visualizations and structured outputs for analyzing companies and sectors.

## Contributors
- Wang Yiting (Rebecca)
- Le Ha Minh Anh (Mia) 
- Chen Hou Hsuan (Rock)   
- Ng Limeylia   
- Wu Weiqiang  

