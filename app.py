# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 12:28:49 2025

@author: reddy
"""

# AI-Powered Market Access & Transport Optimizer for Farmers (Karnataka Version)

import streamlit as st
st.set_page_config(page_title="AI Market Access & Transport", layout="wide")

import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
from prophet import Prophet
from datetime import timedelta
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load Data
@st.cache_data
def load_data():
    part1 = pd.read_csv(r"C:\Users\chethan\Desktop\Market Access\split_part1.csv")
    part2 = pd.read_csv(r"C:\Users\chethan\Desktop\Market Access\split_part2.csv")
    df = pd.concat([part1, part2], ignore_index=True)
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    return df

data = load_data()

st.title("🚜 AI Market Access & Transport Optimizer for  Farmers")

# Sidebar - User Input
st.sidebar.header("Select Your District and Crop")
districts = data['district'].unique()
selected_district = st.sidebar.selectbox("District", districts)
district_data = data[data['district'] == selected_district]
crops = district_data['crop'].unique()
selected_crop = st.sidebar.selectbox("Crop", crops)

selected_data = district_data[district_data['crop'] == selected_crop]

# Prophet Forecasting
st.subheader(f"📈 Price Forecast for {selected_crop} in {selected_district}")

selected_data['ds'] = pd.to_datetime(selected_data['date'])
selected_data['y'] = selected_data['price']

# Allow year selection up to 2050
selected_year = st.selectbox("Select forecast year (up to 2050)", list(range(2025, 2051)))
future_days = (pd.Timestamp(f"{selected_year}-12-31") - selected_data['ds'].max()).days

# Add extra regressors if available
regressors = []
if 'rainfall' in selected_data.columns:
    regressors.append('rainfall')
if 'festival' in selected_data.columns:
    regressors.append('festival')
if 'demand_index' in selected_data.columns:
    regressors.append('demand_index')

model = Prophet()
for reg in regressors:
    model.add_regressor(reg)

model.fit(selected_data[['ds', 'y'] + regressors])

# Future DataFrame
future = pd.date_range(start=selected_data['ds'].max() + timedelta(days=1), periods=future_days)
future_df = pd.DataFrame({'ds': future})

for reg in regressors:
    future_df[reg] = selected_data[reg].mean()

forecast = model.predict(future_df)

# Display Predictions
forecast_display = forecast[['ds', 'yhat']].copy()
forecast_display['ds'] = forecast_display['ds'].dt.date
forecast_display.columns = ['Date', 'Forecasted Price']
st.dataframe(forecast_display.head(7), use_container_width=True)

# Summary Metrics
st.markdown("### 📊 Forecast Summary")
tomorrow_price = forecast_display.iloc[0]['Forecasted Price']
day_after_price = forecast_display.iloc[1]['Forecasted Price']
week_avg = forecast_display.iloc[:7]['Forecasted Price'].mean()
month_avg = forecast_display['Forecasted Price'].mean()

st.metric(label="🌤️ Tomorrow's Price", value=f"₹{tomorrow_price:.2f}")
st.metric(label="🌦️ Day After Tomorrow", value=f"₹{day_after_price:.2f}")
st.metric(label="📅 7-Day Avg Price", value=f"₹{week_avg:.2f}")
st.metric(label="🗓️ Monthly Avg Price", value=f"₹{month_avg:.2f}")

# Forecast Accuracy
if len(selected_data) >= 30:
    train = selected_data.iloc[:-7]
    test = selected_data.iloc[-7:]

    model_eval = Prophet()
    for reg in regressors:
        model_eval.add_regressor(reg)

    model_eval.fit(train[['ds', 'y'] + regressors])
    future_test = test[['ds']].copy()
    for reg in regressors:
        future_test[reg] = test[reg]

    forecast_test = model_eval.predict(future_test)
    y_true = test['y'].values
    y_pred = forecast_test['yhat'].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    st.markdown("### ✅ Forecast Accuracy")
    col1, col2 = st.columns(2)
    col1.metric(label="📉 MAE", value=f"₹{mae:.2f}")
    col2.metric(label="📈 RMSE", value=f"₹{rmse:.2f}")

# Forecast Plot
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                    subplot_titles=("📈 Historical vs Forecasted Price", "📊 Forecast Component Trends"))
fig.add_trace(go.Scatter(x=selected_data['ds'], y=selected_data['y'], mode='lines+markers', name='Historical'), row=1, col=1)
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecasted Price', line=dict(color='orange')), row=1, col=1)

fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Price (₹)',
    template='plotly_white',
    showlegend=True,
    height=700
)
st.plotly_chart(fig, use_container_width=True)

# Suggested Transport Route
st.subheader("🚚 Suggested Transport Route")
st.markdown("_(Using OpenRouteService)_")

location_data = pd.read_csv(r"C:\Users\chethan\Desktop\Market Access\karnataka_district_mandis2.csv")
mandi_location = location_data[location_data['district'] == selected_district].iloc[0]
lat, lon = mandi_location['latitude'], mandi_location['longitude']

m = folium.Map(location=[lat, lon], zoom_start=10)
folium.Marker([lat, lon], popup=f"{selected_district} Mandi", tooltip="Recommended Mandi", icon=folium.Icon(color='green')).add_to(m)
folium_static(m)

# Local Buyer Info
st.subheader("🧑‍🌾 Local Buyers in Your District")
buyers_df = pd.read_csv(r"C:\Users\chethan\Desktop\Market Access\karnataka_crop_buyers.csv")
district_buyers = buyers_df[(buyers_df['district'] == selected_district) & (buyers_df['Crop'].str.lower() == selected_crop.lower())]
st.dataframe(district_buyers, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ for Farmers | Powered by Prophet & OpenRouteService")
