# Auto-generated Python script from Jupyter Notebook

# Core libraries
import streamlit as st
import pandas as pd
import numpy as np

# Data visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Geospatial (optional)
import folium
from folium.plugins import HeatMap

# Machine learning / stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Settings
matplotlib inline
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Load the Global Land Temperatures by Country dataset
url = "GlobalLandTemperaturesByCountry.csv"
df = pd.read_csv(url)

# Quick look
print(df.shape)
df.head()

# Basic info
df.info()
df.describe()

# Convert date column to datetime
df['dt'] = pd.to_datetime(df['dt'])

# Extract year for easier analysis
df['year'] = df['dt'].dt.year

# Drop rows with missing temperatures (about 10% of data)
df_clean = df.dropna(subset=['AverageTemperature']).copy()

# Remove extreme outliers if any (e.g., temperatures outside plausible range)
df_clean = df_clean[(df_clean['AverageTemperature'] > -50) & 
                    (df_clean['AverageTemperature'] < 60)]

print(f"Remaining rows: {len(df_clean):,}")

# Group by year and compute global mean temperature
global_yearly = df_clean.groupby('year')['AverageTemperature'].mean().reset_index()

# Plot with Seaborn
plt.figure(figsize=(12,6))
sns.lineplot(data=global_yearly, x='year', y='AverageTemperature')
plt.title('Global Average Land Temperature (1750–2013)', fontsize=14)
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.show()

fig = px.line(global_yearly, x='year', y='AverageTemperature',
              title='Global Average Land Temperature Over Time',
              labels={'AverageTemperature':'°C', 'year':'Year'},
              template='plotly_dark')
fig.show()

country_avg = df_clean.groupby('Country')['AverageTemperature'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,6))
sns.barplot(x=country_avg.values, y=country_avg.index, palette='Reds_r')
plt.title('Top 10 Warmest Countries (Overall Average)')
plt.xlabel('Average Temperature (°C)')
plt.show()

# Baseline period
baseline = df_clean[(df_clean['year'] >= 1951) & (df_clean['year'] <= 1980)]
baseline_avg = baseline.groupby('Country')['AverageTemperature'].mean().reset_index()
baseline_avg.rename(columns={'AverageTemperature':'baseline_temp'}, inplace=True)

# Merge baseline into main data
df_clean = df_clean.merge(baseline_avg, on='Country', how='left')
df_clean['anomaly'] = df_clean['AverageTemperature'] - df_clean['baseline_temp']

# Pivot for heatmap: years vs countries (top 10 most populated countries, etc.)
# For demonstration, we'll take a subset of major countries
top_countries = ['United States', 'China', 'India', 'Brazil', 'Russia', 'Germany', 'France', 'Canada', 'Australia', 'South Africa']
df_heat = df_clean[df_clean['Country'].isin(top_countries)]
heat_data = df_heat.pivot_table(index='year', columns='Country', values='anomaly', aggfunc='mean')

# Plot heatmap
plt.figure(figsize=(14,10))
sns.heatmap(heat_data, cmap='coolwarm', center=0, annot=False, cbar_kws={'label':'Temperature Anomaly (°C)'})
plt.title('Annual Temperature Anomalies by Country (relative to 1951–1980)')
plt.ylabel('Year')
plt.xlabel('Country')
plt.show()

# Compute average temperature per country
country_avg = df_clean.groupby('Country')['AverageTemperature'].mean().reset_index()

# Load a world geometry (GeoJSON) – we'll use a built‑in one from Folium or an external file.
# For simplicity, we'll use a pre‑stored GeoJSON (you can download from https://github.com/python-visualization/folium/tree/main/examples/data)
# In practice, you might use 'geopandas' and 'folium'.

# For demonstration, we'll create a choropleth using plotly instead (easier)
fig = px.choropleth(country_avg,
                    locations='Country',
                    locationmode='country names',
                    color='AverageTemperature',
                    hover_name='Country',
                    color_continuous_scale='RdBu_r',
                    title='Average Land Temperature by Country')
fig.show()

# This requires a world GeoJSON file. Example:
# world_geo = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json'
# m = folium.Map(location=[0,0], zoom_start=2)
# folium.Choropleth(geo_data=world_geo, data=country_avg,
#                   columns=['Country','AverageTemperature'],
#                   key_on='feature.properties.name',
#                   fill_color='YlOrRd', fill_opacity=0.7,
#                   line_opacity=0.2,
#                   legend_name='Average Temperature (°C)').add_to(m)
# m

# Prepare time series (yearly global average)
ts = df_clean.groupby('year')['AverageTemperature'].mean()
ts.index = pd.date_range(start=str(ts.index.min()), periods=len(ts), freq='Y')

# Fit Holt‑Winters model (additive trend, multiplicative seasonality not needed as yearly)
model = ExponentialSmoothing(ts, trend='add', seasonal=None, initialization_method='estimated')
fit = model.fit()

# Forecast next 10 years
forecast = fit.forecast(10)

# Plot results
plt.figure(figsize=(12,6))
plt.plot(ts.index, ts, label='Historical')
plt.plot(forecast.index, forecast, label='Forecast', linestyle='--', color='red')
plt.title('Global Average Temperature Forecast (Holt‑Winters)')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()

# Print forecasted values
print(forecast)

# Prepare data
X = ts.index.year.values.reshape(-1,1)
y = ts.values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predictions
y_pred = lr.predict(X_test)

# Evaluation
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f} °C")
print(f"R²: {r2_score(y_test, y_pred):.3f}")

# Forecast future
future_years = np.arange(ts.index.year.max()+1, ts.index.year.max()+11).reshape(-1,1)
future_temp = lr.predict(future_years)
print("Forecast:", future_temp)


st.title("Global Climate Change Dashboard")
df = pd.read_csv("GlobalLandTemperaturesByCountry.csv")
# ... analysis and interactive widgets
st.plotly_chart(fig)
