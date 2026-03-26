import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Global Climate Change Dashboard")
df = pd.read_csv("GlobalLandTemperaturesByCountry.csv")
# ... analysis and interactive widgets
st.plotly_chart(fig)