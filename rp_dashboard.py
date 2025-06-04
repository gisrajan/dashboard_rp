#!/usr/bin/env python
# coding: utf-8

# To see the latest numbers please use **Ctrl + F5** or **Shift + F5**

# **S1 : Puddled Transplanted**<br>
# **S2 : Mechanical Transplanted**<br>
# **S3 : DSR**<br>
# **S4 : DSR**

# **Conditions for the irrigation recomendation :**<br>
# * Full Irrigation plots :<br>
#     * Recommendation Criteria:<br>
#         * If there are three consecutive 0 readings in the Paani Pipe readings, irrigation is recommended.<br>
# * Farmer Irrigation plots :<br>
#     * Recommendation Criteria:<br>
#         * If the cumulative rainfall in the past 10 days is less than 20mm, irrigation is recommended.<br>
#         * However, if the recommended irrigation is within 3 days of the last irrigation applied in the field, the recommended irrigation is not considered.<br>
#         * The 10-day period for cumulative rainfall calculation will be reset based on the following:<br>
#           * From the date of a valid irrigation recommendation.<br>
#           * From the date when irrigation is applied in the field.<br>
#           * Or from the 11th day after the last reset.<br>

# [Download CSV file](https://drive.google.com/uc?export=download&id=1-7ilg8GUT_Kym7QOb_cHMfHj3NSS9ykX)

import sys

required_packages = ['pandas', 'pyodk', 'plotly']

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"{package} is not installed. Installing now...")
        get_ipython().system('pip install {package}')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import requests
import pandas as pd
pd.options.mode.chained_assignment = None

import os
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
from pyodk.client import Client

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import numpy as np
import seaborn as sns
from datetime import datetime, timedelta

import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default='notebook_connected+plotly_mimetype'
from plotly.subplots import make_subplots

client = Client(config_path='/home/jovyan/pyodk_config.toml').open()

json = client.submissions.get_table(form_id = 'Pani_Pipe_daily_record_RP')["value"]
df = pd.json_normalize(json)

df =  df.loc[df['__system.reviewState'] != 'rejected'].reset_index()
df.to_csv("RP_paani_pipe.csv")

def send_dataframe_as_email(df, sender_email, receiver_email, subject, smtp_server, smtp_port, username, password):
    if not df.empty:
        html = df.to_html()

    # Create email message
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = ", ".join(receiver_emails)

    # Add HTML part to email
    part1 = MIMEText(html, 'html')
    msg.attach(part1)

    # Send email
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(username, password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_emails, text)

def is_within_three_days(date1, date2):
    delta = abs(date2 - date1)
    return delta <= timedelta(days=3)

# Function to check and append irrigation date in full irrigation plot
def farmer_irrigation(df):
    # Load or create irrigation dates DataFrame
    pred_irrigation_dates = pd.DataFrame(columns=['date'])

    start_date = df['date'].min()
    end_date = df['date'].max()

    current_date = start_date

    while current_date <= end_date:
        cumulative_rainfall = df[(df['date'] >= start_date) & (df['date'] <= current_date)]
        cumulative_rainfall =  cumulative_rainfall.sort_values(by='date')

        # Check for missing dates within the subset
        subset_complete_dates = pd.date_range(cumulative_rainfall['date'].min(), current_date)
        subset_missing_dates = subset_complete_dates[~subset_complete_dates.isin(cumulative_rainfall['date'])]
        subset_num_missing_dates = len(subset_missing_dates)
        act_len = 10-subset_num_missing_dates

        if (cumulative_rainfall['rainfall'].sum() < 20) and (len(cumulative_rainfall) >= act_len):
            irrigation_date = cumulative_rainfall['date'].max() #+ timedelta(days=1)
            # Check for previous irrigation within 3 days
            if is_within_three_days(irrigation_date, cumulative_rainfall['irrigation_date'].max()):
                start_date = cumulative_rainfall['irrigation_date'].max() + timedelta(days=1)
            else:
                pred_irrigation_dates = pd.concat([pred_irrigation_dates, pd.DataFrame([{'date': irrigation_date}])], ignore_index=True)
                start_date = irrigation_date
            current_date += timedelta(days=1)
        else:
            if (cumulative_rainfall['rainfall'].sum() >= 20):
                start_date = cumulative_rainfall['date'].max() #+ timedelta(days=1)
            current_date += timedelta(days=1)
    return pred_irrigation_dates

# Function to check and append irrigation date in farmer field
def full_irrigation(df, irrigation_dates=None):
    if irrigation_dates is None:
        irrigation_dates = pd.DataFrame(columns=['date'])

    last_irrigation_date = irrigation_dates['date'].max()

    if pd.isnull(last_irrigation_date):
        start_date = df['date'].min()
    end_date = df['date'].max()
    current_date = start_date

    while current_date <= end_date:
        consecutive_zeros = 0
        warning_issued = False

        for i in range(len(df)):
            reading_date = df['date'].iloc[i]
            if reading_date < current_date:
                continue

            reading = df['paani_pipe'].iloc[i]
            if reading == 0:
                consecutive_zeros += 1
                if consecutive_zeros == 3:
                    irrigation_date = reading_date #+ timedelta(days=1)
                    irrigation_dates = pd.concat([irrigation_dates, pd.DataFrame([{'date': irrigation_date}])], ignore_index=True)
                    current_date = irrigation_date + timedelta(days=3)
                    warning_issued = True
                    break
            else:
                consecutive_zeros = 0

        if not warning_issued:
            current_date += timedelta(days=1)

    irrigation_dates = irrigation_dates[irrigation_dates['date'] <= end_date]
    return irrigation_dates

df.loc[:, 'collectionDate'] = pd.to_datetime(df['collectionDate'], format='%Y-%m-%d')

df.rename(columns={'collectionDate':'date', 'water_level_in_mm_rain_gauge':'rainfall','D.water_level_in_cm_Pani_pipe':'paani_pipe' ,'D.date_of_irrigation':'irrigation_date'}, inplace=True)

df['irrigation_date'] = pd.to_datetime(df['irrigation_date'], format = '%Y-%m-%d')

df_rain = df[df['A.PP_raingauge']=='Raingauge']
df_rain = df[['date','rainfall']]
df_rain = df_rain.dropna()

plot_id = pd.read_csv("/home/jovyan/Plot_code.csv")
plot_id['Area_ID'] = plot_id['Area_ID'].astype('string')

df = df.merge(plot_id, left_on='A.area', right_on='Area_ID')
df = df.drop('rainfall', axis=1)

df = df.merge(df_rain, on =['date'])

df_far = df[df['hydrology']=='Farmer_Practice_Irrigation']
df_full= df[df['hydrology']=='Full_Irrigation']

# Number of rows and columns
rows = 12
cols = 2

# Fixed size for each subplot
subplot_width = 600
subplot_height = 300

# Overall figure size
fig_width = cols * subplot_width
fig_height = rows * subplot_height

# Ensure vertical_spacing is within allowable limits
max_vertical_spacing = 1 / (rows - 1) if rows > 1 else 0
vertical_spacing = min(0.1, max_vertical_spacing)  # Adjust this value if necessary

# Scenarios, replications, and plots
scenarios = ["S1", "S2", "S3", "S4"]
replications = ["R1", "R2", "R3"]
plots = ["Full Irrigation", "Farmer Practice"]

# Generate names
names = [f"{scenario}_{replication}_{plot}" for scenario in scenarios for replication in replications for plot in plots]

# Create an empty DataFrame with specified columns
irri_msg = pd.DataFrame(columns=['Scenario', 'Replication', 'Hydrology'])
data_list = []

# Create subplot figure
fig = go.Figure()

fig = make_subplots(rows=rows, cols=cols, subplot_titles=names)

irrigation_dates = pd.DataFrame(columns=['date'])
for i, v in enumerate(np.sort(df_full['Recreated_ID'].unique())):
    nd = df_full[df_full['Recreated_ID']== v]
    nd['date'] = pd.to_datetime(nd['date'], format='%Y-%m-%d')
    df_predicted = full_irrigation(nd)

    df_predicted['predicted_irrigation_date'] = 1

    if not df_predicted.empty :
        if (df_predicted['date'].iloc[-1] >= nd['date'].max()):

            scenario_value = nd['A.Scenario'].unique()
            replication_value = nd['A.Replication_no'].unique()
            hydrology_value =nd['hydrology'].unique()  # Replace with actual hydrology data

            # Create a new row as a dictionary
            new_row = {'Scenario': scenario_value, 'Replication': replication_value, 'Hydrology': hydrology_value}
            data_list.append(new_row)
            # Append the new row to the DataFrame
            irri_msg = pd.concat([irri_msg, pd.DataFrame(data_list)], ignore_index=True)

            # Merge both DataFrames
            df_combined = pd.merge(nd, df_predicted, on='date', how='left')
            df_combined['irrigation_id'] = df_combined['irrigation_date'].apply(lambda x: 1 if pd.notnull(x) else 0)

            row = i+1
            col = 1

            fig.add_trace(go.Scatter(x=df_combined['date'], y=df_combined['rainfall'],
                                     mode='lines', name='Rainfall (mm)', showlegend=False, line=dict(color='green')), row=row, col=col)
            fig.add_trace(go.Scatter(x=df_combined['date'], y=df_combined['paani_pipe'],
                                     mode='lines', name='Paani Pipe (cm)',showlegend=False, line=dict(color='orange')), row=row, col=col)

            # Add vertical lines for irrigation dates
            for date in df_combined[df_combined['irrigation_id'] == 1]['date']:
                fig.add_vline(x=date, row = row, col =col, line=dict(color='blue', width=1, dash='dash'), name='Irrigation Date')

            # Add vertical lines for predicted irrigation dates
            for date in df_combined[df_combined['predicted_irrigation_date'] == 1]['date']:
                fig.add_vline(x=date, row = row, col=col, line=dict(color='red', width=1, dash='dash'), name='Predicted Irrigation Date')

            # Update layout with the same axis titles
            fig.update_xaxes(title_text='Date', row=row, col=col)
            fig.update_yaxes(title_text='Measurement', row=row, col=col)
            fig.update_yaxes(range = [0,40])

for i, v in enumerate(np.sort(df_far['Recreated_ID'].unique())):
    nd = df_far[df_far['Recreated_ID']== v]
    nd['date'] = pd.to_datetime(nd['date'], format='%Y-%m-%d')
    nd = nd.sort_values(by='date')
    # Check and append irrigation dates
    nd['irrigation_date'].where(nd['irrigation_date'] >= nd['date'], np.nan)
    irrigation_dates = nd['irrigation_date']
    df_predicted = farmer_irrigation(nd)
    df_predicted['predicted_irrigation_date'] = 1

    if not df_predicted.empty :
        if (df_predicted['date'].iloc[-1] >= nd['date'].max()):

            scenario_value = nd['A.Scenario'].unique()
            replication_value = nd['A.Replication_no'].unique()
            hydrology_value =nd['hydrology'].unique()  # Replace with actual hydrology data

            # Create a new row as a dictionary
            new_row = {'Scenario': scenario_value, 'Replication': replication_value, 'Hydrology': hydrology_value}
            data_list.append(new_row)
            # Append the new row to the DataFrame
            irri_msg = pd.concat([irri_msg, pd.DataFrame(data_list)], ignore_index=True)

            # Merge both DataFrames
            df_combined = pd.merge(nd, df_predicted, on='date', how='left')
            df_combined['irrigation_id'] = df_combined['irrigation_date'].apply(lambda x: 1 if pd.notnull(x) else 0)

            row = i+1
            col = 2

            fig.add_trace(go.Scatter(x=df_combined['date'], y=df_combined['rainfall'],
                                     mode='lines', name='Rainfall (mm)', showlegend=False, line=dict(color='green')), row=row, col=col)
            fig.add_trace(go.Scatter(x=df_combined['date'], y=df_combined['paani_pipe'],
                                     mode='lines', name='Paani Pipe (cm)',showlegend=False, line=dict(color='orange')), row=row, col=col)

            # Add vertical lines for irrigation dates
            for date in df_combined[df_combined['irrigation_id'] == 1]['date']:
                fig.add_vline(x=date, row = row, col =col, line=dict(color='blue', width=1, dash='dash'), name='Irrigation Date')

            # Add vertical lines for predicted irrigation dates
            for date in df_combined[df_combined['predicted_irrigation_date'] == 1]['date']:
                fig.add_vline(x=date, row=row, col=col, line=dict(color='red', width=1, dash='dash'), name='Recommended Irrigation Date')

            # Update layout with the same axis titles
            fig.update_xaxes(title_text='Date', row=row, col=col)
            fig.update_yaxes(title_text='Measurement', row=row, col=col)
            fig.update_yaxes(range = [0,40])

# Add dummy traces for the legend
fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='green'), name='Rainfall (mm)'))
fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='orange'), name='Paani Pipe (cm)'))
fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='blue', width=1, dash='dash'), name='Irrigation Date'))
fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='red', width=1, dash='dash'), name='Predicted Irrigation Date'))

fig.update_layout(
    height=fig_height,
    width=fig_width,
    xaxis=dict(title="Date"),
    yaxis=dict(title="Measurement"),
    legend=dict(
        yanchor="top",
        y=1,
        xanchor="left",
        x=1.05  # Position the legend outside the plot
    ),
)

# Replace with your email credentials and SMTP server details
sender_email = "sis.patna23@gmail.com"
receiver_emails = ["h.rajan@cgiar.org"]
subject = "Urgent: Irrigate fields"
smtp_server = 'smtp.gmail.com'
smtp_port = 587
username = "sis.patna23@gmail.com"
password = "ndxw faif awkg ovnv"

send_dataframe_as_email(irri_msg, sender_email, receiver_emails, subject, smtp_server, smtp_port, username, password)

# Save the Plotly figure as HTML
pio.write_html(fig, file='index.html', auto_open=False)

import datetime
from pytz import timezone
utc_now = datetime.datetime.utcnow()

ist_now = utc_now.astimezone(timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
print('Last updated :%s' %ist_now)