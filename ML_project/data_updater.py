import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
import os

def get_bitcoin_difficulty():
    """Get Bitcoin mining difficulty from Blockchain.info API"""
    try:
        response = requests.get('https://blockchain.info/q/difficulty')
        return float(response.text)
    except Exception as e:
        print(f"Error getting difficulty: {e}")
        return None

def get_hash_rate():
    """Get Bitcoin hash rate from Blockchain.info API"""
    try:
        response = requests.get('https://blockchain.info/q/hashrate')
        return float(response.text) / 1000  # Ajustar según escala del dataset original
    except Exception as e:
        print(f"Error getting hash rate: {e}")
        return None

def get_fear_greed_index():
    """Get current fear and greed index"""
    try:
        response = requests.get("https://api.alternative.me/fng/?limit=1")
        data = response.json()
        return int(data['data'][0]['value'])
    except Exception as e:
        print(f"Error getting fear and greed index: {e}")
        return None

def update_dataset():
    """Update the merged_data.csv with latest data"""
    # Path to the merged data file
    file_path = 'data/merged_data.csv'
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        return None, False
    
    # Load existing data
    df = pd.read_csv(file_path)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Get the last date in the dataset
    last_date = df['date'].max()
    start_date = last_date + timedelta(days=1)
    
    # Set end date to yesterday
    end_date = datetime.now() - timedelta(days=1)
    
    # Skip update if we're already up to date
    if start_date > end_date:
        print("Dataset already up to date.")
        return df, False
    
    print(f"Fetching new data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Get data for all required assets
    btc_data = yf.download("BTC-USD", start=start_date, end=end_date)
    if btc_data.empty:
        print("No new Bitcoin data available")
        return df, False
    
    # Get other assets data
    oil_data = yf.download("CL=F", start=start_date, end=end_date)  # Oil WTI
    vix_data = yf.download("^VIX", start=start_date, end=end_date)  # VIX
    qqq_data = yf.download("QQQ", start=start_date, end=end_date)  # QQQ
    int_rate_data = yf.download("^IRX", start=start_date, end=end_date)  # Interest rates
    
    # Create new dataframe with all required columns
    new_data = pd.DataFrame(index=btc_data.index)
    new_data['date'] = new_data.index
    
    # Add Bitcoin data
    new_data['close_btc'] = btc_data['Close']
    new_data['volume_btc'] = btc_data['Volume']
    new_data['btc_change'] = btc_data['Close'].pct_change() * 100
    
    # Add Oil data
    new_data['close_WTI'] = oil_data['Close']
    new_data['volume'] = oil_data['Volume']
    
    # Add VIX data
    new_data['close_VIX'] = vix_data['Close']
    
    # Add QQQ data
    new_data['close_qqq'] = qqq_data['Close']
    new_data['volume_qqq'] = qqq_data['Volume']
    
    # Add Interest Rate data
    new_data['close_int'] = int_rate_data['Close']
    
    # Get CPI data - use last value as it doesn't change daily
    if 'close_cpi' in df.columns:
        new_data['close_cpi'] = df['close_cpi'].iloc[-1]
    
    # Get API data (difficulty, hash rate, fear/greed)
    # Get latest values
    difficulty = get_bitcoin_difficulty()
    hash_rate = get_hash_rate()
    fear_greed = get_fear_greed_index()
    
    # Use last values from dataset if APIs fail
    if difficulty is None and 'difficulty' in df.columns:
        difficulty = df['difficulty'].iloc[-1]
    if hash_rate is None and 'hash_rate' in df.columns:
        hash_rate = df['hash_rate'].iloc[-1]
    if fear_greed is None and 'value_fear_greed' in df.columns:
        fear_greed = df['value_fear_greed'].iloc[-1]
    
    # Add these values to new_data
    new_data['difficulty'] = difficulty
    new_data['hash_rate'] = hash_rate
    new_data['value_fear_greed'] = fear_greed
    
    # Drop rows with missing values
    new_data = new_data.dropna()
    
    # Make sure we have all required columns
    required_columns = df.columns
    for col in required_columns:
        if col not in new_data.columns and col != 'date':
            print(f"Adding missing column: {col}")
            if col in df.columns:
                new_data[col] = df[col].iloc[-1]  # Use last value from existing data
            else:
                new_data[col] = np.nan  # Or use NaN
    
    # Filter new_data to only include columns from original dataset
    new_data = new_data[required_columns]
    
    # Append new data to existing dataframe
    updated_df = pd.concat([df, new_data], ignore_index=True)
    
    # Sort by date
    updated_df = updated_df.sort_values('date')
    
    # Save the updated dataframe
    os.makedirs('data', exist_ok=True)
    updated_df.to_csv(file_path, index=False)
    
    # Record when we last updated
    with open('data/last_updated.txt', 'w') as f:
        f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    print(f"Added {len(new_data)} new rows to the dataset")
    return updated_df, True

def get_last_update_time():
    """Get the time when the data was last updated"""
    try:
        with open('data/last_updated.txt', 'r') as f:
            return f.read().strip()
    except:
        return "Never"

# If run directly, perform update
if __name__ == "__main__":
    df, updated = update_dataset()
    if updated:
        print(f"Data updated successfully! Total rows: {len(df)}")
    else:
        print("No update needed.")