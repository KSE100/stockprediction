
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, timedelta
from psx import stocks
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- Configuration ---
STOCK_TICKER = "UBL"
DATA_FILE = "data/KSE30 Data.csv"
PREDICTIONS_FILE = "data/predictions.csv" # File to store historical predictions

# --- Helper Functions ---

@st.cache_data(ttl=timedelta(hours=12)) # Cache data for 12 hours
def load_and_prepare_data(ticker, data_file):
    """Loads historical data, fetches today's data, combines, and adds features."""
    try:
        # Load historical data if it exists
        if os.path.exists(data_file):
            historical_data = pd.read_csv(data_file, index_col='Date', parse_dates=True)
            st.write(f"Loaded {len(historical_data)} days of historical data from {data_file}")
        else:
            historical_data = pd.DataFrame()
            st.write(f"No historical data found at {data_file}. Starting fresh.")

        # Fetch today's data
        today = date.today()
        st.write(f"Fetching data for today: {today}")
        ubl_today_data = stocks(ticker, start=today, end=today)

        if ubl_today_data.empty:
            st.warning(f"No data found for {ticker} for today's date. Using historical data only.")
            combined_data = historical_data
        else:
            st.success(f"Successfully fetched today's data for {ticker}.")
            # Drop today's data from historical_data if it exists to prevent duplicates
            historical_data = historical_data[historical_data.index.date != today]
            # Combine historical data with today's data
            combined_data = pd.concat([historical_data, ubl_today_data])
            combined_data.sort_index(inplace=True)
            st.write("Combined historical and today's data:")
            st.dataframe(combined_data.tail())

        if combined_data.empty:
             st.error("No data available to process.")
             return pd.DataFrame()


        # Calculate Technical Indicators (RSI and Moving Averages)
        combined_data['MA_20'] = combined_data['Close'].rolling(window=20).mean()
        combined_data['MA_50'] = combined_data['Close'].rolling(window=50).mean()

        delta = combined_data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / avg_loss
        combined_data['RSI'] = 100 - (100 / (1 + rs))

        # Feature Engineering
        combined_data['Close_Lag1'] = combined_data['Close'].shift(1)
        combined_data['Close_Lag2'] = combined_data['Close'].shift(2)
        combined_data['Close_Lag3'] = combined_data['Close'].shift(3)
        combined_data['Volume_MA_5'] = combined_data['Volume'].rolling(window=5).mean()

        # New features for classification
        combined_data['MA_5'] = combined_data['Close'].rolling(window=5).mean()
        combined_data['MA_5_vs_20'] = combined_data['MA_5'] - combined_data['MA_20']
        combined_data['Volatility_14'] = combined_data['Close'].rolling(window=14).std()
        combined_data['Volume_Change_Lag1'] = combined_data['Volume'].pct_change().shift(1)
        combined_data['Price_Change_1d'] = combined_data['Close'].diff(1).shift(1)
        combined_data['Price_Change_3d'] = combined_data['Close'].diff(3).shift(1)


        # Create target variable for classification (next day's price direction)
        combined_data['Target'] = combined_data['Close'].shift(-1) # Keep for potential future use or debugging
        price_difference = combined_data['Target'] - combined_data['Close']
        combined_data['Price_Direction'] = np.where(price_difference > 0, 'Up', 'Down')


        # Drop rows with NaN values introduced by lagging and rolling windows
        combined_data.dropna(inplace=True)

        # Save updated data
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        combined_data.to_csv(data_file)
        st.write(f"Updated data saved to {data_file}")

        return combined_data

    except Exception as e:
        st.error(f"An error occurred during data loading and preparation: {e}")
        return pd.DataFrame()


def train_classification_model(data):
    """Trains a classification model to predict price direction."""
    if data.empty:
        st.warning("No data available to train the model.")
        return None, None, None, None, None, None, None

    # Define features (X) and target (y) for classification
    features_classification = data.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'Price_Direction'], axis=1)
    target_classification = data['Price_Direction']

    # Use the last row for prediction, the rest for training
    # Ensure there's enough data for both training and prediction
    if len(features_classification) < 2:
         st.warning("Not enough data to train the model and make a prediction.")
         return None, None, None, None, None, None, None


    X_train_classification = features_classification.iloc[:-1]
    y_train_classification = target_classification.iloc[:-1]
    X_latest = features_classification.iloc[-1:] # Features for the latest day
    y_latest_actual = target_classification.iloc[-1:] # Actual direction for the latest day (for evaluation)


    # Initialize and train the Random Forest Classifier model
    model_classification = RandomForestClassifier(random_state=42)
    model_classification.fit(X_train_classification, y_train_classification)

    st.write("✅ Classification Model training complete.")

    # Evaluate the model on the training data (optional, but good for debugging)
    y_train_pred = model_classification.predict(X_train_classification)
    train_accuracy = accuracy_score(y_train_classification, y_train_pred)

    # Evaluate the model on the latest available data point
    y_latest_pred = model_classification.predict(X_latest)
    latest_accuracy = accuracy_score(y_latest_actual, y_latest_pred)


    return model_classification, X_latest, train_accuracy, latest_accuracy, y_latest_actual.iloc[0], y_latest_pred[0], model_classification.classes_.tolist()


def make_prediction(model, X_latest, model_classes):
    """Makes a prediction and gets confidence score for the latest data."""
    if model is None or X_latest.empty:
        st.warning("Model not trained or no data for prediction.")
        return None, None, None

    # Predict the direction
    predicted_direction = model.predict(X_latest)[0]

    # Get prediction probabilities
    y_prob_classification = model.predict_proba(X_latest)

    # Get confidence score for the predicted class
    # Find the index of the predicted_direction in the model's classes
    predicted_class_index = model_classes.index(predicted_direction)
    confidence_score = y_prob_classification[0, predicted_class_index]

    st.write(f"🔮 Prediction made: {predicted_direction} with confidence {confidence_score:.2f}")

    return predicted_direction, confidence_score, X_latest.index[0] # Return the date of the prediction


def store_prediction(prediction_date, predicted_direction, confidence_score):
    """Stores the prediction in a CSV file."""
    try:
        if os.path.exists(PREDICTIONS_FILE):
            predictions_df = pd.read_csv(PREDICTIONS_FILE, index_col='Date', parse_dates=True)
        else:
            predictions_df = pd.DataFrame(columns=['Predicted_Direction', 'Confidence_Score', 'Actual_Outcome'])

        # Ensure the index is datetime
        predictions_df.index = pd.to_datetime(predictions_df.index)

        # Add the new prediction
        # Check if a prediction for this date already exists
        if prediction_date in predictions_df.index:
             # Update existing prediction if needed (e.g., if you want to overwrite)
             # For now, let's assume we don't overwrite and just log that it exists
             st.write(f"Prediction for {prediction_date.date()} already exists. Skipping storage of new prediction.")
        else:
            new_prediction = pd.DataFrame([{
                'Predicted_Direction': predicted_direction,
                'Confidence_Score': confidence_score,
                'Actual_Outcome': np.nan # Actual outcome is unknown at prediction time
            }], index=[prediction_date])
            predictions_df = pd.concat([predictions_df, new_prediction])
            predictions_df.sort_index(inplace=True)
            predictions_df.to_csv(PREDICTIONS_FILE)
            st.write(f"Prediction for {prediction_date.date()} stored.")

    except Exception as e:
        st.error(f"An error occurred while storing the prediction: {e}")


def update_actual_outcomes(data):
    """Updates the actual outcomes for historical predictions."""
    try:
        if not os.path.exists(PREDICTIONS_FILE):
            return

        predictions_df = pd.read_csv(PREDICTIONS_FILE, index_col='Date', parse_dates=True)

        # Ensure indices are of comparable types
        predictions_df.index = pd.to_datetime(predictions_df.index)
        data.index = pd.to_datetime(data.index)

        updated_count = 0
        # Iterate through predictions and find corresponding actual outcomes in data
        for index, row in predictions_df.iterrows():
            if pd.isna(row['Actual_Outcome']): # Only update if actual outcome is missing
                # The actual outcome for a prediction made on date X is the direction on date X+1
                next_day_date = index + timedelta(days=1)
                if next_day_date in data.index:
                    actual_direction = data.loc[next_day_date, 'Price_Direction']
                    predictions_df.loc[index, 'Actual_Outcome'] = actual_direction
                    updated_count += 1

        if updated_count > 0:
            predictions_df.to_csv(PREDICTIONS_FILE)
            st.write(f"Updated {updated_count} historical prediction outcomes.")

        return predictions_df # Return updated predictions dataframe

    except Exception as e:
        st.error(f"An error occurred while updating actual outcomes: {e}")
        return pd.DataFrame()


# --- Streamlit App ---

st.title(f"Stock Price Direction Prediction Dashboard ({STOCK_TICKER})")

st.write("Click the button below to fetch the latest data, update the model, and get a prediction for the next trading day's price direction.")

if st.button("Run Analysis and Get Prediction"):
    st.write("Running analysis...")

    # 1. Load and Prepare Data
    combined_data = load_and_prepare_data(STOCK_TICKER, DATA_FILE)

    if not combined_data.empty:
        # 2. Update Actual Outcomes for past predictions
        historical_predictions_df = update_actual_outcomes(combined_data) # Capture the updated predictions

        # 3. Train Classification Model and get latest features
        model_classification, X_latest, train_accuracy, latest_accuracy, latest_actual_direction, latest_predicted_direction, model_classes = train_classification_model(combined_data)

        if model_classification is not None and not X_latest.empty:
            # 4. Make Prediction for the next day (using the trained model on the latest data)
            # Note: The prediction date will be X_latest.index[0] + timedelta(days=1)
            # The make_prediction function currently returns the date of X_latest, which is the date the prediction is *based on*.
            # Let's adjust make_prediction or how we call it to reflect the *predicted* date.
            # For simplicity, let's assume the prediction is for the day after the latest data point.

            predicted_direction_tomorrow, confidence_score_tomorrow, date_of_latest_data = make_prediction(model_classification, X_latest, model_classes)
            predicted_date = date_of_latest_data + timedelta(days=1) # The date the prediction is for

            if predicted_direction_tomorrow is not None:
                st.subheader(f"Prediction for {predicted_date.date()}:")
                st.write(f"**Predicted Direction:** {predicted_direction_tomorrow}")
                st.write(f"**Confidence Score:** {confidence_score_tomorrow:.2f}")

                # 5. Store the new prediction
                store_prediction(predicted_date, predicted_direction_tomorrow, confidence_score_tomorrow)

            # Display model evaluation metrics on the latest data point
            st.subheader(f"Model Performance on Latest Data ({date_of_latest_data.date()}):")
            st.write(f"Actual Direction: {latest_actual_direction}")
            st.write(f"Predicted Direction: {latest_predicted_direction}")
            st.write(f"Accuracy on latest data: {latest_accuracy:.2f}")
            st.write(f"Accuracy on training data: {train_accuracy:.2f}") # Display training accuracy as a reference

            # Display historical predictions and outcomes (using the updated dataframe)
            st.subheader("Historical Predictions and Outcomes:")
            if not historical_predictions_df.empty:
                 # Calculate accuracy of historical predictions with known outcomes
                evaluated_predictions = historical_predictions_df.dropna(subset=['Actual_Outcome'])
                if not evaluated_predictions.empty:
                    historical_accuracy = accuracy_score(evaluated_predictions['Actual_Outcome'], evaluated_predictions['Predicted_Direction'])
                    st.write(f"Historical Prediction Accuracy (evaluated outcomes): {historical_accuracy:.2f}")

                st.dataframe(historical_predictions_df.sort_index(ascending=False))
            else:
                st.write("No historical predictions found.")


    else:
        st.error("Could not load or prepare data. Please check the logs.")


# Display historical stock data and charts initially (even before button click, if data file exists)
st.subheader("Historical Stock Data:")
if os.path.exists(DATA_FILE):
    historical_data_display = pd.read_csv(DATA_FILE, index_col='Date', parse_dates=True)
    if not historical_data_display.empty:
        st.line_chart(historical_data_display['Close'])
        st.dataframe(historical_data_display.tail())
    else:
        st.write("No historical stock data available.")
else:
     st.write("No historical stock data file found.")

# Display historical predictions and outcomes initially (even before button click, if file exists)
# This section is already present in the 'if st.button' block, let's move a version outside
# so it's visible on initial load if the file exists.
st.subheader("All Recorded Predictions and Outcomes:")
if os.path.exists(PREDICTIONS_FILE):
    all_predictions_df = pd.read_csv(PREDICTIONS_FILE, index_col='Date', parse_dates=True)
    if not all_predictions_df.empty:
        st.dataframe(all_predictions_df.sort_index(ascending=False))
    else:
        st.write("No recorded predictions found.")
else:
    st.write("No recorded predictions file found.")

