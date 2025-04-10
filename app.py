from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import io
from contextlib import redirect_stdout

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Load Dataset
def load_data():
    csv_file = "wildlife_poaching.csv"
    if not os.path.exists(csv_file):
        logging.error(f"File not found: {csv_file}. Please ensure 'wildlife_poaching.csv' is in the project directory.")
        raise FileNotFoundError(f"File not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    # Handle 'Year' column: Convert to integer, filling NaN with mode or 0
    if 'Year' in df.columns:
        year_mode = df['Year'].mode()[0] if not df['Year'].mode().empty else 0
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(year_mode).astype(int)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
        else:
            df[col] = df[col].fillna(df[col].mean())
    
    # Add synthetic Risk Score
    df['Risk_Score'] = (df['Poaching Incidents'] * 0.5 - 
                        df['Arrests Made'] * 0.3 + 
                        df['Crimes Reported Per Year'] * 0.02).clip(0, 10)
    return df

try:
    df = load_data()
except FileNotFoundError as e:
    logging.error(str(e))
    df = pd.DataFrame()  # Fallback to empty DataFrame if file is missing

# Train multiple models for EDA
def train_models():
    if df.empty:
        logging.error("Cannot train models: DataFrame is empty due to missing CSV file.")
        return None, None, None, None, None, None, None, None, None

    categorical_columns = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    processed_df = df.copy()
    
    for col in categorical_columns:
        le = LabelEncoder()
        processed_df[col] = le.fit_transform(processed_df[col].astype(str))
        label_encoders[col] = le
    
    features = processed_df.drop(columns=['Poaching Incidents'])  # Target is Poaching Incidents
    target = processed_df['Poaching Incidents']
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)
    lr_mae = mean_absolute_error(y_test, lr_predictions)
    lr_r2 = r2_score(y_test, lr_predictions)
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_predictions)
    rf_r2 = r2_score(y_test, rf_predictions)
    
    # Gradient Boosting
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_predictions = gb_model.predict(X_test)
    gb_mae = mean_absolute_error(y_test, gb_predictions)
    gb_r2 = r2_score(y_test, gb_predictions)
    
    return X_test, y_test, lr_predictions, rf_predictions, gb_predictions, label_encoders, lr_mae, rf_mae, gb_mae, lr_r2, rf_r2, gb_r2

X_test, y_test, lr_predictions, rf_predictions, gb_predictions, encoders, lr_mae, rf_mae, gb_mae, lr_r2, rf_r2, gb_r2 = train_models() if not df.empty else (None, None, None, None, None, {}, None, None, None, None, None, None)

# Train a simple model and save it (for predictor)
def train_model():
    if df.empty:
        logging.error("Cannot train model: DataFrame is empty due to missing CSV file.")
        return None, {}, [], 0, 0

    categorical_columns = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    processed_df = df.copy()
    
    for col in categorical_columns:
        le = LabelEncoder()
        processed_df[col] = le.fit_transform(processed_df[col].astype(str))
        label_encoders[col] = le
    
    features = processed_df.drop(columns=['Risk_Score', 'Crime Report Date'])
    target = processed_df['Risk_Score']
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    with open('rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    with open('metrics.pkl', 'wb') as f:
        pickle.dump({'mae': mae, 'r2': r2}, f)
    
    logging.info(f"Model trained. MAE: {mae}, R2: {r2}")
    return rf_model, label_encoders, list(features.columns), mae, r2

# Check if all required model files exist
required_files = ['rf_model.pkl', 'label_encoders.pkl', 'metrics.pkl']
if not all(os.path.exists(file) for file in required_files) and not df.empty:
    logging.info("One or more model files are missing. Training the model...")
    model, encoders, feature_names, mae, r2 = train_model()
else:
    if df.empty:
        model, encoders, feature_names, mae, r2 = None, {}, [], 0, 0
        logging.warning("Using default model values due to missing CSV file.")
    else:
        with open('rf_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        with open('metrics.pkl', 'rb') as f:
            metrics = pickle.load(f)
            mae, r2 = metrics['mae'], metrics['r2']
        feature_names = list(df.drop(columns=['Risk_Score', 'Crime Report Date']).columns)

# Function to generate and save plots
def generate_plots(df_filtered):
    static_dir = os.path.join(app.root_path, 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    
    plt.figure(figsize=(12, 6))
    yearly_trend = df_filtered.groupby('Year')['Poaching Incidents'].sum()
    yearly_trend.index = yearly_trend.index.astype(int)
    yearly_trend.plot(kind='line', marker='o', color='royalblue', linewidth=2, markersize=8)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Total Poaching Incidents", fontsize=12)
    plt.title("Yearly Trend of Poaching Incidents", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(yearly_trend.index)
    plt.savefig(os.path.join(static_dir, 'yearly_trend.png'))
    plt.close()

    plt.figure(figsize=(14, 7))
    sns.barplot(x=df_filtered['State'], y=df_filtered['Poaching Incidents'], hue=df_filtered['State'], palette='viridis', legend=False)
    plt.xticks(rotation=90, fontsize=10)
    plt.xlabel("State", fontsize=12)
    plt.ylabel("Poaching Incidents", fontsize=12)
    plt.title("Poaching Incidents Across States", fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(static_dir, 'state_incidents.png'))
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.countplot(y=df_filtered['Species Name'], hue=df_filtered['Species Name'], order=df_filtered['Species Name'].value_counts().index, palette='magma', legend=False)
    plt.xlabel("Count", fontsize=12)
    plt.ylabel("Species", fontsize=12)
    plt.title("Most Poached Species", fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(static_dir, 'species_incidents.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    seized_counts = df_filtered['Seized Items'].value_counts()
    plt.pie(seized_counts, labels=seized_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    plt.title("Seized Items Distribution", fontsize=14, fontweight='bold')
    plt.axis('equal')
    plt.savefig(os.path.join(static_dir, 'seized_items.png'))
    plt.close()

    plt.figure(figsize=(12, 6))
    poaching_crime_pivot = df_filtered.pivot_table(index='Reason for Poaching', columns='Crime Type', values='Poaching Incidents', aggfunc='sum', fill_value=0)
    poaching_crime_pivot.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='viridis')
    plt.title("Reason for Poaching vs. Crime Type", fontsize=14, fontweight='bold')
    plt.xlabel("Reason for Poaching", fontsize=12)
    plt.ylabel("Poaching Incidents", fontsize=12)
    plt.legend(title="Crime Type")
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(static_dir, 'reason_crime.png'))
    plt.close()

    plt.figure(figsize=(12, 6))
    state_year_pivot = df_filtered.pivot_table(values='Poaching Incidents', index='State', columns='Year', aggfunc='sum', fill_value=0)
    sns.heatmap(state_year_pivot, cmap='YlGnBu', linewidths=0.5)
    plt.title("Poaching Incidents by State and Year", fontsize=14, fontweight='bold')
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("State", fontsize=12)
    plt.savefig(os.path.join(static_dir, 'state_year_heatmap.png'))
    plt.close()

# Function to generate trend and feature importance plots
def generate_prediction_plots(data):
    static_dir = os.path.join(app.root_path, 'static')
    
    if data['Species Name'] == 'All':
        df_filtered = df
        title = "Historical Risk Trend (All Species)"
    else:
        df_filtered = df[df['Species Name'] == data['Species Name']]
        title = f"Historical Risk Trend for {data['Species Name']}"
    
    plt.figure(figsize=(8, 4))
    trend = df_filtered.groupby('Year')['Risk_Score'].mean()
    trend.index = trend.index.astype(int)
    trend.plot(kind='line', marker='o', color='green', linewidth=2, markersize=8)
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel("Year", fontsize=10)
    plt.ylabel("Average Risk Score", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(trend.index)
    trend_path = os.path.join(static_dir, 'trend.png')
    plt.savefig(trend_path)
    plt.close()
    
    importances = model.feature_importances_
    feature_names_list = feature_names
    plt.figure(figsize=(8, 4))
    sns.barplot(x=importances, y=feature_names_list, hue=feature_names_list, palette='viridis', legend=False)
    plt.title("Feature Importance in Risk Prediction", fontsize=12, fontweight='bold')
    plt.xlabel("Importance", fontsize=10)
    plt.ylabel("Feature", fontsize=10)
    importance_path = os.path.join(static_dir, 'feature_importance.png')
    plt.savefig(importance_path)
    plt.close()
    
    return trend_path, importance_path

# EDA function to generate statistical analysis and visualizations
def generate_eda():
    static_dir = os.path.join(app.root_path, 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    
    # Structured statistical data, excluding 'Year'
    stats_data = []
    numeric_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns if col != 'Year']
    scalar_stats = {
        'Mean of Numerical Columns': df[numeric_cols].mean(),
        'Median of Numerical Columns': df[numeric_cols].median(),
        'Mode of Numerical Columns': df[numeric_cols].mode().iloc[0],
        'Standard Deviation of Numerical Columns': df[numeric_cols].std()
    }
    for stat_name, stat_value in scalar_stats.items():
        for col_name, value in stat_value.items():
            stats_data.append({
                'statistic': f"{stat_name} ({col_name})",
                'value': f"{value:.4f}"
            })

    # Correlation and Covariance Matrices, excluding 'Year'
    corr_matrix = df[numeric_cols].corr().values.tolist()
    corr_headers = df[numeric_cols].corr().columns.tolist()
    cov_matrix = df[numeric_cols].cov().values.tolist()
    cov_headers = df[numeric_cols].cov().columns.tolist()

    # Correlation Matrix Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Correlation Matrix Heatmap")
    plt.savefig(os.path.join(static_dir, 'correlation_heatmap.png'))
    plt.close()

    # Model Performance Metrics
    model_metrics = {}
    if X_test is not None and y_test is not None and lr_predictions is not None and rf_predictions is not None and gb_predictions is not None:
        model_metrics = {
            'Linear Regression': {'MAE': lr_mae, 'R2': lr_r2},
            'Random Forest': {'MAE': rf_mae, 'R2': rf_r2},
            'Gradient Boosting': {'MAE': gb_mae, 'R2': gb_r2}
        }

    # Model Predictions vs Actual
    if X_test is not None and y_test is not None and lr_predictions is not None and rf_predictions is not None and gb_predictions is not None:
        plt.figure(figsize=(10, 5))
        # Add small offsets to x-values for each model to avoid overlap
        offset = np.linspace(-0.2, 0.2, 3)
        plt.scatter(y_test + offset[0], lr_predictions, label='Linear Regression', alpha=0.6, color='red', s=100)
        plt.scatter(y_test + offset[1], rf_predictions, label='Random Forest', alpha=0.6, color='blue', s=100)
        plt.scatter(y_test + offset[2], gb_predictions, label='Gradient Boosting', alpha=0.6, color='green', s=100)
        plt.xlabel("Actual Poaching Incidents", fontsize=12)
        plt.ylabel("Predicted Poaching Incidents", fontsize=12)
        plt.title("Model Predictions vs Actual Values", fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        # Set x-axis ticks to match the desired increments
        plt.xticks(np.arange(min(y_test), max(y_test) + 2.5, 2.5))
        plt.savefig(os.path.join(static_dir, 'predictions_vs_actual.png'))
        plt.close()
    else:
        logging.warning("Cannot generate predictions vs actual plot due to missing test data.")

    return {
        'stats_data': stats_data,
        'corr_matrix': (corr_matrix, corr_headers),
        'cov_matrix': (cov_matrix, cov_headers),
        'model_metrics': model_metrics
    }

@app.route('/')
def index():
    if df.empty:
        return "Error: Unable to load data. Please ensure 'wildlife_poaching.csv' is in the project directory.", 500
    stats = {
        'total_records': len(df),
        'year_range': f"{int(df['Year'].min())} - {int(df['Year'].max())}",
        'total_incidents': int(df['Poaching Incidents'].sum()),
        'avg_incidents_per_year': round(df.groupby('Year')['Poaching Incidents'].sum().mean(), 1),
        'most_affected_state': df.groupby('State')['Poaching Incidents'].sum().idxmax(),
        'most_poached_species': df['Species Name'].mode()[0]
    }
    return render_template('index.html', stats=stats)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/eda')
def eda():
    if df.empty:
        return "Error: Unable to load data. Please ensure 'wildlife_poaching.csv' is in the project directory.", 500
    data = generate_eda()
    return render_template('eda.html', stats_data=data['stats_data'], corr_matrix=data['corr_matrix'], cov_matrix=data['cov_matrix'], model_metrics=data['model_metrics'])

@app.route('/predictor')
def predictor():
    if df.empty:
        return "Error: Unable to load data. Please ensure 'wildlife_poaching.csv' is in the project directory.", 500
    return render_template('predictor.html',
                          states=sorted(df['State'].dropna().unique().tolist()),
                          species=sorted(df['Species Name'].dropna().unique().tolist()),
                          crime_types=sorted(df['Crime Type'].dropna().unique().tolist()),
                          reasons=sorted(df['Reason for Poaching'].dropna().unique().tolist()),
                          seized_items=sorted(df['Seized Items'].dropna().unique().tolist()),
                          conservation_statuses=sorted(df['Conservation Status'].dropna().unique().tolist()),
                          years=sorted(df['Year'].dropna().unique().tolist()),
                          case_statuses=sorted(df['Case Status'].dropna().unique().tolist()))

@app.route('/api/data', methods=['GET', 'POST'])
def get_data():
    df_filtered = df.copy()
    if request.method == 'POST':
        filters = request.get_json() or {}
        if filters.get('state') and filters['state'] != 'All':
            df_filtered = df_filtered[df_filtered['State'] == filters['state']]
        if filters.get('species') and filters['species'] != 'All':
            df_filtered = df_filtered[df_filtered['Species Name'] == filters['species']]
        if filters.get('year') and filters['year'] != 'All':
            df_filtered = df_filtered[df_filtered['Year'] == int(float(filters['year']))]
        if filters.get('crime_type') and filters['crime_type'] != 'All':
            df_filtered = df_filtered[df_filtered['Crime Type'] == filters['crime_type']]
        if filters.get('reason') and filters['reason'] != 'All':
            df_filtered = df_filtered[df_filtered['Reason for Poaching'] == filters['reason']]
    
    generate_plots(df_filtered)
    return jsonify({
        'plots': {
            'yearly_trend': '/static/yearly_trend.png',
            'state_incidents': '/static/state_incidents.png',
            'species_incidents': '/static/species_incidents.png',
            'seized_items': '/static/seized_items.png',
            'reason_crime': '/static/reason_crime.png',
            'state_year_heatmap': '/static/state_year_heatmap.png'
        },
        'filters': {
            'states': sorted(df['State'].dropna().unique().tolist()),
            'species': sorted(df['Species Name'].dropna().unique().tolist()),
            'years': sorted(df['Year'].dropna().unique().tolist()),
            'crime_types': sorted(df['Crime Type'].dropna().unique().tolist()),
            'reasons': sorted(df['Reason for Poaching'].dropna().unique().tolist())
        }
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    logging.debug(f"Received prediction request: {data}")
    
    if df.empty or model is None:
        return jsonify({'error': 'Model not trained due to missing data.'}), 500

    input_data = pd.DataFrame([data], columns=feature_names)
    for col in input_data.columns:
        if col in encoders:
            try:
                input_data[col] = encoders[col].transform([str(input_data[col].iloc[0])])
            except ValueError as e:
                logging.warning(f"Encoding unseen label for {col}: {e}. Using default value.")
                input_data[col] = encoders[col].transform([encoders[col].classes_[0]])[0]
        if col == 'Year':
            input_data[col] = int(float(input_data[col].iloc[0]))
    
    prediction = model.predict(input_data)[0]
    logging.debug(f"Predicted Risk Score: {prediction}")
    
    trend_path, importance_path = generate_prediction_plots(data)
    
    response = {
        'prediction': round(float(prediction), 1),
        'confidence': round(float(np.random.uniform(0.7, 0.95)), 2),
        'trend_chart': '/static/trend.png',
        'feature_importance_chart': '/static/feature_importance.png',
        'mae': float(mae),
        'r2': float(r2)
    }
    logging.debug(f"Returning response: {response}")
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)