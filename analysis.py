"""
analysis.py

Provides pitch recommendation logic using a trained XGBoost model and Statcast data.
It loads the preprocessed dataset and model artifacts to simulate pitch outcomes and
suggest the mos successful pitch based on game context.

"""

import pandas as pd
import joblib

# Load data and model

filtered_df = pd.read_csv('statcast_filtered.csv')
pitch_mix_with_cluster = pd.read_csv("pitch_mix_with_cluster.csv")
best_model = joblib.load("best_model.pkl")
X_columns = joblib.load("X_columns.pkl")


def get_pitch_options(pitcher_name, df, min_usage=50):
    """
    Returns the pitch types a given pitcher uses above a minimum threshold.

    :param pitcher_name: Full name of the pitcher (ex. "Webb, Logan").
    :param df: Filtered Statcast dataset.
    :param min_usage: Minimum number of pitches thrown to be included.
    :return: List of pitch types that meet the minimum threshold.
    """
    pitcher_data = df[df['player_name'] == pitcher_name]
    pitch_counts = pitcher_data['pitch_type'].value_counts()
    return pitch_counts[pitch_counts >= min_usage].index.tolist()


def recommend_pitch(pitcher_name, current_context, model, X_columns):
    """

    :param pitcher_name: Full name of the pitcher.
    :param current_context: Dictionary of current pitch context features.
    :param model: Trained XGBoost model for pitch success.
    :param X_columns: List of feature columns expected by the model
    :return: Pitch types with their predicted success probabilities,
             sorted from highest to lowest.
    """
    # Get valid pitch options for this pitcher
    pitch_options = get_pitch_options(pitcher_name, filtered_df)
    # Get the cluster assignment for the pitcher
    cluster = pitch_mix_with_cluster.loc[
        pitch_mix_with_cluster['player_name'] == pitcher_name, 'cluster'
    ].values[0]

    simulated_rows = []
    # Simulate each pitch option in the current context
    for pitch in pitch_options:
        sim_row = current_context.copy()
        sim_row['pitch_type'] = pitch
        sim_row['cluster'] = cluster
        simulated_rows.append(sim_row)

    sim_df = pd.DataFrame(simulated_rows)
    # One-Hot encode and align columns with model expectations
    sim_df_encoded = pd.get_dummies(sim_df)
    sim_df_encoded = sim_df_encoded.reindex(columns=X_columns, fill_value=0)
    # Predict success probability for each pitch
    probs = model.predict_proba(sim_df_encoded)[:, 1]
    # Return a sorted dataframe of pitch recommendations
    recommendations = pd.DataFrame({
        'pitch_type': pitch_options,
        'predicted_success_prob': probs
    }).sort_values(by='predicted_success_prob', ascending=False)

    return recommendations
