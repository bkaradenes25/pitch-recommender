import pandas as pd
import joblib

filtered_df = pd.read_csv('statcast_filtered.csv')
pitch_mix_with_cluster = pd.read_csv("pitch_mix_with_cluster.csv")
best_model = joblib.load( "best_model.pkl")
X_columns = joblib.load( "X_columns.pkl")

def get_pitch_options(pitcher_name, df, min_usage = 50):
    pitcher_data = df[df['player_name'] == pitcher_name]
    pitch_counts = pitcher_data['pitch_type'].value_counts()
    return pitch_counts[pitch_counts >= min_usage].index.tolist()

def recommend_pitch(pitcher_name, current_context, model, X_columns):
    pitch_options = get_pitch_options(pitcher_name, filtered_df)
    cluster = pitch_mix_with_cluster.loc[
        pitch_mix_with_cluster['player_name'] == pitcher_name, 'cluster'
    ].values[0]

    simulated_rows = []

    for pitch in pitch_options:
        sim_row = current_context.copy()
        sim_row['pitch_type'] = pitch
        sim_row['cluster'] = cluster
        simulated_rows.append(sim_row)

    sim_df = pd.DataFrame(simulated_rows)

    sim_df_encoded = pd.get_dummies(sim_df)
    sim_df_encoded = sim_df_encoded.reindex(columns=X_columns, fill_value=0)

    probs = model.predict_proba(sim_df_encoded)[:, 1]

    recommendations = pd.DataFrame({
        'pitch_type': pitch_options,
        'predicted_success_prob': probs
    }).sort_values(by='predicted_success_prob', ascending=False)

    return recommendations
