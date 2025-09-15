"""
app.py

Streamlit web app for recommending baseball pitches using a trained machine learning
model. Users can select a pitcher and input game context (count, runners on, location),
and the app will return pitch type recommendations ranked by predicted success
probability.
"""

import streamlit as st
from analysis import recommend_pitch, filtered_df, best_model, X_columns, hitter_clusters
from data_prep import hitters_df

# Title and description
st.title("Pitch Recommender System")
st.markdown("This tool uses Machine Learning to make pitch recommendations based on given situations.")

# User input widgets for context features
pitcher_name = st.selectbox("Select Pitcher", sorted(filtered_df["player_name"].unique()))
stand = st.selectbox("Batter Stance (Left/Right)", ['L', 'R'])
p_throws = st.selectbox("Pitcher Throws (Left/Right)", ['L', 'R'])
on_base = st.checkbox("Is there a runner on base?")
count = st.slider("Count (Strikes - Balls)", -3, 3, 0)
outs_when_up = st.slider("Outs", 0, 2, 1)
inning = st.slider("Inning", 1, 9, 1)
plate_x = st.slider("Plate X Location", -1.5, 1.5, 0.0)
plate_z = st.slider("Plate Z Location", 0.0, 5.0, 2.5)

hitter_clusters_label = {
    0: "Balanced Power Hitter",
    1: "Light-Hitting Contact",
    2: "Elite Power/On-Base",
    3: "High Contact / Low Power",
    4: "Disciplined Power Hitter"
}

hitter_cluster = st.selectbox(
    "Select Hitter Cluster",
    options = [0,1,2,3,4],
    format_func=lambda x: hitter_clusters_label[x]
)

# When user clicks button, generate recommendations
if st.button("Recommend Pitch"):
    # Assemble game context into dictionary
    current_context = {
        'stand': stand,
        'p_throws': p_throws,
        'on_base': int(on_base),
        'count': count,
        'outs_when_up': outs_when_up,
        'inning': inning,
        'plate_x': plate_x,
        'plate_z': plate_z,
        'hitter_cluster': hitter_cluster
    }
    # Generate recommendations using trained model
    recommendations = recommend_pitch(pitcher_name, current_context, best_model, X_columns)
    # Display results
    st.subheader("Recommended Pitches (Ranked By Success Probability)")
    st.dataframe(recommendations)
    # Bar chart of predicted success probabilities
    st.bar_chart(recommendations.set_index('pitch_type'))
