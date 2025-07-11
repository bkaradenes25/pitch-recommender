import streamlit as st
import joblib
from analysis import recommend_pitch, filtered_df, best_model, X

joblib.dump(best_model, "best_model.pk1")
joblib.dump(X.columns, "X_columns.pk1")

model = joblib.load("best_model.pk1")
X_columns = joblib.load("X_columns.pk1")

st.title("Pitch Recommender System")
st.markdown("This tool uses Machine Learning to make pitch recommendations based on given situations.")

pitcher_name = st.selectbox("Select Pitcher", sorted(filtered_df["player_name"].unique()))
stand = st.selectbox("Batter Stance (Left/Right)", ['L', 'R'])
p_throws = st.selectbox("Pitcher Throws (Left/Right)", ['L', 'R'])
on_base = st.checkbox("Is there a runner on base?")
count = st.slider("Count (Strikes - Balls)", -3, 3, 0)
outs_when_up = st.slider("Outs", 0, 2, 1)
inning = st.slider("Inning", 1, 9, 1)
plate_x = st.slider("Plate X Location", -1.5, 1.5, 0.0)
plate_z = st.slider("Plate Z Location", 0.0, 5.0, 2.5)

if st.button("Recommend Pitch"):
    current_context = {
        'stand': stand,
        'p_throws': p_throws,
        'on_base': int(on_base),
        'count': count,
        'outs_when_up': outs_when_up,
        'inning': inning,
        'plate_x': plate_x,
        'plate_z': plate_z
    }

    recommendations = recommend_pitch(pitcher_name, current_context, model, X_columns)
    st.subheader("Recommended Pitches (Ranked By Success Probability)")
    st.dataframe(recommendations)

    st.bar_chart(recommendations.set_index('pitch_type'))

