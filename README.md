# Pitch Recommender System

A Streamlit app that uses machine learning and Statcast data to recommend pitch types based on given situation.

Here is the link: https://pitchrecommender.streamlit.app/

## How It Works

Data: Pulled from MLB Statcast using PyBaseball
Pitcher Clustering: Pitchers are clustered based on general usage
Model: A tuned XGBoost classifier is trained to predict pitch success.
Recommendations: For a given pitcher and situation, the model simulates each pitch option and outputs success probabilities

## Features

- Selected from 50+ qualified 2024 pitchers
- Game situation: count, inning, runners on base
- View ranked pitch selections based on probability scores
- See bar chart of recs

## Note

This model uses wOBA <= .300 to define success + called/swinging strikes

Made by Brendan Karadenes


