import pandas as pd
from pybaseball import statcast
from datetime import datetime, timedelta
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
start_date = datetime(2024, 3, 28)
end_date = datetime(2024, 10, 1)
all_data = []

delta = timedelta(days = 7)
current_date = start_date

while current_date < end_date:
    next_date = min(current_date + delta, end_date)
    print(f"Downloading: {current_date.date()} to {next_date.date()}")

    try:
        chunk = statcast(start_dt = current_date.strftime("%Y-%m-%d"),
                         end_dt = next_date.strftime("%Y-%m-%d"))
        all_data.append(chunk)
    except Exception as e:
        print(f"Failed on chunk {current_date.date()} to {next_date.date()}: {e}")

    current_date = next_date + timedelta(days=1)

full_df = pd.concat(all_data, ignore_index=True)
full_df.to_csv("statcast_2024_full.csv", index = False)

qualified_pitchers = [
    "Gilbert, Logan", "Lugo, Seth", "Webb, Logan", "Wheeler, Zack", "Nola, Aaron",
    "Burnes, Corbin", "Berríos, José", "Skubal, Tarik", "Kirby, George", "Cease, Dylan",
    "Irvin, Jake", "Ragans, Cole", "López, Pablo", "Crawford, Kutter", "Severino, Luis",
    "Manaea, Sean", "Pfaadt, Brandon", "Sánchez, Cristopher", "Gausman, Kevin", "Sears, JP",
    "Miller, Bryce", "Singer, Brady", "Anderson, Tyler", "Houck, Tanner", "Ober, Bailey",
    "Keller, Mitch", "Sale, Chris", "Fedde, Erick", "Valdez, Framber", "Kikuchi, Yusei",
    "Castillo, Luis", "Rodón, Carlos", "Corbin, Patrick", "Cortes, Nestor", "Fried, Max",
    "Bibee, Tanner", "King, Michael", "Peralta, Freddy", "Imanaga, Shota", "Canning, Griffin",
    "Mikolas, Miles", "Bassitt, Chris", "Eovaldi, Nathan", "Quintana, Jose", "Brown, Hunter",
    "Gibson, Kyle", "Rea, Colin", "Blanco, Ronel", "Wacha, Michael", "Gore, MacKenzie",
    "Gray, Sonny", "Eflin, Zach", "Morton, Charlie", "Taillon, Jameson", "Gomber, Austin",
    "Bello, Brayan", "Feltner, Ryan", "Flaherty, Jack"
]
filtered_df = full_df[full_df['player_name'].isin(qualified_pitchers)]
filtered_df.to_csv("statcast_filtered.csv", index = False)
filtered_df = pd.read_csv('statcast_filtered.csv')
missing = filtered_df[(filtered_df['type'] == 'X') & (filtered_df['estimated_woba_using_speedangle'].isna())]
print(f"Missing expected_woba rows: {len(missing)}")
filtered_df = filtered_df[~((filtered_df['type'] == 'X') & (filtered_df['estimated_woba_using_speedangle'].isna()))]
filtered_df['success'] = np.where(
    (
        filtered_df['estimated_woba_using_speedangle'].notna() &
        (filtered_df['estimated_woba_using_speedangle'] <= 0.300)
    ) | (filtered_df['description'].isin(['called_strike', 'swinging_strike'])),
    1,
    np.where(
        filtered_df['estimated_woba_using_speedangle'].notna() &
        (filtered_df['estimated_woba_using_speedangle'] >= 0.370),
        0,
        0
    )
)

pitch_types = filtered_df['pitch_type'].dropna().unique().tolist()
print(pitch_types)

fastballs = ['FF', 'FT', 'SI', 'FC', 'FA']
breaking_balls = ['SL', 'CU', 'KC', 'KN', 'ST', 'SB']
offspeed = ['FS', 'CH', 'EP', 'SC']
conditions = [
    filtered_df['pitch_type'].isin(fastballs),
    filtered_df['pitch_type'].isin(breaking_balls),
    filtered_df['pitch_type'].isin(offspeed)
]
choices = ['fastballs', 'breaking_balls', 'offspeed']

filtered_df['pitch_cat'] = np.select(conditions, choices, default = 'other')

pitch_mix = (
    filtered_df
    .groupby(['player_name', 'pitch_cat'])
    .size()
    .unstack(fill_value = 0)
)

pitch_mix = pitch_mix.div(pitch_mix.sum(axis=1), axis=0)

inertia = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(pitch_mix)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'o-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow method for optimal k')
plt.xticks(K)
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42)
pitch_mix['cluster'] = kmeans.fit_predict(pitch_mix)

pitch_mix_with_cluster = pitch_mix.copy().reset_index()
filtered_df = filtered_df.merge(
    pitch_mix_with_cluster[['player_name', 'cluster']],
    on='player_name',
    how='left'
)

filtered_df['count'] = filtered_df['strikes'] - filtered_df['balls']

filtered_df['on_base'] = np.where(
    filtered_df[['on_1b', 'on_2b', 'on_3b']].notna().any(axis=1),
    1,
    0
)

features = [
    'pitch_type', 'stand', 'p_throws', 'on_base',
    'count', 'outs_when_up', 'inning', 'plate_x', 'plate_z', 'cluster'
]

target = 'success'

df_model = filtered_df.dropna(subset=features + [target])
df_encoded = pd.get_dummies(df_model[features + [target]], drop_first=True)

X = df_encoded.drop(columns=[target])
y = df_encoded[target]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.30, random_state = 42, stratify=y)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify = y_temp)

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train)

y_val_pred = model.predict(X_val)
y_val_prob = model.predict_proba(X_val)[:, 1]

print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Validation ROC AUC:", roc_auc_score(y_val, y_val_prob))
print(classification_report(y_val, y_val_pred))

y_test_pred = model.predict(X_test)
y_test_prob = model.predict_proba(X_test)[:, 1]

print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Test ROC AUC:", roc_auc_score(y_test, y_test_prob))

param_grid = {
    'max_depth': [5,7,9],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 150],
}

grid_search = GridSearchCV(
    estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    param_grid=param_grid,
    scoring='roc_auc',
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best cross-val AUC:", grid_search.best_score_)

best_model = grid_search.best_estimator_

xgb.plot_importance(best_model, max_num_features=10)
plt.title('Top 10 Feature Importances')
plt.tight_layout()
plt.show()

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
    sim_df_encoded = sim_df_encoded.reindex(columns=X.columns, fill_value=0)

    probs = model.predict_proba(sim_df_encoded)[:, 1]

    recommendations = pd.DataFrame({
        'pitch_type': pitch_options,
        'predicted_success_prob': probs
    }).sort_values(by='predicted_success_prob', ascending=False)

    return recommendations

X_columns = X.columns

current_context = {
    'stand': 'L',
    'p_throws': 'R',
    'on_base': 1,
    'count': 1,
    'outs_when_up': 1,
    'inning': 4,
    'plate_x': 0.1,
    'plate_z': 2.5
}
print(recommend_pitch('Webb, Logan', current_context, best_model, X_columns))




