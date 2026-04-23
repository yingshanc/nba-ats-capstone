import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. LOAD DATA (Frozen Source)
df = pd.read_csv('/Users/yingshanchen/Desktop/STAT-390/nba-ats-capstone/data:raw/nba_2008-2025.csv')

# 2. DATA CLEANING
# Dropping rows with missing spread values to ensure baseline integrity
df = df.dropna(subset=['spread'])

# 3. DEFINE THE TARGET: Did the favored team cover the spread?
def calculate_ats_cover(row):
    # Calculate the margin of victory/loss
    margin_home = row['score_home'] - row['score_away']
    margin_away = row['score_away'] - row['score_home']
    
    if row['whos_favored'] == 'home':
        # Home team covers if they win by MORE than the spread
        return 1 if margin_home > row['spread'] else 0
    else:
        # Away team covers if they win by MORE than the spread
        return 1 if margin_away > row['spread'] else 0

df['fav_covered'] = df.apply(calculate_ats_cover, axis=1)

# 4. BASELINE FEATURE ENGINEERING
# Feature 1: The spread itself
# Feature 2: Is the favorite playing at home? (Binary)
df['fav_is_home'] = (df['whos_favored'] == 'home').astype(int)

# 5. TIME-SERIES SPLIT (Protecting the Week 7 "Confirmation" data)
# We train on everything up to 2023 and lock 2024-2025 for final verification.
train_df = df[df['season'] < 2024]
test_df = df[df['season'] >= 2024]

X_train = train_df[['spread', 'fav_is_home']]
y_train = train_df['fav_covered']

X_test = test_df[['spread', 'fav_is_home']]
y_test = test_df['fav_covered']

# 6. TRAIN BASELINE MODEL
model = LogisticRegression()
model.fit(X_train, y_train)

# 7. OUTPUT RESULTS
test_preds = model.predict(X_test)
accuracy = accuracy_score(y_test, test_preds)

print(f"--- Baseline Experiment 001 ---")
print(f"ATS Win Rate on Frozen Test Set: {accuracy:.2%}")
print("\nDetailed Performance:")
print(classification_report(y_test, test_preds))