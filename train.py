from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
import time

print('Loading full_data.csv ...')
df = pd.read_csv('full_data.csv')
print(f'Loaded {len(df):,} windows across {df["Participant ID"].nunique()} subjects.')
print(f'Label distribution: {dict(df["Label"].value_counts().rename({0: "non-stress", 1: "stress"}))}')
print()

# Per-subject z-score normalisation — removes inter-subject physiological
# baseline differences so LDA focuses on relative within-subject changes.
# Each subject's features are normalised using only that subject's own stats,
# so there is no data leakage across subjects.
feature_cols = [c for c in df.columns if c not in ['Participant ID', 'Window ID', 'Label']]
for subj in df['Participant ID'].unique():
    mask = df['Participant ID'] == subj
    mu   = df.loc[mask, feature_cols].mean()
    sig  = df.loc[mask, feature_cols].std().replace(0, 1)
    df.loc[mask, feature_cols] = (df.loc[mask, feature_cols] - mu) / sig

participants = sorted(df['Participant ID'].unique())
n = len(participants)
accuracies = []
f1_scores  = []
pipeline_start = time.time()

for idx, test_participant in enumerate(participants, 1):
    t0 = time.time()
    train_df = df[df['Participant ID'] != test_participant]
    test_df  = df[df['Participant ID'] == test_participant]

    X_train = train_df.drop(['Participant ID', 'Window ID', 'Label'], axis=1)
    y_train = train_df['Label']
    X_test  = test_df.drop(['Participant ID', 'Window ID', 'Label'], axis=1)
    y_test  = test_df['Label']

    print(f'[{idx}/{n}] Test: {test_participant}  |  '
          f'train={len(X_train):,} windows  test={len(X_test):,} windows')

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = LinearDiscriminantAnalysis()
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average='macro')
    accuracies.append(acc)
    f1_scores.append(f1)

    print(f'         Accuracy: {acc:.4f}  |  F1 (macro): {f1:.4f}  |  {time.time()-t0:.1f}s')

print()
print('='*45)
print(f'LOSO Mean Accuracy : {np.mean(accuracies):.4f}  ({np.mean(accuracies)*100:.2f}%)')
print(f'LOSO Mean F1 (macro): {np.mean(f1_scores):.4f}')
print(f'Total time         : {time.time()-pipeline_start:.1f}s')
print('='*45)