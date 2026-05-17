from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--no-temp', action='store_true', help='Exclude temperature features from training and testing')
parser.add_argument('--no-resp', action='store_true', help='Exclude respiration features from training and testing')
parser.add_argument('--balanced', action='store_true', help='Weight classes inversely to their frequency to handle imbalance')
parser.add_argument('--min-resp-coverage', type=float, default=0.0,
                    help='Drop participants whose RESP coverage falls below this fraction (e.g. 0.5 keeps only those with >=50%% non-NaN RESP)')
parser.add_argument('csv', help='Path to input CSV file')
args = parser.parse_args()

# ── load & normalise format ─────────────────────────────────────────────────

def _load_csv(path):
    df = pd.read_csv(path)
    # New format uses underscores; rename to the canonical space-separated names
    if 'Participant_ID' in df.columns:
        df = df.rename(columns={'Participant_ID': 'Participant ID', 'Window_ID': 'Window ID'})
        print(f'  Detected new format (stress_features_all)')
    else:
        print(f'  Detected old format (full_data)')
    if 'Visit_Type' in df.columns:
        df = df.drop(columns=['Visit_Type'])

    if args.min_resp_coverage > 0 and 'RESP_inhal_mean' in df.columns:
        coverage = df.groupby('Participant ID')['RESP_inhal_mean'].apply(
            lambda s: s.notna().mean()
        )
        keep = coverage[coverage >= args.min_resp_coverage].index
        dropped = sorted(set(coverage.index) - set(keep))
        before = len(df)
        df = df[df['Participant ID'].isin(keep)]
        print(f'  RESP coverage filter (>= {args.min_resp_coverage:.0%}): '
              f'kept {len(keep)} subjects, dropped {len(dropped)} ({dropped})')
        print(f'  Rows: {before:,} -> {len(df):,}')

    feature_cols = [c for c in df.columns if c not in ['Participant ID', 'Window ID', 'Label']]
    nan_count = df[feature_cols].isna().sum().sum()
    if nan_count:
        df[feature_cols] = df[feature_cols].fillna(0)
        print(f'  Imputed {nan_count:,} NaN feature values with 0')
    return df

csv_path = args.csv
print(f'Loading {csv_path} ...')
df = _load_csv(csv_path)
print(f'Loaded {len(df):,} windows across {df["Participant ID"].nunique()} subjects.')
print(f'Label distribution: {dict(df["Label"].value_counts().rename({0: "non-stress", 1: "stress"}))}')

if args.no_temp:
    temp_cols = [c for c in df.columns if c.startswith('TEMP_')]
    if temp_cols:
        df = df.drop(columns=temp_cols)
        print(f'Temperature features disabled: dropped {temp_cols}')

if args.no_resp:
    resp_cols = [c for c in df.columns if c.startswith('RESP_') or c.startswith('RSP_')]
    if resp_cols:
        df = df.drop(columns=resp_cols)
        print(f'Respiration features disabled: dropped {resp_cols}')

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

    priors = [0.5, 0.5] if args.balanced else None
    model = LinearDiscriminantAnalysis(priors=priors)
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