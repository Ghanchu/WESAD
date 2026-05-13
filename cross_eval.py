from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('train_csv', help='CSV to train on (e.g. full_data.csv)')
parser.add_argument('test_csv',  help='CSV to test on (e.g. stress_features_all.csv)')
parser.add_argument('--no-pdst', action='store_true', help='Exclude PDST windows from MOXIE data')
args = parser.parse_args()

META = {'Participant ID', 'Window ID', 'Label', 'Visit_Type'}

def load(path):
    df = pd.read_csv(path)
    if 'Participant_ID' in df.columns:
        df = df.rename(columns={'Participant_ID': 'Participant ID', 'Window_ID': 'Window ID'})
        print(f'  Detected new format')
        if args.no_pdst and 'Visit_Type' in df.columns:
            before = len(df)
            df = df[df['Visit_Type'] != 'PDST']
            print(f'  Excluded PDST: {before - len(df):,} rows removed, {len(df):,} remaining')
    else:
        print(f'  Detected old format')
    if 'Visit_Type' in df.columns:
        df = df.drop(columns=['Visit_Type'])
    feature_cols = [c for c in df.columns if c not in META]
    nan_count = df[feature_cols].isna().sum().sum()
    if nan_count:
        df[feature_cols] = df[feature_cols].fillna(0)
        print(f'  Imputed {nan_count:,} NaN values with 0')
    return df

print(f'\nLoading train: {args.train_csv}')
train_df = load(args.train_csv)
print(f'  {len(train_df):,} windows, {train_df["Participant ID"].nunique()} subjects')

print(f'\nLoading test: {args.test_csv}')
test_df = load(args.test_csv)
print(f'  {len(test_df):,} windows, {test_df["Participant ID"].nunique()} subjects')

# Use only features present in both datasets
train_feats = set(c for c in train_df.columns if c not in META)
test_feats  = set(c for c in test_df.columns  if c not in META)
common = sorted(train_feats & test_feats)
dropped = (train_feats | test_feats) - train_feats & test_feats
print(f'\nCommon features: {len(common)}')
if train_feats - test_feats:
    print(f'  Dropped (train-only): {sorted(train_feats - test_feats)}')
if test_feats - train_feats:
    print(f'  Dropped (test-only):  {sorted(test_feats - train_feats)}')

# Per-subject z-score normalisation within each dataset independently
for df in (train_df, test_df):
    for subj in df['Participant ID'].unique():
        mask = df['Participant ID'] == subj
        mu  = df.loc[mask, common].mean()
        sig = df.loc[mask, common].std().replace(0, 1)
        df.loc[mask, common] = (df.loc[mask, common] - mu) / sig

X_train = train_df[common].values
y_train = train_df['Label'].values
X_test  = test_df[common].values
y_test  = test_df['Label'].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f'\nTraining LDA on {len(X_train):,} windows ...')
t0 = time.time()
model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train)
print(f'  Done in {time.time()-t0:.1f}s')

# Overall test performance
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
f1  = f1_score(y_test, preds, average='macro')
print(f'\nOverall test performance ({len(X_test):,} windows):')
print(f'  Accuracy : {acc:.4f}  ({acc*100:.2f}%)')
print(f'  F1 (macro): {f1:.4f}')

# Per-subject breakdown on test set
print(f'\nPer-subject breakdown:')
print(f'{"Subject":<15} {"Windows":>8} {"Accuracy":>10} {"F1":>10}')
print('-' * 45)
participants = sorted(test_df['Participant ID'].unique())
accs, f1s = [], []
for subj in participants:
    mask   = test_df['Participant ID'] == subj
    X_s    = scaler.transform(test_df.loc[mask, common].values)

    # re-apply per-subject normalisation already done in-place above
    y_s    = test_df.loc[mask, 'Label'].values
    p_s    = model.predict(X_s)
    a      = accuracy_score(y_s, p_s)
    f      = f1_score(y_s, p_s, average='macro')
    accs.append(a); f1s.append(f)
    print(f'{str(subj):<15} {mask.sum():>8,} {a:>10.4f} {f:>10.4f}')

print('=' * 45)
print(f'{"Mean":<15} {"":>8} {np.mean(accs):>10.4f} {np.mean(f1s):>10.4f}')
