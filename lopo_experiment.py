#!/usr/bin/env python3
# ============================================================================
# Leave-One-Person-Out (LOPO) 15-fold experiment
# ============================================================================
# Pipeline:
#   - Reads Data/Original/pXX/dayYY/annotation.txt (gaze = columns 26-28)
#   - Splits strictly by participant: windows are built within a participant,
#     so all windows of one person fall entirely in either train or test
#     (never both), eliminating cross-participant leakage
#   - 15-fold LOPO: each fold holds out one participant for testing, trains
#     on the remaining 14
#   - Three models per fold: CNN-only, LSTM-only, CNN+LSTM
#   - Each fold result is written to CSV immediately (crash-resilient; the run
#     can be resumed)
#   - Final statistics: mean accuracy + 95% CI, McNemar, Friedman, Cohen's d,
#     and the aggregated confusion matrix
#
# Leakage controls:
#   1. No window overlap between train and test (per-participant windowing)
#   2. No participant appears in both train and test (LOPO)
#   3. Windows are kept in temporal order within each participant
#
# Label thresholds use the raw gaze-magnitude standard deviation (37 / 50).
# ============================================================================

import os, json, time
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)

# Adjust ORIG to the location of the MPIIGaze "Data/Original" directory.
ORIG = Path('/kaggle/input/datasets/dhruv413/mpiigaze/MPIIGaze/Data/Original')
OUT  = Path('/kaggle/working')
WIN  = 10
EPOCHS = 30
BATCH = 128

# Label thresholds on the gaze-magnitude standard deviation (sigma):
# sigma < 37 -> Fixation (0); 37 <= sigma <= 50 -> Smooth Pursuit (2);
# sigma > 50 -> Saccade (1).
THR_LOW, THR_HIGH = 37, 50

# ============================================================================
# 1. READ DATA PER PARTICIPANT (preserving participant identity)
# ============================================================================
print("="*70); print("[1/4] Reading data (per participant)..."); print("="*70)

def read_person(p_folder):
    """Return all gaze vectors of one participant, in temporal order."""
    gaze = []
    for day in sorted(p_folder.iterdir()):
        if not day.is_dir():
            continue
        ann = day / 'annotation.txt'
        if not ann.exists():          # skip calibration etc.
            continue
        with open(ann) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 29:
                    continue
                try:
                    gaze.append([float(parts[26]), float(parts[27]), float(parts[28])])
                except ValueError:
                    continue
    return np.array(gaze)

def make_sequences(gaze):
    """Build 10-frame sliding windows + sigma labels within one participant."""
    X, y = [], []
    for i in range(len(gaze) - WIN):
        w = gaze[i:i+WIN]
        mags = np.linalg.norm(w, axis=1)
        s = np.std(mags)
        if   s < THR_LOW:  lab = 0   # Fixation
        elif s > THR_HIGH: lab = 1   # Saccade
        else:              lab = 2   # Smooth Pursuit
        X.append(w); y.append(lab)
    if not X:
        return np.empty((0,WIN,3)), np.empty((0,),dtype=int)
    return np.array(X), np.array(y)

persons = sorted([d for d in ORIG.iterdir() if d.is_dir()])
person_data = {}
for p in persons:
    g = read_person(p)
    X, y = make_sequences(g)
    person_data[p.name] = (X, y)
    print(f"  {p.name}: {len(g)} frames -> {len(X)} sequences, "
          f"class distribution {np.bincount(y, minlength=3).tolist()}")

all_names = list(person_data.keys())
print(f"\n  {len(all_names)} participants total")

# ============================================================================
# 2. MODEL DEFINITIONS (three variants for the ablation)
# ============================================================================
def build_cnn():
    m = keras.Sequential([
        layers.Input(shape=(WIN,3)),
        layers.Conv1D(64,2,padding='same',activation='relu'),
        layers.BatchNormalization(), layers.Dropout(0.2),
        layers.Conv1D(128,2,padding='same',activation='relu'),
        layers.BatchNormalization(), layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128,activation='relu'), layers.Dropout(0.4),
        layers.Dense(3,activation='softmax')])
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

def build_lstm():
    m = keras.Sequential([
        layers.Input(shape=(WIN,3)),
        layers.LSTM(128,return_sequences=True), layers.Dropout(0.3),
        layers.LSTM(64), layers.Dropout(0.3),
        layers.Dense(128,activation='relu'), layers.Dropout(0.4),
        layers.Dense(3,activation='softmax')])
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

def build_hybrid():
    m = keras.Sequential([
        layers.Input(shape=(WIN,3)),
        layers.Conv1D(64,3,padding='same',activation='relu'),
        layers.BatchNormalization(), layers.Dropout(0.2),
        layers.Conv1D(128,3,padding='same',activation='relu'),
        layers.BatchNormalization(), layers.Dropout(0.2),
        layers.LSTM(128,return_sequences=True), layers.Dropout(0.3),
        layers.LSTM(64), layers.Dropout(0.3),
        layers.Dense(128,activation='relu'), layers.Dropout(0.4),
        layers.Dense(3,activation='softmax')])
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

builders = {'CNN': build_cnn, 'LSTM': build_lstm, 'CNN+LSTM': build_hybrid}

# ============================================================================
# 3. LOPO LOOP (each fold result written immediately to CSV)
# ============================================================================
print("\n"+"="*70); print("[2/4] Starting 15-fold LOPO..."); print("="*70)

fold_csv = OUT/'lopo_fold_results.csv'
preds_dir = OUT/'lopo_preds'; preds_dir.mkdir(exist_ok=True)
rows = []

# Resume support: skip folds that are already done.
done = set()
if fold_csv.exists():
    prev = pd.read_csv(fold_csv)
    rows = prev.to_dict('records')
    done = set(prev['test_person'].tolist())
    print(f"  Resuming: {len(done)} folds already done, skipping: {sorted(done)}")

for fi, test_p in enumerate(all_names):
    if test_p in done:
        continue
    t0 = time.time()
    # train = all others, test = test_p
    Xtr = np.concatenate([person_data[n][0] for n in all_names if n != test_p])
    ytr = np.concatenate([person_data[n][1] for n in all_names if n != test_p])
    Xte, yte = person_data[test_p]
    if len(Xte) == 0:
        print(f"  [{test_p}] empty, skipped"); continue

    print(f"\n  Fold {fi+1}/15 - test={test_p} "
          f"(train {len(Xtr)}, test {len(Xte)})")

    row = {'fold': fi+1, 'test_person': test_p,
           'n_train': int(len(Xtr)), 'n_test': int(len(Xte))}
    fold_preds = {'y_true': yte.tolist()}

    for mname, build in builders.items():
        np.random.seed(SEED); tf.random.set_seed(SEED)
        model = build()
        model.fit(Xtr, ytr, validation_split=0.1, epochs=EPOCHS,
                  batch_size=BATCH, verbose=0)
        yp = np.argmax(model.predict(Xte, verbose=0), axis=1)
        acc = float(np.mean(yp == yte))
        row[f'acc_{mname}'] = acc
        fold_preds[mname] = yp.tolist()
        print(f"    {mname:10}: {acc*100:.2f}%")
        del model; keras.backend.clear_session()

    row['time_sec'] = round(time.time()-t0, 1)
    rows.append(row)
    # write immediately
    pd.DataFrame(rows).to_csv(fold_csv, index=False)
    with open(preds_dir/f'preds_{test_p}.json','w') as f:
        json.dump(fold_preds, f)
    print(f"    -> saved ({row['time_sec']}s)")

df = pd.DataFrame(rows)
print("\n"+"="*70); print("[3/4] LOPO complete - summary"); print("="*70)
print(df.to_string(index=False))

# ============================================================================
# 4. STATISTICS (over the pooled predictions of all folds)
# ============================================================================
print("\n"+"="*70); print("[4/4] Statistics"); print("="*70)

# Macro (unweighted) and fold-size-weighted mean accuracy.
for m in builders:
    accs = df[f'acc_{m}'].values
    ns   = df['n_test'].values
    macro = accs.mean()
    weighted = np.average(accs, weights=ns)
    sd = accs.std(ddof=1)
    ci = 1.96*sd/np.sqrt(len(accs))
    print(f"  {m:10}: macro {macro*100:.2f}% (SD {sd*100:.2f}) "
          f"95%CI [{(macro-ci)*100:.2f}, {(macro+ci)*100:.2f}] | "
          f"weighted {weighted*100:.2f}%")

# Pool predictions across folds (for McNemar / Friedman / confusion matrix).
yt_all, cnn_all, lstm_all, hyb_all = [], [], [], []
for test_p in all_names:
    fp = preds_dir/f'preds_{test_p}.json'
    if not fp.exists(): continue
    d = json.load(open(fp))
    yt_all  += d['y_true']
    cnn_all += d['CNN']
    lstm_all+= d['LSTM']
    hyb_all += d['CNN+LSTM']
yt_all=np.array(yt_all); cnn_all=np.array(cnn_all)
lstm_all=np.array(lstm_all); hyb_all=np.array(hyb_all)

print(f"\n  Aggregate test N = {len(yt_all)}")
print(f"  CNN+LSTM aggregate accuracy: {np.mean(hyb_all==yt_all)*100:.2f}%")

# McNemar's test (continuity-corrected).
from statsmodels.stats.contingency_tables import mcnemar
def mcnemar_test(a, b, yt, na, nb):
    a_ok=(a==yt); b_ok=(b==yt)
    n01=int(np.sum(~a_ok&b_ok)); n10=int(np.sum(a_ok&~b_ok))
    tbl=np.array([[int(np.sum(a_ok&b_ok)), n10],[n01, int(np.sum(~a_ok&~b_ok))]])
    r=mcnemar(tbl, exact=False, correction=True)
    print(f"  McNemar {na} vs {nb}: chi2={r.statistic:.2f}, p={r.pvalue:.4e} "
          f"(discordant n01={n01}, n10={n10})")
    return r.statistic, r.pvalue
print()
mcnemar_test(cnn_all,  hyb_all, yt_all, "CNN",  "CNN+LSTM")
mcnemar_test(lstm_all, hyb_all, yt_all, "LSTM", "CNN+LSTM")

# Friedman test (over the per-fold accuracies).
from scipy.stats import friedmanchisquare
fr = friedmanchisquare(df['acc_CNN'], df['acc_LSTM'], df['acc_CNN+LSTM'])
print(f"\n  Friedman: chi2={fr.statistic:.2f}, p={fr.pvalue:.4e}")

# Cohen's d (over the per-fold accuracy distributions).
def cohens_d(x,y):
    nx,ny=len(x),len(y); vx,vy=np.var(x,ddof=1),np.var(y,ddof=1)
    sp=np.sqrt(((nx-1)*vx+(ny-1)*vy)/(nx+ny-2))
    return (np.mean(x)-np.mean(y))/sp if sp>0 else 0
print(f"\n  Cohen's d (fold-level):")
print(f"    CNN+LSTM vs CNN : {cohens_d(df['acc_CNN+LSTM'], df['acc_CNN']):.4f}")
print(f"    CNN+LSTM vs LSTM: {cohens_d(df['acc_CNN+LSTM'], df['acc_LSTM']):.4f}")

# Aggregated confusion matrix and classification report (CNN+LSTM).
print(f"\n  CNN+LSTM confusion matrix (aggregate):")
print(confusion_matrix(yt_all, hyb_all))
print(f"\n  Classification report (CNN+LSTM):")
print(classification_report(yt_all, hyb_all,
      target_names=['Fixation','Saccade','SmoothPursuit'], digits=4))

# Save pooled predictions.
np.save(OUT/'yt_all.npy', yt_all)
np.save(OUT/'cnn_all.npy', cnn_all)
np.save(OUT/'lstm_all.npy', lstm_all)
np.save(OUT/'hyb_all.npy', hyb_all)
print("\n  Saved: lopo_fold_results.csv, lopo_preds/, *.npy")
print("\n"+"="*70); print("DONE"); print("="*70)
