#!/usr/bin/env python3
# ============================================================================
# Training history for a single representative LOPO fold
# ============================================================================
# Pipeline:
#   - Same data reading and per-participant split as the main LOPO experiment
#   - Trains a single fold: TEST_PERSON is the held-out participant
#     (default p11, a representative median-performance fold)
#   - Records the 30-epoch loss & accuracy history of the CNN+LSTM model
#   - Output: fig4_training_history.csv (train/val loss + acc, 30 rows)
#
# Runtime: a few minutes (one fold, one model), not the full 15-fold run.
# ============================================================================

import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)

# Adjust ORIG to the location of the MPIIGaze "Data/Original" directory.
ORIG = Path('/kaggle/input/datasets/dhruv413/mpiigaze/MPIIGaze/Data/Original')
OUT  = Path('/kaggle/working')
WIN  = 10
EPOCHS = 30
BATCH = 128
THR_LOW, THR_HIGH = 37, 50

TEST_PERSON = 'p11'   # representative median-performance fold (change if needed)

print("="*70)
print(f"Training history for a single fold - held-out = {TEST_PERSON}")
print("="*70)

# --- Data reading (same as the main LOPO experiment) ---
def read_person(p_folder):
    gaze = []
    for day in sorted(p_folder.iterdir()):
        if not day.is_dir(): continue
        ann = day/'annotation.txt'
        if not ann.exists(): continue
        with open(ann) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 29: continue
                try:
                    gaze.append([float(parts[26]),float(parts[27]),float(parts[28])])
                except ValueError: continue
    return np.array(gaze)

def make_seq(gaze):
    X,y = [],[]
    for i in range(len(gaze)-WIN):
        w = gaze[i:i+WIN]
        s = np.std(np.linalg.norm(w,axis=1))
        if   s < THR_LOW:  lab=0
        elif s > THR_HIGH: lab=1
        else:              lab=2
        X.append(w); y.append(lab)
    return np.array(X), np.array(y)

persons = sorted([d for d in ORIG.iterdir() if d.is_dir()])
pdata = {}
for p in persons:
    X,y = make_seq(read_person(p))
    pdata[p.name] = (X,y)
    print(f"  {p.name}: {len(X)} sequences")

# --- train = all others, test = TEST_PERSON ---
names = list(pdata.keys())
Xtr = np.concatenate([pdata[n][0] for n in names if n != TEST_PERSON])
ytr = np.concatenate([pdata[n][1] for n in names if n != TEST_PERSON])
Xte, yte = pdata[TEST_PERSON]
print(f"\n  Train: {len(Xtr)}, Test ({TEST_PERSON}): {len(Xte)}")

# --- CNN+LSTM model (identical to the main hybrid model) ---
np.random.seed(SEED); tf.random.set_seed(SEED)
model = keras.Sequential([
    layers.Input(shape=(WIN,3)),
    layers.Conv1D(64,3,padding='same',activation='relu'),
    layers.BatchNormalization(), layers.Dropout(0.2),
    layers.Conv1D(128,3,padding='same',activation='relu'),
    layers.BatchNormalization(), layers.Dropout(0.2),
    layers.LSTM(128,return_sequences=True), layers.Dropout(0.3),
    layers.LSTM(64), layers.Dropout(0.3),
    layers.Dense(128,activation='relu'), layers.Dropout(0.4),
    layers.Dense(3,activation='softmax')
])
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(f"  Params: {model.count_params():,}")

print("\n  Training (30 epochs)...")
history = model.fit(Xtr, ytr, validation_split=0.1, epochs=EPOCHS,
                    batch_size=BATCH, verbose=1)

# --- Test accuracy ---
te_loss, te_acc = model.evaluate(Xte, yte, verbose=0)
print(f"\n  {TEST_PERSON} test accuracy: {te_acc*100:.2f}%")

# --- Save history ---
h = history.history
df = pd.DataFrame({
    'epoch': range(1, EPOCHS+1),
    'train_loss': h['loss'],
    'val_loss':   h['val_loss'],
    'train_acc':  h['accuracy'],
    'val_acc':    h['val_accuracy'],
})
df.to_csv(OUT/'fig4_training_history.csv', index=False)

print("\n" + "="*70)
print("First and last epoch values:")
print("="*70)
print(df.iloc[[0,-1]].to_string(index=False))
print(f"\n  Lowest val_loss:  epoch {df['val_loss'].idxmin()+1}, {df['val_loss'].min():.4f}")
print(f"  Highest val_acc:  epoch {df['val_acc'].idxmax()+1}, {df['val_acc'].max():.4f}")
print(f"\n  Saved: fig4_training_history.csv")
print(f"  TEST_PERSON = {TEST_PERSON}, test_acc = {te_acc*100:.2f}%")
print("="*70)
