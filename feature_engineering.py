import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import time

print("Loading clean data...")
start = time.time()

df = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\Btech Project\Crispr_code\data\changeseq_clean.csv",
                 engine='python', encoding='latin-1')

print(f"Loaded {len(df):,} rows in {time.time()-start:.1f}s")

# ─────────────────────────────────────────────────────────────
# SECTION 1 — ONE-HOT ENCODING OF SEQUENCES
# Each position in the 23-nt sequence gets 4 binary features
# A=[1,0,0,0]  T=[0,1,0,0]  G=[0,0,1,0]  C=[0,0,0,1]
# Unknown / gap character '-' → [0,0,0,0]
# ─────────────────────────────────────────────────────────────

BASES   = ['A', 'T', 'G', 'C']
SEQ_LEN = 23

def one_hot_sequence(series, seq_len, prefix):
    """
    Takes a pandas Series of strings, returns a DataFrame
    with seq_len * 4 one-hot columns named prefix_pos{i}_{base}
    """
    col_names = [f"{prefix}_pos{i}_{b}" for i in range(seq_len) for b in BASES]
    matrix = np.zeros((len(series), seq_len * 4), dtype=np.int8)

    for row_idx, seq in enumerate(series):
        seq = str(seq).upper()
        for pos in range(min(len(seq), seq_len)):
            base = seq[pos]
            if base in BASES:
                matrix[row_idx, pos * 4 + BASES.index(base)] = 1

    return pd.DataFrame(matrix, columns=col_names, index=series.index)

print("\nOne-hot encoding sgRNA sequences (23 positions × 4 bases = 92 features)...")
sgrna_ohe   = one_hot_sequence(df['Align.sgRNA'],       SEQ_LEN, 'sgrna')

print("One-hot encoding off-target sequences (92 features)...")
offtgt_ohe  = one_hot_sequence(df['Align.off-target'],  SEQ_LEN, 'offtgt')

print(f"  sgRNA OHE shape   : {sgrna_ohe.shape}")
print(f"  Off-target OHE shape : {offtgt_ohe.shape}")

# ─────────────────────────────────────────────────────────────
# SECTION 2 — POSITION-WISE MISMATCH VECTOR
# For each position, is the sgRNA base ≠ off-target base?
# Gives 23 binary features + extra mismatch-derived features
# ─────────────────────────────────────────────────────────────

print("\nBuilding mismatch features (23 positional + derived)...")

def mismatch_vector(sgrna_series, offtgt_series, seq_len):
    """
    Returns a matrix of shape (n_rows, seq_len) where 
    1 = mismatch at that position, 0 = match
    """
    col_names = [f"mismatch_pos{i}" for i in range(seq_len)]
    matrix    = np.zeros((len(sgrna_series), seq_len), dtype=np.int8)

    for row_idx, (sg, ot) in enumerate(zip(sgrna_series, offtgt_series)):
        sg = str(sg).upper()
        ot = str(ot).upper()
        for pos in range(min(len(sg), len(ot), seq_len)):
            if sg[pos] != ot[pos]:
                matrix[row_idx, pos] = 1

    return pd.DataFrame(matrix, columns=col_names, index=sgrna_series.index)

mismatch_df = mismatch_vector(df['Align.sgRNA'], df['Align.off-target'], SEQ_LEN)

# PAM is at the 3' end — positions 20,21,22 are PAM
# Seed region is positions 13-19 (PAM-proximal, biologically critical)
# Positions 0-12 are PAM-distal (more tolerant of mismatches)

mismatch_df['mismatch_in_seed']    = mismatch_df[[f'mismatch_pos{i}' for i in range(13, 20)]].sum(axis=1)
mismatch_df['mismatch_in_distal']  = mismatch_df[[f'mismatch_pos{i}' for i in range(0, 13)]].sum(axis=1)
mismatch_df['mismatch_in_pam']     = mismatch_df[[f'mismatch_pos{i}' for i in range(20, 23)]].sum(axis=1)

# Position of first mismatch from PAM (lower = more PAM-proximal = more tolerated)
def first_mismatch_from_pam(row):
    for i in range(SEQ_LEN-1, -1, -1):   # scan from PAM end backward
        if row[f'mismatch_pos{i}'] == 1:
            return SEQ_LEN - 1 - i
    return SEQ_LEN  # no mismatch = furthest possible distance

mismatch_df['first_mismatch_from_pam'] = mismatch_df.apply(first_mismatch_from_pam, axis=1)

print(f"  Mismatch feature shape : {mismatch_df.shape}")

# ─────────────────────────────────────────────────────────────
# SECTION 3 — SUMMARY / SCALAR FEATURES
# GC content, bulge count, PAM identity, raw mismatch count
# ─────────────────────────────────────────────────────────────

print("\nBuilding summary features...")

def gc_content(seq):
    seq = str(seq).upper()
    if len(seq) == 0:
        return 0.0
    return (seq.count('G') + seq.count('C')) / len(seq)

def pam_identity(seq):
    """
    Extract last 3 nt (PAM). Score: NGG=1.0, NAG=0.5, other=0.0
    """
    seq = str(seq).upper()
    pam = seq[-3:] if len(seq) >= 3 else 'NNN'
    if pam[1:] == 'GG':
        return 1.0
    elif pam[1:] == 'AG':
        return 0.5
    else:
        return 0.0

scalar_df = pd.DataFrame(index=df.index)

scalar_df['gc_sgrna']          = df['Align.sgRNA'].apply(gc_content)
scalar_df['gc_offtarget']      = df['Align.off-target'].apply(gc_content)
scalar_df['pam_score']         = df['Align.off-target'].apply(pam_identity)
scalar_df['n_mismatches']      = df['Align.#Mismatches']    # already in data
scalar_df['n_bulges']          = df['Align.#Bulges']        # already in data
scalar_df['total_edits']       = df['Align.#Mismatches'] + df['Align.#Bulges']

print(f"  Scalar feature shape : {scalar_df.shape}")

# ─────────────────────────────────────────────────────────────
# SECTION 4 — COMBINE ALL SEQUENCE FEATURES
# ─────────────────────────────────────────────────────────────

print("\nCombining all sequence features...")

feature_df = pd.concat([sgrna_ohe, offtgt_ohe, mismatch_df, scalar_df], axis=1)

# Add genomic coordinates (needed for Step 4 chromatin lookup)
feature_df['chrom']            = df['chrom'].values
feature_df['chromStart']       = df['Align.chromStart'].values
feature_df['chromEnd']         = df['Align.chromEnd'].values
feature_df['strand']           = df['Align.strand'].values

# Add the label
feature_df['label']            = df['label'].values
feature_df['reads']            = df['reads'].values

print(f"\n=== FEATURE MATRIX SUMMARY ===")
print(f"Total rows              : {len(feature_df):,}")
print(f"Total columns           : {len(feature_df.columns)}")
print(f"Sequence features       : {sgrna_ohe.shape[1] + offtgt_ohe.shape[1]} (OHE)")
print(f"Mismatch features       : {mismatch_df.shape[1]}")
print(f"Scalar features         : {scalar_df.shape[1]}")
print(f"Positives (label=1)     : {(feature_df['label']==1).sum():,}")
print(f"Negatives (label=0)     : {(feature_df['label']==0).sum():,}")
print(f"\nSample of scalar features:")
print(feature_df[['gc_sgrna','gc_offtarget','pam_score',
                   'n_mismatches','n_bulges','total_edits']].describe().round(3))

# ─────────────────────────────────────────────────────────────
# SECTION 5 — SAVE
# ─────────────────────────────────────────────────────────────

out = r"C:\Users\HP\OneDrive\Desktop\Btech Project\Crispr_code\data\changeseq_features.csv"
print(f"\nSaving feature matrix...")
feature_df.to_csv(out, index=False)
print(f"✓ Saved → {out}")
print(f"Total time: {time.time()-start:.1f}s")