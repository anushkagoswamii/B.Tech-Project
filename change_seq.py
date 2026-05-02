import pandas as pd
import numpy as np

# ── 1. Load ──────────────────────────────────────────────
df = pd.read_csv(
    r"C:\Users\HP\OneDrive\Desktop\Btech Project\CHANGEseq\include_on_targets\CHANGEseq_CR_Lazzarotto_2020_dataset.csv",
    engine='python',
    encoding='utf-8',
    on_bad_lines='skip'
)
print("=== BASIC INFO ===")
print(f"Total rows     : {len(df):,}")
print(f"Column names   : {df.columns.tolist()}")

# ── 2. Check missing values ───────────────────────────────
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

# ── 3. Reads distribution ─────────────────────────────────
print("\n=== READS DISTRIBUTION ===")
print(f"Positives (reads > 0) : {(df['reads'] > 0).sum():,}")
print(f"Negatives (reads = 0) : {(df['reads'] == 0).sum():,}")
print(f"Positive rate         : {(df['reads'] > 0).mean()*100:.2f}%")
print(f"Max reads             : {df['reads'].max():,}")

# ── 4. Sequence length check ──────────────────────────────
print("\n=== SEQUENCE LENGTHS ===")
print("sgRNA lengths:\n",
      df['Align.sgRNA'].astype(str).str.len().value_counts().head())
print("Off-target lengths:\n",
      df['Align.off-target'].astype(str).str.len().value_counts().head())

# ── 5. Create binary label ────────────────────────────────
df['label'] = (df['reads'] > 0).astype(int)

# ── 6. Drop rows missing key columns ─────────────────────
before = len(df)
df = df.dropna(subset=[
    'Align.sgRNA',
    'Align.off-target',
    'chrom',
    'Align.chromStart',
    'Align.chromEnd'
])
print(f"\nDropped {before - len(df):,} incomplete rows")
print(f"Clean dataset  : {len(df):,} rows")

# ── 7. Save ───────────────────────────────────────────────
df.to_csv("data/changeseq_clean.csv", index=False)
print("\n✓ Saved → data/changeseq_clean.csv")

print("\n=== FULL LABEL DISTRIBUTION ===")
print(f"Positives (reads > 0) : {(df['reads'] > 0).sum():,}")
print(f"Negatives (reads = 0) : {(df['reads'] == 0).sum():,}")
print(f"Positive rate         : {(df['reads'] > 0).mean()*100:.3f}%")
print(f"\nClass imbalance ratio : 1 positive : {int((df['reads']==0).sum()/(df['reads']>0).sum())} negatives")