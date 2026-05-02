import pandas as pd
import numpy as np
from itertools import combinations
import time

DATA = r"C:\Users\HP\OneDrive\Desktop\Btech Project\Crispr_code\data"
OUT  = r"C:\Users\HP\OneDrive\Desktop\Btech Project\Crispr_code\outputs"
start = time.time()

# ══════════════════════════════════════════════════════════════
# SICKLE CELL ANAEMIA CONTEXT
# Gene     : HBB (Haemoglobin Beta) on chromosome 11
# Mutation : c.20A>T — GAG→GTG at codon 6 (Glu6Val)
# Effect   : Mutant haemoglobin polymerises → sickle-shaped
#            red blood cells → vaso-occlusion → organ damage
# Strategy : CRISPR-Cas9 with this sgRNA cuts near the mutation
#            A repair template corrects GTG back to GAG
# Risk     : sgRNA may cut elsewhere in genome (off-targets)
# Our job  : Predict which off-target sites are highest risk
# ══════════════════════════════════════════════════════════════

HBB_SGRNA = "CTTGCCCCACAGGGCAGTAA"  # 20-nt guide targeting HBB
PAM       = "NGG"
SEQ_LEN   = 20
BASES     = ['A', 'T', 'G', 'C']

print("="*60)
print("SICKLE CELL ANAEMIA — CRISPR OFF-TARGET SAFETY ANALYSIS")
print("="*60)
print(f"sgRNA          : {HBB_SGRNA}")
print(f"Target gene    : HBB (Haemoglobin Beta)")
print(f"Chromosome     : chr11p15.4")
print(f"Disease        : Sickle Cell Anaemia (HbSS)")
print(f"Mutation       : c.20A>T (GAG→GTG, Glu6Val)")
print(f"CRISPR goal    : Correct mutant HBB allele via HDR")

# ══════════════════════════════════════════════════════════════
# GENERATE CANDIDATE OFF-TARGET SEQUENCES
# WHY: Cas9 can tolerate mismatches between sgRNA and DNA.
# Standard cutoff is up to 4 mismatches (beyond that, cleavage
# is very rare). We generate ALL possible sequences with 1-4
# mismatches from the HBB sgRNA. Each is a candidate off-target.
# Total: ~425,000 candidates — we predict risk for all of them.
# ══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("Generating candidate off-target sequences...")
print("="*60)

def generate_mismatch_variants(sgrna, max_mismatches=4):
    """
    Generate all sequences with 1 to max_mismatches substitutions
    from the input sgRNA. Each variant = one candidate off-target.

    For each combination of positions, we substitute each position
    with all 3 alternative bases (not the original base).

    Returns list of (variant_sequence, mismatch_positions, n_mismatches)
    """
    variants = []
    seq = list(sgrna)
    n   = len(seq)

    # On-target itself (0 mismatches) — always risk score should be high
    variants.append({
        'offtarget_seq'   : sgrna,
        'n_mismatches'    : 0,
        'mismatch_positions': '',
        'is_ontarget'     : True
    })

    for n_mm in range(1, max_mismatches + 1):
        print(f"  Generating {n_mm}-mismatch variants...", end=' ')
        count = 0

        for positions in combinations(range(n), n_mm):
            # For each combination of mismatch positions,
            # substitute each with all 3 alternative bases
            alt_bases_per_pos = []
            for p in positions:
                orig = seq[p]
                alts = [b for b in BASES if b != orig]
                alt_bases_per_pos.append(alts)

            # Generate all combinations of alternative bases
            from itertools import product
            for alt_combo in product(*alt_bases_per_pos):
                new_seq = seq.copy()
                for p, b in zip(positions, alt_combo):
                    new_seq[p] = b

                variants.append({
                    'offtarget_seq'    : ''.join(new_seq),
                    'n_mismatches'     : n_mm,
                    'mismatch_positions': ','.join(map(str, positions)),
                    'is_ontarget'      : False
                })
                count += 1

        print(f"{count:,} generated")

    return variants

variants = generate_mismatch_variants(HBB_SGRNA, max_mismatches=4)
df = pd.DataFrame(variants)
df['sgrna_seq'] = HBB_SGRNA

print(f"\nTotal candidate off-target sequences: {len(df):,}")
print(f"Breakdown:")
for n in range(5):
    c = (df['n_mismatches']==n).sum()
    print(f"  {n} mismatches: {c:,}")

# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# Same exact features as our CHANGE-seq pipeline
# so the model input matches what it was trained on
# ══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("Extracting sequence features...")
print("="*60)

# ── One-hot encoding ──────────────────────────────────────────
def one_hot_sequence(seqs, seq_len, prefix):
    col_names = [f"{prefix}_pos{i}_{b}"
                 for i in range(seq_len) for b in BASES]
    matrix = np.zeros((len(seqs), seq_len * 4), dtype=np.int8)
    for row_idx, seq in enumerate(seqs):
        seq = str(seq).upper()
        for pos in range(min(len(seq), seq_len)):
            b = seq[pos]
            if b in BASES:
                matrix[row_idx, pos*4 + BASES.index(b)] = 1
    return pd.DataFrame(matrix, columns=col_names)

print("  One-hot encoding sgRNA sequences (92 features)...")
sgrna_ohe  = one_hot_sequence(df['sgrna_seq'],    SEQ_LEN, 'sgrna')

print("  One-hot encoding off-target sequences (92 features)...")
offtgt_ohe = one_hot_sequence(df['offtarget_seq'], SEQ_LEN, 'offtgt')

# ── Mismatch vector ───────────────────────────────────────────
print("  Building mismatch position features (23 features)...")

def mismatch_vector(sgrna_seqs, offtgt_seqs, seq_len):
    col_names = [f"mismatch_pos{i}" for i in range(seq_len)]
    matrix    = np.zeros((len(sgrna_seqs), seq_len), dtype=np.int8)
    for i, (sg, ot) in enumerate(zip(sgrna_seqs, offtgt_seqs)):
        sg = str(sg).upper()
        ot = str(ot).upper()
        for pos in range(min(len(sg), len(ot), seq_len)):
            if sg[pos] != ot[pos]:
                matrix[i, pos] = 1
    return pd.DataFrame(matrix, columns=col_names)

mm_df = mismatch_vector(df['sgrna_seq'], df['offtarget_seq'], SEQ_LEN)

# Biologically important derived mismatch features
# Seed region = positions 13-19 (PAM-proximal) — mismatches here
# are MORE tolerated by Cas9 than in the distal region
mm_df['mismatch_in_seed']   = mm_df[[f'mismatch_pos{i}'
                                      for i in range(13,20)]].sum(axis=1)
mm_df['mismatch_in_distal'] = mm_df[[f'mismatch_pos{i}'
                                      for i in range(0,13)]].sum(axis=1)
mm_df['mismatch_in_pam']    = mm_df[[f'mismatch_pos{i}'
                                      for i in range(17,20)]].sum(axis=1)

def first_mm_from_pam(row):
    for i in range(SEQ_LEN-1, -1, -1):
        if row[f'mismatch_pos{i}'] == 1:
            return SEQ_LEN - 1 - i
    return SEQ_LEN
mm_df['first_mismatch_from_pam'] = mm_df.apply(first_mm_from_pam, axis=1)

# ── Scalar features ───────────────────────────────────────────
print("  Computing scalar features (GC, PAM, bulges)...")

def gc_content(seq):
    seq = str(seq).upper()
    return (seq.count('G') + seq.count('C')) / max(len(seq), 1)

def pam_score(seq):
    seq = str(seq).upper()
    pam = seq[-3:] if len(seq) >= 3 else 'NNN'
    if pam[1:] == 'GG': return 1.0
    elif pam[1:] == 'AG': return 0.5
    return 0.0

scalar = pd.DataFrame({
    'gc_sgrna'     : df['sgrna_seq'].apply(gc_content),
    'gc_offtarget' : df['offtarget_seq'].apply(gc_content),
    'pam_score'    : df['offtarget_seq'].apply(pam_score),
    'n_mismatches' : df['n_mismatches'],
    'n_bulges'     : 0,  # no bulges in substitution-only variants
    'total_edits'  : df['n_mismatches']
})

# ── Combine all features ──────────────────────────────────────
print("  Combining feature matrix...")
features = pd.concat([sgrna_ohe, offtgt_ohe, mm_df, scalar], axis=1)

# Keep metadata alongside features
features['offtarget_seq']     = df['offtarget_seq'].values
features['sgrna_seq']         = df['sgrna_seq'].values
features['n_mismatches']      = df['n_mismatches'].values
features['mismatch_positions'] = df['mismatch_positions'].values
features['is_ontarget']       = df['is_ontarget'].values

print(f"\n=== FEATURE MATRIX SUMMARY ===")
print(f"Candidates     : {len(features):,}")
print(f"Feature columns: {len(features.columns)}")
print(f"On-target      : {features['is_ontarget'].sum()}")
print(f"1-mismatch     : {(features['n_mismatches']==1).sum():,}")
print(f"2-mismatch     : {(features['n_mismatches']==2).sum():,}")
print(f"3-mismatch     : {(features['n_mismatches']==3).sum():,}")
print(f"4-mismatch     : {(features['n_mismatches']==4).sum():,}")

# Save
out_path = f"{DATA}\\scd_candidates_features.csv"
features.to_csv(out_path, index=False)
print(f"\n✓ Saved → {out_path}")
print(f"Runtime: {time.time()-start:.1f}s")