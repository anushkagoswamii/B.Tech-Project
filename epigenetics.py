import pandas as pd
import numpy as np
import time

start = time.time()

DATA = r"C:\Users\HP\OneDrive\Desktop\Btech Project\Crispr_code\data"

# ══════════════════════════════════════════════════════════════
# STEP 1 — Load CHANGE-seq coordinates only (chunked)
# WHY: File has 4.9M rows with occasional malformed cells
# from Excel export. Chunked reading skips bad lines safely.
# ══════════════════════════════════════════════════════════════
print("="*60)
print("Loading CHANGE-seq coordinates...")
print("="*60)

CHUNKSIZE = 500_000
chunks = []

for i, chunk in enumerate(pd.read_csv(
        f"{DATA}\\changeseq_features.csv",
        usecols=['chrom', 'chromStart', 'chromEnd', 'label', 'reads'],
        engine='python',
        encoding='latin-1',
        on_bad_lines='skip',
        chunksize=CHUNKSIZE)):
    chunks.append(chunk)
    print(f"  Loaded chunk {i+1}: {len(chunk):,} rows")

df = pd.concat(chunks, ignore_index=True)
df = df.reset_index(drop=True)

# Ensure correct types
df['chromStart'] = pd.to_numeric(df['chromStart'], errors='coerce')
df['chromEnd']   = pd.to_numeric(df['chromEnd'],   errors='coerce')
df['label']      = pd.to_numeric(df['label'],      errors='coerce')
df               = df.dropna(subset=['chrom','chromStart','chromEnd','label'])
df['chromStart'] = df['chromStart'].astype(int)
df['chromEnd']   = df['chromEnd'].astype(int)
df['label']      = df['label'].astype(int)
df               = df.reset_index(drop=True)

print(f"\nLoaded        : {len(df):,} rows")
print(f"Positives     : {df['label'].sum():,}")
print(f"Negatives     : {(df['label']==0).sum():,}")
print(f"Chromosomes   : {sorted(df['chrom'].unique())[:6]} ...")
# ══════════════════════════════════════════════════════════════
# STEP 2 — Load narrowPeak BED files
# WHY: These are the experimentally validated regions where
# each epigenetic mark is present in K562 cells.
# Each file is ~10-50MB — loads in seconds.
# ══════════════════════════════════════════════════════════════
def load_narrowpeak(path, name):
    """
    Load a narrowPeak BED file from ENCODE.
    Columns: chrom, start, end, name, score, strand,
             signalValue, pValue, qValue, peak
    We only need chrom, start, end, signalValue.
    """
    print(f"\nLoading {name}...")
    bed = pd.read_csv(
        path, sep='\t', header=None,
        names=['chrom','start','end','name','score','strand',
               'signal','pValue','qValue','peak'],
        usecols=['chrom','start','end','signal'],
        dtype={'chrom': str, 'start': int,
               'end': int, 'signal': float}
    )
    # Keep only standard chromosomes — removes random contigs
    valid = [f'chr{i}' for i in list(range(1, 23)) + ['X', 'Y']]
    bed   = bed[bed['chrom'].isin(valid)].reset_index(drop=True)
    print(f"  Peaks loaded  : {len(bed):,}")
    print(f"  Signal range  : {bed['signal'].min():.2f} – "
          f"{bed['signal'].max():.2f}")
    return bed

# ── Load H3K27ac (active enhancer / promoter mark) ────────────
# WHY: H3K27ac marks regions where chromatin is open and
# transcriptionally active. Sites in H3K27ac peaks are more
# accessible to Cas9 → higher off-target cleavage risk.
# This is one of the core epigenetic features in your synopsis.
try:
    h3k27ac_bed = load_narrowpeak(
        f"{DATA}\\h3k27ac_k562.bed", "H3K27ac K562")
    has_h3k27ac = True
except FileNotFoundError:
    print("\n  WARNING: h3k27ac_k562.bed not found in data folder.")
    print("  Skipping H3K27ac. Download from ENCODE ENCSR000AKP.")
    has_h3k27ac = False

# ── Load DNase-seq (chromatin accessibility) ──────────────────
# WHY: DNase-seq identifies open chromatin regions genome-wide.
# Open chromatin = nucleosomes removed = Cas9 can bind freely.
# This is the most direct measure of physical DNA accessibility.
try:
    dnase_bed = load_narrowpeak(
        f"{DATA}\\dnase_k562.bed", "DNase-seq K562")
    has_dnase = True
except FileNotFoundError:
    print("\n  WARNING: dnase_k562.bed not found.")
    print("  Skipping DNase. Download from ENCODE ENCSR000EMT.")
    has_dnase = False

# ── Load H3K4me3 (active promoter mark) ───────────────────────
# WHY: H3K4me3 marks gene promoters that are actively transcribed.
# Transcriptionally active regions have open chromatin →
# more Cas9 access → higher off-target risk.
try:
    h3k4me3_bed = load_narrowpeak(
        f"{DATA}\\h3k4me3_k562.bed", "H3K4me3 K562")
    has_h3k4me3 = True
except FileNotFoundError:
    print("\n  WARNING: h3k4me3_k562.bed not found.")
    print("  Skipping H3K4me3.")
    has_h3k4me3 = False

# ══════════════════════════════════════════════════════════════
# STEP 3 — Compute peak overlap for each off-target site
# WHY: For every one of the 4.9M off-target candidate sites,
# we check whether a ±500bp window around it overlaps any peak.
#
# ±500bp window rationale: Cas9 needs physical access to ~20bp
# of DNA, but chromatin state is determined by the local region
# (~500bp = ~3 nucleosomes either side). This is the standard
# window used in Mak et al. 2022 and Kimata & Satou 2025,
# which your synopsis cites.
#
# Algorithm: binary search on sorted peak coordinates per chrom.
# WHY binary search: naive loop = O(n×m) = too slow for 4.9M
# sites × 100k peaks. Binary search = O(n × log m) = fast.
# ══════════════════════════════════════════════════════════════
def compute_overlap(df, peaks_df, col_prefix, window=500):
    """
    For each site in df, checks whether a ±window bp region
    overlaps any peak in peaks_df.

    Adds two columns to df:
      {col_prefix}_overlap : 1 if any peak overlaps, else 0
      {col_prefix}_signal  : max signal value in window (0 if none)

    Uses chromosome-wise binary search for efficiency.
    Processes 4.9M sites in ~10-20 minutes.
    """
    print(f"\nComputing {col_prefix} overlap (±{window}bp window)...")
    print(f"  Processing {len(df):,} sites across "
          f"{df['chrom'].nunique()} chromosomes...")

    overlap = np.zeros(len(df), dtype=np.int8)
    signal  = np.zeros(len(df), dtype=np.float32)

    # Pre-sort peaks by chromosome for binary search
    peaks_by_chrom = {}
    for chrom, grp in peaks_df.groupby('chrom'):
        sorted_grp = grp.sort_values('start').reset_index(drop=True)
        peaks_by_chrom[chrom] = {
            'start' : sorted_grp['start'].values,
            'end'   : sorted_grp['end'].values,
            'signal': sorted_grp['signal'].values
        }

    chroms = df['chrom'].unique()
    for chrom_idx, chrom in enumerate(chroms):
        site_mask = (df['chrom'] == chrom).values
        site_indices = np.where(site_mask)[0]

        if chrom not in peaks_by_chrom:
            continue  # no peaks on this chrom → features stay 0

        p = peaks_by_chrom[chrom]
        p_start  = p['start']
        p_end    = p['end']
        p_signal = p['signal']

        chrom_starts = df['chromStart'].values[site_indices]
        chrom_ends   = df['chromEnd'].values[site_indices]

        for i, idx in enumerate(site_indices):
            s = chrom_starts[i] - window
            e = chrom_ends[i]   + window

            # Binary search: first peak ending after window start
            lo = np.searchsorted(p_end,   s, side='right')
            # Binary search: first peak starting after window end
            hi = np.searchsorted(p_start, e, side='left')

            if lo < hi:
                overlap[idx] = 1
                signal[idx]  = p_signal[lo:hi].max()

        if (chrom_idx + 1) % 5 == 0:
            print(f"  Processed {chrom_idx+1}/{len(chroms)} chromosomes...")

    df[f'{col_prefix}_overlap'] = overlap
    df[f'{col_prefix}_signal']  = signal

    # Biological validation check
    # If the feature is predictive, positives should have
    # HIGHER overlap rate than negatives
    pos_rate = df.loc[df['label']==1, f'{col_prefix}_overlap'].mean()*100
    neg_rate = df.loc[df['label']==0, f'{col_prefix}_overlap'].mean()*100
    diff     = pos_rate - neg_rate

    print(f"\n  ── {col_prefix} Biological Validation ──")
    print(f"  Overlap in positives (real off-targets): {pos_rate:.2f}%")
    print(f"  Overlap in negatives (non-off-targets) : {neg_rate:.2f}%")
    print(f"  Difference                             : {diff:+.2f}%")
    if diff > 2:
        print(f"  ✓ PREDICTIVE — positives more likely in {col_prefix} peaks")
        print(f"    This confirms {col_prefix} is biologically relevant")
        print(f"    for CRISPR off-target activity → supports your hypothesis")
    elif diff < -2:
        print(f"  ✓ PROTECTIVE — {col_prefix} peaks REDUCE off-target risk")
        print(f"    (consistent with closed/repressive chromatin marks)")
    else:
        print(f"  ~ NEUTRAL — {col_prefix} shows little difference")
        print(f"    May still help model as interaction feature")

    return df

# Apply each available epigenetic feature
if has_h3k27ac:
    df = compute_overlap(df, h3k27ac_bed, 'h3k27ac')

if has_dnase:
    df = compute_overlap(df, dnase_bed, 'dnase')

if has_h3k4me3:
    df = compute_overlap(df, h3k4me3_bed, 'h3k4me3')

# If no BED files were found at all, exit with instructions
epi_cols_added = [c for c in df.columns
                  if c.endswith('_overlap') or c.endswith('_signal')]

if len(epi_cols_added) == 0:
    print("\n" + "="*60)
    print("NO EPIGENETIC BED FILES FOUND")
    print("="*60)
    print("Download these files from ENCODE and put in data folder:")
    print("\n1. H3K27ac K562 hg38:")
    print("   https://www.encodeproject.org/experiments/ENCSR000AKP/")
    print("   → File details → bed narrowPeak → optimal IDR peaks")
    print("   → Rename to: h3k27ac_k562.bed")
    print("\n2. DNase-seq K562 hg38:")
    print("   https://www.encodeproject.org/experiments/ENCSR000EMT/")
    print("   → File details → bed narrowPeak")
    print("   → Rename to: dnase_k562.bed")
    print("\n3. H3K4me3 K562 hg38:")
    print("   https://www.encodeproject.org/experiments/ENCSR000AKU/")
    print("   → File details → bed narrowPeak → optimal IDR peaks")
    print("   → Rename to: h3k4me3_k562.bed")
    exit()

# ══════════════════════════════════════════════════════════════
# STEP 4 — Save epigenetic annotation file
# WHY: We save ONLY the new epigenetic columns + coordinates.
# During modelling (step5_models.py) we will merge this file
# with the full 223-column sequence feature matrix.
# Keeping them separate means we never have to reload 4.9M×223
# all at once — we merge only what we need.
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Saving epigenetic annotation file...")
print("="*60)

save_cols = ['chrom', 'chromStart', 'chromEnd',
             'label', 'reads'] + epi_cols_added

out_path = (r"C:\Users\HP\OneDrive\Desktop\Btech Project"
            r"\Crispr_code\data\changeseq_epigenetic_annotations.csv")

df[save_cols].to_csv(out_path, index=False)

print(f"\n=== EPIGENETIC ANNOTATION SUMMARY ===")
print(f"Total sites annotated : {len(df):,}")
print(f"Epigenetic cols added : {epi_cols_added}")
print(f"Output file           : changeseq_epigenetic_annotations.csv")
print(f"Total runtime         : {time.time()-start:.1f}s")
print(f"\n✓ Done. Next step: run step5_models.py")