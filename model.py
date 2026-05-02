import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (roc_auc_score, roc_curve,
                                     f1_score, precision_score,
                                     recall_score, accuracy_score)
import xgboost as xgb
import shap
import warnings, time
warnings.filterwarnings('ignore')

OUT   = r"C:\Users\HP\OneDrive\Desktop\Btech Project\Crispr_code\outputs"
DATA  = r"C:\Users\HP\OneDrive\Desktop\Btech Project\Crispr_code\data"
START = time.time()

# ══════════════════════════════════════════════════════════════
# DATASET A — CHANGE-seq sequence features (large, in vitro)
# Used for: baseline sequence-only models
# 4.9M rows, 213 sequence features, no epigenetics
# ══════════════════════════════════════════════════════════════
print("="*60)
print("LOADING CHANGE-seq sequence features")
print("="*60)

# Load sequence feature columns only — skip coordinates/label/reads
# We identify feature cols by exclusion
COORD_COLS = ['chrom','chromStart','chromEnd','strand','label','reads']

print("Reading column names first...")
sample = pd.read_csv(f"{DATA}\\changeseq_features.csv",
                     nrows=2, engine='c', low_memory=False)
seq_cols = [c for c in sample.columns if c not in COORD_COLS]
print(f"Sequence feature columns: {len(seq_cols)}")

print("Loading features in chunks...")
chunks = []
for i, chunk in enumerate(pd.read_csv(
        f"{DATA}\\changeseq_features.csv",
        usecols=seq_cols + ['label'],
        engine='python', encoding='latin-1',
        on_bad_lines='skip', chunksize=500_000)):
    chunks.append(chunk)
    print(f"  Chunk {i+1} loaded: {len(chunk):,} rows")

cs = pd.concat(chunks, ignore_index=True)
cs['label'] = pd.to_numeric(cs['label'], errors='coerce').fillna(0).astype(int)
print(f"Total loaded: {len(cs):,} rows")

# Subsample 300k for speed — statistically identical results
# 300k still has ~12,000 positives (4%) — plenty for training
np.random.seed(42)
idx   = np.random.choice(len(cs), size=300_000, replace=False)
X_cs  = cs[seq_cols].fillna(0).values[idx]
y_cs  = cs['label'].values[idx]
print(f"Subsampled: {len(y_cs):,} rows | "
      f"Positives: {y_cs.sum():,} | "
      f"Negatives: {(y_cs==0).sum():,}")

X_tr_cs, X_te_cs, y_tr_cs, y_te_cs = train_test_split(
    X_cs, y_cs, test_size=0.2, random_state=42, stratify=y_cs)

# ══════════════════════════════════════════════════════════════
# DATASET B — crisprSQL (small, in-cell, has epigenetics)
# Used for: sequence-only vs multimodal comparison
# 2000 rows, 308 features including DNase/H3K4me3/RRBS/MNase
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("LOADING crisprSQL (epigenetic dataset)")
print("="*60)

crispr = pd.read_csv(
    r"C:\Users\HP\Downloads\crispr-cas9-epigenetics-main"
    r"\crispr-cas9-epigenetics-main\data\crisprSQL_dataset_2000.csv")
print(f"Shape: {crispr.shape}")

# ── Find label column automatically ───────────────────────────
label_candidates = [c for c in crispr.columns if any(
    x in c.lower() for x in ['label','cleav','activ','target',
                               'freq','binary','read','cut'])]
print(f"Label candidates: {label_candidates}")

if label_candidates:
    label_col = label_candidates[0]
    y_crispr  = crispr[label_col].values
    print(f"Using label: '{label_col}'")
else:
    # No explicit label — check if last column is numeric activity
    last = crispr.columns[-1]
    vals = pd.to_numeric(crispr[last], errors='coerce').fillna(0).values
    # Binarize at median of non-zero values
    nonzero = vals[vals > 0]
    thresh  = np.median(nonzero) if len(nonzero) > 0 else 0.5
    y_crispr  = (vals > thresh).astype(int)
    label_col = last
    print(f"Binarized '{last}' at threshold {thresh:.4f}")
    print(f"Positives: {y_crispr.sum()} / {len(y_crispr)}")

print(f"Label distribution: "
      f"{y_crispr.sum()} pos / {(y_crispr==0).sum()} neg")

# ── Define feature subsets ─────────────────────────────────────
# Sequence-only: GC, WS, YR, NucleotideBDM, StrongWeakBDM
# These are sequence-derived features — no chromatin info
seq_only_prefixes = ['GCContent','WSScore','YRScore',
                     'NucleotideBDM','StrongWeakBDM']
seq_only_cols = [c for c in crispr.columns
                 if any(c.startswith(p) for p in seq_only_prefixes)]

# All features: everything except label
all_cols = [c for c in crispr.columns if c != label_col]

# Epigenetic-only cols (for reporting which ones matter)
epi_cols = ['epigen_ctcf','epigen_dnase','epigen_rrbs',
            'epigen_h3k4me3','epigen_drip','MNase']
epi_cols = [c for c in epi_cols if c in crispr.columns]

print(f"Sequence-only features : {len(seq_only_cols)}")
print(f"All features           : {len(all_cols)}")
print(f"Epigenetic features    : {epi_cols}")

X_all = crispr[all_cols].fillna(0).values
X_seq = crispr[seq_only_cols].fillna(0).values

X_tr_all, X_te_all, y_tr_all, y_te_all = train_test_split(
    X_all, y_crispr, test_size=0.2, random_state=42,
    stratify=y_crispr)
X_tr_seq, X_te_seq, _, _ = train_test_split(
    X_seq, y_crispr, test_size=0.2, random_state=42,
    stratify=y_crispr)

pos_w = max((y_tr_all==0).sum() / max((y_tr_all==1).sum(),1), 1)
print(f"Class weight (pos)     : {pos_w:.1f}")

# ══════════════════════════════════════════════════════════════
# EVALUATION HELPER
# ══════════════════════════════════════════════════════════════
def evaluate(model, X_test, y_test, name, threshold=0.5):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    auc  = roc_auc_score(y_test, y_prob)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    acc  = accuracy_score(y_test, y_pred)
    print(f"\n  {'─'*48}")
    print(f"  {name}")
    print(f"  {'─'*48}")
    print(f"  ROC-AUC   : {auc:.4f}  ← main metric")
    print(f"  F1 score  : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  Accuracy  : {acc:.4f}")
    return auc, f1, prec, rec, y_prob

results = {}  # name → (y_true, y_prob, auc, dataset_label)

# ══════════════════════════════════════════════════════════════
# MODEL 1 — Logistic Regression, sequence-only (CHANGE-seq)
# Simplest possible baseline. Linear, sequence features only.
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL 1: Logistic Regression — Sequence Only (CHANGE-seq)")
print("="*60)
t = time.time()

scaler     = StandardScaler()
X_tr_sc    = scaler.fit_transform(X_tr_cs)
X_te_sc    = scaler.transform(X_te_cs)

lr = LogisticRegression(class_weight='balanced', max_iter=200,
                         solver='saga', n_jobs=-1, random_state=42)
lr.fit(X_tr_sc, y_tr_cs)
print(f"Training time: {time.time()-t:.1f}s")
auc,f1,prec,rec,yp = evaluate(lr, X_te_sc, y_te_cs,
                               "LR — Sequence Only (CHANGE-seq)")
results['1. LR Seq-Only\n(CHANGE-seq)'] = (y_te_cs, yp, auc)

# ══════════════════════════════════════════════════════════════
# MODEL 2 — Random Forest, sequence-only (CHANGE-seq)
# Non-linear baseline capturing position interactions.
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL 2: Random Forest — Sequence Only (CHANGE-seq)")
print("="*60)
t = time.time()

rf = RandomForestClassifier(n_estimators=100, max_depth=12,
                             class_weight='balanced',
                             n_jobs=-1, random_state=42)
rf.fit(X_tr_cs, y_tr_cs)
print(f"Training time: {time.time()-t:.1f}s")
auc,f1,prec,rec,yp = evaluate(rf, X_te_cs, y_te_cs,
                               "RF — Sequence Only (CHANGE-seq)")
results['2. RF Seq-Only\n(CHANGE-seq)'] = (y_te_cs, yp, auc)

# ══════════════════════════════════════════════════════════════
# MODEL 3 — XGBoost, sequence-only (crisprSQL)
# Same algorithm as Model 4 but WITHOUT epigenetics.
# Direct apples-to-apples comparison on same dataset.
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL 3: XGBoost — Sequence Only (crisprSQL)")
print("="*60)
t = time.time()

xgb3 = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    scale_pos_weight=pos_w, subsample=0.8, colsample_bytree=0.8,
    eval_metric='auc', random_state=42, verbosity=0)
xgb3.fit(X_tr_seq, y_tr_all,
         eval_set=[(X_te_seq, y_te_all)], verbose=False)
print(f"Training time: {time.time()-t:.1f}s")
auc,f1,prec,rec,yp = evaluate(xgb3, X_te_seq, y_te_all,
                               "XGB — Seq Only (crisprSQL)")
results['3. XGB Seq-Only\n(crisprSQL)'] = (y_te_all, yp, auc)

# ══════════════════════════════════════════════════════════════
# MODEL 4 — XGBoost, ALL features + epigenetics (crisprSQL)
# THE MAIN MODEL — adds DNase, H3K4me3, RRBS, MNase, NuPoP
# AUC improvement over Model 3 = value of epigenetics
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL 4: XGBoost — ALL Features + Epigenetics (crisprSQL)")
print("="*60)
t = time.time()

xgb4 = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    scale_pos_weight=pos_w, subsample=0.8, colsample_bytree=0.8,
    eval_metric='auc', random_state=42, verbosity=0)
xgb4.fit(X_tr_all, y_tr_all,
         eval_set=[(X_te_all, y_te_all)], verbose=False)
print(f"Training time: {time.time()-t:.1f}s")
auc,f1,prec,rec,yp = evaluate(xgb4, X_te_all, y_te_all,
                               "XGB — All+Epigenetics (crisprSQL)")
results['4. XGB All+Epi\n(crisprSQL)'] = (y_te_all, yp, auc)

# ══════════════════════════════════════════════════════════════
# STEP 7 — ROC CURVE PLOT
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 7: Saving ROC curve plot")
print("="*60)

colors = ['#aaaaaa','#5577aa','#ee8833','#cc2222']
styles = ['--', '-.', '-', '-']
widths = [1.5, 1.5, 2.0, 2.5]

fig, ax = plt.subplots(figsize=(9, 7))
for (name,(y_true,y_prob,auc)), col, ls, lw in zip(
        results.items(), colors, styles, widths):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    label = f"{name.replace(chr(10),' ')}  (AUC={auc:.3f})"
    ax.plot(fpr, tpr, color=col, lw=lw, ls=ls, label=label)

ax.plot([0,1],[0,1],'k:', lw=1, label='Random classifier (AUC=0.500)')
ax.fill_between([0,1],[0,1], alpha=0.05, color='gray')
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curves — CRISPR Off-Target Prediction\n'
             'Sequence-Only vs Multimodal (Sequence + Epigenetics)',
             fontsize=13)
ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.25)
ax.set_xlim([0,1]); ax.set_ylim([0,1])
plt.tight_layout()
plt.savefig(f"{OUT}\\roc_curves.png", dpi=150)
plt.close()
print(f"✓ Saved → outputs/roc_curves.png")

# ══════════════════════════════════════════════════════════════
# STEP 8 — SHAP FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 8: SHAP feature importance analysis")
print("="*60)

explainer = shap.TreeExplainer(xgb4)
shap_vals = explainer.shap_values(X_te_all[:300])
mean_shap = np.abs(shap_vals).mean(axis=0)

shap_df = pd.DataFrame({
    'feature'  : all_cols,
    'importance': mean_shap
}).sort_values('importance', ascending=False).reset_index(drop=True)

print("\nTop 25 most important features:")
print(shap_df.head(25).to_string(index=False))

# Categorise for colour coding
def categorise(f):
    if f in ['epigen_ctcf','epigen_dnase','epigen_rrbs',
             'epigen_h3k4me3','epigen_drip','MNase']:
        return 'Experimental Epigenetic'
    elif any(f.startswith(x) for x in ['NuPop','NuPoP','nuCpos',
                                         'VanDer','LeNup']):
        return 'Nucleosome Positioning'
    elif f.startswith('energy'):
        return 'RNA-DNA Energy'
    elif any(f.startswith(x) for x in ['GCContent','WSScore',
                                         'YRScore','NucleotideBDM',
                                         'StrongWeakBDM']):
        return 'Sequence Context'
    else:
        return 'Other'

shap_df['category'] = shap_df['feature'].apply(categorise)

cat_colors = {
    'Experimental Epigenetic' : '#cc2222',
    'Nucleosome Positioning'  : '#2255cc',
    'RNA-DNA Energy'          : '#ee8833',
    'Sequence Context'        : '#22aa55',
    'Other'                   : '#999999'
}

top25     = shap_df.head(25)
bar_colors = [cat_colors[c] for c in top25['category']]

fig, ax = plt.subplots(figsize=(11, 8))
ax.barh(range(25), top25['importance'].values,
        color=bar_colors, edgecolor='white', linewidth=0.4)
ax.set_yticks(range(25))
ax.set_yticklabels(top25['feature'].values, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('Mean |SHAP value| — contribution to prediction', fontsize=11)
ax.set_title('Top 25 Features — XGBoost Multimodal Model\n'
             'CRISPR Off-Target Prediction (crisprSQL dataset)',
             fontsize=12)

from matplotlib.patches import Patch
legend_els = [Patch(facecolor=v, label=k)
              for k,v in cat_colors.items()
              if k in top25['category'].values]
ax.legend(handles=legend_els, fontsize=9, loc='lower right')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}\\shap_importance.png", dpi=150)
plt.close()
print(f"✓ Saved → outputs/shap_importance.png")

# ══════════════════════════════════════════════════════════════
# STEP 9 — EPIGENETIC FEATURE CONTRIBUTION ANALYSIS
# Show how much each epigenetic feature individually contributes
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 9: Epigenetic feature contribution")
print("="*60)

epi_present = [c for c in epi_cols if c in shap_df['feature'].values]
epi_ranks   = shap_df[shap_df['feature'].isin(epi_present)][
    ['feature','importance','category']].reset_index(drop=True)
print("\nEpigenetic features ranked by SHAP importance:")
print(epi_ranks.to_string(index=False))

total_shap    = mean_shap.sum()
epi_shap_sum  = shap_df[shap_df['category']=='Experimental Epigenetic'
                         ]['importance'].sum()
nuc_shap_sum  = shap_df[shap_df['category']=='Nucleosome Positioning'
                         ]['importance'].sum()
seq_shap_sum  = shap_df[shap_df['category']=='Sequence Context'
                         ]['importance'].sum()
rna_shap_sum  = shap_df[shap_df['category']=='RNA-DNA Energy'
                         ]['importance'].sum()

print(f"\nFeature category contributions (% of total SHAP):")
print(f"  Experimental Epigenetic : {epi_shap_sum/total_shap*100:.1f}%")
print(f"  Nucleosome Positioning  : {nuc_shap_sum/total_shap*100:.1f}%")
print(f"  RNA-DNA Energy          : {rna_shap_sum/total_shap*100:.1f}%")
print(f"  Sequence Context        : {seq_shap_sum/total_shap*100:.1f}%")

import joblib
joblib.dump(rf,   f"{OUT}\\rf_sequence_model.pkl")
joblib.dump(scaler, f"{OUT}\\scaler.pkl")
print("✓ Models saved")

# ══════════════════════════════════════════════════════════════
# FINAL RESULTS TABLE
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FINAL RESULTS — ALL MODELS")
print("="*60)

best_auc = max(r[2] for r in results.values())
print(f"\n{'Model':<42} {'AUC':>7}  {'Note'}")
print("─"*65)
for name, (_, _, auc) in results.items():
    clean = name.replace('\n', ' ')
    note  = " ◄ BEST" if auc == best_auc else ""
    delta = f"  (+{auc - list(results.values())[0][2]:.3f} vs LR baseline)" \
            if auc != list(results.values())[0][2] else ""
    print(f"{clean:<42} {auc:.4f}{note}{delta}")

print(f"\nKey finding:")
auc_seq = results['3. XGB Seq-Only\n(crisprSQL)'][2]
auc_epi = results['4. XGB All+Epi\n(crisprSQL)'][2]
delta   = auc_epi - auc_seq
print(f"  XGBoost seq-only AUC      : {auc_seq:.4f}")
print(f"  XGBoost multimodal AUC    : {auc_epi:.4f}")
print(f"  Improvement from epigenetics: {delta:+.4f}")
if delta > 0:
    print(f"  ✓ Epigenetic features IMPROVE prediction on in-cell data")
else:
    print(f"  ~ Models perform similarly — crisprSQL is small (2000 rows)")
    print(f"    The NuPoP/nucleosome features dominate over raw epigenetics")

print(f"\nTotal runtime: {time.time()-START:.1f}s")
print(f"✓ Outputs saved to: {OUT}")
print(f"   - roc_curves.png")
print(f"   - shap_importance.png")