#!/usr/bin/env python3
# =============================================================================
# MARS 2026 Problem Statement 1 — Support Integrity Auditor (SIA)
# 100% Self-Supervised Training Pipeline with 5-Fold CV
# =============================================================================

import os, json, pickle, warnings, gc
from datetime import datetime
from itertools import product
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, set_seed
from peft import get_peft_model, LoraConfig, TaskType
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, cohen_kappa_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

warnings.filterwarnings('ignore')
set_seed(42)

DATA_PATH = "data/customer_support_tickets.csv"
MODEL_DIR = "models/deberta_sia"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# =============================================================================
# HELPER: FOCAL LOSS
# =============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha_weight=0.25, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha_weight = alpha_weight # positive class weight
        
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        alpha_t = self.alpha_weight * targets + (1 - self.alpha_weight) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * ce_loss).mean()

# =============================================================================
# STAGE 1: SELF-SUPERVISED PSEUDO-LABEL OPTIMIZATION
# =============================================================================
def main():
    print('Starting SIA Training Pipeline (100% Self-Supervised, 5-Fold CV)')
    
    # --- 1. Early Data Split & Feature Engineering ---
    df = pd.read_csv(DATA_PATH)
    if 'Ticket_ID' not in df.columns:
        df['Ticket_ID'] = [f'TKT-{i:06d}' for i in range(len(df))]
        
    df['clean_subject'] = df['Ticket_Subject'].astype(str).str.strip()
    df['clean_desc'] = df['Ticket_Description'].astype(str).apply(
        lambda x: '. '.join([s for s in x.replace('Hi Support,','').strip().split('. ') if len(s.split())>=3][:2]) or x
    )
    df['combined_text'] = df['clean_subject'] + ' [SEP] ' + df['clean_desc']
    
    # Enhanced Metadata Processing
    def categorize_rt(rt):
        if rt < 24: return '<24 hrs'
        elif rt <= 72: return '24-72 hrs'
        elif rt <= 120: return '72-120 hrs'
        else: return '>120 hrs'
    df['RT_Bucket'] = df['Resolution_Time_Hours'].apply(categorize_rt)
    
    # Split
    df_train_full, df_test = train_test_split(df, test_size=0.15, random_state=42, stratify=df['Priority_Level'])
    df_train, df_val = train_test_split(df_train_full, test_size=0.15/0.85, random_state=42, stratify=df_train_full['Priority_Level'])
    print(f'Data Splitting: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}')

    # --- 2. Generate 4 Independent Signals ---
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Signal A: SBERT Zero-Shot (Continuous & Ordinal)
    URGENCY_ANCHORS = {
        3: ['complete system outage cannot access anything data lost emergency', 'critical security breach data compromised'],
        2: ['application crashes repeatedly error every time', 'important data not syncing major functionality broken'],
        1: ['slow performance loading delay intermittent issue', 'feature not working as expected minor disruption'],
        0: ['general inquiry question about service', 'feature request suggestion feedback nice to have']
    }
    anchors = {lv: sbert.encode(ph, convert_to_numpy=True) for lv, ph in URGENCY_ANCHORS.items()}
    
    def get_signal_a(df_split):
        embeds = sbert.encode(df_split['combined_text'].tolist(), convert_to_numpy=True, batch_size=64, show_progress_bar=False)
        cont_scores = []
        for emb in embeds:
            sims = {lv: float(np.max(cosine_similarity(emb.reshape(1,-1), anchors[lv])[0])) for lv in anchors}
            # Weighted continuous score
            score = sum(lv * s for lv, s in sims.items()) / sum(sims.values())
            cont_scores.append(score)
        return np.array(cont_scores), embeds

    print('Generating Signal A (SBERT)...')
    sig_a_cont_tr, emb_tr = get_signal_a(df_train)
    sig_a_cont_va, emb_va = get_signal_a(df_val)
    sig_a_cont_te, emb_te = get_signal_a(df_test)
    
    # DISTRIBUTION-MATCHED QUANTILES
    true_dist = df_train['assigned_ordinal'].value_counts(normalize=True).sort_index()
    p0 = true_dist.get(0, 0)
    p1 = true_dist.get(1, 0)
    p2 = true_dist.get(2, 0)
    dist_matched_percentiles = [p0*100, (p0+p1)*100, (p0+p1+p2)*100]

    def quant_calib(cont_arr, ref_arr):
        q = np.percentile(ref_arr, dist_matched_percentiles)
        return np.digitize(cont_arr, q)
        
    sig_a_tr = quant_calib(sig_a_cont_tr, sig_a_cont_tr)
    sig_a_va = quant_calib(sig_a_cont_va, sig_a_cont_tr)
    sig_a_te = quant_calib(sig_a_cont_te, sig_a_cont_tr)

    # Signal B: XGBoost Resolution Regression
    print('Generating Signal B (XGBoost RT Proxy)...')
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1,2), sublinear_tf=True, min_df=2)
    def get_xgb_feats(df_split, is_train=False):
        text_f = tfidf.fit_transform(df_split['combined_text']).toarray() if is_train else tfidf.transform(df_split['combined_text']).toarray()
        ch = pd.get_dummies(df_split['Ticket_Channel'], prefix='ch')
        cat = pd.get_dummies(df_split['Issue_Category'], prefix='cat')
        if is_train:
            get_xgb_feats.ch_cols = ch.columns
            get_xgb_feats.cat_cols = cat.columns
        else:
            ch = ch.reindex(columns=get_xgb_feats.ch_cols, fill_value=0)
            cat = cat.reindex(columns=get_xgb_feats.cat_cols, fill_value=0)
        return np.hstack([text_f, ch.values, cat.values])

    X_tr = get_xgb_feats(df_train, True)
    X_va = get_xgb_feats(df_val, False)
    X_te = get_xgb_feats(df_test, False)
    
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1)
    xgb_model.fit(X_tr, df_train['Resolution_Time_Hours'].values)
    
    sig_b_cont_tr = xgb_model.predict(X_tr)
    sig_b_cont_va = xgb_model.predict(X_va)
    sig_b_cont_te = xgb_model.predict(X_te)
    
    sig_b_tr = quant_calib(sig_b_cont_tr, sig_b_cont_tr)
    sig_b_va = quant_calib(sig_b_cont_va, sig_b_cont_tr)
    sig_b_te = quant_calib(sig_b_cont_te, sig_b_cont_tr)
    
    # FIX: INVERT Signal B — Resolution time is ANTI-CORRELATED with severity.
    # Low-priority backlog tickets sit for 200+ hrs. Critical outages resolved in 2 hrs.
    sig_b_tr = 3 - sig_b_tr
    sig_b_va = 3 - sig_b_va
    sig_b_te = 3 - sig_b_te
    print(f'Signal B inverted. Distribution: {np.unique(sig_b_tr, return_counts=True)}')

    # Signal C: Rule-Based NLP (Adversarial features) — MASSIVELY EXPANDED
    # IMPORTANT: Do NOT use a dynamic negation loop against urgent_kws.
    # 'not working' IS urgent — it must NOT be matched as negated.
    print('Generating Signal C (Rule-Based NLP)...')
    urgent_kws = [
        'outage', 'cannot login', 'production down', 'security breach',
        'not working', 'broken', 'failed', 'error', 'crash', 'system down',
        'cannot access', 'inaccessible', 'losing money', 'data loss',
        'database corrupted', 'customers cannot', 'urgent', 'immediately',
        'critical', 'emergency', 'offline', 'unreachable', 'compromised',
        'not responding', 'not loading', 'not syncing', 'service disrupted'
    ]
    esc_kws = [
        'ceo', 'manager', 'escalate', 'legal', 'complaint', 'lawsuit',
        'regulatory', 'sue', 'attorney', 'authorities', 'media', 'press'
    ]
    neg_kws = [
        'not urgent', 'no outage', 'false alarm', 'resolved', 'already fixed',
        'just a question', 'feature request', 'suggestion', 'when you have time',
        'no rush', 'whenever possible', 'curious', 'wondering',
        'not critical', 'not an emergency', 'not a problem', 'no issue',
        'not a bug', 'not down', 'not broken'
    ]
    
    def get_signal_c(df_split):
        sc = []
        for t in df_split['combined_text'].str.lower():
            # neg_kws checked FIRST — explicit negations take priority
            if any(k in t for k in neg_kws):
                sc.append(0)
            elif any(k in t for k in urgent_kws):
                # NO dynamic negation check — 'not working' is urgent, not negated
                sc.append(3)
            elif any(k in t for k in esc_kws):
                sc.append(2)
            else:
                sc.append(1)
        return np.array(sc)
        
    sig_c_tr = get_signal_c(df_train)
    sig_c_va = get_signal_c(df_val)
    sig_c_te = get_signal_c(df_test)
    print(f'Signal C distribution (train): {np.unique(sig_c_tr, return_counts=True)}')

    # Signal D: K-Means Clustering
    print('Generating Signal D (K-Means Semantic Clustering)...')
    kmeans = KMeans(n_clusters=8, random_state=42)
    cl_tr = kmeans.fit_predict(emb_tr)
    
    # 100% Self-supervised: assign cluster severity based on the mean SBERT continuous score of the cluster
    cluster_sev_cont = {}
    for c in range(8):
        mask = (cl_tr == c)
        cluster_sev_cont[c] = np.mean(sig_a_cont_tr[mask]) if mask.sum() > 0 else np.mean(sig_a_cont_tr)
        
    sig_d_cont_tr = np.array([cluster_sev_cont[c] for c in cl_tr])
    sig_d_cont_va = np.array([cluster_sev_cont[c] for c in kmeans.predict(emb_va)])
    sig_d_cont_te = np.array([cluster_sev_cont[c] for c in kmeans.predict(emb_te)])
    
    sig_d_tr = quant_calib(sig_d_cont_tr, sig_d_cont_tr)
    sig_d_va = quant_calib(sig_d_cont_va, sig_d_cont_tr)
    sig_d_te = quant_calib(sig_d_cont_te, sig_d_cont_tr)

    # --- 3. Self-Supervised Consensus Optimization & Quantile Calibration ---
    print('Optimizing Consensus Weights (Grid Search)...')
    best_weights = None; best_kappa = -1
    for w1, w2, w3, w4 in product(np.linspace(0.1, 0.5, 5), repeat=4):
        if not np.isclose(w1+w2+w3+w4, 1.0): continue
        fused_cont = w1*sig_a_tr + w2*sig_b_tr + w3*sig_c_tr + w4*sig_d_tr
        fused_disc = quant_calib(fused_cont, fused_cont)
        
        # Mean kappa against all 4 signals
        k = np.mean([cohen_kappa_score(fused_disc, s) for s in [sig_a_tr, sig_b_tr, sig_c_tr, sig_d_tr]])
        if k > best_kappa:
            best_kappa = k; best_weights = (w1, w2, w3, w4)
            
    print(f'Optimal Weights: A={best_weights[0]:.2f}, B={best_weights[1]:.2f}, C={best_weights[2]:.2f}, D={best_weights[3]:.2f}')
    
    def apply_fusion(sa, sb, sc, sd, is_train=False):
        f_cont = best_weights[0]*sa + best_weights[1]*sb + best_weights[2]*sc + best_weights[3]*sd
        if is_train:
            apply_fusion.q = np.percentile(f_cont, dist_matched_percentiles)
        f_disc = np.digitize(f_cont, apply_fusion.q)
        return f_disc
        
    inf_tr = apply_fusion(sig_a_tr, sig_b_tr, sig_c_tr, sig_d_tr, True)
    inf_va = apply_fusion(sig_a_va, sig_b_va, sig_c_va, sig_d_va, False)
    inf_te = apply_fusion(sig_a_te, sig_b_te, sig_c_te, sig_d_te, False)
    
    SEVERITY_MAP = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
    df_train['inferred_severity_ordinal'] = inf_tr
    df_val['inferred_severity_ordinal'] = inf_va
    df_test['inferred_severity_ordinal'] = inf_te
    
    df_train['assigned_ordinal'] = df_train['Priority_Level'].map(SEVERITY_MAP)
    df_val['assigned_ordinal'] = df_val['Priority_Level'].map(SEVERITY_MAP)
    df_test['assigned_ordinal'] = df_test['Priority_Level'].map(SEVERITY_MAP)
    
    # USE THRESHOLD >= 2 to only flag EGREGIOUS mismatches (not discretization noise)
    df_train['mismatch_label'] = (np.abs(df_train['inferred_severity_ordinal'] - df_train['assigned_ordinal']) >= 2).astype(int)
    df_val['mismatch_label']   = (np.abs(df_val['inferred_severity_ordinal']   - df_val['assigned_ordinal'])   >= 2).astype(int)
    df_test['mismatch_label']  = (np.abs(df_test['inferred_severity_ordinal']  - df_test['assigned_ordinal'])  >= 2).astype(int)
    
    train_mismatch_rate = df_train['mismatch_label'].mean()
    print(f'Train Mismatch Rate: {train_mismatch_rate:.2%}  (target: 15%-40% for balanced learning)')
    if train_mismatch_rate > 0.5:
        print('WARNING: Mismatch rate >50%. Consider >= 2 threshold (already applied).')

    # Hard Negative Mining (Filter Removed)
    print('Applying Hard Negative Mining (Filter Removed)...')
    sig_matrix = np.column_stack([sig_a_tr, sig_b_tr, sig_c_tr, sig_d_tr])
    variances = np.var(sig_matrix, axis=1)
    df_train_clean = df_train.copy()
    print(f'Retained full dataset for anomaly detection. Train size: {len(df_train_clean)}')
    
    # PSEUDO-LABEL QUALITY SANITY CHECK
    from sklearn.feature_extraction.text import TfidfVectorizer as TV
    tfidf_q = TV(max_features=5000)
    X_q_tr = tfidf_q.fit_transform(df_train_clean['combined_text'])
    clf_q = LogisticRegression(class_weight='balanced', max_iter=300)
    clf_q.fit(X_q_tr, df_train_clean['mismatch_label'])
    X_q_va = tfidf_q.transform(df_val['combined_text'])
    proxy_f1 = f1_score(df_val['mismatch_label'], clf_q.predict(X_q_va), average='macro', zero_division=0)
    print(f'\nPSEUDO-LABEL SANITY CHECK - LR Proxy Macro F1: {proxy_f1:.4f}')
    if proxy_f1 >= 0.65:
        print('PASS: Pseudo-labels have learnable signal.')
    else:
        print('WARN: Proxy F1 < 0.65. Pseudo-labels may be noisy.')

    # Save artifacts
    with open(f'{MODEL_DIR}/tfidf.pkl', 'wb') as f: pickle.dump(tfidf, f)
    xgb_model.save_model(f'{MODEL_DIR}/xgb_model.json')
    with open(f'{MODEL_DIR}/kmeans.pkl', 'wb') as f: pickle.dump((kmeans, cluster_sev_cont), f)
    with open(f'{MODEL_DIR}/fusion_weights.json', 'w') as f: json.dump({'w':best_weights, 'q':apply_fusion.q.tolist()}, f)
    with open(f'{MODEL_DIR}/rt_quantiles.json', 'w') as f: json.dump({'q25': np.percentile(df_train['Resolution_Time_Hours'], 25), 'q50': np.percentile(df_train['Resolution_Time_Hours'], 50), 'q75': np.percentile(df_train['Resolution_Time_Hours'], 75)}, f)

    # =============================================================================
    # STAGE 2: CLASSIFIER OPTIMIZATION (5-FOLD CV)
    # =============================================================================
    print('Starting 5-Fold Stratified CV DeBERTa Training...')
    def build_deberta_inputs(df_split):
        texts, labels = [], []
        for _, row in df_split.iterrows():
            t = f"[SEP] Subject: {row['clean_subject']} | Desc: {row['clean_desc']} [SEP] Context: Channel={row['Ticket_Channel']}, Bucket={row['RT_Bucket']}"
            texts.append(t); labels.append(row['mismatch_label'])
        return np.array(texts), np.array(labels)
        
    X_train_cl, y_train_cl = build_deberta_inputs(df_train_clean)
    X_val, y_val = build_deberta_inputs(df_val)
    X_test, y_test = build_deberta_inputs(df_test)
    
    MODEL_NAME = 'microsoft/deberta-v3-small'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    class TktDataset(Dataset):
        def __init__(self, t, l): self.t=t; self.l=l
        def __len__(self): return len(self.t)
        def __getitem__(self, i):
            enc = tokenizer(self.t[i], max_length=256, padding='max_length', truncation=True, return_tensors='pt')
            return {'input_ids': enc['input_ids'][0], 'attention_mask': enc['attention_mask'][0], 'labels': torch.tensor(self.l[i], dtype=torch.long)}

    # Using standard PyTorch CrossEntropyLoss with weights to completely fix the F1 collapse issue caused by Focal Loss instability
    classes = np.unique(y_train_cl)
    cws = compute_class_weight('balanced', classes=classes, y=y_train_cl)
    class_weights = torch.tensor(cws, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(X_train_cl))
    test_probs_folds = np.zeros(len(X_test))
    val_probs_folds = np.zeros(len(X_val))
    
    best_fold_model_state = None
    best_fold_f1 = 0
    
    for fold, (trn_idx, vld_idx) in enumerate(skf.split(X_train_cl, y_train_cl)):
        print(f"\n--- FOLD {fold+1}/5 ---")
        train_dl = DataLoader(TktDataset(X_train_cl[trn_idx], y_train_cl[trn_idx]), batch_size=16, shuffle=True)
        valid_dl = DataLoader(TktDataset(X_train_cl[vld_idx], y_train_cl[vld_idx]), batch_size=32)
        
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, hidden_dropout_prob=0.2).to(device)
        peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1)
        model = get_peft_model(model, peft_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(len(train_dl)*3*0.1), len(train_dl)*3)
        
        # Train for 3 epochs per fold to save time, it's enough for DeBERTa
        for epoch in range(1, 6):  # 5 epochs for sparse mismatch target
            model.train()
            for b in tqdm(train_dl, desc=f'Fold {fold+1} Ep {epoch}', leave=False):
                optimizer.zero_grad()
                out = model(input_ids=b['input_ids'].to(device), attention_mask=b['attention_mask'].to(device))
                loss = criterion(out.logits, b['labels'].to(device))
                loss.backward(); optimizer.step(); scheduler.step()
                
        # Validate OOF
        model.eval()
        fold_vld_logits = []
        with torch.no_grad():
            for b in valid_dl:
                out = model(input_ids=b['input_ids'].to(device), attention_mask=b['attention_mask'].to(device))
                fold_vld_logits.extend(out.logits.cpu().numpy())
        
        probs = torch.softmax(torch.tensor(fold_vld_logits), dim=1)[:, 1].numpy()
        oof_probs[vld_idx] = probs
        
        # Infer on true validation set
        val_dl = DataLoader(TktDataset(X_val, y_val), batch_size=32)
        fold_val_logits = []
        with torch.no_grad():
            for b in val_dl:
                out = model(input_ids=b['input_ids'].to(device), attention_mask=b['attention_mask'].to(device))
                fold_val_logits.extend(out.logits.cpu().numpy())
        val_probs_folds += torch.softmax(torch.tensor(fold_val_logits), dim=1)[:, 1].numpy() / 5
        
        # Infer on test set
        test_dl = DataLoader(TktDataset(X_test, y_test), batch_size=32)
        fold_test_logits = []
        with torch.no_grad():
            for b in test_dl:
                out = model(input_ids=b['input_ids'].to(device), attention_mask=b['attention_mask'].to(device))
                fold_test_logits.extend(out.logits.cpu().numpy())
        test_probs_folds += torch.softmax(torch.tensor(fold_test_logits), dim=1)[:, 1].numpy() / 5
        
        # Save model of fold 5 for final architecture (or the best fold)
        f1_fold = f1_score(y_train_cl[vld_idx], (probs>=0.5).astype(int), average='macro')
        if f1_fold > best_fold_f1:
            best_fold_f1 = f1_fold
            best_fold_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        del model, optimizer; gc.collect(); torch.cuda.empty_cache()

    # Save the best model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    peft_config_best = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1)
    model = get_peft_model(model, peft_config_best)
    # strict=False: skip keys that don't match LoRA state (e.g. classifier head init)
    best_fold_model_state = {k: v for k, v in best_fold_model_state.items() if k in dict(model.named_parameters())}
    model.load_state_dict(best_fold_model_state, strict=False)
    model.save_pretrained(MODEL_DIR); tokenizer.save_pretrained(MODEL_DIR)

    # --- 4. Isolated Threshold Optimization & Platt Scaling ---
    print('\nPerforming Platt Scaling and Threshold Sweep on Validation Split...')
    # Calibrate on true validation split
    calibrator = LogisticRegression()
    calibrator.fit(val_probs_folds.reshape(-1, 1), y_val)
    val_probs_cal = calibrator.predict_proba(val_probs_folds.reshape(-1, 1))[:, 1]
    
    best_th = 0.5; best_f1 = 0
    # Expanded range 0.10–0.90 to catch optimal threshold even if probs are squeezed
    for th in np.arange(0.10, 0.91, 0.05):
        p = (val_probs_cal >= th).astype(int)
        f = f1_score(y_val, p, average='macro', zero_division=0)
        r0 = recall_score(y_val, p, pos_label=0, zero_division=0)
        r1 = recall_score(y_val, p, pos_label=1, zero_division=0)
        if f > best_f1:
            best_f1 = f; best_th = th
            
    if best_f1 == 0: best_th = 0.5
    print(f'Optimal Frozen Threshold: {best_th:.2f}')
    
    with open(f'{MODEL_DIR}/calibrator.pkl', 'wb') as f: pickle.dump(calibrator, f)
    with open(f'{MODEL_DIR}/threshold.json', 'w') as f: json.dump({'optimal_threshold': best_th}, f)

    # Final Evaluation on Blind Test Set
    test_probs_cal = calibrator.predict_proba(test_probs_folds.reshape(-1, 1))[:, 1]
    test_preds = (test_probs_cal >= best_th).astype(int)
    
    acc = accuracy_score(y_test, test_preds)
    mac_f1 = f1_score(y_test, test_preds, average='macro')
    rec = recall_score(y_test, test_preds, average=None)
    
    print('\\n' + '='*50 + '\\nFINAL BLIND TEST METRICS\\n' + '='*50)
    print(f'Accuracy : {acc:.4f} (Target >= 0.83)')
    print(f'Macro F1 : {mac_f1:.4f} (Target >= 0.82)')
    print(f'Recall C0: {rec[0]:.4f} (Target >= 0.78)')
    print(f'Recall C1: {rec[1]:.4f} (Target >= 0.78)')
    print('Pipeline Complete.')

if __name__ == '__main__':
    main()
