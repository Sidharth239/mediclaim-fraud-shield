"""
ML Pipeline — trains all models and saves them for Flask API use.
"""
import pandas as pd
import numpy as np
import pickle, os, json
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.metrics import (roc_auc_score, f1_score, average_precision_score,
                              classification_report, confusion_matrix, roc_curve,
                              precision_recall_curve)
import warnings
warnings.filterwarnings('ignore')

FEATURE_COLS = [
    'log_claim_amount','amount_zscore','is_amount_outlier','days_admitted',
    'num_procedures_same_day','prior_claims_90d','diagnosis_mismatch',
    'days_to_submission','is_duplicate','self_referral','patient_age',
    'specialty_enc','state_enc','patient_gender_enc','claim_month',
    'claim_dayofweek','is_weekend','claim_quarter','prov_total_claims',
    'prov_avg_amount','prov_avg_procedures','prov_amount_ratio','risk_score',
]

def engineer_features(df, prov_stats=None):
    df = df.copy()
    df['claim_date'] = pd.to_datetime(df['claim_date'])
    df['claim_month'] = df['claim_date'].dt.month
    df['claim_dayofweek'] = df['claim_date'].dt.dayofweek
    df['is_weekend'] = (df['claim_dayofweek'] >= 5).astype(int)
    df['claim_quarter'] = df['claim_date'].dt.quarter
    df['log_claim_amount'] = np.log1p(df['claim_amount'])
    df['amount_zscore'] = np.abs(stats.zscore(df['claim_amount']))
    df['is_amount_outlier'] = (df['amount_zscore'] > 3).astype(int)

    if prov_stats is None:
        prov_stats = df.groupby('provider_id').agg(
            prov_total_claims=('claim_id','count'),
            prov_avg_amount=('claim_amount','mean'),
            prov_fraud_rate=('is_fraud','mean') if 'is_fraud' in df.columns else ('claim_amount','count'),
            prov_avg_procedures=('num_procedures_same_day','mean')
        ).reset_index()
    df = df.merge(prov_stats, on='provider_id', how='left')
    df['prov_total_claims'] = df['prov_total_claims'].fillna(1)
    df['prov_avg_amount'] = df['prov_avg_amount'].fillna(df['claim_amount'].mean())
    df['prov_avg_procedures'] = df['prov_avg_procedures'].fillna(1)
    df['prov_amount_ratio'] = df['claim_amount'] / (df['prov_avg_amount'] + 1)

    df['risk_score'] = (
        df['diagnosis_mismatch'] * 3 + df['is_duplicate'] * 4 +
        df['self_referral'] * 2 + df['is_amount_outlier'] * 2 +
        (df['prior_claims_90d'] > 3).astype(int) * 2 +
        (df['days_to_submission'] > 90).astype(int) * 1 +
        (df['num_procedures_same_day'] > 3).astype(int) * 2
    )

    le_spec = LabelEncoder()
    le_state = LabelEncoder()
    le_gender = LabelEncoder()
    df['specialty_enc'] = le_spec.fit_transform(df['specialty'].astype(str))
    df['state_enc'] = le_state.fit_transform(df['state'].astype(str))
    df['patient_gender_enc'] = le_gender.fit_transform(df['patient_gender'].astype(str))

    return df, prov_stats, {'specialty': le_spec, 'state': le_state, 'gender': le_gender}

def oversample(X, y, ratio=0.35):
    fraud_idx = np.where(y==1)[0]
    legit_idx = np.where(y==0)[0]
    target = int(len(legit_idx) * ratio)
    extra = target - len(fraud_idx)
    if extra > 0:
        resample = np.random.choice(fraud_idx, size=extra, replace=True)
        return np.vstack([X, X[resample]]), np.concatenate([y, y[resample]])
    return X, y

def train_and_save(df):
    os.makedirs("models", exist_ok=True)

    df_feat, prov_stats, encoders = engineer_features(df)
    X = df_feat[FEATURE_COLS].fillna(0).values
    y = df_feat['is_fraud'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    X_train_os, y_train_os = oversample(X_train_sc, y_train)

    models = {}

    lr = LogisticRegression(max_iter=1000, C=0.5, class_weight='balanced', random_state=42)
    lr.fit(X_train_os, y_train_os)
    models['Logistic Regression'] = lr

    rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train_os, y_train_os)
    models['Random Forest'] = rf

    gb = GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1,
                                    subsample=0.8, random_state=42)
    gb.fit(X_train_os, y_train_os)
    models['Gradient Boosting'] = gb

    iso = IsolationForest(contamination=0.08, n_estimators=150, random_state=42)
    iso.fit(X_train_sc)
    models['Isolation Forest'] = iso

    # Evaluate & build metrics JSON
    metrics = {}
    roc_data = {}
    cm_data = {}
    for name, model in models.items():
        if name == 'Isolation Forest':
            preds = (model.predict(X_test_sc) == -1).astype(int)
            probs = -model.score_samples(X_test_sc)
        else:
            preds = model.predict(X_test_sc)
            probs = model.predict_proba(X_test_sc)[:,1]
        auc = roc_auc_score(y_test, probs)
        f1 = f1_score(y_test, preds)
        ap = average_precision_score(y_test, probs)
        fpr, tpr, _ = roc_curve(y_test, probs)
        cm = confusion_matrix(y_test, preds).tolist()
        metrics[name] = {'auc': round(auc,4), 'f1': round(f1,4), 'ap': round(ap,4),
                         'report': classification_report(y_test, preds, target_names=['Legit','Fraud'], output_dict=True)}
        roc_data[name] = {'fpr': fpr.tolist()[::5], 'tpr': tpr.tolist()[::5]}
        cm_data[name] = cm

    # Cross-validation
    cv = cross_val_score(rf, X_train_sc, y_train, cv=5, scoring='roc_auc')
    metrics['cv_auc_mean'] = round(cv.mean(), 4)
    metrics['cv_auc_std'] = round(cv.std(), 4)

    # Feature importances
    feat_imp = dict(zip(FEATURE_COLS, rf.feature_importances_.tolist()))
    feat_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

    # Monthly trend
    df_feat['month'] = df_feat['claim_date'].dt.to_period('M').astype(str)
    monthly = df_feat.groupby('month').agg(total=('is_fraud','count'), fraud=('is_fraud','sum')).reset_index()
    monthly['fraud_rate'] = (monthly['fraud'] / monthly['total'] * 100).round(2)
    monthly_data = monthly.to_dict('records')

    # Specialty fraud rates
    spec_fraud = df_feat.groupby('specialty')['is_fraud'].agg(['mean','sum','count']).reset_index()
    spec_fraud.columns = ['specialty','fraud_rate','fraud_count','total']
    spec_fraud['fraud_rate'] = (spec_fraud['fraud_rate'] * 100).round(2)
    spec_fraud = spec_fraud.sort_values('fraud_rate', ascending=False)
    spec_data = spec_fraud.to_dict('records')

    # Monte Carlo
    fraud_claims = df[df['is_fraud']==1]['claim_amount'].values
    total = len(df)
    np.random.seed(42)
    sim_losses = []
    sim_rates = []
    for _ in range(5000):
        r = np.random.beta(df['is_fraud'].sum(), (df['is_fraud']==0).sum())
        n = np.random.binomial(total, r)
        loss = np.random.choice(fraud_claims, size=max(1,n), replace=True).sum()
        sim_rates.append(round(r*100, 3))
        sim_losses.append(round(loss/1e6, 4))
    mc_data = {
        'rates': sim_rates[::10],
        'losses': sim_losses[::10],
        'rate_ci': [round(np.percentile(sim_rates,2.5),2), round(np.percentile(sim_rates,97.5),2)],
        'loss_median': round(np.percentile(sim_losses,50),2),
        'loss_p95': round(np.percentile(sim_losses,95),2),
    }

    # Summary stats
    summary = {
        'total_claims': len(df),
        'fraud_count': int(df['is_fraud'].sum()),
        'fraud_rate': round(df['is_fraud'].mean()*100, 2),
        'avg_fraud_amount': round(df[df['is_fraud']==1]['claim_amount'].mean(), 2),
        'avg_legit_amount': round(df[df['is_fraud']==0]['claim_amount'].mean(), 2),
        'outlier_fraud_rate': round(df_feat[df_feat['is_amount_outlier']==1]['is_fraud'].mean()*100, 2),
        'date_range': [df['claim_date'].min(), df['claim_date'].max()],
    }

    # Save everything
    with open('models/rf_model.pkl','wb') as f: pickle.dump(rf, f)
    with open('models/gb_model.pkl','wb') as f: pickle.dump(gb, f)
    with open('models/scaler.pkl','wb') as f: pickle.dump(scaler, f)
    with open('models/prov_stats.pkl','wb') as f: pickle.dump(prov_stats, f)
    with open('models/encoders.pkl','wb') as f: pickle.dump(encoders, f)
    with open('models/metrics.json','w') as f: json.dump(metrics, f)
    with open('models/roc_data.json','w') as f: json.dump(roc_data, f)
    with open('models/cm_data.json','w') as f: json.dump(cm_data, f)
    with open('models/feat_imp.json','w') as f: json.dump(feat_imp, f)
    with open('models/monthly_data.json','w') as f: json.dump(monthly_data, f)
    with open('models/spec_data.json','w') as f: json.dump(spec_data, f)
    with open('models/mc_data.json','w') as f: json.dump(mc_data, f)
    with open('models/summary.json','w') as f: json.dump(summary, f)

    print("✅ All models and data saved.")
    return metrics, summary

if __name__ == "__main__":
    from generate_data import generate_dataset
    df = generate_dataset()
    train_and_save(df)
