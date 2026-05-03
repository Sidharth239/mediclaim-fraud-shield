"""
MediClaim Fraud Shield — Flask Backend
Fully working API + serves the frontend
"""
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle, json, os
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB upload limit

# ── Load models & data on startup ──
def load_artifacts():
    artifacts = {}
    try:
        with open('models/rf_model.pkl','rb') as f: artifacts['rf'] = pickle.load(f)
        with open('models/gb_model.pkl','rb') as f: artifacts['gb'] = pickle.load(f)
        with open('models/scaler.pkl','rb') as f: artifacts['scaler'] = pickle.load(f)
        with open('models/prov_stats.pkl','rb') as f: artifacts['prov_stats'] = pickle.load(f)
        with open('models/encoders.pkl','rb') as f: artifacts['encoders'] = pickle.load(f)
        with open('models/metrics.json') as f: artifacts['metrics'] = json.load(f)
        with open('models/roc_data.json') as f: artifacts['roc_data'] = json.load(f)
        with open('models/cm_data.json') as f: artifacts['cm_data'] = json.load(f)
        with open('models/feat_imp.json') as f: artifacts['feat_imp'] = json.load(f)
        with open('models/monthly_data.json') as f: artifacts['monthly_data'] = json.load(f)
        with open('models/spec_data.json') as f: artifacts['spec_data'] = json.load(f)
        with open('models/mc_data.json') as f: artifacts['mc_data'] = json.load(f)
        with open('models/summary.json') as f: artifacts['summary'] = json.load(f)
        print("✅ All artifacts loaded.")
    except Exception as e:
        print(f"⚠️  Artifacts not found, run ml_pipeline.py first: {e}")
    return artifacts

A = load_artifacts()

FEATURE_COLS = [
    'log_claim_amount','amount_zscore','is_amount_outlier','days_admitted',
    'num_procedures_same_day','prior_claims_90d','diagnosis_mismatch',
    'days_to_submission','is_duplicate','self_referral','patient_age',
    'specialty_enc','state_enc','patient_gender_enc','claim_month',
    'claim_dayofweek','is_weekend','claim_quarter','prov_total_claims',
    'prov_avg_amount','prov_avg_procedures','prov_amount_ratio','risk_score',
]

SPECIALTIES = ["Cardiology","Orthopedics","Oncology","Neurology","General Practice",
               "Dermatology","Psychiatry","Radiology","Emergency Medicine","Surgery"]
STATES = ["CA","TX","FL","NY","PA","OH","IL","GA","NC","MI","NJ","VA","WA","AZ","MA"]


def prepare_single_claim(data):
    """Convert form input dict into model-ready feature vector."""
    enc = A['encoders']
    prov_stats = A['prov_stats']

    claim_amount = float(data.get('claim_amount', 1000))
    claim_date = pd.to_datetime(data.get('claim_date', '2023-01-01'))
    provider_id = data.get('provider_id', 'PROV-9999')
    specialty = data.get('specialty', 'General Practice')
    state = data.get('state', 'CA')
    gender = data.get('patient_gender', 'M')
    age = int(data.get('patient_age', 45))
    days_admitted = int(data.get('days_admitted', 0))
    num_procs = int(data.get('num_procedures_same_day', 1))
    prior_claims = int(data.get('prior_claims_90d', 0))
    diag_mismatch = int(data.get('diagnosis_mismatch', 0))
    days_sub = int(data.get('days_to_submission', 30))
    is_dup = int(data.get('is_duplicate', 0))
    self_ref = int(data.get('self_referral', 0))

    # Provider stats
    prov_row = prov_stats[prov_stats['provider_id'] == provider_id]
    prov_total = float(prov_row['prov_total_claims'].values[0]) if len(prov_row) else 1.0
    prov_avg_amt = float(prov_row['prov_avg_amount'].values[0]) if len(prov_row) else claim_amount
    prov_avg_procs = float(prov_row['prov_avg_procedures'].values[0]) if len(prov_row) else 1.0

    # Engineered
    log_amount = np.log1p(claim_amount)
    amount_zscore = abs((claim_amount - 4485) / 9885)
    is_outlier = 1 if amount_zscore > 3 else 0
    prov_ratio = claim_amount / (prov_avg_amt + 1)
    risk_score = (diag_mismatch*3 + is_dup*4 + self_ref*2 + is_outlier*2 +
                  (1 if prior_claims>3 else 0)*2 + (1 if days_sub>90 else 0) +
                  (1 if num_procs>3 else 0)*2)

    # Encode
    try: spec_enc = enc['specialty'].transform([specialty])[0]
    except: spec_enc = 0
    try: state_enc = enc['state'].transform([state])[0]
    except: state_enc = 0
    try: gender_enc = enc['gender'].transform([gender])[0]
    except: gender_enc = 0

    features = [
        log_amount, amount_zscore, is_outlier, days_admitted, num_procs,
        prior_claims, diag_mismatch, days_sub, is_dup, self_ref, age,
        spec_enc, state_enc, gender_enc,
        claim_date.month, claim_date.dayofweek,
        1 if claim_date.dayofweek >= 5 else 0,
        claim_date.quarter,
        prov_total, prov_avg_amt, prov_avg_procs, prov_ratio, risk_score,
    ]
    return np.array(features).reshape(1, -1), risk_score


# ══════════════════════════════════════════
# ROUTES — Pages
# ══════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


# ══════════════════════════════════════════
# API — Dashboard Data
# ══════════════════════════════════════════

@app.route('/api/summary')
def api_summary():
    return jsonify(A.get('summary', {}))

@app.route('/api/metrics')
def api_metrics():
    return jsonify(A.get('metrics', {}))

@app.route('/api/roc')
def api_roc():
    return jsonify(A.get('roc_data', {}))

@app.route('/api/confusion_matrix')
def api_cm():
    return jsonify(A.get('cm_data', {}))

@app.route('/api/feature_importance')
def api_feat_imp():
    fi = A.get('feat_imp', {})
    top15 = dict(list(fi.items())[:15])
    return jsonify(top15)

@app.route('/api/monthly_trend')
def api_monthly():
    return jsonify(A.get('monthly_data', []))

@app.route('/api/specialty_fraud')
def api_specialty():
    return jsonify(A.get('spec_data', []))

@app.route('/api/monte_carlo')
def api_mc():
    return jsonify(A.get('mc_data', {}))


# ══════════════════════════════════════════
# API — Single Claim Prediction
# ══════════════════════════════════════════

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features, risk_score = prepare_single_claim(data)
        scaled = A['scaler'].transform(features)

        gb_prob = float(A['gb'].predict_proba(scaled)[0][1])
        rf_prob = float(A['rf'].predict_proba(scaled)[0][1])
        ensemble_prob = round((gb_prob * 0.6 + rf_prob * 0.4), 4)

        if ensemble_prob >= 0.7:
            risk_level = "HIGH"
            recommendation = "Flag for immediate investigation"
        elif ensemble_prob >= 0.4:
            risk_level = "MEDIUM"
            recommendation = "Schedule for manual review"
        else:
            risk_level = "LOW"
            recommendation = "Approve with standard audit"

        # Generate explanation
        flags = []
        d = data
        if int(d.get('is_duplicate',0)): flags.append("Duplicate claim detected")
        if int(d.get('diagnosis_mismatch',0)): flags.append("Diagnosis-procedure mismatch")
        if int(d.get('self_referral',0)): flags.append("Self-referral pattern")
        if int(d.get('prior_claims_90d',0)) > 3: flags.append(f"High prior claims: {d.get('prior_claims_90d')} in 90 days")
        if int(d.get('days_to_submission',30)) > 90: flags.append("Late claim submission")
        if int(d.get('num_procedures_same_day',1)) > 3: flags.append(f"Excessive procedures: {d.get('num_procedures_same_day')} on same day")
        if float(d.get('claim_amount',0)) > 20000: flags.append(f"High claim amount: ${float(d.get('claim_amount',0)):,.0f}")

        return jsonify({
            'fraud_probability': ensemble_prob,
            'gb_probability': round(gb_prob, 4),
            'rf_probability': round(rf_prob, 4),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'risk_score': risk_score,
            'flags': flags,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ══════════════════════════════════════════
# API — Upload CSV & Batch Predict
# ══════════════════════════════════════════

@app.route('/api/upload', methods=['POST'])
def upload_csv():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Please upload a CSV file'}), 400

        df = pd.read_csv(file)
        required = ['claim_amount','days_admitted','num_procedures_same_day',
                    'prior_claims_90d','diagnosis_mismatch','days_to_submission',
                    'is_duplicate','self_referral','patient_age']
        missing = [c for c in required if c not in df.columns]
        if missing:
            return jsonify({'error': f'Missing columns: {missing}'}), 400

        # Fill defaults for optional columns
        df['claim_date'] = df.get('claim_date', '2023-01-01')
        df['provider_id'] = df.get('provider_id', 'PROV-9999')
        df['specialty'] = df.get('specialty', 'General Practice')
        df['state'] = df.get('state', 'CA')
        df['patient_gender'] = df.get('patient_gender', 'M')

        results = []
        for _, row in df.iterrows():
            feats, risk_score = prepare_single_claim(row.to_dict())
            scaled = A['scaler'].transform(feats)
            gb_p = float(A['gb'].predict_proba(scaled)[0][1])
            rf_p = float(A['rf'].predict_proba(scaled)[0][1])
            prob = round(gb_p*0.6 + rf_p*0.4, 4)
            level = "HIGH" if prob>=0.7 else ("MEDIUM" if prob>=0.4 else "LOW")
            results.append({
                'claim_id': row.get('claim_id', f'CLM-{_+1:04d}'),
                'claim_amount': row['claim_amount'],
                'fraud_probability': prob,
                'risk_level': level,
                'risk_score': risk_score,
            })

        results_df = pd.DataFrame(results)
        high = int((results_df['risk_level']=='HIGH').sum())
        medium = int((results_df['risk_level']=='MEDIUM').sum())
        low = int((results_df['risk_level']=='LOW').sum())
        avg_prob = round(results_df['fraud_probability'].mean(), 4)

        return jsonify({
            'total': len(results),
            'high_risk': high,
            'medium_risk': medium,
            'low_risk': low,
            'avg_fraud_probability': avg_prob,
            'results': results[:200],  # cap at 200 rows for response size
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ══════════════════════════════════════════
# API — Claims Table with Filters
# ══════════════════════════════════════════

@app.route('/api/claims')
def api_claims():
    try:
        df = pd.read_csv('data/claims_raw.csv')
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        search = request.args.get('search', '').lower()
        fraud_filter = request.args.get('fraud', '')
        specialty_filter = request.args.get('specialty', '')
        min_amount = request.args.get('min_amount', '')
        max_amount = request.args.get('max_amount', '')

        if search:
            mask = (df['claim_id'].str.lower().str.contains(search) |
                    df['provider_id'].str.lower().str.contains(search) |
                    df['specialty'].str.lower().str.contains(search))
            df = df[mask]
        if fraud_filter in ['0','1']:
            df = df[df['is_fraud'] == int(fraud_filter)]
        if specialty_filter:
            df = df[df['specialty'] == specialty_filter]
        if min_amount:
            df = df[df['claim_amount'] >= float(min_amount)]
        if max_amount:
            df = df[df['claim_amount'] <= float(max_amount)]

        total = len(df)
        start = (page - 1) * per_page
        end = start + per_page
        page_df = df.iloc[start:end].copy()

        records = page_df[['claim_id','patient_id','provider_id','claim_date','specialty',
                           'procedure','claim_amount','diagnosis_mismatch','is_duplicate',
                           'self_referral','prior_claims_90d','is_fraud']].to_dict('records')
        return jsonify({'total': total, 'page': page, 'per_page': per_page,
                        'total_pages': (total + per_page - 1) // per_page, 'claims': records})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ══════════════════════════════════════════
# API — Retrain trigger
# ══════════════════════════════════════════

@app.route('/api/retrain', methods=['POST'])
def retrain():
    try:
        from generate_data import generate_dataset
        from ml_pipeline import train_and_save
        df = generate_dataset()
        metrics, summary = train_and_save(df)
        global A
        A = load_artifacts()
        return jsonify({'status': 'success', 'message': 'Models retrained successfully',
                        'best_auc': metrics['Gradient Boosting']['auc']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Auto-train if models don't exist
    if not os.path.exists('models/gb_model.pkl'):
        print("🔄 No models found. Training now...")
        from generate_data import generate_dataset
        from ml_pipeline import train_and_save
        df = generate_dataset()
        train_and_save(df)
        A = load_artifacts()
    app.run(debug=True, port=5000)
