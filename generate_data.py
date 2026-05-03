"""Generate synthetic health insurance claims dataset."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

N_CLAIMS = 15000
FRAUD_RATE = 0.08

SPECIALTIES = ["Cardiology","Orthopedics","Oncology","Neurology",
               "General Practice","Dermatology","Psychiatry","Radiology",
               "Emergency Medicine","Surgery"]

PROCEDURES = {
    "Cardiology":        [("EKG",120),("Echo",850),("Angioplasty",12000),("Stress Test",600)],
    "Orthopedics":       [("X-Ray",200),("MRI Knee",1400),("Hip Replacement",25000),("Physical Therapy",180)],
    "Oncology":          [("Chemo Session",4000),("CT Scan",1200),("Biopsy",2200),("PET Scan",5000)],
    "Neurology":         [("EEG",700),("MRI Brain",1800),("Nerve Conduction",900),("Lumbar Puncture",1500)],
    "General Practice":  [("Office Visit",150),("Blood Panel",300),("Vaccination",80),("Urinalysis",60)],
    "Dermatology":       [("Skin Biopsy",400),("Laser Treatment",600),("Patch Test",250),("Excision",900)],
    "Psychiatry":        [("Therapy Session",200),("Psych Eval",350),("Medication Mgmt",150),("Crisis Eval",500)],
    "Radiology":         [("Chest X-Ray",250),("Abdominal CT",1500),("Bone Scan",800),("Ultrasound",400)],
    "Emergency Medicine":[("ER Visit L1",800),("ER Visit L4",2500),("Trauma Care",8000),("Observation",3000)],
    "Surgery":           [("Appendectomy",15000),("Gallbladder Removal",12000),("Hernia Repair",9000),("CABG",40000)],
}

STATES = ["CA","TX","FL","NY","PA","OH","IL","GA","NC","MI","NJ","VA","WA","AZ","MA"]
DIAGNOSIS_CODES = [f"ICD-{chr(65+i)}{random.randint(10,99)}" for i in range(30)]

def generate_claim(claim_id, is_fraud):
    specialty = random.choice(SPECIALTIES)
    proc_name, base_cost = random.choice(PROCEDURES[specialty])
    state = random.choice(STATES)
    age = max(18, min(90, int(np.random.normal(52, 18))))
    gender = random.choice(["M","F"])
    base_date = datetime(2022, 1, 1)
    claim_date = base_date + timedelta(days=random.randint(0, 730))
    provider_id = f"PROV-{random.randint(1000,1200):04d}" if is_fraud else f"PROV-{random.randint(1000,5000):04d}"
    noise = np.random.normal(1.0, 0.1)
    claim_amount = round(base_cost * noise * (np.random.uniform(1.5,4.0) if is_fraud else 1.0), 2)
    days_admitted = random.choices([0,random.randint(1,3),random.randint(8,30)],[0.3,0.3,0.4])[0] if is_fraud \
                else random.choices([0,random.randint(1,5),random.randint(6,15)],[0.5,0.35,0.15])[0]
    num_procedures = random.choices([1,2,3,4,5,6],[0.05,0.1,0.15,0.2,0.25,0.25])[0] if is_fraud \
                else random.choices([1,2,3,4,5,6],[0.5,0.25,0.12,0.08,0.03,0.02])[0]
    prior_claims_90d = random.choices(range(9),[0.05,0.05,0.1,0.1,0.15,0.2,0.15,0.1,0.1])[0] if is_fraud \
                   else random.choices(range(9),[0.45,0.25,0.12,0.08,0.04,0.03,0.01,0.01,0.01])[0]
    diagnosis_mismatch = 1 if (is_fraud and random.random()<0.6) else (1 if random.random()<0.05 else 0)
    days_to_submission = random.randint(91,365) if (is_fraud and random.random()<0.45) else random.randint(1,90)
    is_duplicate = 1 if (is_fraud and random.random()<0.3) else 0
    self_referral = 1 if (is_fraud and random.random()<0.4) else (1 if random.random()<0.03 else 0)
    return {
        "claim_id": f"CLM-{claim_id:06d}", "patient_id": f"PAT-{random.randint(10000,99999)}",
        "provider_id": provider_id, "claim_date": claim_date.strftime("%Y-%m-%d"),
        "state": state, "specialty": specialty, "procedure": proc_name,
        "diagnosis_code": random.choice(DIAGNOSIS_CODES), "patient_age": age,
        "patient_gender": gender, "claim_amount": claim_amount, "days_admitted": days_admitted,
        "num_procedures_same_day": num_procedures, "prior_claims_90d": prior_claims_90d,
        "diagnosis_mismatch": diagnosis_mismatch, "days_to_submission": days_to_submission,
        "is_duplicate": is_duplicate, "self_referral": self_referral, "is_fraud": int(is_fraud),
    }

def generate_dataset():
    n_fraud = int(N_CLAIMS * FRAUD_RATE)
    n_legit = N_CLAIMS - n_fraud
    records = [generate_claim(i+1, False) for i in range(n_legit)]
    records += [generate_claim(n_legit+i+1, True) for i in range(n_fraud)]
    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/claims_raw.csv", index=False)
    return df

if __name__ == "__main__":
    df = generate_dataset()
    print(f"Generated {len(df)} claims, {df['is_fraud'].sum()} fraud")
