"""
Secure Medical Prescription Data System
========================================
- Data extraction from JSON prescription records
- Shamir's Secret Sharing on medicine details
- ElGamal homomorphic encryption on shares
- Doctor OTP authentication & decryption
- Skyline query for medicine recommendation (BNL, Divide & Conquer, Dominance-Based)
"""

# ─────────────────────────────────────────────
# DEPENDENCIES
# ─────────────────────────────────────────────
import builtins
builtins.long = int  # Python 3 compatibility for secretsharing library

import ast
import hashlib
import json
import os
import random
import re
import sys
import time

import pandas as pd
from Crypto.PublicKey import ElGamal
from Crypto.Random import random as crypto_random
from Crypto.Util import number
from Crypto.Util.number import getPrime, bytes_to_long, long_to_bytes
from secretsharing import SecretSharer


# ─────────────────────────────────────────────
# SECTION 1: DATA EXTRACTION
# ─────────────────────────────────────────────

def clean_json_text(text):
    """Clean raw OCR JSON strings to valid JSON format."""
    if not isinstance(text, str):
        return text
    text = text.replace("'", '"')
    text = re.sub(r",\s*([\]}])", r"\1", text)
    text = text.replace("None", "null").replace("True", "true").replace("False", "false")
    return text


def safe_json_load(text):
    """Safely parse a JSON string after cleaning."""
    try:
        return json.loads(clean_json_text(text))
    except json.JSONDecodeError:
        return None


def extract_details(row):
    """Extract structured fields from a JSON results column row."""
    try:
        cleaned = clean_json_text(row['results'])
        details = json.loads(cleaned)

        patient  = details.get('patient_details', {})
        clinic   = details.get('clinic_pharmacy_details', {})
        doctor   = details.get('doctor_details', {})
        meds     = details.get('medicine_details', [])
        lab      = details.get('lab_test_details', {})

        extracted = {
            'text':                       row['text'],
            'clinic_pharmacy_name':       clinic.get('clinic_pharmacy_name', ''),
            'clinic_pharmacy_address':    clinic.get('clinic_pharmacy_address', ''),
            'patient_name':               patient.get('patient_name', ''),
            'doctor_name':                doctor.get('doctor_name', ''),
            'doctor_qualifications':      doctor.get('doctor_qualifications', ''),
            'doctor_registration_number': doctor.get('doctor_registration_number', ''),
            'chief_complaints_diagnosis': doctor.get('chief_complaints_diagnosis', ''),
        }

        for i in range(3):
            med = meds[i] if i < len(meds) else {}
            extracted[f'medicine_{i+1}_name']            = med.get('medicine_name', '')
            extracted[f'medicine_{i+1}_dosage']          = med.get('medicine_dosage', '')
            extracted[f'medicine_{i+1}_frequency']       = med.get('medicine_frequency', '')
            extracted[f'medicine_{i+1}_course_duration'] = med.get('course_duration', '')

        if isinstance(lab, dict):
            extracted['lab_test_names']   = lab.get('lab_test_names', '')
            extracted['procedure_request'] = lab.get('procedure_request', '')

        return pd.Series(extracted)

    except (json.JSONDecodeError, AttributeError):
        return pd.Series({
            'text': row.get('text', ''),
            **{k: '' for k in [
                'clinic_pharmacy_name','clinic_pharmacy_address','patient_name',
                'doctor_name','doctor_qualifications','doctor_registration_number',
                'chief_complaints_diagnosis','lab_test_names','procedure_request',
                'medicine_1_name','medicine_1_dosage','medicine_1_frequency','medicine_1_course_duration',
                'medicine_2_name','medicine_2_dosage','medicine_2_frequency','medicine_2_course_duration',
                'medicine_3_name','medicine_3_dosage','medicine_3_frequency','medicine_3_course_duration',
            ]}
        })


def combine_medicine_details(df):
    """Combine medicine name, dosage, frequency, and duration into a single string."""
    return df.apply(
        lambda row: (
            f"{row.get('medicine_name','Unknown')} "
            f"({row.get('medicine_dosage','Unknown')}, "
            f"{row.get('medicine_frequency','Unknown')}, "
            f"{row.get('medicine_course_duration','Unknown')})"
            if pd.notna(row.get('medicine_name')) else ""
        ),
        axis=1
    )


def extract_and_save(input_csv='pre.csv',
                     output_cleaned='modified_pre.csv',
                     output_combined='retrieved_data_with_combined_medicine.csv'):
    """Full data extraction pipeline from raw prescription CSV."""
    df = pd.read_csv(input_csv, encoding='ISO-8859-1')
    df['parsed'] = df['results'].apply(safe_json_load)
    expanded = pd.json_normalize(df['parsed'])

    if 'medicine_details' in expanded.columns:
        expanded = expanded.explode('medicine_details')
        expanded['medicine_details'] = expanded['medicine_details'].apply(
            lambda x: x if isinstance(x, dict) else {}
        )
        med_df = pd.json_normalize(expanded['medicine_details'])
        expanded = pd.concat([expanded.drop(columns=['medicine_details']).reset_index(drop=True), med_df], axis=1)

    required = [
        'procedure.chief_complaints_diagnosis',
        'patient_details.patient_name',
        'doctor_details.doctor_name',
        'doctor_details.doctor_registration_number',
    ]
    missing = [c for c in required if c not in expanded.columns]
    if missing:
        print(f"⚠ Missing columns: {missing}")
        return

    med_cols = ['medicine_name', 'medicine_dosage', 'medicine_frequency', 'medicine_course_duration']
    if all(c in expanded.columns for c in med_cols):
        expanded['medicine_details_combined'] = combine_medicine_details(expanded)
    else:
        expanded['medicine_details_combined'] = 'Unknown'

    relevant = expanded[required + ['medicine_details_combined']].dropna(subset=required)
    for col in required:
        relevant[col] = relevant[col].apply(lambda x: str(x) if isinstance(x, list) else x)

    relevant = relevant.groupby(required, as_index=False).agg(
        {'medicine_details_combined': lambda x: "; ".join(map(str, x.dropna()))}
    ).drop_duplicates()

    relevant.to_csv(output_combined, index=False)
    print(f"✅ Combined data saved to '{output_combined}'. Records: {len(relevant)}")


# ─────────────────────────────────────────────
# SECTION 2: SHAMIR'S SECRET SHARING
# ─────────────────────────────────────────────

def hash_secret(secret):
    """SHA-256 hash of a normalised string."""
    normalised = " ".join(str(secret).strip().lower().split())
    return hashlib.sha256(normalised.encode('utf-8')).hexdigest()


def generate_shamir_shares(hashed_secret, threshold=3, num_shares=5):
    return SecretSharer.split_secret(hashed_secret, threshold, num_shares)


def reconstruct_shamir_secret(shares):
    return SecretSharer.recover_secret(shares)


def shamir_encrypt_csv(input_csv, output_csv, column='medicine_details_combined',
                       threshold=3, num_shares=5):
    """Hash the target column and generate Shamir shares for each row."""
    df = pd.read_csv(input_csv, encoding='latin1')
    share_cols = [f"share_{i+1}" for i in range(num_shares)]

    for idx, row in df.iterrows():
        val = row.get(column)
        if pd.isna(val) or str(val).strip().lower() == 'unknown':
            df.loc[idx, share_cols] = [None] * num_shares
            df.loc[idx, 'hashed_secret'] = None
        else:
            h = hash_secret(str(val))
            df.loc[idx, 'hashed_secret'] = h
            shares = generate_shamir_shares(h, threshold, num_shares)
            for i, s in enumerate(shares):
                df.loc[idx, share_cols[i]] = s

    df.drop(columns=[column], inplace=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Shamir shares saved to '{output_csv}'")


def shamir_reconstruct_and_verify(shares_csv, output_csv):
    """Reconstruct secrets from shares and verify against stored hash."""
    df = pd.read_csv(shares_csv, dtype=str)
    share_cols = [c for c in df.columns if c.startswith('share_')]
    total, verified = 0, 0

    for idx, row in df.iterrows():
        shares = [s for s in row[share_cols].dropna() if '-' in str(s)]
        expected = str(row.get('hashed_secret', '')).strip().zfill(64)
        if len(shares) < 3:
            df.at[idx, 'verification_status'] = '❌ Insufficient Shares'
            continue
        try:
            reconstructed = reconstruct_shamir_secret(shares[:3]).zfill(64)
            match = reconstructed == expected
            df.at[idx, 'reconstructed_hash'] = reconstructed
            df.at[idx, 'verification_status'] = '✅ Verified' if match else '❌ Failed'
            if match:
                verified += 1
            total += 1
        except Exception as e:
            df.at[idx, 'verification_status'] = f'Error: {e}'

    accuracy = (verified / total * 100) if total else 0
    df.to_csv(output_csv, index=False)
    print(f"✅ Verification done — Accuracy: {accuracy:.2f}% ({verified}/{total})")


# ─────────────────────────────────────────────
# SECTION 3: ELGAMAL ENCRYPTION
# ─────────────────────────────────────────────

def elgamal_keygen(bits=2048):
    p = getPrime(bits)
    g = random.randint(2, p - 2)
    x = random.randint(1, p - 2)
    h = pow(g, x, p)
    return (p, g, h), x


def elgamal_encrypt(m, pub):
    p, g, h = pub
    y = random.randint(1, p - 2)
    c1 = pow(g, y, p)
    c2 = (m * pow(h, y, p)) % p
    return (c1, c2)


def elgamal_decrypt(c1, c2, priv, pub):
    p, g, h = pub
    s = pow(c1, priv, p)
    s_inv = number.inverse(s, p)
    return (c2 * s_inv) % p


def save_elgamal_keys(pub, priv, path='elgamal_keys.json'):
    p, g, h = pub
    with open(path, 'w') as f:
        json.dump({'p': str(p), 'g': str(g), 'h': str(h), 'x': str(priv)}, f)


def load_elgamal_keys(path='elgamal_keys.json'):
    with open(path) as f:
        d = json.load(f)
    pub = (int(d['p']), int(d['g']), int(d['h']))
    priv = int(d['x'])
    return pub, priv


# ─────────────────────────────────────────────
# SECTION 4: MULTIPLICATIVE SECRET SHARING + ELGAMAL
# ─────────────────────────────────────────────

def multiplicative_shares(secret, p, num_shares=3):
    """Split secret multiplicatively mod p."""
    shares = []
    for _ in range(num_shares - 1):
        while True:
            s = random.randint(1, p - 2)
            if number.GCD(s, p) == 1:
                shares.append(s)
                break
    product = 1
    for s in shares:
        product = (product * s) % p
    inv_product = number.inverse(product, p)
    shares.append((secret * inv_product) % p)
    return shares


def string_to_chunks(s, chunk_size=16):
    b = s.encode()
    return [int.from_bytes(b[i:i+chunk_size], 'big') for i in range(0, len(b), chunk_size)]


def chunks_to_string(chunks):
    result = b''.join(int(c).to_bytes((int(c).bit_length() + 7) // 8, 'big') for c in chunks)
    return result.decode(errors='ignore')


def encrypt_csv(input_csv, output_csv, column, num_shares=3):
    """Encrypt medicine data column using multiplicative shares + ElGamal."""
    df = pd.read_csv(input_csv, encoding='latin1')
    pub, priv = elgamal_keygen(bits=2048)
    p = pub[0]

    df['chunk_count'] = None
    for i in range(num_shares):
        df[f'enc_chunks_c1_{i+1}'] = None
        df[f'enc_chunks_c2_{i+1}'] = None

    for idx, row in df.iterrows():
        text = row.get(column, '')
        if pd.isna(text) or str(text).lower() == 'unknown':
            continue
        try:
            chunks = string_to_chunks(str(text))
            if any(c >= p for c in chunks):
                print(f"⚠ Row {idx}: chunk too large, skipping")
                continue

            all_c1 = [[] for _ in range(num_shares)]
            all_c2 = [[] for _ in range(num_shares)]

            for chunk in chunks:
                shares = multiplicative_shares(chunk, p, num_shares)
                encrypted = [elgamal_encrypt(s, pub) for s in shares]
                for i, (c1, c2) in enumerate(encrypted):
                    all_c1[i].append(c1)
                    all_c2[i].append(c2)

            for i in range(num_shares):
                df.at[idx, f'enc_chunks_c1_{i+1}'] = json.dumps(all_c1[i])
                df.at[idx, f'enc_chunks_c2_{i+1}'] = json.dumps(all_c2[i])
            df.at[idx, 'chunk_count'] = len(chunks)

        except Exception as e:
            print(f"❌ Row {idx}: {e}")

    df.to_csv(output_csv, index=False)
    save_elgamal_keys(pub, priv)
    print(f"✅ Encrypted CSV saved to '{output_csv}'")
    print("✅ Keys saved to 'elgamal_keys.json'")


def decrypt_and_verify(csv_file, num_shares=3, output_csv='decrypted_output.csv'):
    """Decrypt encrypted shares and reconstruct original text."""
    pub, priv = load_elgamal_keys()
    p = pub[0]
    df = pd.read_csv(csv_file)
    success, total, rows = 0, 0, []

    for idx, row in df.iterrows():
        try:
            chunk_count = int(row['chunk_count'])
            decrypted_chunks = []

            for ci in range(chunk_count):
                c1_mul, c2_mul = 1, 1
                for i in range(num_shares):
                    c1 = int(json.loads(row[f'enc_chunks_c1_{i+1}'])[ci])
                    c2 = int(json.loads(row[f'enc_chunks_c2_{i+1}'])[ci])
                    c1_mul = (c1_mul * c1) % p
                    c2_mul = (c2_mul * c2) % p
                decrypted_chunks.append(elgamal_decrypt(c1_mul, c2_mul, priv, pub))

            recovered = chunks_to_string(decrypted_chunks)
            row = row.copy()
            row['decrypted_data'] = recovered
            rows.append(row)
            success += 1
            total += 1
        except Exception as e:
            print(f"❌ Row {idx}: {e}")
            total += 1

    if rows:
        result_df = pd.DataFrame(rows)
        enc_cols = [f'enc_chunks_c1_{i+1}' for i in range(num_shares)] + \
                   [f'enc_chunks_c2_{i+1}' for i in range(num_shares)]
        result_df.drop(columns=enc_cols, inplace=True, errors='ignore')
        result_df.to_csv(output_csv, index=False)
        print(f"✅ Decrypted data saved to '{output_csv}'")

    print(f"🔁 Finished: {success}/{total} records decrypted.")


# ─────────────────────────────────────────────
# SECTION 5: DOCTOR AUTHENTICATION (OTP / MFA)
# ─────────────────────────────────────────────

def generate_doctor_id():
    return "D" + str(random.randint(100000, 999999))


def register_doctor(name, registry='doctor_registry.csv'):
    try:
        df = pd.read_csv(registry)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['doctor_name', 'doctor_number'])

    if name in df['doctor_name'].values:
        doc_id = df[df['doctor_name'] == name]['doctor_number'].values[0]
        print(f"✅ Doctor '{name}' already registered: {doc_id}")
    else:
        doc_id = generate_doctor_id()
        df = pd.concat([df, pd.DataFrame([{'doctor_name': name, 'doctor_number': doc_id}])], ignore_index=True)
        df.to_csv(registry, index=False)
        print(f"✅ New doctor '{name}' registered: {doc_id}")
    return doc_id


def authenticate_doctor(name, registry='doctor_registry.csv'):
    try:
        df = pd.read_csv(registry)
    except FileNotFoundError:
        print("❌ No doctors registered.")
        return False

    if name not in df['doctor_name'].values:
        print(f"❌ Doctor '{name}' not found.")
        return False

    pin = str(random.randint(1000, 9999))
    print(f"🔐 OTP sent: {pin}")
    time.sleep(0.3)
    entered = input("Enter OTP: ")
    return entered == pin


def decrypt_for_doctor(csv_file, doctor_name, num_shares=3):
    """Decrypt records belonging to a specific doctor after OTP auth."""
    pub, priv = load_elgamal_keys()
    p = pub[0]
    df = pd.read_csv(csv_file)
    rows = []
    success, total = 0, 0

    doctor_col = 'doctor_details.doctor_name'
    if doctor_col not in df.columns:
        print(f"⚠ Column '{doctor_col}' not found; processing all rows.")
        doctor_col = None

    for idx, row in df.iterrows():
        if doctor_col and row.get(doctor_col) != doctor_name:
            continue
        try:
            chunk_count = int(row['chunk_count'])
            decrypted_chunks = []
            for ci in range(chunk_count):
                c1_mul, c2_mul = 1, 1
                for i in range(num_shares):
                    c1 = int(json.loads(row[f'enc_chunks_c1_{i+1}'])[ci])
                    c2 = int(json.loads(row[f'enc_chunks_c2_{i+1}'])[ci])
                    c1_mul = (c1_mul * c1) % p
                    c2_mul = (c2_mul * c2) % p
                decrypted_chunks.append(elgamal_decrypt(c1_mul, c2_mul, priv, pub))
            row = row.copy()
            row['medicine_details_combined'] = chunks_to_string(decrypted_chunks)
            rows.append(row)
            success += 1
            total += 1
        except Exception as e:
            print(f"❌ Row {idx}: {e}")
            total += 1

    if rows:
        out = f"reconstructed_records_{doctor_name}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"✅ Records saved to '{out}'")
    print(f"🔁 {success}/{total} decrypted.")


# ─────────────────────────────────────────────
# SECTION 6: SKYLINE QUERY ALGORITHMS
# ─────────────────────────────────────────────

FREQUENCY_ORDER = ["once daily", "od", "bd", "tds", "thrice daily", "qid"]


def parse_dosage(val):
    try:
        return float(str(val).strip().split()[0])
    except:
        return float('inf')


def parse_frequency(val):
    v = str(val).strip().lower()
    return FREQUENCY_ORDER.index(v) if v in FREQUENCY_ORDER else len(FREQUENCY_ORDER)


def parse_duration(val):
    try:
        return int(''.join(filter(str.isdigit, str(val))))
    except:
        return float('inf')


def extract_metrics(row):
    dosage_vals = [parse_dosage(row.get(f'medicine_{i}_dosage', 'inf')) for i in range(1, 4)]
    freq_vals   = [parse_frequency(row.get(f'medicine_{i}_frequency', '')) for i in range(1, 4)]
    dur_vals    = [parse_duration(row.get(f'medicine_{i}_course_duration', 'inf')) for i in range(1, 4)]
    return {
        'avg_dosage':   sum(dosage_vals) / 3,
        'avg_freq':     sum(freq_vals)   / 3,
        'avg_duration': sum(dur_vals)    / 3,
    }


def dominates(row1, row2, cols):
    """Return True if row1 dominates row2 (better or equal in all, strictly better in at least one)."""
    better = False
    for c in cols:
        v1, v2 = row1[c], row2[c]
        if pd.isna(v1) or pd.isna(v2):
            continue
        if v1 > v2:
            return False
        if v1 < v2:
            better = True
    return better


# --- Block Nested Loop (BNL) ---
def skyline_bnl(data, cols):
    skyline = []
    for _, row in data.iterrows():
        dominated = False
        to_remove = []
        for existing in skyline:
            if dominates(existing, row, cols):
                dominated = True
                break
            elif dominates(row, existing, cols):
                to_remove.append(existing)
        if not dominated:
            skyline = [r for r in skyline if not any(r.equals(rem) for rem in to_remove)]
            skyline.append(row)
    return pd.DataFrame(skyline)


# --- Divide and Conquer ---
def skyline_divide_conquer(data, cols):
    if len(data) <= 1:
        return data
    mid = len(data) // 2
    left  = skyline_divide_conquer(data.iloc[:mid], cols)
    right = skyline_divide_conquer(data.iloc[mid:], cols)
    return _merge_skylines(left, right, cols)


def _merge_skylines(left, right, cols):
    merged = pd.DataFrame(columns=left.columns)
    to_remove = set()
    for idx_r, row_r in right.iterrows():
        dominated = False
        for idx_l, row_l in left.iterrows():
            if dominates(row_l, row_r, cols):
                dominated = True
                break
            elif dominates(row_r, row_l, cols):
                to_remove.add(idx_l)
        if not dominated:
            merged = pd.concat([merged, pd.DataFrame([row_r])], ignore_index=True)
    left = left.drop(list(to_remove), errors='ignore')
    return pd.concat([merged, left], ignore_index=True)


# --- Dominance-Based Frequency ---
def skyline_dominance_frequency(data, cols, top_n=5):
    counts = []
    for i, r1 in data.iterrows():
        cnt = sum(1 for j, r2 in data.iterrows() if i != j and dominates(r1, r2, cols))
        counts.append((i, cnt))
    counts.sort(key=lambda x: -x[1])
    top_idx = [i for i, _ in counts[:top_n]]
    return data.loc[top_idx]


# --- Pareto / Frequency-Based ---
def skyline_pareto(data, symptom):
    """Simple Pareto dominance based on prescription frequency and doctor count."""
    filtered = data[data['procedure.chief_complaints_diagnosis']
                    .str.contains(symptom, case=False, na=False)].copy()
    if filtered.empty:
        return f"No results for: {symptom}"

    filtered['_med'] = filtered['medicine_details_combined'].astype(str).str.strip()
    filtered = filtered[filtered['_med'].apply(
        lambda m: len(m) > 2 and re.search(r'[a-zA-Z]', m)
                  and m.lower() not in ['daily','once','twice','thrice','na','unknown']
    )]
    if filtered.empty:
        return "No valid medicines found."

    freq  = filtered['_med'].value_counts().to_dict()
    docs  = filtered.groupby('_med')['doctor_details.doctor_name'].nunique().to_dict()
    meds  = list(freq.keys())

    rank1, rank3 = [], []
    for i, m1 in enumerate(meds):
        dominated = any(
            (freq.get(m2, 0) >= freq.get(m1, 0) and docs.get(m2, 0) >= docs.get(m1, 0))
            and (freq.get(m2, 0) > freq.get(m1, 0) or docs.get(m2, 0) > docs.get(m1, 0))
            for j, m2 in enumerate(meds) if i != j
        )
        (rank3 if dominated else rank1).append(m1)
    rank2 = list(set(meds) - set(rank1) - set(rank3))
    return {'Rank 1 (Skyline)': rank1, 'Rank 2 (Intermediate)': rank2, 'Rank 3 (Dominated)': rank3}


def run_skyline_query(data_csv, symptom, algorithm='bnl'):
    """
    Run a skyline query on medicine data.

    Parameters
    ----------
    data_csv   : str  — path to retrieved_data_with_combined_medicine.csv
    symptom    : str  — symptom keyword to filter on
    algorithm  : str  — 'bnl' | 'dc' | 'dominance' | 'pareto'
    """
    data = pd.read_csv(data_csv, encoding='latin1')
    filtered = data[data['procedure.chief_complaints_diagnosis']
                    .str.contains(symptom, case=False, na=False)].copy()

    if filtered.empty:
        print(f"No results for: {symptom}")
        return pd.DataFrame()

    if algorithm == 'pareto':
        result = skyline_pareto(data, symptom)
        if isinstance(result, str):
            print(result)
        else:
            for rank, meds in result.items():
                print(f"\n{rank}:")
                for m in meds:
                    print(f"  - {m}")
        return result

    # Compute metrics for numeric skyline methods
    metrics = [extract_metrics(row) for _, row in filtered.iterrows()]
    enriched = pd.concat([filtered.reset_index(drop=True), pd.DataFrame(metrics)], axis=1)
    cols = ['avg_dosage', 'avg_freq', 'avg_duration']

    if algorithm == 'bnl':
        result = skyline_bnl(enriched, cols)
    elif algorithm == 'dc':
        result = skyline_divide_conquer(enriched, cols)
    elif algorithm == 'dominance':
        result = skyline_dominance_frequency(enriched, cols)
    else:
        raise ValueError("algorithm must be 'bnl', 'dc', 'dominance', or 'pareto'")

    print(f"\n🌟 Skyline [{algorithm.upper()}] for '{symptom}' — {len(result)} results")
    if not result.empty:
        display_cols = [c for c in ['procedure.chief_complaints_diagnosis',
                                    'doctor_details.doctor_name',
                                    'medicine_details_combined'] if c in result.columns]
        print(result[display_cols].to_string(index=False))

    out = f"skyline_{algorithm}_{symptom.replace(' ','_')}.csv"
    result.to_csv(out, index=False)
    print(f"✅ Saved to '{out}'")
    return result


def jaccard_similarity(set1, set2):
    """Jaccard similarity between two sets (as percentage)."""
    s1, s2 = set(set1), set(set2)
    if not s1 and not s2:
        return 100.0
    return round(len(s1 & s2) / len(s1 | s2) * 100, 2)


# ─────────────────────────────────────────────
# SECTION 7: MAIN WORKFLOW
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Secure Medical Prescription System")
    print("=" * 55)

    # ── Step 1: Extract data ──────────────────────────────
    if os.path.exists('pre.csv'):
        print("\n[1] Extracting prescription data...")
        extract_and_save(
            input_csv='pre.csv',
            output_combined='retrieved_data_with_combined_medicine.csv'
        )
    else:
        print("\n[1] Skipping extraction — 'pre.csv' not found.")

    # ── Step 2: Shamir secret sharing ────────────────────
    if os.path.exists('retrieved_data_with_combined_medicine.csv'):
        print("\n[2] Generating Shamir secret shares...")
        shamir_encrypt_csv(
            input_csv='retrieved_data_with_combined_medicine.csv',
            output_csv='patient_records_shares.csv'
        )

    # ── Step 3: ElGamal encryption ───────────────────────
    if os.path.exists('retrieved_data_with_combined_medicine.csv'):
        print("\n[3] Encrypting with ElGamal + multiplicative secret sharing...")
        encrypt_csv(
            input_csv='retrieved_data_with_combined_medicine.csv',
            output_csv='patient_records_encrypted_shares.csv',
            column='medicine_details_combined',
            num_shares=3
        )

    # ── Step 4: Doctor authentication & decryption ───────
    if os.path.exists('patient_records_encrypted_shares.csv') and os.path.exists('elgamal_keys.json'):
        print("\n[4] Doctor authentication...")
        doctor_name = input("Enter your name (or press Enter to skip): ").strip()
        if doctor_name:
            register_doctor(doctor_name)
            if authenticate_doctor(doctor_name):
                print(f"✅ Authenticated. Decrypting records for Dr. {doctor_name}...")
                decrypt_for_doctor('patient_records_encrypted_shares.csv', doctor_name)
            else:
                print("❌ Authentication failed.")

    # ── Step 5: Skyline medicine query ───────────────────
    if os.path.exists('retrieved_data_with_combined_medicine.csv'):
        print("\n[5] Skyline medicine query")
        symptom = input("Enter symptom to query (or press Enter to skip): ").strip()
        if symptom:
            for algo in ['bnl', 'dc', 'pareto']:
                print(f"\n--- Algorithm: {algo.upper()} ---")
                run_skyline_query(
                    'retrieved_data_with_combined_medicine.csv',
                    symptom,
                    algorithm=algo
                )

    print("\n✅ All steps complete.")


if __name__ == "__main__":
    main()
