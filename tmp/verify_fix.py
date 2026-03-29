import requests
import json

BASE_URL = "http://localhost:5000"

def test_api():
    session = requests.Session()
    
    # 1. Login
    print("--- Testing Login ---")
    login_resp = session.post(f"{BASE_URL}/login", json={"username": "admin", "password": "admin123"})
    print(f"Login Status: {login_resp.status_code}")
    
    # 2. Test Get Patients (Verify Fallback)
    print("\n--- Testing Get Patients ---")
    patients_resp = session.get(f"{BASE_URL}/api/patients")
    print(f"Status: {patients_resp.status_code}")
    print(f"Content: {patients_resp.text[:500]}")
    try:
        patients = patients_resp.json()
        print(f"Found {len(patients.get('patients', []))} patients.")
        if patients.get('patients'):
            print(f"First patient: {patients['patients'][0]['name']} ({patients['patients'][0]['patient_id']})")
    except Exception as e:
        print(f"Error parsing patients: {e}")

    # 3. Test Submit Vitals
    print("\n--- Testing Submit Vitals ---")
    vitals_data = {
        "patient_id": "PATIENT_001",
        "spo2": 98.0,
        "systolic": 120.0,
        "diastolic": 80.0,
        "heart_rate": 72.0,
        "temperature": 36.5
    }
    submit_resp = session.post(f"{BASE_URL}/api/vitals/submit", json=vitals_data)
    print(f"Status: {submit_resp.status_code}")
    print(f"Content: {submit_resp.text[:500]}")
    try:
        result = submit_resp.json()
        print(f"Result: {result.get('success', False)}")
        if not result.get('success'):
            print(f"Error: {result.get('error')}")
    except Exception as e:
        print(f"Error parsing submit response: {e}")

if __name__ == "__main__":
    test_api()
