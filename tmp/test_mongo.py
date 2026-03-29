import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import pymongo
import dns.resolver

load_dotenv()

MONGO_URI = os.environ.get('MONGO_URI')
DB_NAME = os.environ.get('MONGO_DB', 'cardiac_monitor')

print(f"Testing connection to: {MONGO_URI}")

try:
    # 1. Test DNS resolution (SRV check)
    cluster = MONGO_URI.replace('mongodb+srv://', '').split('/', 1)[0].split('@')[-1]
    print(f"Resolving SRV for: {cluster}")
    answers = dns.resolver.resolve(f'_mongodb._tcp.{cluster}', 'SRV')
    for rdata in answers:
        print(f"Found shard: {rdata.target}:{rdata.port}")

    # 2. Connect
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    
    print(f"Collections: {db.list_collection_names()}")
    
    patients = db['patients'].count_documents({})
    print(f"Patient count: {patients}")
    
    analyses = db['ecg_analyses'].count_documents({})
    print(f"Analysis count: {analyses}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
