import random
import string
import time
import base64
from hashlib import sha256
from hmac import HMAC
import argparse

import requests
import pandas as pd

APPID = "tubmafwrzhpgfiuf"
SECRET = "eotpcqbvhycdshwscqnytiwzbgonposs"

def generate_nonce(length=32):
    """Generate a random nonce."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def get_timestamp():
    """Get the current timestamp."""
    return int(time.time())

def build_sign_str(appid, nonce, timestamp):
    """Build the string to be signed."""
    return f'appid={appid}&nonce={nonce}&timestamp={timestamp}'

def calculate_signature(secret, message):
    """Calculate the HMAC SHA-256 signature."""
    return base64.urlsafe_b64encode(
        HMAC(secret.encode('utf-8'), message.encode('utf-8'), sha256).digest()
    ).decode('utf-8')

def fetch_indicator_details(indicator_id):
    nonce = generate_nonce()
    timestamp = get_timestamp()
    sign_str = build_sign_str(APPID, nonce, timestamp)
    signature = calculate_signature(SECRET, sign_str)

    headers = {
        'nonce': nonce,
        'timestamp': str(timestamp),
        'appid': APPID,
        'signature': signature,
        'Accept': "*/*",
        'Accept-Encoding': "gzip, deflate, br",
        'User-Agent': "PostmanRuntime-ApipostRuntime/1.1.0",
        'Connection': "keep-alive",
    }
    url = f"https://etahub.hzinsights.com/v1/edb/data?EdbCode={indicator_id}"
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        return resp.json().get('Data')
    else:
        print(f"[ERROR] Fetch {indicator_id} → {resp.status_code}")
        return None

def fetch_indicator_name(indicator_id):
    nonce = generate_nonce()
    timestamp = get_timestamp()
    sign_str = build_sign_str(APPID, nonce, timestamp)
    signature = calculate_signature(SECRET, sign_str)

    headers = {
        'nonce': nonce,
        'timestamp': str(timestamp),
        'appid': APPID,
        'signature': signature,
        'Accept': "*/*",
        'Accept-Encoding': "gzip, deflate, br",
        'User-Agent': "PostmanRuntime-ApipostRuntime/1.1.0",
        'Connection': "keep-alive",
    }
    url = f"https://etahub.hzinsights.com/v1/edb/detail?EdbCode={indicator_id}"
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        return resp.json().get('Data', {}).get('EdbName')
    else:
        print(f"[ERROR] Fetch name {indicator_id} → {resp.status_code}")
        return indicator_id  # 若失败，返回 ID 以免报错

def main(indicator_ids, output_path):
    data_frames = {}
    for eid in indicator_ids:
        data = fetch_indicator_details(eid)
        if not data:
            continue
        df = pd.DataFrame(data)
        df['DataTime'] = pd.to_datetime(df['DataTime'])
        df.set_index('DataTime', inplace=True)
        df.sort_index(inplace=True)

        name = fetch_indicator_name(eid)
        df = df[['Value']].rename(columns={'Value': name})
        data_frames[eid] = df

    if not data_frames:
        print("No data fetched, exiting.")
        return

    result_df = pd.concat(data_frames.values(), axis=1)
    print(result_df.info())
    result_df.to_excel(output_path)
    print(f"Data saved successfully → {output_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        '--ids', '-i',
        nargs='+',
        required=True
    )
    p.add_argument(
        '--output', '-o',
        required=True
    )
    args = p.parse_args()
    main(args.ids, args.output)
