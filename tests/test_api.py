# test_api.py
import requests

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.bilibili.com",
    "Accept": "application/json, text/plain, */*",
}

try:
    session = requests.Session()
    session.get("https://www.bilibili.com", headers=headers, timeout=10)
    
    response = session.get(
        "https://api.bilibili.com/x/web-interface/popular",
        params={"ps": 20, "pn": 1},
        headers=headers,
        timeout=30
    )
    response.raise_for_status()
    data = response.json()
    if data.get("code") != 0:
        raise RuntimeError(f"B站API错误: {data.get('message')} (code: {data.get('code')})")
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()