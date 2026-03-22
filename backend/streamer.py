"""
Parquet Data Streamer
Streams raw parquet rows with varied, unique alerts based on actual data.
"""

import pandas as pd
import requests
import time
import random
from datetime import datetime

BACKEND_URL = "http://localhost:5000"
PARQUET_FILES = ["data/user_features_research.parquet", "data/ldap_features.parquet", "data/features_only.parquet"]
STREAM_INTERVAL = 10.0

# Varied alert templates based on actual behavior patterns
ALERT_TEMPLATES = {
    'after_hours': [
        "Unusual after-hours activity detected",
        "Login activity outside business hours",
        "Late night system access flagged",
        "Off-hours access pattern identified",
        "Suspicious after-hours session",
    ],
    'high_volume': [
        "High volume of file transfers",
        "Unusual data access frequency",
        "Large number of file operations",
        "Excessive resource utilization",
        "Data movement spike detected",
    ],
    'security': [
        "Security role privilege escalation",
        "Administrative access anomaly",
        "Elevated permissions usage",
        "Critical system access detected",
        "Privileged account activity",
    ],
    'device': [
        "Removable device connection",
        "USB device activity detected",
        "External storage access",
        "Device event spike",
        "Hardware peripheral anomaly",
    ],
    'email': [
        "Suspicious email pattern",
        "Unusual email recipient",
        "Attachment anomaly detected",
        "High email volume flagged",
        "External communication alert",
    ],
    'http': [
        "Unusual web browsing pattern",
        "Access to sensitive domains",
        "Cloud service activity",
        "Job site access detected",
        "Social media activity",
    ],
    'data_exfil': [
        "Potential data exfiltration",
        "Large data transfer flagged",
        "Sensitive file access",
        "Bulk download detected",
        "Data extraction pattern",
    ],
    'sequence': [
        "Sequence anomaly detected",
        "Behavioral pattern break",
        "Unusual activity sequence",
        "Temporal pattern deviation",
        "Anomalous user behavior",
    ],
}

ANOMALY_INDICATORS = {
    'after_hours': ['After-hours activity', 'Off-hours access', 'Late session', 'Night access'],
    'high_logons': ['Multiple logins', 'Login spike', 'Session flood', 'Auth anomaly'],
    'device': ['Device event', 'USB activity', 'Storage access', 'Hardware change'],
    'email': ['Email anomaly', 'Attachment spike', 'External send', 'Mail flood'],
    'http': ['Web anomaly', 'Domain access', 'Cloud activity', 'URL pattern'],
    'file': ['File operation', 'Data access', 'Bulk file', 'Transfer spike'],
    'security': ['Security event', 'Privilege use', 'Admin action', 'Access anomaly'],
}

def load_parquet_files():
    dfs = []
    for file in PARQUET_FILES:
        try:
            df = pd.read_parquet(file)
            df['_source_file'] = file.split('/')[-1]
            dfs.append(df)
            print(f"Loaded {file}: {len(df)} rows")
        except Exception as e:
            print(f"Could not load {file}: {e}")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None

def analyze_row_for_alerts(row):
    """Analyze row data and return relevant alert type and indicators"""
    alerts = []
    indicators = []
    
    # Check after hours ratio
    after_hours_ratio = row.get('after_hours_ratio_24h', 0) or 0
    if after_hours_ratio > 0.3:
        alerts.append(random.choice(ALERT_TEMPLATES['after_hours']))
        indicators.extend(random.sample(ANOMALY_INDICATORS['after_hours'], min(2, len(ANOMALY_INDICATORS['after_hours']))))
    
    # Check logon count
    logon_count = row.get('logon_count_24h', 0) or 0
    if logon_count > 10:
        alerts.append(random.choice(ALERT_TEMPLATES['high_volume']))
        indicators.extend(random.sample(ANOMALY_INDICATORS['high_logons'], 1))
    
    # Check device events
    device_events = row.get('device_events_24h', 0) or 0
    if device_events > 2:
        alerts.append(random.choice(ALERT_TEMPLATES['device']))
        indicators.extend(random.sample(ANOMALY_INDICATORS['device'], 1))
    
    # Check emails
    emails = row.get('emails_sent_24h', 0) or 0
    if emails > 15:
        alerts.append(random.choice(ALERT_TEMPLATES['email']))
        indicators.extend(random.sample(ANOMALY_INDICATORS['email'], 1))
    
    # Check HTTP
    http = row.get('http_requests_24h', 0) or 0
    if http > 100:
        alerts.append(random.choice(ALERT_TEMPLATES['http']))
        indicators.extend(random.sample(ANOMALY_INDICATORS['http'], 1))
    
    # Check security role
    is_security = row.get('is_security_role_24h', False)
    if is_security:
        alerts.append(random.choice(ALERT_TEMPLATES['security']))
        indicators.extend(random.sample(ANOMALY_INDICATORS['security'], 1))
    
    # Check high after hours flag
    high_after = row.get('high_after_hours', 0) or 0
    if high_after == 1:
        alerts.append(random.choice(ALERT_TEMPLATES['sequence']))
        indicators.append('High after-hours indicator')
    
    return alerts, list(set(indicators))

def format_event(row, idx):
    user_hash = row.get('user_hash', f'USR{idx}')
    date = row.get('date', datetime.now().date())
    source_file = row.get('_source_file', 'unknown')
    
    # Determine base log type and action from data
    logon_count = row.get('logon_count_24h', 0) or 0
    device_events = row.get('device_events_24h', 0) or 0
    emails = row.get('emails_sent_24h', 0) or 0
    http = row.get('http_requests_24h', 0) or 0
    
    if logon_count > 0:
        log_type = 'logon'
        action = f"User activity: {int(logon_count)} logons"
    elif device_events > 0:
        log_type = 'device'
        action = f"Device events: {int(device_events)}"
    elif emails > 0:
        log_type = 'email'
        action = f"Emails: {int(emails)} sent"
    elif http > 0:
        log_type = 'http'
        action = f"HTTP requests: {int(http)}"
    else:
        log_type = 'process'
        action = f"Activity recorded from {source_file}"
    
    # Get unique alerts based on actual data
    alerts, indicators = analyze_row_for_alerts(row)
    primary_alert = alerts[0] if alerts else "Normal behavior pattern"
    
    # Raw features for model
    raw_features = {}
    for key, value in row.items():
        if pd.notna(value):
            if isinstance(value, (int, float, bool)):
                raw_features[key] = float(value)
            else:
                raw_features[key] = str(value)
    
    return {
        "id": f"LOG-{date}-{idx}",
        "timestamp": datetime.now().isoformat(),
        "logType": log_type,
        "userId": str(user_hash)[:12],
        "userName": f"User_{str(user_hash)[:6]}",
        "department": f"Dept_{int(row.get('department_id_24h', 0))}",
        "action": primary_alert,  # Use actual alert from data
        "details": f"{action} | {source_file}",
        "sourceIp": f"192.168.{int(row.get('department_id_24h', 1))}.{idx % 255}",
        "resource": f"/data/{source_file}",
        "sessionId": f"SES-{int(hash(str(user_hash)) % 1000000)}",
        "rawFeatures": raw_features,
        "sourceFile": source_file,
        "anomalyFactors": indicators,
        "alerts": alerts,
    }

def stream_events(df):
    total_rows = len(df)
    print(f"\nStreaming {total_rows} rows with varied alerts...")
    print(f"Backend URL: {BACKEND_URL}")
    print("-" * 60)
    
    idx = 0
    row_idx = 0
    
    while True:
        if row_idx >= total_rows:
            row_idx = 0
            print("Looping back...")
        
        row = df.iloc[row_idx]
        event = format_event(row, idx)
        
        try:
            response = requests.post(
                f"{BACKEND_URL}/api/stream/event",
                json=event,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                risk = result.get('event', {}).get('riskScore', 0)
                threat = result.get('event', {}).get('threatLevel', 'normal')
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {event['userId']} | "
                      f"Alert: {event['action'][:50]} | Risk: {risk}%")
            else:
                print(f"Error: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"Backend not available, waiting...")
            time.sleep(5)
        except Exception as e:
            print(f"Error: {e}")
        
        idx += 1
        row_idx += 1
        time.sleep(STREAM_INTERVAL)

def main():
    print("=" * 60)
    print("SENTINEL - Varied Alert Streamer")
    print("=" * 60)
    
    df = load_parquet_files()
    
    if df is None or len(df) == 0:
        print("ERROR: No data loaded!")
        return
    
    print(f"\nTotal rows: {len(df)}")
    print("\nStarting stream with unique alerts based on data...")
    
    stream_events(df)

if __name__ == "__main__":
    main()
