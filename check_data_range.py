"""
Check earliest available data date for key sensors across cities.
Helps determine realistic training window.
"""
import requests, json, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OPENAQ_API_KEY = "97df45e8a9386c64e83d37ce26c23254d42948567f4d48d1ecef928152abea98"
headers = {"X-API-Key": OPENAQ_API_KEY}
BASE = "https://api.openaq.org/v3"

# Key sensors across cities we care about
TEST_SENSORS = {
    # Delhi DPCC
    "delhi_rk_puram":       12234787,
    "delhi_punjabi_bagh":   12234760,   # guessed - try nearby
    "delhi_anand_vihar":    12234791,
    # US Embassy stations (usually best historical data)
    "delhi_us_embassy":     5398,
    "mumbai_us_consulate":  None,  # need to find
    # Kolkata
    "kolkata_victoria":     None,
    # Hyderabad
    "hyderabad_icrisat":    None,
}

# Simpler: just check the 30 most-recently-created sensors from our registry
import os
with open(r"c:\Users\laksh\Desktop\ML_Py\BTP\project3\pm25_india\data\india_stations.json",
          encoding="utf-8") as f:
    stations = json.load(f)

# Get all unique sensor IDs, sorted descending (newest first)
seen_sids = set()
all_sids = []
for s in stations:
    for sid in s.get("sensor_ids", [s.get("sensor_id")] if s.get("sensor_id") else []):
        if sid and sid not in seen_sids:
            seen_sids.add(sid)
            all_sids.append((sid, s["station"], s.get("city", "?")))

# Sort by sensor_id descending (higher = newer)
all_sids.sort(key=lambda x: -x[0])

print(f"Total unique sensor IDs: {len(all_sids)}")
print(f"\nChecking first available date for top sensors (highest IDs = newest):")
print(f"{'SensorID':>12}  {'Station':>40}  {'City':>15}  {'EarliestDate':>22}  {'LatestDate':>22}  {'Hrs':>6}")
print("-" * 130)

found_count = 0
for sid, station, city in all_sids[:60]:  # check top 60 newest sensors
    try:
        # Check data range by fetching oldest and newest available
        r = requests.get(f"{BASE}/sensors/{sid}/hours",
            headers=headers,
            params={"limit": 1, "order_by": "datetime", "sort_order": "asc"},
            timeout=15)
        data = r.json()
        if data.get("meta", {}).get("found", 0) == 0:
            print(f"{sid:>12}  {station:>40}  {city:>15}  {'NO DATA':>22}")
            time.sleep(0.3)
            continue

        first = data["results"][0]
        first_ts = first.get("period", {}).get("datetimeFrom", {}).get("utc", "?")

        # Also get latest
        r2 = requests.get(f"{BASE}/sensors/{sid}/hours",
            headers=headers,
            params={"limit": 1, "order_by": "datetime", "sort_order": "desc"},
            timeout=15)
        data2 = r2.json()
        last_ts = data2["results"][0].get("period", {}).get("datetimeFrom", {}).get("utc", "?") if data2.get("results") else "?"

        # Estimate hours
        n = data.get("meta", {}).get("found", "?")

        print(f"{sid:>12}  {station[:40]:>40}  {city[:15]:>15}  {first_ts:>22}  {last_ts:>22}  {str(n):>6}")
        found_count += 1
        time.sleep(0.5)
    except Exception as e:
        print(f"{sid:>12}  ERROR: {e}")
        time.sleep(1)

print(f"\nSensors with data: {found_count}")
