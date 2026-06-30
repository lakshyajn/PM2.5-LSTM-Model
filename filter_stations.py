"""
filter_stations.py
──────────────────
Pre-screen the station registry to identify which stations actually have
/hours data. Makes a quick 1-request probe per station and marks them.
Only keeps stations with confirmed data, saving time in the full build.

Run once:
  python filter_stations.py --cities Delhi Mumbai Kolkata Chennai Bengaluru Hyderabad Pune Lucknow Patna Jaipur
"""
import os, sys, json, time, argparse
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None

from config import STATIONS_JSON, OPENAQ_BASE, OPENAQ_HEADERS, DATA_DIR

SCREENED_JSON = os.path.join(DATA_DIR, "screened_stations.json")


def _get_sensor_data_range(sensor_id: int, retries: int = 2) -> tuple:
    """
    Probe /sensors/{id}/hours to check if data exists AND find the earliest date.
    Returns (has_data: bool, earliest_date: str or None, latest_date: str or None).
    """
    for attempt in range(retries):
        try:
            # Check earliest record
            r_asc = requests.get(
                f"{OPENAQ_BASE}/sensors/{sensor_id}/hours",
                headers=OPENAQ_HEADERS,
                params={"limit": 1, "order_by": "datetime", "sort_order": "asc"},
                timeout=15,
            )
            if r_asc.status_code == 429:
                wait = int(r_asc.headers.get("Retry-After", 60))
                print(f"  [RATE LIMIT] sleeping {wait}s ...")
                time.sleep(wait)
                continue
            if r_asc.status_code != 200:
                return False, None, None
            results_asc = r_asc.json().get("results", [])
            if not results_asc:
                return False, None, None

            earliest = results_asc[0].get("period", {}).get("datetimeFrom", {}).get("utc", None)

            # Check latest record
            r_desc = requests.get(
                f"{OPENAQ_BASE}/sensors/{sensor_id}/hours",
                headers=OPENAQ_HEADERS,
                params={"limit": 1, "order_by": "datetime", "sort_order": "desc"},
                timeout=15,
            )
            results_desc = r_desc.json().get("results", []) if r_desc.status_code == 200 else []
            latest = results_desc[0].get("period", {}).get("datetimeFrom", {}).get("utc", None) if results_desc else None

            return True, earliest, latest
        except Exception:
            time.sleep(2)
    return False, None, None


def screen_stations(city_filter=None):
    with open(STATIONS_JSON, encoding="utf-8") as f:
        stations = json.load(f)

    if city_filter:
        city_set = {c.lower() for c in city_filter}
        stations = [s for s in stations if s.get("city", "").lower() in city_set]

    print(f"Screening {len(stations)} stations for /hours data availability ...")
    print(f"{'#':>4}  {'Station':>45}  {'City':>15}  {'SensorID':>10}  Status")
    print("-" * 100)

    active = []
    pre2025_count = 0
    for i, stn in enumerate(stations, 1):
        name   = stn["station"]
        city   = stn.get("city", "?")
        sids   = stn.get("sensor_ids", [stn.get("sensor_id")] if stn.get("sensor_id") else [])

        has_data = False
        working_sid = None
        earliest_date = None
        latest_date   = None

        for sid in sids:
            if sid is None:
                continue
            found, earliest, latest = _get_sensor_data_range(sid)
            if found:
                has_data      = True
                working_sid   = sid
                earliest_date = earliest
                latest_date   = latest
                break
            time.sleep(0.5)

        if has_data:
            # Determine if station has pre-2025 data
            pre2025 = ""
            if earliest_date and earliest_date[:4] < "2025":
                pre2025 = f"  [PRE-2025: {earliest_date[:10]}]"
                pre2025_count += 1
            status = f"OK (sid={working_sid}) {earliest_date[:10] if earliest_date else '?'} -> {latest_date[:10] if latest_date else '?'}{pre2025}"
        else:
            status = "NO DATA"

        print(f"{i:>4}  {name[:45]:>45}  {city[:15]:>15}  {str(sids[0] if sids else '?'):>10}  {status}")

        if has_data:
            stn_copy = dict(stn)
            stn_copy["sensor_id"]   = working_sid
            stn_copy["sensor_ids"]  = [working_sid]
            stn_copy["data_start"]  = earliest_date  # store for build date range
            stn_copy["data_end"]    = latest_date
            active.append(stn_copy)

        time.sleep(0.4)


    print(f"\n{'='*60}")
    print(f"Active stations: {len(active)} / {len(stations)}")
    print(f"Stations with pre-2025 data: {pre2025_count}")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SCREENED_JSON, "w", encoding="utf-8") as f:
        json.dump(active, f, indent=2, ensure_ascii=False)
    print(f"Saved -> {SCREENED_JSON}")

    # City breakdown with data range
    print("\nActive stations per city (earliest data -> latest):")
    city_data = {}
    for s in active:
        c = s.get("city", "Unknown")
        ds = s.get("data_start", "?")
        de = s.get("data_end", "?")
        if c not in city_data:
            city_data[c] = {"count": 0, "earliest": ds, "latest": de}
        city_data[c]["count"] += 1
        if ds and ds < (city_data[c]["earliest"] or "9999"):
            city_data[c]["earliest"] = ds
        if de and de > (city_data[c]["latest"] or "0000"):
            city_data[c]["latest"] = de

    for c, info in sorted(city_data.items(), key=lambda x: -x[1]["count"]):
        earliest = info["earliest"][:10] if info["earliest"] else "?"
        latest   = info["latest"][:10] if info["latest"] else "?"
        pre = " [PRE-2025]" if earliest < "2025" else ""
        print(f"  {c:<25} {info['count']:>3} stations  {earliest} -> {latest}{pre}")

    return active



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", nargs="+", default=None)
    args = parser.parse_args()
    screen_stations(city_filter=args.cities)
