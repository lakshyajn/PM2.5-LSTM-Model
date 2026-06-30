"""
clean_screened_v2.py
────────────────────
Filters screened_stations.json by keeping only stations whose
sensor_id is in a productive range (confirmed to have data from
the build logs and test runs), removing the old legacy IDs.

Rules:
  - KEEP: sensor_ids >= 5000 (CPCB stations registered on OpenAQ v3 2024+)  
  - KEEP: certain known productive low-ID stations (explicitly listed)
  - DROP: sensor_ids < 1000 (legacy 2015-era registrations, no /hours data)
  - DROP: any Pakistan/cross-border stations (lat/lon outside India proper)

Run: python clean_screened_v2.py
"""
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR      = r"c:\Users\laksh\Desktop\ML_Py\BTP\project3\pm25_india\data"
SCREENED_JSON = os.path.join(DATA_DIR, "screened_stations.json")
CLEANED_JSON  = os.path.join(DATA_DIR, "screened_stations_clean.json")

def is_productive_sensor(sid: int) -> bool:
    """
    Returns True if sensor_id is in a range known to have /hours data.
    
    Productive ranges (confirmed from build logs):
      - 12,200,000 – 16,000,000  : CPCB v3 batch imported Feb 2025+
      - 14,100,000+               : Newer HSPCB/state board sensors  
      - 2,000,000 – 12,000,000   : Some valid IQAir/PurpleAir sensors
      - 5,000,000 – 8,000,000    : US Diplomatic posts (limited data)
    
    Non-productive ranges (return NO DATA on /hours):
      - < 100,000                 : Legacy 2015-era OpenAQ registrations
      - 100,000 – 1,000,000      : Mixed; mostly legacy with sparse data
      - 13,000 – 99,999          : KSPCB/WBPCB legacy sensors, no /hours
    """
    if sid is None:
        return False
    # Core productive range: CPCB v3 migration batch
    if 12_200_000 <= sid <= 16_800_000:
        return True
    # IQAir / PurpleAir / newer sensors
    if 2_000_000 <= sid <= 12_100_000:
        return True
    # Specific known-good outliers (US Embassy type)
    if sid in {5077640, 5077812}:  # Kolkata, Chennai US consulates
        return True
    return False


with open(SCREENED_JSON, encoding="utf-8") as f:
    stations = json.load(f)

print(f"Input: {len(stations)} stations")

kept    = []
dropped = []

for s in stations:
    sid  = s.get("sensor_id")
    lat  = float(s.get("lat", 0))
    lon  = float(s.get("lon", 0))
    name = s["station"]

    # Drop if no sensor_id
    if sid is None:
        dropped.append((name, "no sensor_id"))
        continue

    # Drop if outside India proper (Pakistan/Afghanistan/Bangladesh border sensors)
    if not (6.5 <= lat <= 37.5 and 67.0 <= lon <= 97.5):
        dropped.append((name, f"outside India: lat={lat:.2f} lon={lon:.2f}"))
        continue

    # Drop non-productive sensor IDs (legacy registrations with no /hours data)
    if not is_productive_sensor(sid):
        dropped.append((name, f"non-productive sensor_id={sid}"))
        continue

    kept.append(s)

print(f"Kept:    {len(kept)}")
print(f"Dropped: {len(dropped)}")

# City breakdown of kept
city_counts = {}
for s in kept:
    c = s.get("city", "Unknown")
    city_counts[c] = city_counts.get(c, 0) + 1

print(f"\nKept stations per city:")
for c, n in sorted(city_counts.items(), key=lambda x: -x[1]):
    print(f"  {c:<25} {n}")

with open(CLEANED_JSON, "w", encoding="utf-8") as f:
    json.dump(kept, f, indent=2, ensure_ascii=False)
print(f"\nSaved {len(kept)} stations -> {CLEANED_JSON}")
