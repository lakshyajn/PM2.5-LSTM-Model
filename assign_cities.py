"""
assign_cities.py
───────────────
Post-processes india_stations.json to assign city/state labels
based on nearest major city using haversine distance.
Run once after discovery.
"""
import json, sys
import numpy as np
sys.path.insert(0, '.')
from config import STATIONS_JSON, MAJOR_CITIES_FALLBACK

# Build reference city list from config
CITY_REFS = []
for key, info in MAJOR_CITIES_FALLBACK.items():
    CITY_REFS.append({
        "city":  info["city"],
        "state": info["state"],
        "lat":   info["lat"],
        "lon":   info["lon"],
    })

# Additional cities not in fallback
EXTRA_CITIES = [
    {"city": "Delhi",       "state": "Delhi",         "lat": 28.613,  "lon": 77.209},
    {"city": "Mumbai",      "state": "Maharashtra",   "lat": 19.076,  "lon": 72.878},
    {"city": "Kolkata",     "state": "West Bengal",   "lat": 22.572,  "lon": 88.364},
    {"city": "Chennai",     "state": "Tamil Nadu",    "lat": 13.083,  "lon": 80.270},
    {"city": "Bengaluru",   "state": "Karnataka",     "lat": 12.972,  "lon": 77.594},
    {"city": "Hyderabad",   "state": "Telangana",     "lat": 17.385,  "lon": 78.487},
    {"city": "Ahmedabad",   "state": "Gujarat",       "lat": 23.023,  "lon": 72.572},
    {"city": "Lucknow",     "state": "Uttar Pradesh", "lat": 26.847,  "lon": 80.947},
    {"city": "Kanpur",      "state": "Uttar Pradesh", "lat": 26.449,  "lon": 80.331},
    {"city": "Patna",       "state": "Bihar",         "lat": 25.594,  "lon": 85.137},
    {"city": "Jaipur",      "state": "Rajasthan",     "lat": 26.913,  "lon": 75.787},
    {"city": "Amritsar",    "state": "Punjab",        "lat": 31.634,  "lon": 74.872},
    {"city": "Ludhiana",    "state": "Punjab",        "lat": 30.901,  "lon": 75.857},
    {"city": "Chandigarh",  "state": "Chandigarh",    "lat": 30.733,  "lon": 76.779},
    {"city": "Gurugram",    "state": "Haryana",       "lat": 28.459,  "lon": 77.029},
    {"city": "Faridabad",   "state": "Haryana",       "lat": 28.408,  "lon": 77.317},
    {"city": "Bhopal",      "state": "Madhya Pradesh","lat": 23.259,  "lon": 77.412},
    {"city": "Indore",      "state": "Madhya Pradesh","lat": 22.719,  "lon": 75.857},
    {"city": "Pune",        "state": "Maharashtra",   "lat": 18.521,  "lon": 73.857},
    {"city": "Nagpur",      "state": "Maharashtra",   "lat": 21.146,  "lon": 79.089},
    {"city": "Bhubaneswar", "state": "Odisha",        "lat": 20.296,  "lon": 85.825},
    {"city": "Raipur",      "state": "Chhattisgarh",  "lat": 21.251,  "lon": 81.630},
    {"city": "Ranchi",      "state": "Jharkhand",     "lat": 23.344,  "lon": 85.309},
    {"city": "Dhanbad",     "state": "Jharkhand",     "lat": 23.799,  "lon": 86.433},
    {"city": "Visakhapatnam","state":"Andhra Pradesh", "lat": 17.686,  "lon": 83.218},
    {"city": "Vijayawada",  "state": "Andhra Pradesh","lat": 16.506,  "lon": 80.648},
    {"city": "Thiruvananthapuram","state":"Kerala",   "lat": 8.526,   "lon": 76.933},
    {"city": "Kochi",       "state": "Kerala",        "lat": 9.931,   "lon": 76.268},
    {"city": "Kozhikode",   "state": "Kerala",        "lat": 11.259,  "lon": 75.781},
    {"city": "Guwahati",    "state": "Assam",         "lat": 26.144,  "lon": 91.736},
    {"city": "Dehradun",    "state": "Uttarakhand",   "lat": 30.316,  "lon": 78.032},
    {"city": "Varanasi",    "state": "Uttar Pradesh", "lat": 25.317,  "lon": 82.974},
    {"city": "Agra",        "state": "Uttar Pradesh", "lat": 27.176,  "lon": 78.008},
    {"city": "Ghaziabad",   "state": "Uttar Pradesh", "lat": 28.670,  "lon": 77.432},
    {"city": "Noida",       "state": "Uttar Pradesh", "lat": 28.535,  "lon": 77.391},
    {"city": "Meerut",      "state": "Uttar Pradesh", "lat": 28.989,  "lon": 77.709},
    {"city": "Jodhpur",     "state": "Rajasthan",     "lat": 26.295,  "lon": 73.024},
    {"city": "Udaipur",     "state": "Rajasthan",     "lat": 24.585,  "lon": 73.712},
    {"city": "Surat",       "state": "Gujarat",       "lat": 21.170,  "lon": 72.831},
    {"city": "Vadodara",    "state": "Gujarat",       "lat": 22.308,  "lon": 73.182},
    {"city": "Rajkot",      "state": "Gujarat",       "lat": 22.303,  "lon": 70.802},
    {"city": "Coimbatore",  "state": "Tamil Nadu",    "lat": 11.017,  "lon": 76.955},
    {"city": "Madurai",     "state": "Tamil Nadu",    "lat": 9.919,   "lon": 78.119},
    {"city": "Mangaluru",   "state": "Karnataka",     "lat": 12.914,  "lon": 74.856},
    {"city": "Mysuru",      "state": "Karnataka",     "lat": 12.295,  "lon": 76.639},
    {"city": "Hubli",       "state": "Karnataka",     "lat": 15.364,  "lon": 75.124},
    {"city": "Shimla",      "state": "Himachal Pradesh","lat": 31.104, "lon": 77.173},
    {"city": "Srinagar",    "state": "J&K",           "lat": 34.073,  "lon": 74.797},
    {"city": "Jammu",       "state": "J&K",           "lat": 32.727,  "lon": 74.858},
    {"city": "Panaji",      "state": "Goa",           "lat": 15.499,  "lon": 73.826},
    {"city": "Imphal",      "state": "Manipur",       "lat": 24.819,  "lon": 93.937},
    {"city": "Shillong",    "state": "Meghalaya",     "lat": 25.578,  "lon": 91.883},
    {"city": "Agartala",    "state": "Tripura",       "lat": 23.831,  "lon": 91.286},
]

ALL_CITY_REFS = EXTRA_CITIES  # use the comprehensive list

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def nearest_city(lat, lon, max_dist_km=80):
    best_city  = "Unknown"
    best_state = "Unknown"
    best_dist  = max_dist_km + 1

    for ref in ALL_CITY_REFS:
        d = haversine_km(lat, lon, ref["lat"], ref["lon"])
        if d < best_dist:
            best_dist  = d
            best_city  = ref["city"]
            best_state = ref["state"]

    return best_city, best_state

def main():
    print(f"Loading {STATIONS_JSON} ...")
    with open(STATIONS_JSON, encoding="utf-8") as f:
        stations = json.load(f)

    print(f"  {len(stations)} stations — assigning city labels ...")
    assigned = 0
    city_counts = {}

    for s in stations:
        lat = s.get("lat")
        lon = s.get("lon")
        if lat is None or lon is None:
            continue
        city, state = nearest_city(lat, lon)
        s["city"]  = city
        s["state"] = state
        if city != "Unknown":
            assigned += 1
            city_counts[city] = city_counts.get(city, 0) + 1

    # Save
    with open(STATIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(stations, f, indent=2, ensure_ascii=False)

    print(f"  Assigned city to {assigned}/{len(stations)} stations")
    print(f"  Unique cities covered: {len(city_counts)}")
    print(f"\n  Top 30 cities by station count:")
    for city, n in sorted(city_counts.items(), key=lambda x: -x[1])[:30]:
        print(f"    {city:<25} {n:>4} stations")

if __name__ == "__main__":
    main()
