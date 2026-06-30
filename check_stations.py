import json, sys
sys.path.insert(0,'.')
with open('data/india_stations.json', encoding='utf-8') as f:
    stns = json.load(f)

cities = {}
for s in stns:
    c = s.get('city','Unknown')
    cities[c] = cities.get(c,0)+1

target = ['Delhi','Mumbai','Kolkata','Chennai','Bengaluru','Hyderabad',
          'Ahmedabad','Lucknow','Patna','Jaipur','Amritsar','Chandigarh',
          'Bhopal','Pune','Nagpur']

print(f'Total stations in registry: {len(stns)}')
print(f'Unique cities: {len(cities)}')
print()
print(f"{'City':<22} {'Stations':>8}")
print('-'*32)
total_filtered = 0
for city in target:
    n = cities.get(city, 0)
    total_filtered += n
    note = '  OK' if n > 0 else '  MISSING - fallback'
    print(f'  {city:<20} {n:>6}{note}')
print(f"{'TOTAL filtered':<22} {total_filtered:>8}")
print()
print('Top 20 cities by station count:')
for c,n in sorted(cities.items(), key=lambda x:-x[1])[:20]:
    print(f'  {c:<30} {n}')
