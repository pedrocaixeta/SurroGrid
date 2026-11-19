"""Vollstaendigkeitspruefung: Vergleich mit OpenPLZ API (Bayern, federalState 09)."""
import csv, json, os, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ref = {}
page = 1
while True:
    req = urllib.request.Request(f"https://openplzapi.org/de/FederalStates/09/Localities?page={page}&pageSize=50",
                                 headers={"Accept": "application/json"})
    r = urllib.request.urlopen(req, timeout=60)
    data = json.load(r)
    for d in data:
        ref.setdefault(d["postalCode"], set()).add(d["name"])
    tp = int(r.headers["x-total-pages"])
    if page >= tp:
        break
    page += 1
print("OpenPLZ Bayern: PLZ", len(ref))
json.dump({k: sorted(v) for k, v in ref.items()}, open(os.path.join(HERE, "data", "openplz_bayern.json"), "w"), ensure_ascii=False)

mine = {r["plz"] for r in csv.DictReader(open(os.path.join(HERE, "plz_vnb_bayern.csv"), encoding="utf-8-sig"), delimiter=";")}
print("Meine Tabelle: PLZ", len(mine))
miss = sorted(set(ref) - mine)
extra = sorted(mine - set(ref))
print("In OpenPLZ aber NICHT in meiner Tabelle:", len(miss))
for p in miss:
    print("  ", p, sorted(ref[p]))
print("In meiner Tabelle aber NICHT in OpenPLZ-Bayern:", len(extra))
print("  ", extra)
with open(os.path.join(HERE, "data", "completeness_missing.txt"), "w") as f:
    f.write("\n".join(miss))

