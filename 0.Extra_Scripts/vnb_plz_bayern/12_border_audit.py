import csv, json, os, shapefile

HERE = os.path.dirname(os.path.abspath(__file__))
IN = "/home/pedro/Linux-AntigravityProjects/SurroGrid-linux/GridExpand/1.grid_sampling/gridreadout/input_data"

openplz = json.load(open(os.path.join(HERE, "data", "openplz_bayern.json"), encoding="utf-8"))
mine = {r["plz"]: r for r in csv.DictReader(open(os.path.join(HERE, "plz_vnb_bayern.csv"), encoding="utf-8-sig"), delimiter=";")}
overlap = {r["plz"]: (float(r["anteil_in_bayern"]), float(r["flaeche_km2"]))
           for r in csv.DictReader(open(os.path.join(HERE, "data", "plz_bayern_overlap.tsv")), delimiter="\t")}
counts = {}
for r in csv.DictReader(open(os.path.join(HERE, "data", "plz_nb_counts.tsv"), encoding="utf-8"), delimiter="\t"):
    counts[r["plz"]] = counts.get(r["plz"], 0) + int(r["n_lokationen"])

p = shapefile.Reader(f"{IN}/plz-5stellig.shp", encoding="utf-8")
shp_notes = {str(sr.record[0]).zfill(5): (sr.record[1], sr.record[2], sr.record[3]) for sr in p.iterShapeRecords()}

print("1. OpenPLZ Bayern total unique PLZ:", len(openplz))
print("2. plz_vnb_bayern.csv total unique PLZ:", len(mine))
print("3. Missing from plz_vnb_bayern.csv vs OpenPLZ:", set(openplz) - set(mine))
print("4. Extra in plz_vnb_bayern.csv vs OpenPLZ:", set(mine) - set(openplz))
for k in sorted(set(mine) - set(openplz)):
    print("   Extra detail:", k, shp_notes.get(k), overlap.get(k), mine[k])

print("\n5. Border PLZ in plz_bayern_overlap.tsv with 0.01 <= anteil_in_bayern < 0.5:")
for k, (fr, area) in sorted(overlap.items(), key=lambda x: -x[1][0]):
    if 0.01 <= fr < 0.5:
        print(f"   PLZ {k}: {fr*100:5.1f}% in BY ({area:6.1f} km2) | in table={k in mine} | in OpenPLZ_BY={k in openplz} | MaStR_BY_units={counts.get(k, 0)} | note={shp_notes.get(k)}")

