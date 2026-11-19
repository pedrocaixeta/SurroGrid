import collections, csv
rows = list(csv.DictReader(open("data/plz_nb_counts.tsv", encoding="utf-8"), delimiter="\t"))
print("Zeilen", len(rows), "PLZ", len({r["plz"] for r in rows}), "Lok", sum(int(r["n_lokationen"]) for r in rows))
agg = collections.defaultdict(lambda: [0, set(), set()])
for r in rows:
    k = (r["nb_mastr"], r["nb_name"])
    agg[k][0] += int(r["n_lokationen"])
    agg[k][1].add((r["netz_bezeichnung"], r["netz_sparte"], r["netz_kunden_ge_100k"], r["netz_geschlossenes_vn"]))
    agg[k][2].add(r["plz"])
print("\nNB mit mind. einem Netz mit Flag KundenAngeschlossen=1:")
for k, (n, nets, plzs) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
    if any(x[2] == "1" for x in nets):
        print(f"{n:8d} PLZ={len(plzs):4d} {k[1]} | {sorted(nets)}")
print("\nNB ohne Flag, > 3000 Lok:")
for k, (n, nets, plzs) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
    if not any(x[2] == "1" for x in nets) and n > 3000:
        print(f"{n:8d} PLZ={len(plzs):4d} {k[1]} | {sorted(nets)}")
