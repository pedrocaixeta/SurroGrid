"""Baut die finale Tabelle PLZ -> VNB (Bayern)."""
import collections
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")

# Veroeffentlichungspflichtige VNB (>=100.000 Kunden, Netzausbauplan nach par. 14d EnWG)
PFLICHT_BAYERN = {  # VNBdigital: Planungsregion Bayern (9 VNB)
    "SNB940352624434": "Bayernwerk Netz GmbH",
    "SNB911705062982": "LEW Verteilnetz GmbH",
    "SNB973875583315": "N-ERGIE Netz GmbH",
    "SNB969473762610": "SWM Infrastruktur GmbH & Co. KG",
    "SNB926394308747": "AllgäuNetz GmbH & Co. KG",
    "SNB941929592729": "Regensburg Netz GmbH",
    "SNB915316807789": "Mainfranken Netze GmbH",
    "SNB958077102830": "swa Netze GmbH",
    "SNB934068635945": "Stadtwerke Ulm/Neu-Ulm Netze GmbH",
}
# >=100.000 Kunden (MaStR-Flag KundenAngeschlossen=1), Planungsregion ausserhalb Bayerns, greift aber in bay. PLZ
PFLICHT_ANDERE = {
    "Netze ODR GmbH": "Netze ODR GmbH",
    "Syna GmbH": "Syna GmbH",
    "Netze BW GmbH": "Netze BW GmbH",
    "OsthessenNetz GmbH": "OsthessenNetz GmbH",
}
MIN_FRAC_ANTEIL = float(os.environ.get("MIN_SHARE", "0.10"))

rows = list(csv.DictReader(open(os.path.join(D, "plz_nb_counts.tsv"), encoding="utf-8"), delimiter="\t"))
cnt = collections.defaultdict(collections.Counter)  # plz -> nb_mastr -> n
names = {}
for r in rows:
    cnt[r["plz"]][r["nb_mastr"]] += int(r["n_lokationen"])
    names[r["nb_mastr"]] = r["nb_name"]
pflicht = dict(PFLICHT_BAYERN)
for nb, nm in names.items():
    if nm in PFLICHT_ANDERE:
        pflicht[nb] = PFLICHT_ANDERE[nm]
missing = [n for n in PFLICHT_BAYERN if n not in names]
print("Pflicht-VNB nicht in Daten:", missing)

overlap = {}
p = os.path.join(D, "plz_bayern_overlap.tsv")
if os.path.exists(p):
    for r in csv.DictReader(open(p), delimiter="\t"):
        overlap[r["plz"]] = float(r["anteil_in_bayern"])

QUELLE_DATEN = ("MaStR-Gesamtdatenexport 12.11.2026, Bundesnetzagentur (DL-DE-BY-2.0): "
                "https://www.marktstammdatenregister.de/MaStR/Datendownload")
QUELLE_PFLICHT = ("VNBdigital Planungsregion Bayern (veröffentlichungspflichtige VNB, § 14d EnWG): https://www.vnbdigital.de")

plz_all = set(cnt) | {k for k, v in overlap.items() if v >= 0.5}
out, dropped = [], []
for plz in sorted(plz_all):
    c = cnt.get(plz)
    fr = overlap.get(plz)
    if not c:
        out.append(dict(plz=plz, vnb="k.A.", dominanter_vnb_real="", anteil_dominant="", n_lokationen=0, weitere_vnb="",
                        status="keine MaStR-Daten", anteil_in_bayern=fr, quelle="PLZ-Geometrie (plz-5stellig.shp / VG250); kein VNB ermittelbar"))
        continue
    tot = sum(c.values())
    # Grenz-/Fehl-PLZ: ausserhalb Bayerns laut Geometrie
    if overlap and (fr is None or fr < 0.5) and tot < 50:
        dropped.append((plz, tot, fr, c.most_common(1)[0][0], names[c.most_common(1)[0][0]]))
        continue
    (nb, n), = c.most_common(1)
    share = n / tot
    pf_total = sum(v for k, v in c.items() if k in pflicht)
    weitere = "; ".join(f"{names[k]} {v/tot:.0%}" for k, v in c.most_common() if k != nb and v / tot >= MIN_FRAC_ANTEIL)
    if nb in pflicht:
        vnb = pflicht[nb]
        status = "veröffentlichungspflichtiger VNB" + (" (anderer Planungsraum)" if nb not in PFLICHT_BAYERN else "")
        quelle = QUELLE_DATEN + " | " + QUELLE_PFLICHT
    else:
        vnb = "nVNB"
        status = "nicht veröffentlichungspflichtig"
        if pf_total / tot >= 0.25:
            status += f" (aber veröffentlichungspflichtige VNB {pf_total/tot:.0%} Anteil)"
        quelle = QUELLE_DATEN
    flags = []
    if share < 0.6:
        flags.append("gemischt (<60% Anteil)")
    if tot < 10:
        flags.append("wenige Datenpunkte (<10)")
    if fr is not None and fr < 0.5:
        flags.append(f"Grenz-PLZ (nur {fr:.0%} in Bayern)")
    out.append(dict(plz=plz, vnb=vnb, dominanter_vnb_real=names[nb], anteil_dominant=f"{share:.0%}", n_lokationen=tot,
                    weitere_vnb=weitere, status=status + ("; " + ", ".join(flags) if flags else ""),
                    anteil_in_bayern=fr, quelle=quelle))

cols = ["plz", "vnb", "dominanter_vnb_real", "anteil_dominant", "n_lokationen", "weitere_vnb", "status", "quelle"]
with open(os.path.join(HERE, "plz_vnb_bayern.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, delimiter=";", extrasaction="ignore")
    w.writeheader()
    w.writerows(out)
with open(os.path.join(D, "dropped_plz.tsv"), "w", encoding="utf-8") as f:
    f.write("plz\tn\tanteil_in_bayern\tnb\tname\n")
    for d in dropped:
        f.write("\t".join(str(x) for x in d) + "\n")
print("PLZ in Tabelle:", len(out), " verworfen:", len(dropped))
print(collections.Counter(r["vnb"] for r in out).most_common(20))
print("k.A.:", sum(1 for r in out if r["vnb"] == "k.A."))

