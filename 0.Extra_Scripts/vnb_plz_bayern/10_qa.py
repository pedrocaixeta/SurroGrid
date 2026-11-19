import csv, collections
rows = list(csv.DictReader(open("plz_vnb_bayern.csv", encoding="utf-8-sig"), delimiter=";"))
print("Gesamt", len(rows))
for key in ("Netze ODR GmbH", "OsthessenNetz GmbH", "Syna GmbH", "Stadtwerke Ulm/Neu-Ulm Netze GmbH"):
    print(key, [(r["plz"], r["anteil_dominant"], r["n_lokationen"]) for r in rows if r["vnb"] == key])
print("\nGemischt:")
for r in rows:
    if "gemischt" in r["status"]:
        print(r["plz"], r["vnb"], "|", r["dominanter_vnb_real"], r["anteil_dominant"], r["n_lokationen"], "|", r["weitere_vnb"])
few = [r for r in rows if int(r["n_lokationen"]) < 10]
print("\nWenige Datenpunkte (<10):", len(few))
print("n<30:", sum(1 for r in rows if int(r["n_lokationen"]) < 30))
print("Anteil dominant >=90%:", sum(1 for r in rows if int(r["anteil_dominant"].rstrip('%')) >= 90))
print("Anteil <80%:", sum(1 for r in rows if int(r["anteil_dominant"].rstrip('%')) < 80))
print("Kategorie:", collections.Counter("pflicht" if r["vnb"] != "nVNB" else "nVNB" for r in rows))

