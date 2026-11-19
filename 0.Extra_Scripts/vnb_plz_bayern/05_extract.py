"""Schritt 1-3: PLZ -> Netzbetreiber aus dem MaStR-Gesamtdatenexport (nur Bayern).

1. Katalogwerte: Code fuer Bundesland 'Bayern' ermitteln
2. EinheitenSolar_*.xml: (LokationMaStRNummer, PLZ) fuer Einheiten in Bayern
3. Netzanschlusspunkte_*.xml: Lokation -> NetzbetreiberMaStRNummer (Anschlussnetzbetreiber)
4. Marktakteure_*.xml: Name der Netzbetreiber (SNB...)
Ergebnis: data/plz_nb_counts.tsv  (plz, nb_mastr, nb_name, n_lokationen)
"""
import collections
import multiprocessing as mp
import os
import re
import sys
import time

from mastr_stream import field, iter_records, open_zip

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data", "Gesamtdatenexport_20261112_26.2.zip")
MAXREC = int(os.environ.get("MAXREC", "0")) or None
OUT = os.path.join(HERE, "data")
UNIT_PREFIXES = os.environ.get("UNITS", "EinheitenSolar").split(",")

LOK2PLZ = {}
BY_CODE = None


def members(z, prefix):
    names = [n for n in z.namelist() if re.fullmatch(prefix + r"(_\d+)?\.xml", n)]
    return sorted(names, key=lambda n: int(re.search(r"_(\d+)\.xml", n).group(1)) if "_" in n else 0)


def find_bayern_code(z):
    for rec in iter_records(z, "Katalogwerte.xml", "Katalogwert"):
        # Kategorie 101 = Bundesland der Einheiten (Kategorie 14 = Bundesland der Netze, Id 335)
        if field(rec, "Wert") == "Bayern" and field(rec, "KatalogKategorieId") == "101":
            return field(rec, "Id")


def scan_units(args):
    src, member, tag, code = args
    z = open_zip(src)
    out = []
    for rec in iter_records(z, member, tag, MAXREC):
        if f"<Bundesland>{code}</Bundesland>" not in rec:
            continue
        lok = field(rec, "LokationMaStRNummer")
        plz = field(rec, "Postleitzahl")
        if lok and plz:
            out.append((lok, plz.strip()))
    return member, out


def scan_nap(args):
    src, member = args
    z = open_zip(src)
    res = []
    for rec in iter_records(z, member, "Netzanschlusspunkt", MAXREC):
        lok = field(rec, "LokationMaStRNummer")
        if lok in LOK2PLZ:
            res.append((lok, field(rec, "NetzbetreiberMaStRNummer"), field(rec, "NetzMaStRNummer")))
    return member, res


def scan_actors(args):
    src, member, wanted = args
    z = open_zip(src)
    res = {}
    for rec in iter_records(z, member, "Marktakteur", MAXREC):
        nr = field(rec, "MastrNummer")
        if nr in wanted:
            res[nr] = (field(rec, "Firmenname"), field(rec, "Ort"), field(rec, "Postleitzahl"))
    return member, res


def main():
    global LOK2PLZ
    t0 = time.time()
    z = open_zip(SRC)
    code = find_bayern_code(z)
    print("Bundesland-Code Bayern:", code, flush=True)

    jobs = []
    for pre in UNIT_PREFIXES:
        tag = pre.replace("Einheiten", "Einheit")
        jobs += [(SRC, m, tag, code) for m in members(z, pre)]
    print(len(jobs), "Einheiten-Dateien", flush=True)
    with mp.Pool(min(10, len(jobs))) as pool:
        for i, (m, out) in enumerate(pool.imap_unordered(scan_units, jobs)):
            for lok, plz in out:
                LOK2PLZ[lok] = plz
            print(f"[units {i+1}/{len(jobs)}] {m}: {len(out)} Bayern  (gesamt {len(LOK2PLZ)})  {time.time()-t0:.0f}s", flush=True)

    jobs = [(SRC, m) for m in members(z, "Netzanschlusspunkte")]
    pairs = collections.Counter()
    nb_nums = set()
    seen = set()
    with mp.Pool(10) as pool:  # fork -> LOK2PLZ wird vererbt
        for i, (m, res) in enumerate(pool.imap_unordered(scan_nap, jobs)):
            for lok, nb, netz in res:
                if lok in seen or not nb:
                    continue
                seen.add(lok)
                pairs[(LOK2PLZ[lok], nb, netz)] += 1
                nb_nums.add(nb)
            print(f"[nap {i+1}/{len(jobs)}] {m}: {len(res)} Treffer  {time.time()-t0:.0f}s", flush=True)

    jobs = [(SRC, m, nb_nums) for m in members(z, "Marktakteure")]
    names = {}
    with mp.Pool(10) as pool:
        for i, (m, res) in enumerate(pool.imap_unordered(scan_actors, jobs)):
            names.update(res)
            print(f"[actors {i+1}/{len(jobs)}] {m} {len(names)}/{len(nb_nums)}  {time.time()-t0:.0f}s", flush=True)

    netze = {}
    for rec in iter_records(z, "Netze.xml", "Netz"):
        netze[field(rec, "MastrNummer")] = (
            field(rec, "Sparte"), field(rec, "Bezeichnung"), field(rec, "KundenAngeschlossen"),
            field(rec, "GeschlossenesVerteilnetz"))
    with open(os.path.join(OUT, "plz_nb_counts.tsv"), "w", encoding="utf-8") as f:
        f.write("plz\tnb_mastr\tnb_name\tnb_ort\tnetz_mastr\tnetz_sparte\tnetz_bezeichnung\tnetz_kunden_ge_100k\tnetz_geschlossenes_vn\tn_lokationen\n")
        for (plz, nb, netz), n in sorted(pairs.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or "")):
            nm = names.get(nb, (None, None, None))
            nz = netze.get(netz, (None, None, None, None))
            f.write(f"{plz}\t{nb}\t{nm[0]}\t{nm[1]}\t{netz}\t{nz[0]}\t{nz[1]}\t{nz[2]}\t{nz[3]}\t{n}\n")
    print("fertig", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
