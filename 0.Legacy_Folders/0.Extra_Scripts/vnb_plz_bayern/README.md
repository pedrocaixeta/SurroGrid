# PLZ → Verteilnetzbetreiber (Bayern)

Ergebnis: `plz_vnb_bayern.csv` (UTF-8, Trennzeichen `;`, 2.063 PLZ)

## Spalten
| Spalte | Bedeutung |
|---|---|
| plz | Postleitzahl |
| vnb | Name des veröffentlichungspflichtigen VNB, sonst `nVNB` |
| dominanter_vnb_real | tatsächlich dominanter Anschlussnetzbetreiber laut MaStR (auch kleine VNB) |
| anteil_dominant | Anteil der dominanten VNB an den Anschluss-Lokationen der PLZ |
| n_lokationen | Anzahl ausgewerteter Lokationen in der PLZ |
| weitere_vnb | weitere VNB mit ≥10 % Anteil |
| status | Pflicht-Status + Hinweise (gemischt, wenige Datenpunkte, ...) |
| quelle | Quellenangabe |

## Methode
1. MaStR-Gesamtdatenexport (BNetzA, Stand 12.11.2026, DL-DE-BY-2.0).
2. Einheiten (Solar, Stromspeicher, Wind, Biomasse, Wasser, Verbrennung, Stromverbraucher) mit Bundesland = Bayern → (Lokation, PLZ).
3. Netzanschlusspunkte: Lokation → NetzbetreiberMaStRNummer.
4. Marktakteure: Name des Netzbetreibers.
5. Je PLZ: VNB mit den meisten Lokationen (Mehrheitsprinzip).
6. PLZ-Liste Bayern: MaStR-PLZ plus PLZ-Polygone (`plz-5stellig.shp`) mit ≥50 % Flächenanteil in Bayern (VG250, SN_L=09).
7. Veröffentlichungspflichtig (≥100.000 Kunden, § 14d EnWG): die 9 VNB der Planungsregion Bayern laut VNBdigital;
   zusätzlich VNB aus anderen Planungsregionen mit MaStR-Flag `KundenAngeschlossen=1` (Netze ODR, Syna, OsthessenNetz, Netze BW).

## Skripte
`01`–`10` im selben Ordner; Rohdaten in `data/` (3,3 GB ZIP, kann gelöscht werden).

