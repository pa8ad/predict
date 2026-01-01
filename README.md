# AI Contest Assistant (CQ WPX / CQ WW / IARU)

Lokale SO2R helper die uitsluitend N1MM UDP XML-berichten verwerkt (RadioInfo, ContactInfo/Replace/Delete, Spot). UI is PySide6 (Qt) voor een compact, lokaal dashboard zonder Streamlit.

## Vereisten
- Python 3.11+
- PySide6 (`pip install PySide6`)

## Starten (GUI)
```
python app.py gui --host 0.0.0.0 --port 12060 --contest "CQ WPX CW"
```
Pas host/port/spot-age in de bovenste balk aan en klik **Start listener**. Het scherm toont:
- Scoreboard (QSOs, punten, prefixes, 1/5/15m rate, band-mults)
- Next best actions met een dominante hoofdaanbeveling en alternatieven; de tekstgrootte is live verstelbaar via de `Text size`-spinner.

In de sectie **Rules & Heuristics** (standaard ingeklapt) kies je de contest (CQ WPX CW, CQ WW DX CW of IARU HF) en stel je de gewichten bij voor mult/freshness/SNR/penalty zodat de spot-score aansluit bij jouw operatiestijl. Gebruik de `run_floor`, `rate5_bias` en `rate15_bias` om het omslagpunt tussen "blijven runnen" en "spot pakken" af te stemmen op jouw gewenste rate: hoe hoger deze waardes, hoe meer het advies bij een goede run blijft. Hover over een veld voor een korte toelichting.

Per contest gelden eigen standaardgewichten, afgeleid van de officiële punt- en multiplier-structuur:
- **CQ WPX CW** – Punten: 1/2/3 per continent. Multipliers: unieke prefixes per band (zeer bepalend). Heuristieken: `mult=16`, `band_mult=8`, `fresh=3.0`, `snr=1.5`, `band_penalty=5`, `dupe_penalty=6`, `run_floor=1.1`, `rate5_bias=0.9`, `rate15_bias=0.6` om prefixes boven run te prioriteren tenzij de run-rate duidelijk wint.
- **CQ WW DX CW** – Punten: 1 (zelfde cont) / 2 (anders) / 3 (NA↔SA/OC/AF/AS/EU). Multipliers: landen + zones per band (hier landenproxy). Heuristieken: `mult=12`, `band_mult=7`, `fresh=2.5`, `snr=1.8`, `band_penalty=5`, `dupe_penalty=6`, `run_floor=1.6`, `rate5_bias=1.2`, `rate15_bias=0.8` zodat een goede run zwaarder telt en enkel sterke mult-spots deze drempel passeren.
- **IARU HF** – Punten: 1 (zelfde ITU-zone), 3 (andere zone zelfde continent), 5 (ander continent/HQ). Multipliers: ITU-zones + HQ per band (landenproxy). Heuristieken: `mult=11`, `band_mult=6`, `fresh=2.8`, `snr=2.0`, `band_penalty=5`, `dupe_penalty=5`, `run_floor=1.4`, `rate5_bias=1.0`, `rate15_bias=0.7` voor balans tussen HQ/zones en behoud van run-rate.

## Console-modus
```
python app.py console --host 0.0.0.0 --port 12060 --contest "CQ WPX CW"
```
Dit draait zonder GUI en print snapshots in de terminal.

## UDP simulator
Stuur voorbeeld-N1MM UDP berichten voor snelle tests:
```
python udp_simulator.py --host 127.0.0.1 --port 12060
```

## Opmerkingen
- Alleen UDP; geen CAT/rig-control, geen kaarten.
- Bandplan en gewichten staan in `app.py` en zijn eenvoudig aanpasbaar.
- De QSO-rates op het scoreboard volgen alleen de gelogde QSO-tijdstempels (1/5/15m windows). De nieuwe run-bias-gewichten (`run_floor`, `rate5_bias`, `rate15_bias`) bepalen hoeveel waarde een lopende run heeft t.o.v. een spot: spots moeten die drempel overstijgen om als advies getoond te worden, anders blijft het advies "Keep running".
- Als de gekozen host/port al bezet is, verschijnt een melding en blijft de **Start listener**-knop actief zodat je een andere poort kunt kiezen.
