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

Per contest gelden eigen standaardgewichten:
- **CQ WPX CW**: prefix-multipliers per band, nadruk op nieuwe prefixes en band-prefixes.
- **CQ WW DX CW**: landen per band (countryprefix als proxy), met iets hogere mult-gewicht en run-bias.
- **IARU HF**: HQ/administratie-prefixen per band (countryprefix als proxy) met gebalanceerde mult/point nadruk.

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
