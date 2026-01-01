# AI Contest Assistant (CQ WPX CW)

Lokale SO2R helper die uitsluitend N1MM UDP XML-berichten verwerkt (RadioInfo, ContactInfo/Replace/Delete, Spot). UI is PySide6 (Qt) voor een compact, lokaal dashboard zonder Streamlit.

## Vereisten
- Python 3.11+
- PySide6 (`pip install PySide6`)

## Starten (GUI)
```
python app.py gui --host 0.0.0.0 --port 12060 --mycall N0CALL
```
Pas host/port/mycall/spot-age in de bovenste balk aan en klik **Start listener**. Het scherm toont:
- Radio 1/2 status (freq, run/TX/focus/active)
- Scoreboard (QSOs, punten, prefixes, 1/5/15m rate, band-mults)
- High value spots met heuristische score (prefix/mult, SNR, WPM, band)
- Next best actions (top 3)
- Event log

In de sectie **Rules & Heuristics** kies je de contest (nu: CQ WPX CW) en stel je de gewichten bij voor mult/freshness/SNR/penalty zodat de spot-score aansluit bij jouw operatiestijl.

## Console-modus
```
python app.py console --host 0.0.0.0 --port 12060
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
