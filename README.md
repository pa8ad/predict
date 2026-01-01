# AI Contest Assistant (CQ WPX CW)

Deze app biedt een lokale SO2R-assistent voor CQ WPX CW die uitsluitend N1MM UDP XML-berichten leest. Er is geen CAT of rig control; alle data komt via UDP binnen en wordt gevisualiseerd voor operator-advies.

## Vereisten
- Python 3.11+
- Streamlit

Installeer dependencies:
```bash
pip install streamlit
```

## Starten
### Streamlit-dashboard
1. Start de Streamlit-app:
```bash
streamlit run app.py
```
2. Stel in de zijbalk host/poort in (standaard `0.0.0.0:12060`) en klik op **Start UDP listener**.
3. Open de getoonde lokale URL in je browser (typisch `http://localhost:8501`).

### Console-headless mode
Wil je zonder Streamlit draaien (bijv. direct in Python), gebruik de consolemodus:
```bash
python app.py console --host 0.0.0.0 --port 12060 --max-age 20 --mycall N0CALL
```
De consoleprint toont elke 5 seconden een snapshot van QSOs, rates, multipliers en de hoogst scorende spots. Stop met `Ctrl+C`.

## UDP simulator
Gebruik de meegeleverde simulator om testberichten te sturen:
```bash
python udp_simulator.py --host 127.0.0.1 --port 12060 --mode all
```
`--mode` kan `radio`, `contact`, `spot` of `all` zijn.

## Functies
- Radio-panels voor 2 radio's (freq, TX, run/tx-status, focus/active).
- Scoreboard met QSOs, punten, multipliers (WPX-prefixes) en 1/5/15m rate.
- High-value spot tabel met heuristische score, dupes en mult-indicaties.
- Next-best-action advies (heuristiek) voor SO2R keuzes.
- Eventlog met debugregels.

## Configuratie
- Bandplan en spot-age drempel staan bovenin `app.py` (pas aan indien nodig).
- Weights voor heuristiek zijn instelbaar in de UI.

## Opmerkingen
- XML parsing is tolerant voor foutieve timestamps en decimalen in bandwaarden ("3,5").
- Contact replace/delete wordt verwerkt zodat multipliers consistent blijven.
- Optionele LLM-ondersteuning kan worden toegevoegd door `OPENAI_API_KEY` te zetten; momenteel werkt de app puur heuristisch.
