# Mundial FIFA 2026 — Modelo Predictivo Cuantitativo

Modelo híbrido Elo → doble Poisson (Dixon-Coles) → Monte Carlo (100,000 simulaciones)
con parámetros calibrados empíricamente y backtest out-of-sample en los Mundiales 2014/2018/2022.

- **UI**: `web/` (estática; se despliega a GitHub Pages)
- **Pipeline**: `model/run_model.py` → genera `web/results.json`
- **Actualizar resultados reales**: `model/update_results.py` (API pública de ESPN) y commit+push para redesplegar
- **Backtest/calibración**: `model/backtest.py`
- **Datos** (con fuente y fecha en cada archivo): `data/`

Setup: `python3 -m venv .venv && .venv/bin/pip install numpy pandas scipy openpyxl`
