"""
Actualizador automático de resultados: consulta la API pública de ESPN
(scoreboard fifa.world), actualiza data/played.json con los partidos
finalizados de fase de grupos y re-ejecuta el pipeline completo.

Uso: .venv/bin/python model/update_results.py [--no-rerun]

Nota: cubre los 72 partidos de fase de grupos (hasta 27-jun). Los resultados
reales de eliminatorias no se condicionan todavía en la simulación (el torneo
entra en KO el 28-jun; extender entonces).
"""

from __future__ import annotations
import argparse
import json
import subprocess
import sys
import urllib.request
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

API = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates={d}"

# ESPN displayName -> nombre del proyecto
NAME_MAP = {
    "Czechia": "Czech Republic", "Czech Republic": "Czech Republic",
    "USA": "United States", "United States": "United States",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Türkiye": "Turkey", "Turkey": "Turkey",
    "Côte d'Ivoire": "Ivory Coast", "Ivory Coast": "Ivory Coast",
    "Curaçao": "Curacao", "Curacao": "Curacao",
    "Cape Verde Islands": "Cape Verde", "Cape Verde": "Cape Verde",
    "IR Iran": "Iran", "Iran": "Iran",
    "South Korea": "South Korea", "Korea Republic": "South Korea",
    "DR Congo": "DR Congo", "Congo DR": "DR Congo",
}


def canon(name: str, valid: set) -> str:
    n = NAME_MAP.get(name, name)
    if n not in valid:
        raise SystemExit(f"Nombre ESPN no mapeado a un clasificado: '{name}' -> '{n}'. "
                         f"Agregalo a NAME_MAP (no se inventan correspondencias).")
    return n


def fetch_day(d: str) -> list:
    with urllib.request.urlopen(API.format(d=d.replace("-", "")), timeout=20) as r:
        return json.load(r).get("events", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rerun", action="store_true")
    args = ap.parse_args()

    fixtures = json.loads((DATA / "fixtures.json").read_text())["matches"]
    valid = {fx["home"] for fx in fixtures} | {fx["away"] for fx in fixtures}
    group_of = {}
    for fx in fixtures:
        group_of[frozenset((fx["home"], fx["away"]))] = fx["group"]

    start = date(2026, 6, 11)
    end = min(date.today(), date(2026, 6, 27))
    results = []
    d = start
    while d <= end:
        ds = d.isoformat()
        try:
            events = fetch_day(ds)
        except Exception as e:
            print(f"  {ds}: error API ({e}), salto")
            d += timedelta(days=1)
            continue
        for ev in events:
            comp = ev["competitions"][0]
            if comp["status"]["type"]["name"] != "STATUS_FULL_TIME":
                continue
            sides = {c["homeAway"]: c for c in comp["competitors"]}
            h = canon(sides["home"]["team"]["displayName"], valid)
            a = canon(sides["away"]["team"]["displayName"], valid)
            key = frozenset((h, a))
            if key not in group_of:
                print(f"  {ds}: {h} vs {a} no es partido de grupos, salto")
                continue
            results.append({
                "date": ds, "group": group_of[key], "home": h, "away": a,
                "home_goals": int(sides["home"]["score"]),
                "away_goals": int(sides["away"]["score"]),
                "source": f"ESPN API scoreboard {ds}",
            })
        d += timedelta(days=1)

    payload = {
        "source": "ESPN public API (site.api.espn.com, fifa.world scoreboard)",
        "as_of": date.today().isoformat(),
        "matches": results,
    }
    (DATA / "played.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    print(f"played.json actualizado: {len(results)} partidos finalizados")
    for r in results:
        print(f"  [{r['group']}] {r['home']} {r['home_goals']}-{r['away_goals']} {r['away']}")

    if not args.no_rerun:
        print("Re-ejecutando pipeline (100,000 simulaciones)…")
        subprocess.run([sys.executable, str(ROOT / "model" / "run_model.py"),
                        "--sims", "100000"], check=True)


if __name__ == "__main__":
    main()
