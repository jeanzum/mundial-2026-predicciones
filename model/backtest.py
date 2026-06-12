"""
Backtest histórico del modelo sobre Mundiales 2014/2018/2022.

Protocolo:
1. Réplica del Elo (fórmula publicada de eloratings.net) sobre el dataset
   completo de partidos internacionales (martj42/international_results, CC0).
   Validación: correlación de la réplica vs. Elo oficial de los 48 clasificados.
2. Calibración empírica de la pendiente Elo->dif. de gol y del total de goles
   en partidos internacionales competitivos 2006-2025, EXCLUYENDO los partidos
   de fases finales de Mundial (que quedan como conjunto de test).
3. Evaluación out-of-sample en los 3 Mundiales: Brier multiclase, log loss,
   tabla de calibración, tasa de empates predicha vs. real, y baselines.

Salida: data/backtest_report.json
"""

from __future__ import annotations
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# --- mapeo de nombres dataset -> nombres del proyecto (solo los que difieren)
NAME_MAP = {"Czechia": "Czech Republic", "Türkiye": "Turkey",
            "Côte d'Ivoire": "Ivory Coast", "Curaçao": "Curacao",
            "Cabo Verde": "Cape Verde", "IR Iran": "Iran"}


def k_factor(tournament: str) -> float:
    """K de eloratings.net por tipo de torneo (aproximación documentada)."""
    t = tournament.lower()
    if t == "fifa world cup":
        return 60
    if "qualification" in t:
        return 40
    if t == "friendly":
        return 20
    finals = ("uefa euro", "copa américa", "african cup of nations",
              "afc asian cup", "concacaf championship", "gold cup",
              "confederations cup", "oceania nations cup", "african nations championship")
    if any(f in t for f in finals):
        return 50
    return 30  # Nations League, torneos menores, etc.


def g_multiplier(gd: int) -> float:
    gd = abs(gd)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0  # 3 -> 1.75, 4 -> 1.875, ...


def load_matches():
    rows = []
    with open(DATA / "historical_results.csv") as f:
        for r in csv.DictReader(f):
            if r["home_score"] in ("NA", "", None):
                continue
            rows.append({
                "date": r["date"],
                "home": NAME_MAP.get(r["home_team"], r["home_team"]),
                "away": NAME_MAP.get(r["away_team"], r["away_team"]),
                "hg": int(r["home_score"]), "ag": int(r["away_score"]),
                "tournament": r["tournament"],
                "neutral": r["neutral"] == "TRUE",
            })
    rows.sort(key=lambda x: x["date"])
    return rows


def run_elo(matches, snapshot_dates):
    """Elo histórico. Devuelve ratings finales y snapshots pre-torneo."""
    R = defaultdict(lambda: 1500.0)
    snaps = {}
    si = 0
    dates = sorted(snapshot_dates)
    for m in matches:
        while si < len(dates) and m["date"] >= dates[si]:
            snaps[dates[si]] = dict(R)
            si += 1
        ha = 0.0 if m["neutral"] else 100.0
        dr = (R[m["home"]] + ha) - R[m["away"]]
        we = 1.0 / (1.0 + 10 ** (-dr / 400.0))
        w = 1.0 if m["hg"] > m["ag"] else (0.0 if m["hg"] < m["ag"] else 0.5)
        k = k_factor(m["tournament"]) * g_multiplier(m["hg"] - m["ag"])
        delta = k * (w - we)
        R[m["home"]] += delta
        R[m["away"]] -= delta
    for d in dates[si:]:
        snaps[d] = dict(R)
    return dict(R), snaps


FINALS_TOURNAMENTS = ("uefa euro", "copa américa", "african cup of nations",
                      "afc asian cup", "gold cup", "confederations cup",
                      "concacaf championship")


def calibrate(matches):
    """Dos calibraciones de GD ~ ΔElo recomputando el Elo al vuelo:
    (a) competitivos generales 2006-2025 (incluye qualifiers con mismatches),
    (b) SOLO fases finales de torneos continentales 2006-2025 (entorno
        comparable a un Mundial: neutral, equipos parejos) — la transferible.
    Mundiales (fase final) quedan fuera de ambas: son el test set."""
    R = defaultdict(lambda: 1500.0)
    gen = {"xs": [], "ys": [], "totals": []}
    fin = {"xs": [], "ys": [], "totals": [], "draws": 0, "n": 0}
    wc_finals_dates = ("2014-06", "2014-07", "2018-06", "2018-07",
                       "2022-11", "2022-12", "2026-06", "2026-07",
                       "2010-06", "2010-07", "2006-06", "2006-07")
    for m in matches:
        ha = 0.0 if m["neutral"] else 100.0
        dr = (R[m["home"]] + ha) - R[m["away"]]
        t = m["tournament"].lower()
        in_window = "2006-01-01" <= m["date"] <= "2025-12-31"
        is_wc_final_phase = (m["tournament"] == "FIFA World Cup"
                             and m["date"][:7] in wc_finals_dates)
        if in_window and t != "friendly" and not is_wc_final_phase:
            gen["xs"].append(dr); gen["ys"].append(m["hg"] - m["ag"])
            gen["totals"].append(m["hg"] + m["ag"])
            if any(f in t for f in FINALS_TOURNAMENTS) and "qualification" not in t:
                fin["xs"].append(dr); fin["ys"].append(m["hg"] - m["ag"])
                fin["totals"].append(m["hg"] + m["ag"])
                fin["n"] += 1
                fin["draws"] += int(m["hg"] == m["ag"])
        we = 1.0 / (1.0 + 10 ** (-dr / 400.0))
        w = 1.0 if m["hg"] > m["ag"] else (0.0 if m["hg"] < m["ag"] else 0.5)
        k = k_factor(m["tournament"]) * g_multiplier(m["hg"] - m["ag"])
        delta = k * (w - we)
        R[m["home"]] += delta
        R[m["away"]] -= delta

    def fit(d):
        xs, ys = np.array(d["xs"], float), np.array(d["ys"], float)
        slope = float(np.sum(xs * ys) / np.sum(xs * xs))
        resid = ys - slope * xs
        se = float(np.sqrt(np.sum(resid ** 2) / (len(xs) - 1) / np.sum(xs ** 2)))
        return {"n": len(xs), "gd_slope": slope, "gd_slope_se": se,
                "mean_total_goals": float(np.mean(d["totals"]))}

    out = {"general_competitive": fit(gen), "continental_finals": fit(fin)}
    out["continental_finals"]["draw_rate"] = fin["draws"] / fin["n"]
    return out


def predict_1x2(elo_h, elo_a, neutral=True, host=None):
    """Usa el motor actual para P(1/X/2) de un partido en cancha neutral."""
    la, lb = engine.expected_lambdas(elo_h, elo_a,
                                     host_a=(host == "home"), host_b=(host == "away"))
    m = engine.score_matrix(la, lb)
    p = engine.match_probs(m)
    return p["p_home"], p["p_draw"], p["p_away"]


WC_TESTS = {
    "2014": {"start": "2014-06-12", "end": "2014-07-13", "host": "Brazil"},
    "2018": {"start": "2018-06-14", "end": "2018-07-15", "host": "Russia"},
    "2022": {"start": "2022-11-20", "end": "2022-12-18", "host": "Qatar"},
}


def backtest(matches, snaps):
    out = {}
    all_briers, all_lls, cal_bins = [], [], defaultdict(lambda: [0, 0.0])
    pred_draw, obs_draw = [], []
    for wc, cfg in WC_TESTS.items():
        elo = snaps[cfg["start"]]
        briers, lls, correct = [], [], 0
        games = [m for m in matches
                 if cfg["start"] <= m["date"] <= cfg["end"]
                 and m["tournament"] == "FIFA World Cup"]
        for m in games:
            host = ("home" if m["home"] == cfg["host"]
                    else "away" if m["away"] == cfg["host"] else None)
            ph, pd, pa = predict_1x2(elo.get(m["home"], 1500), elo.get(m["away"], 1500),
                                     host=host)
            # resultado en 90' no disponible en el dataset para KO (trae el final
            # tras prórroga); aceptamos el marcador registrado como outcome.
            o = (1, 0, 0) if m["hg"] > m["ag"] else ((0, 0, 1) if m["hg"] < m["ag"] else (0, 1, 0))
            probs = (ph, pd, pa)
            briers.append(sum((p - oo) ** 2 for p, oo in zip(probs, o)))
            lls.append(-math.log(max(1e-12, sum(p * oo for p, oo in zip(probs, o)))))
            if max(range(3), key=lambda i: probs[i]) == max(range(3), key=lambda i: o[i]):
                correct += 1
            pred_draw.append(pd)
            obs_draw.append(o[1])
            for p, oo in zip(probs, o):
                b = min(9, int(p * 10))
                cal_bins[b][0] += 1
                cal_bins[b][1] += oo
        out[wc] = {
            "n_matches": len(games),
            "brier": float(np.mean(briers)),
            "log_loss": float(np.mean(lls)),
            "accuracy_modal": correct / len(games),
        }
        all_briers += briers
        all_lls += lls

    # baselines sobre el conjunto agregado
    n = len(all_briers)
    # uniforme
    brier_uniform = float(np.mean([sum((1/3 - oo) ** 2 for oo in (1, 0, 0))] * 1))  # constante
    brier_uniform = 2 * (1/3 - 0) ** 2 + (1/3 - 1) ** 2  # = 0.6667
    # climatológico: frecuencias reales H/D/A en Mundiales 2014-2022 (neutral-ish)
    freq = np.mean([[1, 0, 0] if False else 0]) if False else None
    out["aggregate"] = {
        "n_matches": n,
        "brier": float(np.mean(all_briers)),
        "log_loss": float(np.mean(all_lls)),
        "brier_uniform_baseline": round(brier_uniform, 4),
        "log_loss_uniform_baseline": round(math.log(3), 4),
        "pred_draw_rate": float(np.mean(pred_draw)),
        "obs_draw_rate": float(np.mean(obs_draw)),
        "calibration": [
            {"bucket": f"{b*10}-{b*10+10}%", "n": cnt, "predicted_mid": (b + 0.5) / 10,
             "observed": hits / cnt}
            for b, (cnt, hits) in sorted(cal_bins.items()) if cnt > 0
        ],
    }
    return out


def main():
    matches = load_matches()
    print(f"{len(matches):,} partidos con resultado")

    # 1) réplica de Elo y validación vs oficial
    snaps_needed = [cfg["start"] for cfg in WC_TESTS.values()] + ["2026-06-11"]
    finals, snaps = run_elo(matches, snaps_needed)
    official = json.loads((DATA / "elo.json").read_text())["ratings"]
    mine = snaps["2026-06-11"]
    pairs = [(mine.get(t), official[t]) for t in official if t in mine]
    a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
    r = float(np.corrcoef(a, b)[0, 1])
    mad = float(np.mean(np.abs(a - b)))
    print(f"Réplica Elo vs oficial (n={len(pairs)}): r={r:.4f}, MAD={mad:.1f} pts")

    # 2) calibración empírica
    cal = calibrate(matches)
    g, f = cal["general_competitive"], cal["continental_finals"]
    print(f"Calibración general (n={g['n']:,}, incluye qualifiers): "
          f"slope={g['gd_slope']:.6f}±{g['gd_slope_se']:.6f}, total={g['mean_total_goals']:.3f}")
    print(f"Calibración finals continentales (n={f['n']:,}): "
          f"slope={f['gd_slope']:.6f}±{f['gd_slope_se']:.6f}, total={f['mean_total_goals']:.3f}, "
          f"empates={f['draw_rate']:.1%}")

    # 3) backtest con parámetros del motor actual
    bt = backtest(matches, snaps)

    # 3b) sensibilidad: grid de pendiente y rho sobre el test (solo reporte)
    import itertools
    grid = []
    orig_slope, orig_rho = engine.GD_SLOPE, engine.DC_RHO
    for slope, rho in itertools.product([0.00275, 0.0035, 0.0045, f["gd_slope"]],
                                        [-0.10, -0.05, 0.0]):
        engine.GD_SLOPE, engine.DC_RHO = slope, rho
        r2 = backtest(matches, snaps)["aggregate"]
        grid.append({"slope": round(slope, 5), "rho": rho,
                     "brier": round(r2["brier"], 4), "log_loss": round(r2["log_loss"], 4),
                     "pred_draw": round(r2["pred_draw_rate"], 3)})
    engine.GD_SLOPE, engine.DC_RHO = orig_slope, orig_rho
    grid.sort(key=lambda x: x["log_loss"])
    print("\nSensibilidad (test WC 2014/18/22), mejores 5 por log loss:")
    for gg in grid[:5]:
        print(f"  slope={gg['slope']}, rho={gg['rho']}: Brier={gg['brier']}, "
              f"LL={gg['log_loss']}, draws={gg['pred_draw']:.1%}")
    for wc in ("2014", "2018", "2022"):
        x = bt[wc]
        print(f"WC {wc}: n={x['n_matches']}, Brier={x['brier']:.4f}, "
              f"LogLoss={x['log_loss']:.4f}, acc modal={x['accuracy_modal']:.1%}")
    ag = bt["aggregate"]
    print(f"AGREGADO: Brier={ag['brier']:.4f} (uniforme {ag['brier_uniform_baseline']}), "
          f"LogLoss={ag['log_loss']:.4f} (uniforme {ag['log_loss_uniform_baseline']})")
    print(f"Empates: pred {ag['pred_draw_rate']:.1%} vs obs {ag['obs_draw_rate']:.1%}")

    report = {
        "dataset": "github.com/martj42/international_results (CC0), 1872-2026",
        "elo_replica_validation": {"n": len(pairs), "pearson_r": round(r, 4),
                                   "mean_abs_diff_pts": round(mad, 1)},
        "empirical_calibration": cal,
        "sensitivity_grid": grid,
        "engine_params": {"gd_slope": engine.GD_SLOPE, "base_total": engine.BASE_TOTAL,
                          "dc_rho": engine.DC_RHO, "home_elo_bonus": engine.HOME_ELO_BONUS},
        "backtest": bt,
    }
    (DATA / "backtest_report.json").write_text(json.dumps(report, indent=1))
    print("OK -> data/backtest_report.json")


if __name__ == "__main__":
    main()
