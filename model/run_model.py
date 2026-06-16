"""
Pipeline principal: carga datos verificados -> predicciones por partido ->
Monte Carlo del torneo completo -> web/results.json para la UI.

Uso: .venv/bin/python model/run_model.py [--sims 100000]
"""

from __future__ import annotations
import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import expected_lambdas, score_matrix, match_probs, sim_score, sim_knockout_winner

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
WEB = ROOT / "web"

HOST_COUNTRY = {"Mexico": "Mexico", "United States": "United States", "Canada": "Canada"}

STAGE_ORDER = ["GROUPS", "R32", "R16", "QF", "SF", "F", "CHAMP"]
STAGE_IDX = {s: i for i, s in enumerate(STAGE_ORDER)}


# ----------------------------------------------------------------- carga

def load():
    groups = json.loads((DATA / "groups.json").read_text())["groups"]
    elo = json.loads((DATA / "elo.json").read_text())["ratings"]
    fixtures = json.loads((DATA / "fixtures.json").read_text())["matches"]
    bracket = json.loads((DATA / "bracket.json").read_text())
    fifa = json.loads((DATA / "fifa_ranking.json").read_text()) if (DATA / "fifa_ranking.json").exists() else {"rankings": {}}
    odds = json.loads((DATA / "odds.json").read_text())
    played = json.loads((DATA / "played.json").read_text())["matches"] if (DATA / "played.json").exists() else []
    flags = json.loads((DATA / "flags.json").read_text()) if (DATA / "flags.json").exists() else {}

    all_teams = [t for g in groups.values() for t in g]
    missing = [t for t in all_teams if t not in elo]
    if missing:
        raise SystemExit(f"Equipos sin Elo: {missing}")
    return groups, elo, fixtures, bracket, fifa, odds, played, flags


def is_host(team, country):
    return HOST_COUNTRY.get(team) == country


def compute_live_elo(base_elo, played, fixtures):
    """Elo actualizado tras cada partido real del torneo (orden cronológico).

    Parte del snapshot pre-torneo de eloratings.net y aplica la fórmula Elo
    estándar a cada resultado verificado. Esto hace que el modelo reaccione a
    lo que pasa en la cancha (un favorito que empata pierde rating; una goleada
    sube al ganador) en vez de quedar anclado a la forma previa al Mundial.
    100% basado en resultados reales — no incorpora juicios sobre lesiones."""
    from engine import apply_match_elo
    R = dict(base_elo)
    country_of = {}
    for fx in fixtures:
        country_of[frozenset((fx["home"], fx["away"]))] = fx["country"]
    for p in sorted(played, key=lambda x: x["date"]):
        h, a = p["home"], p["away"]
        country = country_of.get(frozenset((h, a)))
        apply_match_elo(R, h, a, p["home_goals"], p["away_goals"],
                        host_a=is_host(h, country), host_b=is_host(a, country))
    return R


# ------------------------------------------------- predicciones por partido

def predict_fixture(fx, elo):
    ha, hb = is_host(fx["home"], fx["country"]), is_host(fx["away"], fx["country"])
    la, lb = expected_lambdas(elo[fx["home"]], elo[fx["away"]], host_a=ha, host_b=hb)
    m = score_matrix(la, lb)
    p = match_probs(m)
    fav = max(p["p_home"], p["p_draw"], p["p_away"])
    spread = fav - min(p["p_home"], p["p_away"])
    conf = "Alta" if fav >= 0.55 else ("Baja" if spread <= 0.15 else "Media")
    return la, lb, p, conf


# ------------------------------------------------------- reglas de grupo

def rank_group(table, h2h, rng):
    teams = list(table.keys())
    teams.sort(key=lambda t: (-table[t][0], -table[t][1], -table[t][2]))
    out, i = [], 0
    while i < len(teams):
        j = i + 1
        key_i = tuple(table[teams[i]])
        while j < len(teams) and tuple(table[teams[j]]) == key_i:
            j += 1
        tied = teams[i:j]
        if len(tied) > 1:
            sub = defaultdict(lambda: [0, 0, 0])
            for (a, b), (ga, gb) in h2h.items():
                if a in tied and b in tied:
                    if ga > gb: sub[a][0] += 3
                    elif gb > ga: sub[b][0] += 3
                    else: sub[a][0] += 1; sub[b][0] += 1
                    sub[a][1] += ga - gb; sub[b][1] += gb - ga
                    sub[a][2] += ga; sub[b][2] += gb
            tied.sort(key=lambda t: (-sub[t][0], -sub[t][1], -sub[t][2], rng.random()))
        out.extend(tied); i = j
    return out


def assign_thirds(third_groups, slot_allowed, rng):
    """Matching con backtracking: asigna 8 grupos de terceros a slots R32.

    slot_allowed: dict slot_id -> set de grupos elegibles (del calendario FIFA).
    Devuelve dict slot_id -> grupo. FIFA usa una tabla precomputada que
    satisface exactamente estas restricciones; el matching reproduce una
    asignación válida (si hay varias, elige una al azar — el impacto en las
    probabilidades agregadas es de segundo orden).
    """
    slots = list(slot_allowed.keys())
    rng.shuffle(slots)
    groups = list(third_groups)
    assignment = {}

    def bt(k):
        if k == len(slots):
            return True
        sid = slots[k]
        cand = [g for g in groups if g in slot_allowed[sid] and g not in assignment.values()]
        rng.shuffle(cand)
        for g in cand:
            assignment[sid] = g
            if bt(k + 1):
                return True
            del assignment[sid]
        return False

    if not bt(0):
        # No debería ocurrir con la tabla FIFA real; fallback sin restricciones.
        assignment.clear()
        for sid, g in zip(slots, groups):
            assignment[sid] = g
    return assignment


# --------------------------------------------------------------- simulación

def run_simulations(n_sims, groups, elo, fixtures, bracket, played, seed=20260612):
    rng = np.random.default_rng(seed)

    played_map = {}
    for p in played:
        played_map[(p["home"], p["away"])] = (p["home_goals"], p["away_goals"])
        played_map[(p["away"], p["home"])] = (p["away_goals"], p["home_goals"])

    # precomputar lambdas de cada fixture de grupos
    fx_lam = []
    for fx in fixtures:
        ha, hb = is_host(fx["home"], fx["country"]), is_host(fx["away"], fx["country"])
        la, lb = expected_lambdas(elo[fx["home"]], elo[fx["away"]], host_a=ha, host_b=hb)
        fx_lam.append((fx, la, lb, played_map.get((fx["home"], fx["away"]))))

    slot_allowed = {sid: set(allowed) for sid, allowed in bracket["third_slot_allowed_groups"].items()}

    stage_count = {t: Counter() for g in groups.values() for t in g}
    grp_stats = {t: {"pts": 0.0, "first": 0, "second": 0, "advance": 0} for t in stage_count}
    match_appear = defaultdict(Counter)   # match_id -> Counter(equipo)

    for s in range(n_sims):
        tables = {g: {t: [0, 0, 0] for t in members} for g, members in groups.items()}
        h2h = {g: {} for g in groups}
        for fx, la, lb, real in fx_lam:
            if real is not None:
                ga, gb = real
            else:
                ga, gb = sim_score(rng, la, lb)
            g = fx["group"]; a, b = fx["home"], fx["away"]
            h2h[g][(a, b)] = (ga, gb)
            ta, tb = tables[g][a], tables[g][b]
            if ga > gb: ta[0] += 3
            elif gb > ga: tb[0] += 3
            else: ta[0] += 1; tb[0] += 1
            ta[1] += ga - gb; tb[1] += gb - ga
            ta[2] += ga; tb[2] += gb

        standings = {g: rank_group(tables[g], h2h[g], rng) for g in tables}
        slots = {}
        for g, order in standings.items():
            slots[f"1{g}"] = order[0]; slots[f"2{g}"] = order[1]
            grp_stats[order[0]]["first"] += 1
            grp_stats[order[1]]["second"] += 1
            for t in order:
                grp_stats[t]["pts"] += tables[g][t][0]

        thirds = [(g, standings[g][2]) for g in sorted(standings)]
        thirds.sort(key=lambda x: (-tables[x[0]][x[1]][0], -tables[x[0]][x[1]][1],
                                   -tables[x[0]][x[1]][2], rng.random()))
        qual = thirds[:8]
        third_by_group = dict(qual)
        assignment = assign_thirds([g for g, _ in qual], slot_allowed, rng)
        for sid, g in assignment.items():
            slots[sid] = third_by_group[g]

        reached = {t: "GROUPS" for t in stage_count}
        for g, order in standings.items():
            reached[order[0]] = "R32"; reached[order[1]] = "R32"
        for _, t in qual:
            reached[t] = "R32"
        for t in reached:
            if reached[t] == "R32":
                grp_stats[t]["advance"] += 1

        winners = {}
        nxt = {"R32": "R16", "R16": "QF", "QF": "SF", "SF": "F", "F": "CHAMP"}
        for rnd in ["R32", "R16", "QF", "SF", "F"]:
            for m in bracket["rounds"][rnd]:
                a = slots[m["home"]] if m["home"] in slots else winners[m["home"]]
                b = slots[m["away"]] if m["away"] in slots else winners[m["away"]]
                match_appear[m["id"]][a] += 1
                match_appear[m["id"]][b] += 1
                la, lb = expected_lambdas(elo[a], elo[b],
                                          host_a=is_host(a, m["country"]),
                                          host_b=is_host(b, m["country"]),
                                          knockout=True)
                a_wins, _, _ = sim_knockout_winner(rng, elo[a], elo[b], la, lb)
                w, l = (a, b) if a_wins else (b, a)
                winners[m["id"]] = w
                reached[w] = nxt[rnd]
                reached[l] = rnd if rnd != "F" else "F"

        for t, st in reached.items():
            stage_count[t][st] += 1

    return stage_count, grp_stats, match_appear


# ----------------------------------------------------------------- salida

def cumulative(stage_count, t, n_sims, stage):
    idx = STAGE_IDX[stage]
    return sum(c for s, c in stage_count[t].items() if STAGE_IDX[s] >= idx) / n_sims


def implied_no_vig(odds_map):
    inv = {t: 1.0 / o for t, o in odds_map.items()}
    s = sum(inv.values())
    return {t: v / s for t, v in inv.items()}, s


def ensemble_champion(model_p, market_p, w=0.5):
    """Blend log-odds 50/50 modelo+mercado, renormalizado.

    La evidencia (Vaughan Williams & Reade 2016; estudios de forecasting de
    torneos) muestra que combinar un modelo de ratings con odds supera a
    ambos por separado. w=0.5 es la elección neutral estándar.
    """
    import math
    def logit(p): return math.log(max(p, 1e-9) / max(1 - p, 1e-9))
    def sigm(x): return 1 / (1 + math.exp(-x))
    blend = {}
    for t, pm in model_p.items():
        pk = market_p.get(t)
        blend[t] = sigm(w * logit(pm) + (1 - w) * logit(pk)) if pk else pm
    s = sum(blend.values())
    return {t: v / s for t, v in blend.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=100_000)
    args = ap.parse_args()

    groups, elo, fixtures, bracket, fifa, odds, played, flags = load()
    n = args.sims

    # Elo en vivo: parte del snapshot pre-torneo y aplica cada resultado real.
    live_elo = compute_live_elo(elo, played, fixtures)
    moved = sorted(((t, live_elo[t] - elo[t]) for t in elo), key=lambda x: x[1])
    if played:
        print("Elo en vivo aplicado. Mayores caídas/subidas:")
        for t, dv in moved[:3] + moved[-3:]:
            print(f"  {t:<14} {elo[t]:>5} -> {live_elo[t]:>6.0f}  ({dv:+.0f})")

    print(f"Simulando {n:,} torneos con Elo en vivo…")
    stage_count, grp_stats, match_appear = run_simulations(n, groups, live_elo, fixtures, bracket, played)

    # --- partidos (probabilidades exactas, no MC) ---
    played_map = {}
    for p in played:
        played_map[(p["home"], p["away"])] = p
        played_map[(p["away"], p["home"])] = {**p, "home_goals": p["away_goals"], "away_goals": p["home_goals"]}
    matches_out = []
    for fx in fixtures:
        real = played_map.get((fx["home"], fx["away"]))
        # Partidos jugados: predicción con Elo PRE-torneo (lo que el modelo
        # pensaba antes, honesto para evaluar aciertos). Futuros: Elo en vivo.
        used_elo = elo if real is not None else live_elo
        la, lb, p, conf = predict_fixture(fx, used_elo)
        matches_out.append({
            "match": fx["match"], "date": fx["date"], "group": fx["group"],
            "home": fx["home"], "away": fx["away"],
            "elo_home": round(used_elo[fx["home"]]), "elo_away": round(used_elo[fx["away"]]),
            "lam_home": round(la, 3), "lam_away": round(lb, 3),
            "p_home": round(p["p_home"], 4), "p_draw": round(p["p_draw"], 4),
            "p_away": round(p["p_away"], 4),
            "over25": round(p["over25"], 4), "btts": round(p["btts"], 4),
            "top_scores": [[s, round(pp, 4)] for s, pp in p["top_scores"]],
            "confidence": conf,
            "played": real is not None,
            "score": f"{real['home_goals']}-{real['away_goals']}" if real else None,
        })

    # --- precisión real hasta la fecha (transparencia, sin maquillaje) ---
    pc = [m for m in matches_out if m["played"]]
    def outcome(hg, ag): return "1" if hg > ag else ("2" if ag > hg else "X")
    hits = exact = base_hits = draws = 0
    for m in pc:
        hg, ag = map(int, m["score"].split("-"))
        real = outcome(hg, ag)
        modal = max((("1", m["p_home"]), ("X", m["p_draw"]), ("2", m["p_away"])), key=lambda x: x[1])[0]
        hits += modal == real
        exact += m["top_scores"][0][0] == m["score"]
        base_hits += (("1" if m["elo_home"] >= m["elo_away"] else "2") == real)
        draws += real == "X"
    accuracy = {
        "n_played": len(pc),
        "hits_1x2": hits, "hits_exact": exact,
        "baseline_elo_hits": base_hits, "draws_real": draws,
    } if pc else None

    # --- equipos / Monte Carlo ---
    market, overround = implied_no_vig(odds.get("outright_decimal", {}))
    group_of = {t: g for g, members in groups.items() for t in members}
    model_champ = {t: cumulative(stage_count, t, n, "CHAMP") for t in group_of}
    ens = ensemble_champion(model_champ, market)
    teams_out = []
    for t in sorted(group_of):
        fr = fifa.get("rankings", {}).get(t, {})
        teams_out.append({
            "name": t, "group": group_of[t],
            "elo": round(live_elo[t]), "elo_base": elo[t], "elo_delta": round(live_elo[t] - elo[t]),
            "fifa_rank": fr.get("rank"), "fifa_points": fr.get("points"),
            "market_odds": odds["outright_decimal"].get(t),
            "market_implied": round(market.get(t), 5) if t in market else None,
            "mc": {
                "r32": round(cumulative(stage_count, t, n, "R32"), 5),
                "r16": round(cumulative(stage_count, t, n, "R16"), 5),
                "qf": round(cumulative(stage_count, t, n, "QF"), 5),
                "sf": round(cumulative(stage_count, t, n, "SF"), 5),
                "final": round(cumulative(stage_count, t, n, "F"), 5),
                "champion": round(model_champ[t], 5),
            },
            "champion_ensemble": round(ens[t], 5),
        })

    # --- grupos ---
    groups_out = {}
    for g, members in groups.items():
        rows = []
        for t in members:
            st = grp_stats[t]
            rows.append({
                "team": t, "exp_pts": st["pts"] / n,
                "p_first": st["first"] / n, "p_second": st["second"] / n,
                "p_advance": st["advance"] / n,
            })
        rows.sort(key=lambda r: -r["exp_pts"])
        groups_out[g] = rows

    # --- bracket modal ---
    round_labels = {"R32": "Dieciseisavos", "R16": "Octavos", "QF": "Cuartos",
                    "SF": "Semifinales", "F": "Final"}
    bracket_view = []
    for rnd in ["R32", "R16", "QF", "SF", "F"]:
        ties = []
        for m in bracket["rounds"][rnd]:
            top2 = match_appear[m["id"]].most_common(2)
            if len(top2) == 2:
                (a, ca), (b, cb) = top2
                ties.append({"a": a, "pa": ca / n, "b": b, "pb": cb / n})
        bracket_view.append({"round": round_labels[rnd], "ties": ties})

    # --- edges (campeón) ---
    edges = []
    for t in teams_out:
        if t["market_implied"] is None:
            continue
        edge = t["mc"]["champion"] - t["market_implied"]
        if abs(edge) >= 0.005 and (t["mc"]["champion"] >= 0.005 or t["market_implied"] >= 0.005):
            edges.append({
                "type": "Campeón", "team": t["name"],
                "model_p": t["mc"]["champion"], "market_p": t["market_implied"],
                "edge": round(edge, 5), "market_odds": t["market_odds"],
            })
    edges.sort(key=lambda e: -abs(e["edge"]))

    methodology_html = (DATA / "methodology.html").read_text() if (DATA / "methodology.html").exists() else "<p>Pendiente.</p>"
    backtest_report = (json.loads((DATA / "backtest_report.json").read_text())
                       if (DATA / "backtest_report.json").exists() else None)

    out = {
        "model_meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "data_date": json.loads((DATA / "elo.json").read_text())["as_of_date"][:10],
            "n_sims": n,
            "market_overround": round(overround, 4),
        },
        "teams": teams_out,
        "matches": matches_out,
        "groups": groups_out,
        "bracket_view": bracket_view,
        "edges": edges[:25],
        "flags": flags,
        "methodology_html": methodology_html,
        "backtest": backtest_report,
        "accuracy": accuracy,
    }
    (WEB / "results.json").write_text(json.dumps(out, ensure_ascii=False))
    print(f"OK -> web/results.json  ({len(matches_out)} partidos, {len(teams_out)} equipos)")


if __name__ == "__main__":
    main()
