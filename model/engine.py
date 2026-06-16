"""
Motor de predicción Mundial FIFA 2026.

Modelo híbrido: Elo -> goles esperados (doble Poisson) con corrección
Dixon-Coles para marcadores bajos, ventaja de localía para anfitriones,
y modificador de forma opcional.

Calibración documentada (ver README_METODOLOGIA en el Excel):
- P(victoria) desde Elo: expectativa estándar  We = 1/(1+10^(-dr/400)).
- Diferencia de gol esperada: E[GD] = GD_SLOPE * dr.
  Regresiones públicas sobre partidos internacionales sitúan la pendiente
  en ~0.25-0.30 goles por cada 100 puntos Elo. Usamos 0.00275/punto (centro
  del rango) y validamos contra odds de mercado en la fase de validación.
- Total de goles esperado: media histórica de Mundiales recientes
  (2014: 2.67, 2018: 2.64, 2022: 2.69) => BASE_TOTAL = 2.66, con un
  incremento suave en mismatches grandes (los partidos muy desiguales
  históricamente producen más goles totales).
- Dixon-Coles rho para corregir la dependencia en marcadores 0-0/1-0/0-1/1-1.
- Localía: +HOME_ELO_BONUS puntos Elo solo para México/USA/Canadá en su país.
  Elo estándar usa +100 para localía plena; para co-anfitriones de un torneo
  con sedes múltiples usamos un valor reducido.
"""

from __future__ import annotations
import numpy as np

# Calibración empírica (ver model/backtest.py y data/backtest_report.json):
# regresión GD ~ ΔElo en fases finales de torneos continentales 2006-2025
# (n=1,345; entorno comparable a un Mundial) da 0.004987 ± 0.0002 en la escala
# de la réplica Elo, equivalente a 0.00444 en la escala del Elo oficial
# (factor 1.1236 medido). El grid de sensibilidad sobre los Mundiales
# 2014/18/22 (test) confirma el óptimo en ~0.0045: Brier 0.585 vs 0.597 del
# valor anterior 0.00275 tomado de literatura.
GD_SLOPE = 0.00444          # goles de diferencia esperados por punto Elo
BASE_TOTAL = 2.66           # goles totales esperados en partido parejo
MISMATCH_TOTAL_GAIN = 0.45  # goles extra de total cuando |dr| es enorme (saturación)
MISMATCH_SCALE = 350.0      # escala Elo para el efecto anterior
DC_RHO = -0.10              # parámetro Dixon-Coles (valor típico en literatura)
HOME_ELO_BONUS = 65.0       # bonus Elo para anfitrión jugando en su país
MIN_LAMBDA = 0.18           # piso de goles esperados
MAX_GOALS = 10              # truncamiento de la matriz de marcadores

HOSTS = {"Mexico", "United States", "Canada"}


def _g_mult(gd: int) -> float:
    """Multiplicador por margen de gol (eloratings.net)."""
    gd = abs(gd)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0


def apply_match_elo(R: dict, home: str, away: str, hg: int, ag: int,
                    host_a: bool = False, host_b: bool = False, k: float = 60.0) -> None:
    """Actualiza in-place el dict de Elo R tras un partido real (fórmula
    eloratings.net). K=60 es el valor estándar para fase final de Mundial.
    La localía usa el mismo HOME_ELO_BONUS que el modelo (solo anfitriones)."""
    ha = HOME_ELO_BONUS if host_a else 0.0
    hb = HOME_ELO_BONUS if host_b else 0.0
    dr = (R[home] + ha) - (R[away] + hb)
    we = 1.0 / (1.0 + 10 ** (-dr / 400.0))
    w = 1.0 if hg > ag else (0.0 if ag > hg else 0.5)
    delta = k * _g_mult(hg - ag) * (w - we)
    R[home] += delta
    R[away] -= delta


def expected_lambdas(elo_a: float, elo_b: float,
                     form_a: float = 0.0, form_b: float = 0.0,
                     host_a: bool = False, host_b: bool = False,
                     knockout: bool = False) -> tuple[float, float]:
    """Goles esperados (lambda_a, lambda_b) a partir de Elo.

    form_*: modificador aditivo en puntos Elo (acotado fuera de aquí).
    knockout: los partidos eliminatorios son ligeramente más cerrados
    (total observado en KO 90' de Mundiales ~ -0.2 goles vs grupos).
    """
    ra = elo_a + (HOME_ELO_BONUS if host_a else 0.0) + form_a
    rb = elo_b + (HOME_ELO_BONUS if host_b else 0.0) + form_b
    dr = ra - rb

    total = BASE_TOTAL + MISMATCH_TOTAL_GAIN * (1.0 - np.exp(-abs(dr) / MISMATCH_SCALE))
    if knockout:
        total -= 0.20

    gd = GD_SLOPE * dr
    lam_a = max(MIN_LAMBDA, (total + gd) / 2.0)
    lam_b = max(MIN_LAMBDA, (total - gd) / 2.0)
    return lam_a, lam_b


def _poisson_pmf_vec(lam: float, n: int = MAX_GOALS) -> np.ndarray:
    ks = np.arange(n + 1)
    log_pmf = ks * np.log(lam) - lam - np.array([np.sum(np.log(np.arange(1, k + 1))) if k > 0 else 0.0 for k in ks])
    return np.exp(log_pmf)


def score_matrix(lam_a: float, lam_b: float, rho: float = DC_RHO) -> np.ndarray:
    """Matriz P(score = (i, j)) con corrección Dixon-Coles, normalizada."""
    pa = _poisson_pmf_vec(lam_a)
    pb = _poisson_pmf_vec(lam_b)
    m = np.outer(pa, pb)
    # Corrección Dixon-Coles en la esquina de marcadores bajos
    tau = np.ones_like(m)
    tau[0, 0] = 1 - lam_a * lam_b * rho
    tau[0, 1] = 1 + lam_a * rho
    tau[1, 0] = 1 + lam_b * rho
    tau[1, 1] = 1 - rho
    m = m * tau
    m = np.clip(m, 0, None)
    return m / m.sum()


def match_probs(m: np.ndarray) -> dict:
    """Probabilidades 1X2, over/under 2.5, BTTS y top marcadores."""
    home = np.tril(m, -1).sum()   # i > j
    away = np.triu(m, 1).sum()    # j > i
    draw = np.trace(m)
    idx = np.arange(m.shape[0])
    totals = idx[:, None] + idx[None, :]
    over25 = m[totals >= 3].sum()
    btts = m[1:, 1:].sum()
    flat = [((i, j), m[i, j]) for i in idx for j in idx]
    flat.sort(key=lambda x: -x[1])
    top5 = [(f"{i}-{j}", float(p)) for (i, j), p in flat[:5]]
    return {
        "p_home": float(home), "p_draw": float(draw), "p_away": float(away),
        "over25": float(over25), "under25": float(1 - over25), "btts": float(btts),
        "top_scores": top5,
    }


# ----------------------------------------------------------------------
# Simulación

def sim_score(rng: np.random.Generator, lam_a: float, lam_b: float) -> tuple[int, int]:
    """Muestrea un marcador del doble Poisson con corrección DC vía rechazo simple."""
    # Muestreo directo de la matriz sería exacto pero lento por partido único;
    # usamos Poisson independiente y aplicamos DC con un paso de rechazo acotado.
    for _ in range(4):
        ga = rng.poisson(lam_a)
        gb = rng.poisson(lam_b)
        if ga <= 1 and gb <= 1:
            tau = {(0, 0): 1 - lam_a * lam_b * DC_RHO,
                   (0, 1): 1 + lam_a * DC_RHO,
                   (1, 0): 1 + lam_b * DC_RHO,
                   (1, 1): 1 - DC_RHO}[(ga, gb)]
            if rng.random() < min(1.0, tau):
                return ga, gb
            continue
        return ga, gb
    return ga, gb


def sim_knockout_winner(rng: np.random.Generator, elo_a: float, elo_b: float,
                        lam_a: float, lam_b: float) -> tuple[bool, tuple[int, int], str]:
    """Simula un partido KO completo. Devuelve (gana_a, marcador_90, vía)."""
    ga, gb = sim_score(rng, lam_a, lam_b)
    if ga != gb:
        return ga > gb, (ga, gb), "90min"
    # Prórroga: 30 min => lambdas / 3
    ea = rng.poisson(lam_a / 3.0)
    eb = rng.poisson(lam_b / 3.0)
    if ea != eb:
        return ea > eb, (ga + ea, gb + eb), "ET"
    # Penales: leve ventaja por Elo (evidencia empírica débil => casi 50/50)
    p_a = 1.0 / (1.0 + 10 ** (-(elo_a - elo_b) / 2000.0))  # muy plano
    return rng.random() < p_a, (ga + ea, gb + eb), "PEN"
