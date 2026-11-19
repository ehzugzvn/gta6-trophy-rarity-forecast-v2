#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster - "Saturating Share" variant.

For every trophy the earn rate is modelled as a saturating share of the
player base:

    rate_i(t) = A_i * (1 - exp(-k_i * age_days)) + c_i

where c_i is the first rate of the active regime and age is measured in
days from that first active read.  The curve is fitted per trophy by a
grid search over k_i with a least-squares A_i, floored so the implied
earner count never shrinks, and the story chain (trophy ids 3..13) is
forced non-increasing.  Quantiles are logit-normal around the median.

Reads every *.in case in $MAKEFUN_INPUT_DIR (default /input) and writes
<id>.out JSON files to $MAKEFUN_OUTPUT_DIR (default /output).
Standard library only, fully deterministic.
"""

import glob
import json
import math
import os
from datetime import datetime, timezone

LAUNCH = datetime(2026, 11, 19, tzinfo=timezone.utc)
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)  # standard normal quantiles
EPS = 1e-4


def parse_time(value):
    """ISO timestamp -> hours since launch (2026-11-19T00:00:00Z)."""
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt - LAUNCH).total_seconds() / 3600.0


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit(p):
    p = clamp(p, EPS, 1.0 - EPS)
    return math.log(p / (1.0 - p))


def logistic(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def median_of(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def read_series(case):
    """Extract (trophy ids, read times in hours, players, rates per trophy)."""
    trophies = case.get("trophies") or []
    ids = [t.get("id") for t in trophies]
    times, players = [], []
    rates = [[] for _ in ids]
    for read in case.get("history") or []:
        t = parse_time(read.get("at"))
        if t is None:
            t = parse_time(read.get("slot"))
        if t is None:
            continue
        row = read.get("rates") or []
        times.append(t)
        players.append(read.get("players_tracked"))
        for j in range(len(ids)):
            rates[j].append(row[j] if j < len(row) else None)
    return ids, times, players, rates


def active_start(times, players, rates):
    """Index of the first read of the active regime.

    A regime break sits between read k-1 and read k when the tracked
    player count jumps by more than 25% or the median absolute logit
    move across trophies exceeds 0.35.  The active regime is everything
    after the LAST break (a frozen placeholder plateau has no breaks,
    so the whole history stays active).
    """
    start = 0
    for k in range(1, len(times)):
        brk = False
        p0, p1 = players[k - 1], players[k]
        if p0 and p1 and p0 > 0 and abs(p1 - p0) / p0 > 0.25:
            brk = True
        if not brk:
            diffs = []
            for series in rates:
                a, b = series[k - 1], series[k]
                if a is not None and b is not None:
                    diffs.append(abs(logit(b / 100.0) - logit(a / 100.0)))
            if diffs and median_of(diffs) > 0.35:
                brk = True
        if brk:
            start = k
    return start


def players_at(times, players, start, t_target):
    """Forecast tracked players at t_target.

    OLS of ln(players) on ln(1 + hours since launch) over the active
    reads, slope damped by 0.75, result floored at the last observed
    player count and capped at 60x it.
    """
    last = None
    for k in range(len(times) - 1, -1, -1):
        if players[k]:
            last = float(players[k])
            break
    if last is None:
        return 1.0
    xs, ys = [], []
    for k in range(start, len(times)):
        p = players[k]
        if p and p > 0:
            xs.append(math.log1p(max(times[k], 0.0)))
            ys.append(math.log(p))
    if len(xs) < 2:
        return last
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return last
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    slope *= 0.75  # damped growth
    intercept = my - slope * mx
    try:
        pred = math.exp(intercept + slope * math.log1p(max(t_target, 0.0)))
    except OverflowError:
        # Extreme histories can overflow exp(); the cap is the forecast.
        return 60.0 * last
    return clamp(pred, last, 60.0 * last)


def fallback_sigma(horizon_days):
    """Spread of the persistence fallback (house baseline shape)."""
    return 0.5 * math.sqrt(max(horizon_days, 0.0))


def fit_saturating(ages, rates_obs, last_rate):
    """Grid-search fit of rate(age) = A*(1 - exp(-k*age)) + c.

    ages are days since the first active read, c is the first observed
    rate.  k runs over 24 log-spaced values in [0.02, 2.5] per day; for
    each k the amplitude A (the asymptote, c included) is the least
    squares fit clamped to [last_rate, 100].  The k with the lowest
    squared error on the rate scale wins.

    Returns (k, amp, c, rmse_logit) or None when no k is fittable.
    """
    c = rates_obs[0]
    ys = [r - c for r in rates_obs]
    lo_k, hi_k = math.log(0.02), math.log(2.5)
    best = None
    for step in range(24):
        k = math.exp(lo_k + (hi_k - lo_k) * step / 23.0)
        xs = [1.0 - math.exp(-k * a) for a in ages]
        sxx = sum(x * x for x in xs)
        if sxx <= 0:
            continue
        amp = sum(x * y for x, y in zip(xs, ys)) / sxx + c
        amp = clamp(amp, last_rate, 100.0)
        off = amp - c
        err = sum((off * x - y) ** 2 for x, y in zip(xs, ys))
        if best is None or err < best[0]:
            best = (err, k, amp)
    if best is None:
        return None
    _, k, amp = best
    off = amp - c
    sq = 0.0
    for a, r in zip(ages, rates_obs):
        pred = clamp(off * (1.0 - math.exp(-k * a)) + c, 0.01, 99.99)
        sq += (logit(pred / 100.0) - logit(r / 100.0)) ** 2
    rmse_logit = math.sqrt(sq / len(rates_obs))
    return k, amp, c, rmse_logit


def forecast_trophy(times, players, series, start, t_target,
                    players_target, horizon_days):
    """Median + sigma for one trophy, or None to use the fallback."""
    obs = [(times[k], series[k])
           for k in range(start, len(times)) if series[k] is not None]
    if len(obs) < 3:
        return None
    last_rate = obs[-1][1]

    # Earners cannot un-earn: running maximum of rate * players / 100.
    e_last = 0.0
    for k in range(start, len(times)):
        r = series[k]
        if r is not None and players[k]:
            e_last = max(e_last, r * players[k] / 100.0)

    t0 = obs[0][0]
    ages = [(t - t0) / 24.0 for t, _ in obs]
    rates_obs = [r for _, r in obs]
    fit = fit_saturating(ages, rates_obs, last_rate)
    if fit is None:
        return None
    k, amp, c, rmse_logit = fit

    age_target = max((t_target - t0) / 24.0, 0.0)
    share = (amp - c) * (1.0 - math.exp(-k * age_target)) + c
    floor = 100.0 * e_last / players_target if players_target > 0 else 0.0
    med = clamp(max(share, floor), 0.01, 99.99)

    sigma = clamp(0.28 + 2.2 * rmse_logit, 0.30, 1.0)
    sigma *= 0.6 + 0.4 * math.sqrt(max(horizon_days, 0.0) / 4.52)
    return med, sigma


def enforce_chain(medians, ids):
    """Pool-adjacent-violators: medians along trophy ids 3..13 must be
    non-increasing (later story chapters cannot be earned by a larger
    share of players than earlier ones)."""
    pos = {tid: j for j, tid in enumerate(ids)}
    chain = [pos[i] for i in range(3, 14) if i in pos]
    if len(chain) < 2:
        return
    blocks = []  # [sum, count] stack of pooled blocks
    for j in chain:
        blocks.append([medians[j], 1])
        while (len(blocks) >= 2
               and blocks[-2][0] / blocks[-2][1]
               < blocks[-1][0] / blocks[-1][1]):
            s, cnt = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += cnt
    flat = []
    for s, cnt in blocks:
        flat.extend([s / cnt] * cnt)
    for j, v in zip(chain, flat):
        medians[j] = v


def quantile_row(med, sigma):
    base = logit(med / 100.0)
    row = [clamp(100.0 * logistic(base + z * sigma), 0.01, 99.99)
           for z in Z]
    for i in range(1, len(row)):
        if row[i] < row[i - 1]:
            row[i] = row[i - 1]
    return [round(v, 4) for v in row]


def solve_case(case):
    ids, times, players, rates = read_series(case)
    t_cut = parse_time(case.get("at"))
    t_target = parse_time(case.get("target"))
    if t_cut is None:
        t_cut = times[-1] if times else 0.0
    if t_target is None:
        t_target = t_cut + (case.get("horizon_minutes") or 0) / 60.0
    horizon_days = max((t_target - t_cut) / 24.0, 0.0)

    start = active_start(times, players, rates) if times else 0
    p_target = players_at(times, players, start, t_target) if times else 1.0

    medians, sigmas = [], []
    for j in range(len(ids)):
        series = rates[j]
        last = 0.0
        for k in range(len(series) - 1, -1, -1):
            if series[k] is not None:
                last = series[k]
                break
        res = None
        if times:
            try:
                res = forecast_trophy(times, players, series, start,
                                      t_target, p_target, horizon_days)
            except (ValueError, OverflowError, ZeroDivisionError):
                res = None
        if res is None:
            # Persistence fallback (house baseline shape).
            medians.append(clamp(last, 0.01, 99.99))
            sigmas.append(fallback_sigma(horizon_days))
        else:
            medians.append(res[0])
            sigmas.append(res[1])
        if last >= 99.9:
            # Essentially everyone has it; it can only stay ~full.
            medians[-1] = clamp(last, 0.01, 99.99)
            sigmas[-1] = 0.05

    enforce_chain(medians, ids)

    out = []
    for j, tid in enumerate(ids):
        out.append({"id": tid,
                    "q": quantile_row(medians[j], sigmas[j])})
    return {"trophies": out}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(in_dir, "*.in"))):
        with open(path, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = solve_case(case)
        out_path = os.path.join(out_dir, str(case.get("id")) + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))


if __name__ == "__main__":
    main()
