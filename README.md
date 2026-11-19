# GTA VI Trophy Rarity Forecaster — Saturating Share

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ehzugzvn/gta6-trophy-rarity-forecast-v2/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ehzugzvn/gta6-trophy-rarity-forecast-v2/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Entry for the make.fun bounty
[gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
forecast, for each of the 55 GTA VI trophies, the share of players who
will have earned it by the target time, as five quantiles
(5/25/50/75/95) of the earn rate.

## Approach — saturating share curves

`main.py` is a single-file, standard-library-only Python program. For
every case in `$MAKEFUN_INPUT_DIR` it writes one `<id>.out` JSON file to
`$MAKEFUN_OUTPUT_DIR`.

1. **Regime detection.** The read history is scanned for breaks: a jump
   of more than 25% in tracked players, or a median absolute logit move
   across trophies above 0.35. Only reads after the last break (the
   active regime) drive the fit, so a stale placeholder plateau never
   contaminates the real series.
2. **Player growth.** Tracked players are extrapolated with an OLS fit
   of ln(players) on ln(1 + hours since launch), slope damped by 0.75,
   floored at the last observed count and capped at 60× it.
3. **Saturating share per trophy.** Each trophy's rate is modelled as
   `rate(t) = A·(1 − exp(−k·age)) + c`, where `c` is the first active
   rate and age is days since the first active read. `k` is chosen by a
   24-point log-spaced grid search over [0.02, 2.5] per day; for each
   `k`, `A` is the least-squares asymptote clamped to
   [last rate, 100], and the `k` with the lowest squared error on the
   rate scale wins. The median forecast is floored by the earners rule —
   the implied earner count (rate × players, running maximum over the
   regime) can never shrink.
4. **Story-chain constraint.** Trophies 3–13 are story chapters in
   order, so their median rates are forced non-increasing with a
   pool-adjacent-violators pass.
5. **Quantiles.** Logit-normal around the median:
   `q_τ = 100 · logistic(logit(median) + z_τ · σ)`, with
   `σ = clamp(0.28 + 2.2 · rmse_logit, 0.30, 1.0) · (0.6 + 0.4·√(horizon_days/4.52))`
   driven by the fit's residual RMSE on the logit scale. Trophies with
   too little usable history fall back to a persistence row (the house
   baseline shape), and a trophy already at ≥99.9% is pinned there with
   a narrow σ of 0.05.

The program is deterministic, uses no network and no randomness, and
runs each case in well under a second.

## Layout

- `main.py` — the forecaster (the entry program).
- `.makefun/owner.json` — make.fun payout wallet binding.
- `LICENSE` — MIT.

## License

MIT — see `LICENSE`.
