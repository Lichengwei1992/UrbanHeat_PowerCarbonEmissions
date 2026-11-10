#!/usr/bin/env python
"""
47_corrected_power_mass.py -- corrected Power-only mass-equivalent translation
=============================================================================

This is a standalone audit figure. It does not replace or modify Figure 7.

The script refits the continuous-load R3 model for Power, retaining the exact
fixed effects, pooled exposure standardisation, and two-way clustered
covariance used by 16_r3_dynamic_response.py. It then translates the fitted
log-response into CO2 mass equivalents using exp(eta) - 1.

Three quantities are kept conceptually separate:

  a. A one-day +1 pooled-SD exposure shock in every Tier A+B country, summed
     over response lags 0--14.
  b. The same cumulative response for a one-day shock equal to each country's
     own within-country-month exposure SD.
  c. The average annual mass equivalent associated with observed high-load
     anomalies. High-load days exceed one country-specific within-month SD;
     their actual anomaly magnitudes and overlapping lag responses are used.

All quantities are model-implied associations, not attributable emissions.
Only Power is translated. Industry is excluded because its daily series is
constructed from national electricity activity and is not an independent
outcome for this purpose.
"""

from __future__ import annotations

import glob
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("HC_RESULTS_ROOT", BASE / "Results_2019_2022"))
CARBON = Path(os.environ.get("HC_CARBON_DIR", BASE / "CarbonData" / "merged_2019_2022"))
EXPOSURE = BASE / "outputs" / "exposure" / "country_daily_exposure.parquet"
FIG_DIR = RESULTS / "R5" / "figures"
TAB_DIR = RESULTS / "R5" / "tables"

HEAT = "w_SCDH24"
COLD = "w_HDH15"
LEADS = 7
LAGS = 14
N_YEARS = 4.0
Z95 = 1.959963984540054

COLORS = {"heat": "#C34A36", "cold": "#2878B5"}


def load_r3_module():
    os.environ["HC_CARBON_DIR"] = str(CARBON)
    spec = importlib.util.spec_from_file_location(
        "r3_dynamic_response", BASE / "16_r3_dynamic_response.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_emissions() -> pd.DataFrame:
    rows = []
    for file in sorted(glob.glob(str(CARBON / "carbon-monitor-cities-*-v0325.csv"))):
        name = Path(file).stem.replace("carbon-monitor-cities-", "").replace("-v0325", "")
        if "all-cities" in name or "GADM" in name or "Counties" in name:
            continue
        data = pd.read_csv(file)
        data.columns = [column.strip().lower().replace("\ufeff", "") for column in data.columns]
        if "city" not in data.columns:
            continue
        data = data.rename(columns={"value (ktco2 per day)": "v"})
        data = data[data["sector"].isin(
            ["Power", "Ground Transport", "Industry", "Aviation", "Residential"]
        )]
        data["country"] = name
        rows.append(data[["country", "date", "sector", "v"]])
    emissions = pd.concat(rows, ignore_index=True)
    emissions["date"] = pd.to_datetime(emissions["date"])
    return emissions.groupby(["country", "date", "sector"], as_index=False)["v"].sum()


def build_panel() -> tuple[pd.DataFrame, dict[str, float]]:
    emissions = load_emissions()
    exposure = pd.read_parquet(EXPOSURE)
    exposure["date"] = pd.to_datetime(exposure["date"])
    exposure = exposure[exposure["tier"].isin(["A", "B"])].copy()

    panel = emissions.merge(exposure, on=["country", "date"], how="inner")
    panel = panel[panel["v"] > 0].copy()
    panel["y"] = np.log(panel["v"])

    pooled_sd = {column: float(panel[column].std()) for column in (HEAT, COLD)}
    if any(not np.isfinite(value) or value <= 0 for value in pooled_sd.values()):
        raise RuntimeError(f"Invalid pooled exposure SD: {pooled_sd}")
    return panel, pooled_sd


def refit_power(r3, panel: pd.DataFrame, pooled_sd: dict[str, float]):
    data = panel.copy()
    for column in (HEAT, COLD):
        data[column] = data[column] / pooled_sd[column]
    data = data[data["sector"] == "Power"].sort_values(["country", "date"]).reset_index(drop=True)

    names = []
    lag_values = np.arange(-LEADS, LAGS + 1)
    for tag, column in (("H", HEAT), ("C", COLD)):
        grouped = data.groupby("country")[column]
        for lag in lag_values:
            name = f"{tag}{lag:+d}"
            data[name] = grouped.shift(lag)
            names.append(name)
    data = data.dropna(subset=names + ["y"]).reset_index(drop=True)

    matrix = np.column_stack([data[names].to_numpy(dtype="float64"), data["y"].to_numpy()])
    f_cym = pd.factorize(
        data["country"] + "_" + data["date"].dt.to_period("M").astype(str)
    )[0]
    f_date = pd.factorize(data["date"])[0]
    f_cdow = pd.factorize(
        data["country"] + "_" + data["date"].dt.dayofweek.astype(str)
    )[0]
    transformed, convergence = r3.demean_factors(matrix, [f_cym, f_date, f_cdow])
    X, y = transformed[:, :-1], transformed[:, -1]
    beta = np.linalg.solve(X.T @ X + np.eye(X.shape[1]) * 1e-10, X.T @ y)
    residual = y - X @ beta
    f_country = pd.factorize(data["country"])[0]
    covariance, vcov_diag = r3.twoway_vcov(X, residual, f_country, f_date)

    nk = len(lag_values)
    post = lag_values >= 0
    blocks = {}
    for side, offset in (("heat", 0), ("cold", nk)):
        indices = np.arange(offset, offset + nk)[post]
        blocks[side] = {
            "indices": indices,
            "beta": beta[indices],
            "covariance": covariance[np.ix_(indices, indices)],
            "lags": lag_values[post],
        }

    diagnostics = {
        "n": len(data),
        "n_country": int(data["country"].nunique()),
        "r2": float(1 - residual.var() / y.var()),
        "demean_convergence": float(convergence),
        **vcov_diag,
    }
    return blocks, covariance, diagnostics


def validate_refit(blocks: dict, diagnostics: dict) -> pd.DataFrame:
    source = pd.read_csv(RESULTS / "R3" / "tables" / "T3_sectoral_response.csv")
    source = source[(source["sector"] == "Power") & (source["form"] == "continuous")]
    rows = []
    for side in ("heat", "cold"):
        published = source.loc[source["exposure"] == side].iloc[0]
        fitted = float(blocks[side]["beta"].sum())
        difference = fitted - float(published["cum"])
        rows.append({
            "side": side,
            "stored_cumulative_beta": float(published["cum"]),
            "refitted_cumulative_beta": fitted,
            "absolute_difference": abs(difference),
            "stored_n": int(published["n"]),
            "refitted_n": int(diagnostics["n"]),
        })
        if abs(difference) > 1e-10 or int(published["n"]) != int(diagnostics["n"]):
            raise AssertionError(
                f"R3 refit mismatch for {side}: beta difference={difference:.3e}, "
                f"n={diagnostics['n']} versus {int(published['n'])}"
            )
    return pd.DataFrame(rows)


def delta_interval(estimate: float, gradient: np.ndarray, covariance: np.ndarray):
    variance = float(gradient @ covariance @ gradient)
    se = float(np.sqrt(max(variance, 0.0)))
    return se, estimate - Z95 * se, estimate + Z95 * se


def one_day_translation(
    beta: np.ndarray,
    covariance: np.ndarray,
    mean_power: float,
    amplitude: float,
):
    eta = beta * amplitude
    estimate = float(mean_power * np.expm1(eta).sum())
    gradient = mean_power * amplitude * np.exp(eta)
    se, lo, hi = delta_interval(estimate, gradient, covariance)
    return estimate, se, lo, hi, gradient


def observed_translation(
    frame: pd.DataFrame,
    beta: np.ndarray,
    covariance: np.ndarray,
    exposure: str,
    pooled_sd: float,
):
    working = frame.sort_values(["country", "date"]).copy()
    working["ym"] = working["date"].dt.to_period("M").astype(str)
    working["x_month_mean"] = working.groupby(["country", "ym"])[exposure].transform("mean")
    working["x_deviation"] = working[exposure] - working["x_month_mean"]
    sigma = working.groupby("country")["x_deviation"].std()
    working["country_sigma"] = working["country"].map(sigma)
    working["high_load"] = working["x_deviation"] > working["country_sigma"]
    working["amplitude"] = np.where(
        working["high_load"], working["x_deviation"] / pooled_sd, 0.0
    )
    working["power_month_mean"] = working.groupby(["country", "ym"])["v"].transform("mean")

    country_rows = []
    total_gradient = np.zeros(len(beta))
    total_estimate = 0.0
    for country, group in working.groupby("country", sort=True):
        group = group.sort_values("date").copy()
        date_gaps = group["date"].diff().dropna().dt.days
        if len(date_gaps) and int(date_gaps.max()) != 1:
            raise AssertionError(f"Non-daily sequence for {country}")

        amplitude = group["amplitude"]
        design = np.column_stack([
            amplitude.shift(lag).fillna(0.0).to_numpy(dtype="float64")
            for lag in range(LAGS + 1)
        ])
        baseline = group["power_month_mean"].to_numpy(dtype="float64")
        eta = design @ beta
        estimate = float(np.sum(baseline * np.expm1(eta)))
        gradient = design.T @ (baseline * np.exp(eta))
        se, lo, hi = delta_interval(estimate, gradient, covariance)

        total_estimate += estimate
        total_gradient += gradient
        country_rows.append({
            "country": country,
            "country_exposure_sd": float(sigma.loc[country]),
            "n_high_load_days": int(group["high_load"].sum()),
            "mean_high_load_amplitude_pooled_sd": float(
                group.loc[group["high_load"], "amplitude"].mean()
            ),
            "mass_equivalent_4yr_kt": estimate,
            "mass_equivalent_avg_annual_kt": estimate / N_YEARS,
            "se_avg_annual_kt": se / N_YEARS,
            "ci95_lo_avg_annual_kt": lo / N_YEARS,
            "ci95_hi_avg_annual_kt": hi / N_YEARS,
        })

    total_se, total_lo, total_hi = delta_interval(
        total_estimate, total_gradient, covariance
    )
    total = {
        "estimate_kt": total_estimate / N_YEARS,
        "se_kt": total_se / N_YEARS,
        "ci95_lo_kt": total_lo / N_YEARS,
        "ci95_hi_kt": total_hi / N_YEARS,
        "gradient": total_gradient / N_YEARS,
        "n_high_load_days": int(working["high_load"].sum()),
    }
    return pd.DataFrame(country_rows), total, sigma


def spell_translation(
    frame: pd.DataFrame,
    beta: np.ndarray,
    covariance: np.ndarray,
    exposure: str,
    pooled_sd: float,
):
    """Translate one representative observed high-load spell per country.

    A spell is a contiguous run of days whose within-country-month exposure
    anomaly exceeds one country-specific SD. Each spell retains its observed
    daily amplitudes. Its response is propagated through lags 0--14 before
    spells are averaged within a country and then summed across countries.
    """
    working = frame.sort_values(["country", "date"]).copy()
    working["ym"] = working["date"].dt.to_period("M").astype(str)
    working["x_month_mean"] = working.groupby(["country", "ym"])[exposure].transform("mean")
    working["x_deviation"] = working[exposure] - working["x_month_mean"]
    sigma = working.groupby("country")["x_deviation"].std()
    working["country_sigma"] = working["country"].map(sigma)
    working["high_load"] = working["x_deviation"] > working["country_sigma"]
    working["amplitude"] = np.where(
        working["high_load"], working["x_deviation"] / pooled_sd, 0.0
    )
    working["power_month_mean"] = working.groupby(["country", "ym"])["v"].transform("mean")

    country_rows = []
    total_estimate = 0.0
    total_gradient = np.zeros(len(beta))

    for country, group in working.groupby("country", sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        high = group["high_load"].to_numpy(dtype=bool)
        starts = np.flatnonzero(high & ~np.r_[False, high[:-1]])
        ends = np.flatnonzero(high & ~np.r_[high[1:], False])
        if len(starts) == 0:
            continue

        baseline = group["power_month_mean"].to_numpy(dtype="float64")
        observed_amp = group["amplitude"].to_numpy(dtype="float64")
        spell_estimates = []
        spell_gradients = []
        spell_durations = []
        spell_mean_amplitudes = []

        for start, end in zip(starts, ends):
            impulse = np.zeros(len(group), dtype="float64")
            impulse[start:end + 1] = observed_amp[start:end + 1]
            eta = np.convolve(impulse, beta, mode="full")[:len(group)]
            estimate = float(np.sum(baseline * np.expm1(eta)))
            weighted_baseline = baseline * np.exp(eta)
            gradient = np.zeros(len(beta))
            for lag in range(len(beta)):
                valid_end = min(end + 1, len(group) - lag)
                if valid_end > start:
                    gradient[lag] = np.sum(
                        impulse[start:valid_end] *
                        weighted_baseline[start + lag:valid_end + lag]
                    )
            spell_estimates.append(estimate)
            spell_gradients.append(gradient)
            spell_durations.append(int(end - start + 1))
            spell_mean_amplitudes.append(float(observed_amp[start:end + 1].mean()))

        country_estimate = float(np.mean(spell_estimates))
        country_gradient = np.mean(np.vstack(spell_gradients), axis=0)
        se, lo, hi = delta_interval(country_estimate, country_gradient, covariance)
        total_estimate += country_estimate
        total_gradient += country_gradient
        country_rows.append({
            "country": country,
            "n_spells": len(spell_estimates),
            "mean_spell_duration_days": float(np.mean(spell_durations)),
            "mean_spell_amplitude_pooled_sd": float(np.mean(spell_mean_amplitudes)),
            "typical_spell_mass_kt": country_estimate,
            "typical_spell_se_kt": se,
            "typical_spell_ci95_lo_kt": lo,
            "typical_spell_ci95_hi_kt": hi,
        })

    total_se, total_lo, total_hi = delta_interval(
        total_estimate, total_gradient, covariance
    )
    total = {
        "scope": "one_typical_observed_high_load_spell_per_country",
        "estimate_kt": total_estimate,
        "se_kt": total_se,
        "ci95_lo_kt": total_lo,
        "ci95_hi_kt": total_hi,
        "n_high_load_days": np.nan,
        "gradient": total_gradient,
    }
    return pd.DataFrame(country_rows), total


def build_translations(panel, pooled_sd, blocks, full_covariance):
    power = panel[panel["sector"] == "Power"][
        ["country", "date", "v", HEAT, COLD]
    ].copy()
    power = power.sort_values(["country", "date"])
    mean_power = power.groupby("country")["v"].mean()

    totals = []
    country_tables = []
    spell_tables = []
    gradients = {}
    sigma_by_side = {}

    for side, exposure in (("heat", HEAT), ("cold", COLD)):
        block = blocks[side]
        beta = block["beta"]
        covariance = block["covariance"]

        pooled = one_day_translation(
            beta, covariance, float(mean_power.sum()), amplitude=1.0
        )
        totals.append({
            "scope": "simultaneous_one_day_pooled_1sd_cumulative_lag_0_14",
            "side": side,
            "estimate_kt": pooled[0],
            "se_kt": pooled[1],
            "ci95_lo_kt": pooled[2],
            "ci95_hi_kt": pooled[3],
            "n_high_load_days": np.nan,
        })
        gradients[("pooled", side)] = pooled[4]

        observed_by_country, observed_total, sigma = observed_translation(
            power, beta, covariance, exposure, pooled_sd[exposure]
        )
        sigma_by_side[side] = sigma
        observed_total.update({
            "scope": "observed_high_load_anomalies_average_annual",
            "side": side,
        })
        gradients[("observed", side)] = observed_total.pop("gradient")
        totals.append(observed_total)

        spell_by_country, spell_total = spell_translation(
            power, beta, covariance, exposure, pooled_sd[exposure]
        )
        spell_by_country["side"] = side
        spell_by_country["exposure"] = exposure
        spell_tables.append(spell_by_country)
        spell_total["side"] = side
        gradients[("spell", side)] = spell_total.pop("gradient")
        totals.append(spell_total)

        for country, mean_value in mean_power.items():
            amplitude = float(sigma.loc[country] / pooled_sd[exposure])
            own = one_day_translation(beta, covariance, float(mean_value), amplitude)
            row = observed_by_country.loc[
                observed_by_country["country"] == country
            ].iloc[0].to_dict()
            row.update({
                "side": side,
                "exposure": exposure,
                "mean_power_kt_day": float(mean_value),
                "pooled_exposure_sd": pooled_sd[exposure],
                "own_sd_in_pooled_sd_units": amplitude,
                "one_day_own_sd_mass_kt": own[0],
                "one_day_own_sd_se_kt": own[1],
                "one_day_own_sd_ci95_lo_kt": own[2],
                "one_day_own_sd_ci95_hi_kt": own[3],
            })
            country_tables.append(row)

    totals_frame = pd.DataFrame(totals)
    country_frame = pd.DataFrame(country_tables)
    spell_frame = pd.concat(spell_tables, ignore_index=True)
    return power, totals_frame, country_frame, spell_frame, gradients, sigma_by_side


def add_difference_rows(totals, gradients, blocks, full_covariance):
    rows = []
    heat_idx = blocks["heat"]["indices"]
    cold_idx = blocks["cold"]["indices"]
    for key, scope in (
        ("pooled", "simultaneous_one_day_pooled_1sd_cumulative_lag_0_14"),
        ("spell", "one_typical_observed_high_load_spell_per_country"),
        ("observed", "observed_high_load_anomalies_average_annual"),
    ):
        heat = totals[(totals["scope"] == scope) & (totals["side"] == "heat")].iloc[0]
        cold = totals[(totals["scope"] == scope) & (totals["side"] == "cold")].iloc[0]
        gradient = np.zeros(full_covariance.shape[0])
        gradient[cold_idx] = gradients[(key, "cold")]
        gradient[heat_idx] = -gradients[(key, "heat")]
        estimate = float(cold["estimate_kt"] - heat["estimate_kt"])
        se, lo, hi = delta_interval(estimate, gradient, full_covariance)
        rows.append({
            "scope": scope,
            "contrast": "cold_minus_heat",
            "estimate_kt": estimate,
            "se_kt": se,
            "ci95_lo_kt": lo,
            "ci95_hi_kt": hi,
            "cold_to_heat_ratio": float(cold["estimate_kt"] / heat["estimate_kt"]),
        })
    return pd.DataFrame(rows)


def plot_figure(totals: pd.DataFrame, countries: pd.DataFrame):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.dpi": 600,
    })

    fig, axes = plt.subplots(
        1, 3, figsize=(12.8, 4.15), gridspec_kw={"width_ratios": [0.85, 1.18, 0.85]}
    )
    ax_a, ax_b, ax_c = axes

    def interval_panel(ax, scope, scale, ylabel, title):
        data = totals[totals["scope"] == scope].set_index("side")
        x = np.arange(2)
        values = np.array([data.loc[side, "estimate_kt"] for side in ("heat", "cold")]) * scale
        low = np.array([data.loc[side, "ci95_lo_kt"] for side in ("heat", "cold")]) * scale
        high = np.array([data.loc[side, "ci95_hi_kt"] for side in ("heat", "cold")]) * scale
        colors = [COLORS[side] for side in ("heat", "cold")]
        ax.bar(x, values, width=0.58, color=colors, alpha=0.88, edgecolor="white")
        ax.errorbar(x, values, yerr=np.vstack([values - low, high - values]), fmt="none",
                    ecolor="#202020", elinewidth=1.1, capsize=3, capthick=1.1)
        ax.set_xticks(x, ["Cooling load", "Heating load"])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.axhline(0, color="#555555", lw=0.7)
        top = max(high.max(), values.max())
        for xi, value in zip(x, values):
            ax.text(xi, value + top * 0.035, f"{value:,.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(min(0, low.min() * 1.1), top * 1.20)

    interval_panel(
        ax_a,
        "simultaneous_one_day_pooled_1sd_cumulative_lag_0_14",
        1.0,
        "Power CO$_2$ mass equivalent (kt)",
        "a  One-day +1 pooled-SD shock",
    )
    ax_a.text(0.02, 0.98, "cumulative response, lags 0-14",
              transform=ax_a.transAxes, va="top", ha="left", fontsize=8, color="#555555")

    wide = countries.pivot(index="country", columns="side", values="one_day_own_sd_mass_kt").dropna()
    ax_b.scatter(wide["heat"], wide["cold"], s=31, color="#59636E",
                 alpha=0.82, edgecolor="white", linewidth=0.45)
    limit = float(max(wide.max().max(), 1.0) * 1.08)
    ax_b.plot([0, limit], [0, limit], ls="--", lw=0.9, color="#888888")
    label_countries = set(wide.max(axis=1).nlargest(6).index)
    for country in label_countries:
        ax_b.annotate(country, (wide.loc[country, "heat"], wide.loc[country, "cold"]),
                      xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax_b.set_xlim(0, limit)
    ax_b.set_ylim(0, limit)
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.set_xlabel("Cooling-load mass equivalent (kt)")
    ax_b.set_ylabel("Heating-load mass equivalent (kt)")
    ax_b.set_title("b  Country-own-SD shock", loc="left", fontweight="bold")
    ax_b.text(0.98, 0.035, "pooled sensitivity; country-specific exposure scale",
              transform=ax_b.transAxes, va="bottom", ha="right", fontsize=8, color="#555555")
    ax_b.spines[["top", "right"]].set_visible(False)

    interval_panel(
        ax_c,
        "observed_high_load_anomalies_average_annual",
        1e-3,
        "Power CO$_2$ mass equivalent (Mt yr$^{-1}$)",
        "c  Observed high-load anomalies",
    )
    ax_c.text(0.02, 0.98, "actual magnitudes; average, 2019-2022",
              transform=ax_c.transAxes, va="top", ha="left", fontsize=8, color="#555555")

    for ax in axes:
        ax.grid(axis="y", color="#D9D9D9", lw=0.55, alpha=0.7)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}" if abs(value) >= 10 else f"{value:,.1f}"))

    fig.suptitle(
        "Power-sector CO$_2$ mass-equivalent translation of thermal-load responses",
        x=0.06, y=1.01, ha="left", fontsize=12, fontweight="bold"
    )
    fig.text(
        0.5, -0.015,
        "Tier A+B countries. Exact exp(eta)-1 conversion; 95% CIs use the full two-way clustered "
        "lag covariance. Model-implied associations, not attributable emissions.",
        ha="center", va="top", fontsize=8, color="#444444"
    )
    fig.tight_layout(w_pad=2.0)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    stem = FIG_DIR / "Fig_new_power_mass_translation"
    for extension in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{extension}"), bbox_inches="tight", dpi=600)
    plt.close(fig)
    return stem


def plot_three_scope_figure(totals: pd.DataFrame):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.dpi": 600,
    })

    specifications = [
        (
            "simultaneous_one_day_pooled_1sd_cumulative_lag_0_14",
            1.0,
            "Power CO$_2$ mass equivalent (kt)",
            "a  One-day +1 pooled-SD shock",
            "cumulative response, lags 0-14",
        ),
        (
            "one_typical_observed_high_load_spell_per_country",
            1.0,
            "Power CO$_2$ mass equivalent (kt)",
            "b  One typical high-load spell",
            "observed duration and daily magnitudes",
        ),
        (
            "observed_high_load_anomalies_average_annual",
            1e-3,
            "Power CO$_2$ mass equivalent (Mt yr$^{-1}$)",
            "c  Observed annual load",
            "actual magnitudes; average, 2019-2022",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11.9, 4.1))
    for ax, (scope, scale, ylabel, title, subtitle) in zip(axes, specifications):
        data = totals[totals["scope"] == scope].set_index("side")
        sides = ("heat", "cold")
        x = np.arange(2)
        values = np.array([data.loc[side, "estimate_kt"] for side in sides]) * scale
        low = np.array([data.loc[side, "ci95_lo_kt"] for side in sides]) * scale
        high = np.array([data.loc[side, "ci95_hi_kt"] for side in sides]) * scale
        ax.bar(x, values, width=0.58, color=[COLORS[side] for side in sides],
               alpha=0.88, edgecolor="white")
        ax.errorbar(x, values, yerr=np.vstack([values - low, high - values]),
                    fmt="none", ecolor="#202020", elinewidth=1.1,
                    capsize=3, capthick=1.1)
        ax.set_xticks(x, ["Cooling load", "Heating load"])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.text(0.02, 0.98, subtitle, transform=ax.transAxes,
                va="top", ha="left", fontsize=8, color="#555555")
        ax.spines[["top", "right"]].set_visible(False)
        ax.axhline(0, color="#555555", lw=0.7)
        ax.grid(axis="y", color="#D9D9D9", lw=0.55, alpha=0.7)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda value, _: f"{value:,.0f}" if abs(value) >= 10 else f"{value:,.1f}"
        ))
        top = max(high.max(), values.max())
        for xi, value in zip(x, values):
            ax.text(xi, value + top * 0.035, f"{value:,.2f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_ylim(min(0, low.min() * 1.1), top * 1.20)

    fig.suptitle(
        "Power-sector CO$_2$ mass equivalents across three thermal-load scopes",
        x=0.065, y=1.01, ha="left", fontsize=12, fontweight="bold"
    )
    fig.text(
        0.5, -0.015,
        "Tier A+B countries. Exact exp(eta)-1 conversion; 95% CIs use the full two-way clustered "
        "lag covariance. Model-implied associations, not attributable emissions.",
        ha="center", va="top", fontsize=8, color="#444444"
    )
    fig.tight_layout(w_pad=2.0)

    stem = FIG_DIR / "Fig_new_power_mass_three_scopes"
    for extension in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{extension}"), bbox_inches="tight", dpi=600)
    plt.close(fig)
    return stem


def plot_compact_panel_f(totals: pd.DataFrame):
    """Compact three-scope panel designed to occupy the Figure 7f slot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.dpi": 600,
    })

    specifications = [
        (
            "simultaneous_one_day_pooled_1sd_cumulative_lag_0_14",
            1.0,
            "kt CO$_2$",
            "One-day +1\npooled-SD shock",
        ),
        (
            "one_typical_observed_high_load_spell_per_country",
            1.0,
            "kt CO$_2$",
            "One typical\nhigh-load spell",
        ),
        (
            "observed_high_load_anomalies_average_annual",
            1e-3,
            "Mt CO$_2$ yr$^{-1}$",
            "Observed annual\nhigh load",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(6.45, 3.45))
    for ax, (scope, scale, ylabel, xlabel) in zip(axes, specifications):
        data = totals[totals["scope"] == scope].set_index("side")
        sides = ("cold", "heat")
        x = np.arange(2)
        values = np.array([data.loc[side, "estimate_kt"] for side in sides]) * scale
        low = np.array([data.loc[side, "ci95_lo_kt"] for side in sides]) * scale
        high = np.array([data.loc[side, "ci95_hi_kt"] for side in sides]) * scale

        ax.bar(
            x, values, width=0.64,
            color=[COLORS["cold"], COLORS["heat"]],
            alpha=0.90, edgecolor="white", linewidth=0.6,
        )
        ax.errorbar(
            x, values, yerr=np.vstack([values - low, high - values]),
            fmt="none", ecolor="#202020", elinewidth=0.9,
            capsize=2.5, capthick=0.9,
        )
        ax.set_xticks(x, ["Heating\nload", "Cooling\nload"])
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel, labelpad=3)
        ratio = float(data.loc["cold", "estimate_kt"] / data.loc["heat", "estimate_kt"])
        ax.set_title(
            rf"Heating $\times${ratio:.2f}", loc="left", pad=5, fontweight="bold"
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.axhline(0, color="#555555", lw=0.6)
        ax.grid(axis="y", color="#D9D9D9", lw=0.45, alpha=0.65)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda value, _: f"{value:,.0f}" if abs(value) >= 10 else f"{value:,.1f}"
        ))

        top = max(high.max(), values.max())
        for xi, value, side in zip(x, values, sides):
            decimals = 2 if scale == 1e-3 else 0
            ax.text(
                xi, value + top * 0.035, f"{value:,.{decimals}f}",
                ha="center", va="bottom", fontsize=7,
                color=COLORS[side], fontweight="bold",
            )
        ax.set_ylim(min(0, low.min() * 1.08), top * 1.19)

    fig.suptitle(
        "(f) Power CO$_2$ mass equivalents at three temporal scopes",
        x=0.06, y=0.995, ha="left", fontsize=9.5, fontweight="bold",
    )
    fig.text(
        0.5, 0.012,
        "Power only; exact exp(eta)-1 conversion; 95% CI. "
        "Model-implied associations, not attributable emissions.",
        ha="center", va="bottom", fontsize=6.6, color="#4D4D4D",
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.83, bottom=0.27, wspace=0.63)

    stem = FIG_DIR / "Fig_new_power_mass_three_scopes_compact_f"
    for extension in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{extension}"), bbox_inches="tight", dpi=600)
    plt.close(fig)
    return stem


def write_caption():
    caption = (
        "Standalone candidate figure | Power-sector CO2 mass-equivalent translation. "
        "a, Aggregate cumulative mass equivalent across lags 0-14 following a one-day "
        "+1 pooled-SD increase in population-weighted sensible cooling degree-hours "
        "(cooling load) or heating degree-hours (heating load) in every Tier A+B country. "
        "b, Country-level cumulative mass equivalents for a one-day shock equal to each "
        "country's own within-country-month exposure SD; coefficients remain pooled, while "
        "exposure scales and mean Power emissions vary by country. The dashed line denotes "
        "equal cooling- and heating-side mass. c, Average annual mass equivalents associated "
        "with observed high-load anomalies in 2019-2022. High-load days exceed one "
        "country-specific within-month SD, and the calculation uses their actual magnitudes "
        "and overlapping lag responses. Bars and points use exp(eta)-1; error bars are 95% "
        "delta-method confidence intervals based on the full two-way clustered covariance "
        "of lags 0-14. These are model-implied mass equivalents, not causal or attributable "
        "emissions. Industry is excluded because its daily activity profile is not "
        "independent of national electricity activity."
    )
    path = FIG_DIR / "Fig_new_power_mass_translation_caption.txt"
    path.write_text(caption, encoding="utf-8")
    return path


def write_three_scope_caption():
    caption = (
        "Standalone three-scope candidate figure | Power-sector CO2 mass-equivalent "
        "translation. a, Aggregate cumulative response across lags 0-14 following a "
        "one-day +1 pooled-SD increase in cooling or heating load in every Tier A+B "
        "country. b, Aggregate response to one representative observed high-load spell "
        "per country. Spells are contiguous days whose within-country-month load anomaly "
        "exceeds one country-specific SD; observed durations and daily anomaly magnitudes "
        "are retained and spells are averaged within each country. c, Average annual mass "
        "equivalent associated with all observed high-load anomalies in 2019-2022, using "
        "their actual magnitudes and overlapping lag responses. Bars use exp(eta)-1; error "
        "bars are 95% delta-method confidence intervals from the full two-way clustered "
        "lag covariance. Values are model-implied mass equivalents, not causal or "
        "attributable emissions. Only Power is translated."
    )
    path = FIG_DIR / "Fig_new_power_mass_three_scopes_caption.txt"
    path.write_text(caption, encoding="utf-8")
    return path


def write_compact_panel_f_caption():
    caption = (
        "(f) Power-sector CO2 mass equivalents at three thermal-load scopes. "
        "From left to right: the cumulative response across lags 0-14 to a "
        "one-day +1 pooled-SD shock in every Tier A+B country; one representative "
        "observed high-load spell per country, retaining observed duration and "
        "daily magnitudes; and the average annual mass equivalent associated with "
        "all observed high-load anomalies in 2019-2022. Bars use exp(eta)-1 and "
        "error bars show 95% delta-method confidence intervals from the full "
        "two-way clustered lag covariance. Values are model-implied associations, "
        "not causal or attributable emissions."
    )
    path = FIG_DIR / "Fig_new_power_mass_three_scopes_compact_f_caption.txt"
    path.write_text(caption, encoding="utf-8")
    return path


def main():
    print("=" * 78)
    print("Corrected Power-only CO2 mass-equivalent translation")
    print(f"Results root : {RESULTS}")
    print(f"Carbon data : {CARBON}")
    print("=" * 78)

    r3 = load_r3_module()
    panel, pooled_sd = build_panel()
    blocks, full_covariance, diagnostics = refit_power(r3, panel, pooled_sd)
    validation = validate_refit(blocks, diagnostics)

    _, totals, countries, spells, gradients, _ = build_translations(
        panel, pooled_sd, blocks, full_covariance
    )
    contrasts = add_difference_rows(
        totals, gradients, blocks, full_covariance
    )

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    validation_path = TAB_DIR / "T22_corrected_power_mass_refit_validation.csv"
    totals_path = TAB_DIR / "T22b_corrected_power_mass_totals.csv"
    country_path = TAB_DIR / "T22c_corrected_power_mass_by_country.csv"
    contrast_path = TAB_DIR / "T22d_corrected_power_mass_contrasts.csv"
    spell_path = TAB_DIR / "T22e_corrected_power_mass_spells_by_country.csv"
    validation.to_csv(validation_path, index=False)
    totals.to_csv(totals_path, index=False)
    countries.to_csv(country_path, index=False)
    contrasts.to_csv(contrast_path, index=False)
    spells.to_csv(spell_path, index=False)

    figure_stem = plot_figure(totals, countries)
    caption_path = write_caption()
    three_scope_stem = plot_three_scope_figure(totals)
    three_scope_caption = write_three_scope_caption()
    compact_stem = plot_compact_panel_f(totals)
    compact_caption = write_compact_panel_f_caption()

    print("\nRefit validation")
    print(validation.to_string(index=False))
    print("\nMass-equivalent totals")
    print(totals[["scope", "side", "estimate_kt", "se_kt", "ci95_lo_kt", "ci95_hi_kt"]].to_string(index=False))
    print("\nCold-minus-heat contrasts")
    print(contrasts.to_string(index=False))
    print("\nWrote")
    for path in (
        validation_path, totals_path, country_path, contrast_path, spell_path,
        figure_stem.with_suffix(".png"), figure_stem.with_suffix(".pdf"),
        figure_stem.with_suffix(".svg"), caption_path,
        three_scope_stem.with_suffix(".png"), three_scope_stem.with_suffix(".pdf"),
        three_scope_stem.with_suffix(".svg"), three_scope_caption,
        compact_stem.with_suffix(".png"), compact_stem.with_suffix(".pdf"),
        compact_stem.with_suffix(".svg"), compact_caption,
    ):
        print(f"  {path}")


if __name__ == "__main__":
    main()
