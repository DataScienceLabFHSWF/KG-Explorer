"""
Zipf's Law Validation for the Fusion Knowledge Graph
=====================================================
Validates whether entity mention frequencies follow Zipf's law:

    f(r) = C / r^alpha

where *r* is the rank, *f* the frequency, and alpha ~ 1 for natural
language.  A good fit suggests the KG captures the natural distribution
of concepts in the scientific literature.

Outputs
-------
  - 15_zipf_law.png          — log-log rank-frequency plot with fit line
  - 16_zipf_deviation.png    — deviation from ideal Zipf (top/bottom entities)
  - zipf_stats.json          — alpha, xmin, KS statistic, p-value

Computational Complexity
------------------------
  - Frequency counting: O(n)  via Cypher aggregation
  - Power-law fit:      O(n)  via MLE (Clauset et al., 2009)

Sampling Rationale
------------------
  All entities are used (no sampling); the power-law fit runs on the
  full empirical distribution.

References
----------
  Clauset, A., Shalizi, C. R. & Newman, M. E. J. (2009).
    "Power-law distributions in empirical data." SIAM Review, 51(4), 661.
  Zipf, G. K. (1949). Human Behavior and the Principle of Least Effort.
  Piantadosi, S. T. (2014). "Zipf's word frequency law in natural
    language: A critical review and future directions." Psychon. Bull. Rev.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import powerlaw

from analysis.neo4j_utils import get_database, OUTPUT_DIR, save_figure


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_entity_frequencies(driver, database: str | None = None):
    """Count total mentions per entity via Neo4j.

    Returns list of (entity_name, frequency) sorted by frequency descending.
    """
    db = database or get_database()
    query = """
    MATCH (e:Entity)<-[m:MENTIONS]-(p:Paper)
    RETURN e.name_norm AS entity, sum(m.count) AS freq
    ORDER BY freq DESC
    """
    with driver.session(database=db) as session:
        return [(r["entity"], int(r["freq"])) for r in session.run(query)]


# ── Analysis ──────────────────────────────────────────────────────────────────

def fit_zipf(frequencies: list[int]) -> dict:
    """Fit a discrete power-law to the frequency distribution.

    Returns dict with alpha, xmin, KS distance, and p-value.
    """
    data = np.array(frequencies, dtype=float)
    fit = powerlaw.Fit(data, discrete=True, verbose=False)

    # Compare power-law to alternatives
    R_ln, p_ln = fit.distribution_compare("power_law", "lognormal")
    R_exp, p_exp = fit.distribution_compare("power_law", "exponential")

    return {
        "alpha": float(fit.alpha),
        "xmin": float(fit.xmin),
        "D": float(fit.power_law.D),  # KS distance
        "n_tail": int(sum(data >= fit.xmin)),
        "n_total": len(data),
        "comparison_lognormal": {
            "loglikelihood_ratio": float(R_ln),
            "p_value": float(p_ln),
            "preferred": "power_law" if R_ln > 0 else "lognormal",
        },
        "comparison_exponential": {
            "loglikelihood_ratio": float(R_exp),
            "p_value": float(p_exp),
            "preferred": "power_law" if R_exp > 0 else "exponential",
        },
    }


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_zipf(names: list[str], frequencies: list[int], zipf_stats: dict):
    """Log-log rank-frequency plot with fitted power-law overlay."""
    ranks = np.arange(1, len(frequencies) + 1)
    freqs = np.array(frequencies, dtype=float)
    alpha = zipf_stats["alpha"]
    xmin = zipf_stats["xmin"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Empirical data
    ax.scatter(ranks, freqs, s=4, alpha=0.5, color="steelblue", label="Entity frequencies")

    # Fitted power-law (from xmin onward)
    mask = freqs >= xmin
    if mask.any():
        r_fit = ranks[mask]
        C = freqs[mask][0] * r_fit[0] ** alpha  # normalisation constant
        zipf_line = C / r_fit ** alpha
        ax.plot(r_fit, zipf_line, "r-", linewidth=2,
                label=f"Zipf fit: $\\alpha$={alpha:.2f}, $x_{{min}}$={xmin:.0f}")

    # Ideal Zipf (alpha=1) for reference
    C1 = freqs[0]
    ax.plot(ranks[:1000], C1 / ranks[:1000], "--", color="gray", alpha=0.5,
            label="Ideal Zipf ($\\alpha$=1)")

    # Annotate top entities
    n_annotate = min(8, len(names))
    for i in range(n_annotate):
        ax.annotate(names[i], (ranks[i], freqs[i]),
                    fontsize=7, ha="left", va="bottom",
                    xytext=(5, 3), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("Mention frequency", fontsize=12)
    ax.set_title("Zipf's Law Validation — Entity Mention Frequencies", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_deviation(names: list[str], frequencies: list[int], zipf_stats: dict):
    """Show entities that deviate most from the Zipf prediction."""
    alpha = zipf_stats["alpha"]
    ranks = np.arange(1, len(frequencies) + 1, dtype=float)
    freqs = np.array(frequencies, dtype=float)

    # Predicted frequency under fitted power-law
    C = freqs[0] * 1.0 ** alpha  # at rank 1
    predicted = C / ranks ** alpha

    # Relative deviation
    deviation = (freqs - predicted) / predicted

    # Select top over-represented and under-represented
    n_show = 15
    top_over = np.argsort(deviation)[-n_show:][::-1]
    top_under = np.argsort(deviation)[:n_show]

    # Combine, keeping order by magnitude
    indices = np.concatenate([top_over, top_under])
    selected_names = [names[i] for i in indices]
    selected_dev = [deviation[i] for i in indices]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#2ca02c" if d > 0 else "#d62728" for d in selected_dev]
    y_pos = np.arange(len(selected_names))
    ax.barh(y_pos, selected_dev, color=colors, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(selected_names, fontsize=8)
    ax.set_xlabel("Relative deviation from Zipf prediction", fontsize=11)
    ax.set_title("Entities Deviating from Zipf's Law", fontsize=13)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    return fig


# ── Main entry point ──────────────────────────────────────────────────────────

def run(driver, year: int | None = None):
    """Run the Zipf's law analysis.

    Parameters
    ----------
    driver : neo4j.Driver
    year : int | None
        Not used for Zipf (operates on full entity mentions), kept for
        API compatibility with run_analysis.py.
    """
    print("  Fetching entity mention frequencies …")
    data = fetch_entity_frequencies(driver)
    if not data:
        print("  ⚠ No entity mention data found.")
        return {}

    names, freqs = zip(*data)
    names, freqs = list(names), list(freqs)
    print(f"  {len(names)} entities, top-5: {names[:5]}")
    print(f"  Frequency range: {freqs[0]} – {freqs[-1]}")

    print("  Fitting power-law distribution …")
    stats = fit_zipf(freqs)
    print(f"  Alpha = {stats['alpha']:.3f}  (ideal Zipf: 1.0)")
    print(f"  xmin  = {stats['xmin']:.0f}  (tail starts at)")
    print(f"  KS D  = {stats['D']:.4f}")
    print(f"  Tail covers {stats['n_tail']}/{stats['n_total']} entities")
    print(f"  vs. lognormal: {stats['comparison_lognormal']['preferred']} "
          f"(p={stats['comparison_lognormal']['p_value']:.4f})")
    print(f"  vs. exponential: {stats['comparison_exponential']['preferred']} "
          f"(p={stats['comparison_exponential']['p_value']:.4f})")

    # Plot
    fig1 = plot_zipf(names, freqs, stats)
    save_figure(fig1, "15_zipf_law")
    plt.close(fig1)

    fig2 = plot_deviation(names, freqs, stats)
    save_figure(fig2, "16_zipf_deviation")
    plt.close(fig2)

    # Save stats
    stats_path = OUTPUT_DIR / "zipf_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: {stats_path}")

    return stats
