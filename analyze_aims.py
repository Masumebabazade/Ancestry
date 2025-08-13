#!/usr/bin/env python3
"""Analyze genotype distributions for population-specific AIM positions.

This script loads genotype data, AIM positions, and sample labels to
identify the best-fitting statistical distribution for the genotype values
observed across all *samples* in a population at each ancestry-informative
marker (AIM) position.  In addition to per-position fits, the script
records the empirical frequency of complete genotype sequences for the
population, which can be used to synthesise new samples or estimate the
probability of existing ones.

Usage example:
    python analyze_aims.py --genotypes head_train_samples.csv --aims G_const_infs.csv --labels train1.labels
    # optionally add --population ESN to limit the analysis to a single group
"""
import argparse
import concurrent.futures as cf
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


EXPECTED_POPS = [
    "ACB", "ASW", "BEB", "CDX", "CEU", "CHB", "CHS", "CLM",
    "ESN", "FIN", "GBR", "GIH", "GWD", "IBS", "ITU", "JPT",
    "KHV", "LWK", "MSL", "MXL", "PEL", "PJL", "PUR", "STU",
    "TSI", "YRI",
]


def build_pop_mapping(labels_file: str):
    """Return (labels_df, mapping from P(Gi) to population).

    The mapping aligns AIM population codes P(G0) .. P(G25) with the
    alphabetically sorted populations from the training labels.
    """
    labels = pd.read_csv(labels_file, sep="\t", names=["sample", "pop"])
    pops = sorted(labels["pop"].unique())
    if pops != EXPECTED_POPS:
        raise ValueError("labels populations do not match expected order")
    mapping = {f"P(G{i})": pop for i, pop in enumerate(pops)}
    return labels, mapping


def load_aims(aim_file: str, mapping):
    aims = pd.read_csv(aim_file)
    pop_codes = aims["pop"]
    aims["pop"] = pop_codes.map(mapping)
    if aims["pop"].isnull().any():
        unknown = pop_codes[aims["pop"].isnull()].unique()
        raise ValueError(f"unknown AIM population codes: {unknown}")
    aims["pos"] = aims["chromosome"].str.replace("chr", "", regex=False) + "_" + aims["position"].astype(str)
    return aims


def fit_distributions(values: np.ndarray):
    """Fit candidate distributions and return best name, scipy dist, and parameters."""
    if values.size == 0:
        raise ValueError("no genotype values to fit")
    values = values.astype(float)
    candidates = {}
    # Binomial (n=2)
    p = np.clip(values.mean() / 2, 1e-6, 1 - 1e-6)
    candidates["binomial"] = (
        stats.binom(n=2, p=p),
        {"n": 2, "p": float(p)},
        stats.binom.logpmf(values, 2, p).sum(),
    )
    # Poisson
    lam = values.mean()
    candidates["poisson"] = (
        stats.poisson(mu=lam),
        {"mu": float(lam)},
        stats.poisson.logpmf(values, lam).sum(),
    )
    # Normal
    mu, sigma = stats.norm.fit(values)
    candidates["normal"] = (
        stats.norm(loc=mu, scale=sigma),
        {"loc": float(mu), "scale": float(sigma)},
        stats.norm.logpdf(values, mu, sigma).sum(),
    )
    # Uniform
    a, b = values.min(), values.max()
    scale = max(b - a, 1e-6)
    candidates["uniform"] = (
        stats.uniform(loc=a, scale=scale),
        {"loc": float(a), "scale": float(scale)},
        stats.uniform.logpdf(values, a, scale).sum(),
    )
    # Exponential
    loc, scale = stats.expon.fit(values, floc=0)
    candidates["exponential"] = (
        stats.expon(loc=loc, scale=scale),
        {"loc": float(loc), "scale": float(scale)},
        stats.expon.logpdf(values, loc, scale).sum(),
    )
    # Select best by log-likelihood
    best_name, (dist_obj, params, _) = max(candidates.items(), key=lambda kv: kv[1][2])
    return best_name, dist_obj, params


def analyze_population(pop: str, labels: pd.DataFrame, aims: pd.DataFrame, genotypes: str, outdir: Path):
    """Analyze genotype distributions for all samples belonging to a population."""
    samples = labels.loc[labels["pop"] == pop, "sample"].tolist()
    pop_positions = aims.loc[aims["pop"] == pop, "pos"].tolist()
    usecols = ["pos"] + samples
    geno = pd.read_csv(genotypes, sep="\t", usecols=lambda c: c in usecols)
    geno = geno[geno["pos"].isin(pop_positions)]
    if geno.empty:
        raise ValueError("no genotype data for population")

    pop_outdir = outdir / pop
    pop_outdir.mkdir(parents=True, exist_ok=True)

    position_results = {}
    for _, row in geno.iterrows():
        position = row["pos"]
        vals = row[samples].values
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            position_results[position] = {"error": "no genotype values"}
            continue
        best_name, dist_obj, params = fit_distributions(vals)
        fig, ax = plt.subplots()
        bins = np.arange(vals.min(), vals.max() + 2) - 0.5
        ax.hist(vals, bins=bins, density=True, alpha=0.6, label="data")
        if isinstance(dist_obj.dist, stats.rv_discrete):
            xk = np.arange(vals.min(), vals.max() + 1)
            ax.plot(xk, dist_obj.pmf(xk), "r-", label=f"{best_name} fit")
        else:
            x = np.linspace(vals.min(), vals.max(), 100)
            ax.plot(x, dist_obj.pdf(x), "r-", label=f"{best_name} fit")
        ax.set_title(f"{pop} {position} genotype distribution")
        ax.set_xlabel("Genotype value")
        ax.set_ylabel("Density")
        ax.legend()
        plot_path = pop_outdir / f"{position}_distribution.png"
        fig.savefig(plot_path)
        plt.close(fig)
        position_results[position] = {
            "distribution": best_name,
            "params": params,
            "plot": str(plot_path),
            "values": int(vals.size),
        }

    # Build genotype sequence distribution for the population.
    sample_vectors = geno.set_index("pos")[samples].T
    seq_counts = (
        sample_vectors.apply(lambda r: ",".join(map(str, r.values)), axis=1)
        .value_counts()
        .to_dict()
    )

    return {"positions": position_results, "sequence_counts": seq_counts}


def main():
    ap = argparse.ArgumentParser(description="Analyze genotype distributions by population")
    ap.add_argument("--genotypes", required=True, help="Genotype CSV file")
    ap.add_argument("--aims", required=True, help="AIM positions CSV file")
    ap.add_argument("--labels", required=True, help="Sample labels file")
    ap.add_argument("--population", help="Specific population to analyze")
    ap.add_argument("--outdir", default="plots", help="Output directory for plots")
    ap.add_argument("--threads", type=int, default=4, help="Number of threads")
    args = ap.parse_args()

    labels, mapping = build_pop_mapping(args.labels)
    aims = load_aims(args.aims, mapping)
    pops = [args.population] if args.population else sorted(labels["pop"].unique())
    outdir = Path(args.outdir)
    results = {}
    with cf.ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = {ex.submit(analyze_population, pop, labels, aims, args.genotypes, outdir): pop for pop in pops}
        for fut in cf.as_completed(futs):
            pop = futs[fut]
            try:
                results[pop] = fut.result()
            except Exception as exc:
                results[pop] = {"error": str(exc)}
    for pop, data in sorted(results.items()):
        if "error" in data:
            print(f"{pop}: error {data['error']}")
            continue
        for pos, pdata in sorted(data["positions"].items()):
            if "error" in pdata:
                print(f"{pop}/{pos}: error {pdata['error']}")
            else:
                print(
                    f"{pop}/{pos}: best distribution {pdata['distribution']} using {pdata['values']} samples; plot -> {pdata['plot']}"
                )
        print(f"{pop}: {len(data['sequence_counts'])} unique genotype sequences")
    summary_path = outdir / "distribution_summary.json"
    with summary_path.open("w") as fh:
        json.dump(results, fh, indent=2)
    print(f"summary JSON -> {summary_path}")


if __name__ == "__main__":
    main()
