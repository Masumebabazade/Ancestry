#!/usr/bin/env python3
"""Analyze genotype distributions for population-specific AIM positions.

This script loads genotype data, AIM positions, and sample labels to
identify the best-fitting statistical distribution for genotype values of
specific populations at their ancestry-informative markers (AIMs).

Usage example:
    python analyze_aims.py --genotypes head_train_samples.csv --aims G_const_infs.csv --labels train1.labels --population ESN
"""
import argparse
import concurrent.futures as cf
from pathlib import Path

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
    """Fit several candidate distributions and return best name and scipy dist."""
    if values.size == 0:
        raise ValueError("no genotype values to fit")
    values = values.astype(float)
    # Candidate distributions
    candidates = {}
    # Binomial (n=2)
    p = np.clip(values.mean() / 2, 1e-6, 1 - 1e-6)
    candidates["binomial"] = (stats.binom(n=2, p=p), stats.binom.logpmf(values, 2, p).sum())
    # Poisson
    lam = values.mean()
    candidates["poisson"] = (stats.poisson(mu=lam), stats.poisson.logpmf(values, lam).sum())
    # Normal
    mu, sigma = stats.norm.fit(values)
    candidates["normal"] = (stats.norm(loc=mu, scale=sigma), stats.norm.logpdf(values, mu, sigma).sum())
    # Uniform
    a, b = values.min(), values.max()
    scale = max(b - a, 1e-6)
    candidates["uniform"] = (stats.uniform(loc=a, scale=scale), stats.uniform.logpdf(values, a, scale).sum())
    # Exponential
    loc, scale = stats.expon.fit(values, floc=0)
    candidates["exponential"] = (stats.expon(loc=loc, scale=scale), stats.expon.logpdf(values, loc, scale).sum())
    # Select best by log-likelihood
    return max(candidates.items(), key=lambda kv: kv[1][1])


def analyze_population(pop: str, labels: pd.DataFrame, aims: pd.DataFrame, genotypes: str, outdir: Path):
    samples = labels.loc[labels["pop"] == pop, "sample"].tolist()
    pop_positions = aims.loc[aims["pop"] == pop, "pos"].tolist()
    usecols = ["pos"] + samples
    geno = pd.read_csv(genotypes, sep="\t", usecols=lambda c: c in usecols)
    geno = geno[geno["pos"].isin(pop_positions)]
    if geno.empty:
        raise ValueError("no genotype data for population")
    values = geno[samples].values.ravel()
    values = values[~np.isnan(values)]
    if values.size == 0:
        raise ValueError("no genotype values for population")
    (best_name, (dist_obj, _)) = fit_distributions(values)
    # Plot histogram and fitted distribution
    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    bins = np.arange(values.min(), values.max() + 2) - 0.5
    plt.hist(values, bins=bins, density=True, alpha=0.6, label="data")
    x = np.linspace(values.min(), values.max(), 100)
    if isinstance(dist_obj, stats.rv_continuous):
        plt.plot(x, dist_obj.pdf(x), "r-", label=f"{best_name} fit")
    else:
        xk = np.arange(values.min(), values.max() + 1)
        plt.plot(xk, dist_obj.pmf(xk), "r-", label=f"{best_name} fit")
    plt.title(f"{pop} genotype distribution")
    plt.xlabel("Genotype value")
    plt.ylabel("Density")
    plt.legend()
    plot_path = outdir / f"{pop}_distribution.png"
    plt.savefig(plot_path)
    plt.close()
    return best_name, plot_path, len(values)


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
                dist, plot_path, n = fut.result()
                results[pop] = (dist, plot_path, n)
            except Exception as exc:
                results[pop] = ("error", Path(str(exc)), 0)
    for pop, (dist, plot_path, n) in sorted(results.items()):
        if dist == "error":
            print(f"{pop}: error {plot_path}")
        else:
            print(f"{pop}: best distribution {dist} using {n} values; plot -> {plot_path}")


if __name__ == "__main__":
    main()
