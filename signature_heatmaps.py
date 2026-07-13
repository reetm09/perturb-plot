import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


def combined_label(p_value, annot_with_num=True) -> str:
    """
    Format adjusted p-value with stars for significance.
    Optionally, include p-value label with stars.
    """
    if pd.isna(p_value):
        return "NaN"
    significance = ""
    # if p_value < 0.001:
    #    significance = "***"
    if p_value < 0.01:
        significance = "***"
    elif p_value < 0.05:
        significance = "**"
    elif p_value < 0.1:
        significance = "*"
    return f"{p_value:.3f} {significance}" if annot_with_num else f"{significance}"


def calculate_median_differences_and_annotations(
    adata, scores, annot_with_num=True, by_guide=True
) -> [pd.DataFrame(), pd.DataFrame(), list[str]]:
    if by_guide:
        ntc_mdata = adata[adata.obs["perturbed_guide"] == "NTC"]
        guide_mdata = adata[adata.obs["perturbed_guide"] != "NTC"]
        guide_col = "guide"
    else:
        ntc_mdata = adata[adata.obs["perturbed_gene"] == "NTC"]
        guide_mdata = adata[adata.obs["perturbed_gene"] != "NTC"]
        guide_col = "perturbed_gene"

    all_median_differences = pd.DataFrame()
    all_p_values = pd.DataFrame()

    for score in scores:
        if score not in adata.obs:
            print(f"Score {score} not found in adata.obs. Skipping.")
            continue

        ntc_scores = ntc_mdata.obs[score].values
        guide_data = guide_mdata.obs[[guide_col, score]]

        results = []
        median_differences = []
        for guide in guide_data[guide_col].unique():
            guide_scores = guide_data[guide_data[guide_col] == guide][score].values

            if len(guide_scores) > 0:
                median_ntc = np.median(ntc_scores)
                median_guide = np.median(guide_scores)
                median_difference = median_guide - median_ntc
                median_differences.append({guide_col: guide, score: median_difference})
                test_stat, p_value = mannwhitneyu(
                    guide_scores, ntc_scores, alternative="two-sided"
                )
                results.append({guide_col: guide, f"{score}_p_value": p_value})
            else:
                median_differences.append({guide_col: guide, score: np.nan})
                results.append({guide_col: guide, f"{score}_p_value": np.nan})

        median_differences_df = pd.DataFrame(median_differences).set_index(guide_col)
        all_median_differences = pd.concat(
            [all_median_differences, median_differences_df], axis=1
        )

        results_df = pd.DataFrame(results).set_index(guide_col)
        all_p_values = pd.concat([all_p_values, results_df[f"{score}_p_value"]], axis=1)

        # FDR correction for all p-values in results_df
        rejected, pvals_corrected, _, _ = multipletests(
            results_df[f"{score}_p_value"].values, method="fdr_bh"
        )
        results_df[f"{score}_p_value_fdr"] = pvals_corrected
        all_p_values = pd.concat(
            [all_p_values, results_df[f"{score}_p_value_fdr"]], axis=1
        )
    annotations = all_p_values.map(lambda x: combined_label(x, annot_with_num))
    median_differences_df[f"{guide_col}_counts"] = median_differences_df.index.map(
        guide_mdata.obs[guide_col].value_counts()
    )
    row_labels = [f"{guide}" for guide in median_differences_df.index]
    return all_median_differences, annotations, row_labels


def plot_clustermap(
    data,
    annotations,
    row_labels,
    title,
    xlabel,
    ylabel,
    vmin=-0.05,
    vmax=0.05,
    alphabetical: bool = False,
    label: str = "Median",
    figsize=(7, 7),
    by_guide=True,
    rowC=True,
    colC=True,
    filename=".jpeg",
) -> None:
    custom_colors = ["#000063", "#537FE8", "#B54246", "#800000"]
    custom_palette = sns.color_palette(custom_colors, as_cmap=True)

    guide_col = "guide" if by_guide else "perturbed_gene"

    if alphabetical:
        all_mean_differences_sorted_alpha = data.sort_values(by=guide_col)
        annotations_sorted_alpha = annotations.sort_index()
        # Plot heatmap for correlation
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        ax = sns.heatmap(
            all_mean_differences_sorted_alpha,
            annot=annotations_sorted_alpha,
            fmt="",
            cmap="coolwarm",
            cbar_kws={"label": f"{label} Difference"},
            xticklabels=True,
            yticklabels=sorted(row_labels),
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(title + " Alphabetical")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.savefig(filename, bbox_inches="tight")
        plt.show()

    else:
        g = sns.clustermap(
            data,
            annot=annotations,
            fmt="",
            cmap="coolwarm",
            figsize=figsize,
            cbar_kws={"label": f"{label} Difference"},
            xticklabels=True,
            yticklabels=row_labels,
            dendrogram_ratio=(0.2, 0.1),
            method="average",
            metric="euclidean",
            vmin=vmin,
            vmax=vmax,
            row_cluster=rowC,
            col_cluster=colC,
            # dpi=300,
        )

        g.ax_heatmap.set_title(title)
        g.ax_heatmap.set_xlabel(xlabel)
        g.ax_heatmap.set_ylabel(ylabel)
        g.savefig(filename, bbox_inches="tight", dpi=300)
        plt.show()
