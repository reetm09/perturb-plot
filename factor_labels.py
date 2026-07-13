import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

factor_dict = {
    "scHPF_1": "f1",
    "scHPF_2": "f2",
    "scHPF_3": "f3",
    "scHPF_4": "f4",
    "scHPF_5": "f5",
    "scHPF_7": "f7",
    "scHPF_8": "f8",
    "scHPF_9": "f9",
    "scHPF_10": "f10",
    "scHPF_11": "f11",
    "scHPF_14": "f14",
    "scHPF_15": "f15",
    "scHPF_16": "f16",
    "scHPF_17": "f17",
    "scHPF_18": "f18",
    "scHPF_19": "f19",
    "scHPF_20": "f20",
    "scHPF_21": "f21",
    "scHPF_22": "f22",
    "scHPF_23": "f23",
    "scHPF_24": "f24",
    "scHPF_25": "f25",
    "scHPF_26": "f26",
}


factor_map = {
    "f1": "schpf_1",
    "f2": "schpf_2",
    "f3": "schpf_3",
    "f4": "schpf_4",
    "f5": "schpf_5",
    "f6": "schpf_6",
    "f7": "schpf_7",
    "f8": "schpf_8",
    "f9": "schpf_9",
    "f10": "schpf_10",
    "f11": "schpf_11",
    "f12": "schpf_12",
    "f13": "schpf_13",
    "f14": "schpf_14",
    "f15": "schpf_15",
    "f16": "schpf_16",
    "f17": "schpf_17",
    "f18": "schpf_18",
    "f19": "schpf_19",
    "f20": "schpf_20",
    "f21": "schpf_21",
    "f22": "schpf_22",
    "f23": "schpf_23",
    "f24": "schpf_24",
    "f25": "schpf_25",
    "f26": "schpf_26",
}

factor_names = {
    "schpf_1": "il/ifn-γ signaling",
    "schpf_2": "glycolysis",
    "schpf_3": "ciita-high",
    "schpf_4": "npy1r-high",
    "schpf_5": "chemokine",
    "schpf_6": "unknown",
    "schpf_7": "tlr/mapk signaling",
    "schpf_8": "oxphos-1",
    "schpf_9": "motility/adhesion",
    "schpf_10": "grid2-high",
    "schpf_11": "apoe-high",
    "schpf_12": "unknown",
    "schpf_13": "unknown",
    "schpf_14": "stress",
    "schpf_15": "hla-high/apc",
    "schpf_16": "cx3cr1-high",
    "schpf_17": "c1q-high/phagocytic",
    "schpf_18": "oxphos-2",
    "schpf_19": "senescence",
    "schpf_20": "ifn-i response",
    "schpf_21": "oxphos-3",
    "schpf_22": "plcg2-high",
    "schpf_23": "s100/tlr signaling",
    "schpf_24": "actin folding",
    "schpf_25": "immunoregulatory",
    "schpf_26": "gpnmb-high",
}


def get_direct_factor_map() -> dict:
    direct_factor_map = {f: factor_names[factor_map[f]] for f in factor_map}
    return direct_factor_map


def get_factor_nums() -> list:
    return list(factor_dict.values())


def build_df_long(
    adata,
    descriptive_cols,
    diff_type_name="Signature",
    extra_gene_col=None,
) -> pd.DataFrame:

    all_cell_scores = pd.concat(
        [
            adata.obs[descriptive_cols],
            adata.obs["perturbed_gene"],
            adata.obs["perturbed_guide"],
            adata.obs[extra_gene_col] if extra_gene_col else None,
            # adata.obs["is_perturbed"],
        ],
        axis=1,
    )
    df = pd.DataFrame(all_cell_scores)
    df["is_perturbed"] = [
        "NTC" if x == "NTC" else "PERTURBED" for x in df.perturbed_guide
    ]

    df_long = df.melt(
        id_vars=["perturbed_gene", "perturbed_guide"]
        + ([extra_gene_col] if extra_gene_col else []),
        value_vars=descriptive_cols,
        var_name=diff_type_name,
        value_name="Score",
    )
    return df_long


palette_dict_6tf = {
    "ARID2_g1": "#E36769",
    "ARID2_g2": "#9D2937",
    "ARID5B_g1": "#E36769",
    "ARID5B_g2": "#9D2937",
    "ATMIN_g1": "#E36769",
    "ATMIN_g2": "#9D2937",
    "BHLHE40_g1": "#E36769",
    "BHLHE40_g2": "#9D2937",
    "BHLHE41_g1": "#E36769",
    "BHLHE41_g2": "#9D2937",
    "BPTF_g1": "#E36769",
    "BPTF_g2": "#9D2937",
    "CEBPD_g2": "#9D2937",
    "CNOT10_g1": "#E36769",
    "CNOT10_g2": "#9D2937",
    "DEAF1_g1": "#E36769",
    "DEAF1_g2": "#9D2937",
    "DNMT1_g1": "#A685C4",
    "DNMT1_g2": "#5C466F",
    "FOXK1_g1": "#D66BB6",
    "FOXK1_g2": "#BD2F92",
    "IRF9_g1": "#6471A5",
    "IRF9_g2": "#000063",
    "MAF_g1": "#E36769",
    "MAF_g2": "#9D2937",
    "MEF2C_g1": "#E36769",
    "MEF2C_g2": "#9D2937",
    "MEF2D_g2": "#9D2937",
    "MITF_g1": "#E36769",
    "MITF_g2": "#9D2937",
    "NTC": "#CCCCCC",
    # "NTC": "#c2bfbe",
    # "NTC": "#666666",
    "POU5F1_g1": "#E36769",
    "POU5F1_g2": "#9D2937",
    "PRDM1_g1": "#E36769",
    "PRDM1_g2": "#9D2937",
    "RELA_g1": "#E36769",
    "RELA_g2": "#9D2937",
    "RUNX1_g1": "#E36769",
    "RUNX1_g2": "#9D2937",
    "SALL4_g1": "#E36769",
    "SALL4_g2": "#9D2937",
    "SMAD3_g1": "#62A86D",
    "SMAD3_g2": "#1A6321",
    "SPI1_g1": "#E36769",
    "SPI1_g2": "#9D2937",
    "SREBF1_g1": "#E36769",
    "SREBF1_g2": "#9D2937",
    "STAT1_g1": "#E36769",
    "STAT1_g2": "#9D2937",
    "STAT2_g1": "#4C9AE9",
    "STAT2_g2": "#3B5989",
    "TCF4_g1": "#E36769",
    "TCF4_g2": "#9D2937",
    "ZNF148_g1": "#E36769",
    "ZNF148_g2": "#9D2937",
    "ZNF532_g1": "#E9D76F",
    "ZNF532_g2": "#FDAC10",
    "ZNF644_g1": "#E36769",
    "ZNF644_g2": "#9D2937",
    "ZNF783_g1": "#E36769",
    "ZNF783_g2": "#9D2937",
}

palette_dict_imgl = {
    "ARID2_g1": "#E36769",
    "ARID2_g2": "#9D2937",
    "ARID5B_g1": "#E36769",
    "ARID5B_g2": "#9D2937",
    "ATMIN_g1": "#E36769",
    "ATMIN_g2": "#9D2937",
    "BHLHE40_g1": "#E36769",
    "BHLHE40_g2": "#9D2937",
    "BHLHE41_g1": "#E36769",
    "BHLHE41_g2": "#9D2937",
    "BPTF_g1": "#E36769",
    "BPTF_g2": "#9D2937",
    "CEBPD_g1": "#E36769",
    "CEBPD_g2": "#9D2937",
    "CNOT10_g1": "#E36769",
    "CNOT10_g2": "#9D2937",
    "DEAF1_g1": "#E36769",
    "DEAF1_g2": "#9D2937",
    "DNMT1_g1": "#A685C4",
    "DNMT1_g2": "#5C466F",
    "FOXK1_g1": "#D66BB6",
    "FOXK1_g2": "#BD2F92",
    "IRF9_g1": "#6471A5",
    "IRF9_g2": "#000063",
    "MAF_g1": "#E36769",
    "MAF_g2": "#9D2937",
    "MEF2C_g1": "#E36769",
    "MEF2C_g2": "#9D2937",
    "MEF2D_g2": "#9D2937",
    "MITF_g1": "#E36769",
    "MITF_g2": "#9D2937",
    "NTC": "#CCCCCC",
    # "NTC": "#c2bfbe",
    # "NTC": "#666666",
    "POU5F1_g2": "#9D2937",
    "PRDM1_g1": "#E36769",
    "PRDM1_g2": "#9D2937",
    "RELA_g1": "#E36769",
    "RELA_g2": "#9D2937",
    "RUNX1_g1": "#E36769",
    "RUNX1_g2": "#9D2937",
    "SALL4_g1": "#E36769",
    "SALL4_g2": "#9D2937",
    "SMAD3_g1": "#62A86D",
    "SMAD3_g2": "#1A6321",
    "SPI1_g1": "#E36769",
    "SPI1_g2": "#9D2937",
    "SREBF1_g1": "#E36769",
    "SREBF1_g2": "#9D2937",
    "STAT1_g1": "#E36769",
    "STAT1_g2": "#9D2937",
    "STAT2_g1": "#4C9AE9",
    "STAT2_g2": "#3B5989",
    "TCF4_g1": "#E36769",
    "TCF4_g2": "#9D2937",
    "ZNF148_g1": "#E36769",
    "ZNF148_g2": "#9D2937",
    "ZNF532_g1": "#E9D76F",
    "ZNF532_g2": "#FDAC10",
    "ZNF644_g1": "#E36769",
    "ZNF644_g2": "#9D2937",
    "ZNF783_g1": "#E36769",
    "ZNF783_g2": "#9D2937",
}


def draw_pointplot(
    df,
    guides_to_use,
    figsize_num,
    by_guide=True,
    diff_type_name="Signature",
    nrows=3,
    ncols=2,
    palette=None,
    orderlist=None,
    x_val="PctShift",
    filename=None,
) -> None:
    guide_col = "perturbed_guide" if by_guide else "perturbed_gene"

    unique_guides = guides_to_use
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=figsize_num, sharex=True, sharey=True, dpi=300
    )
    axes = axes.flatten()
    for i, guide in enumerate(unique_guides):
        ax = axes[i]

        subset_guide = df[
            (df[guide_col].str.contains(guide)) | (df[guide_col] == "NTC")
        ]
        sns.pointplot(
            data=subset_guide,
            x=x_val,
            y=diff_type_name,
            hue=guide_col,
            dodge=False,
            join=False,
            palette=palette,
            scale=1.5,
            ax=ax,
            # estimator=np.median,
            errorbar=None,
            order=orderlist,
            # order=list(df.sort_values("MedianPct").Factor.unique())
        )
        ax.grid(
            True, which="both", axis="both", linestyle="--", linewidth=0.5, alpha=0.7
        )
        ax.set_title(f"{guide}")
        ax.set_xlabel(f"Median Score Shift vs NTC (pp)")
        ax.set_ylabel(f"{diff_type_name}")

        ntc_rows = subset_guide[subset_guide[guide_col] == "NTC"]
        gene_rows = subset_guide[subset_guide[guide_col] != "NTC"]
        if not ntc_rows.empty:
            x_ntc = ntc_rows[x_val].values
            y_ntc = ntc_rows[diff_type_name].values
            err_low_ntc = ntc_rows[f"NTC_{x_val}_CI_low"].values
            err_high_ntc = ntc_rows[f"NTC_{x_val}_CI_high"].values

            err_low = np.abs(x_ntc - err_low_ntc)
            err_high = np.abs(err_high_ntc - x_ntc)

            y_order = (
                orderlist
                if orderlist is not None
                else sorted(subset_guide[diff_type_name].unique())
            )
            y_pos = [y_order.index(y) for y in y_ntc]
            ax.errorbar(
                x=x_ntc,
                y=y_pos,
                xerr=[err_low, err_high],
                fmt="none",
                ecolor=palette["NTC"],
                elinewidth=2.5,
            )  # capsize=4,

        if guide_col == "perturbed_guide":
            g1_guidename = guide + "_g1"
            guide1_rows = subset_guide[subset_guide["perturbed_guide"] == g1_guidename]

            if not guide1_rows.empty:
                x_g1 = guide1_rows[x_val].values
                y_g1 = guide1_rows[diff_type_name].values
                err_low_g1 = guide1_rows[f"{x_val}_CI_low"].values
                err_high_g1 = guide1_rows[f"{x_val}_CI_high"].values

                err_low = np.abs(x_g1 - err_low_g1)
                err_high = np.abs(err_high_g1 - x_g1)
                y_order = (
                    orderlist
                    if orderlist is not None
                    else sorted(subset_guide[diff_type_name].unique())
                )
                y_pos = [y_order.index(y) for y in y_g1]
                ax.errorbar(
                    x=x_g1,
                    y=y_g1,
                    xerr=[err_low, err_high],
                    fmt="none",
                    ecolor=palette[g1_guidename],
                    elinewidth=2.5,
                )

            g2_guidename = guide + "_g2"
            guide2_rows = subset_guide[subset_guide["perturbed_guide"] == g2_guidename]

            if not guide2_rows.empty:
                x_g2 = guide2_rows[x_val].values
                y_g2 = guide2_rows[diff_type_name].values
                err_low_g2 = guide2_rows[f"{x_val}_CI_low"].values
                err_high_g2 = guide2_rows[f"{x_val}_CI_high"].values

                err_low = np.abs(x_g2 - err_low_g2)
                err_high = np.abs(err_high_g2 - x_g2)
                y_order = (
                    orderlist
                    if orderlist is not None
                    else sorted(subset_guide[diff_type_name].unique())
                )
                y_pos = [y_order.index(y) for y in y_g2]
                ax.errorbar(
                    x=x_g2,
                    y=y_g2,
                    xerr=[err_low, err_high],
                    fmt="none",
                    ecolor=palette[g2_guidename],
                    elinewidth=2.5,
                )  # capsize=4,
            ax.legend().remove()
        else:
            if not gene_rows.empty:
                x_gene = gene_rows[x_val].values
                y_gene = gene_rows[diff_type_name].values
                err_low_gene = gene_rows[f"{x_val}_CI_low"].values
                err_high_gene = gene_rows[f"{x_val}_CI_high"].values

                err_low = np.abs(x_gene - err_low_gene)
                err_high = np.abs(err_high_gene - x_gene)
                y_order = (
                    orderlist
                    if orderlist is not None
                    else sorted(subset_guide[diff_type_name].unique())
                )
                y_pos = [y_order.index(y) for y in y_gene]
                ax.errorbar(
                    x=x_gene,
                    y=y_gene,
                    xerr=[err_low, err_high],
                    fmt="none",
                    ecolor=palette[guide],
                    elinewidth=2.5,
                )  # capsize=4,

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()


def plot_radar_chart(data, metrics, title) -> None:
    N = len(metrics)

    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)

    theta_closed = np.concatenate([theta, [theta[0]]])
    sector_width = 2 * np.pi / N
    bar_width = sector_width * 0.7
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"}, dpi=300)

    ax.set_title(title, y=1.15, fontsize=20)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(90)
    ax.spines["polar"].set_zorder(1)
    ax.spines["polar"].set_color("lightgrey")

    color_palette = ["#9CDADB"]  # optional if you truly only want one color

    for idx, (_, row) in enumerate(data.iterrows()):
        values = row[metrics].to_numpy(dtype=float).flatten()
        values_maxed = [min(i, 0.5) for i in values]
        values_closed = np.concatenate([values_maxed, [values_maxed[0]]])

        ax.bar(
            theta_closed,
            values_closed,
            width=bar_width,
            bottom=0,
            alpha=0.6,
            align="center",
        )

    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_yticklabels(["0.1", "0.2", "0.3", "0.4", "0.5"], color="black", fontsize=12)
    ax.set_xticks(theta)
    ax.set_xticklabels(metrics, color="black", fontsize=16)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    return fig
