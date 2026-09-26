import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "moral_dilemmas_matplotlib")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

MODEL_COLORS = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
)


@dataclass(frozen=True)
class FigureBuildSummary:
    """Short summary of exported final figures."""

    output_dir: Path
    figure_paths: tuple[Path, ...]


def build_final_figures(
    *,
    workbook_path: str | Path,
    output_dir: str | Path,
) -> FigureBuildSummary:
    """Create final scientific figures from the consolidated analysis workbook."""
    workbook = pd.read_excel(workbook_path, sheet_name=None)
    value_model_summary = workbook["value_model_summary"]
    global_glmm_results = workbook["global_glmm_results"]

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    model_order = sorted(value_model_summary["model"].dropna().unique().tolist())
    _validate_model_order(model_order)
    color_map = dict(zip(model_order, MODEL_COLORS, strict=True))

    print("Creating final figures...")
    figure_paths: list[Path] = []

    print("Figure 1: Top-10 prevalence")
    figure_paths.extend(
        _plot_prevalence_top10(
            value_model_summary=value_model_summary,
            model_order=model_order,
            color_map=color_map,
            output_dir=output_directory,
        )
    )

    print("Appendix Figure A1: Eligible-value prevalence")
    figure_paths.extend(
        _plot_prevalence_eligible(
            value_model_summary=value_model_summary,
            model_order=model_order,
            color_map=color_map,
            output_dir=output_directory,
        )
    )

    print("Figure 2: Contrastive selection rates")
    figure_paths.extend(
        _plot_contrastive_rates(
            value_model_summary=value_model_summary,
            model_order=model_order,
            color_map=color_map,
            output_dir=output_directory,
        )
    )

    print("Focused significant-value figure(s)")
    figure_paths.extend(
        _plot_significant_value_focus(
            value_model_summary=value_model_summary,
            global_glmm_results=global_glmm_results,
            model_order=model_order,
            color_map=color_map,
            output_dir=output_directory,
        )
    )

    print("All figures written successfully.")
    return FigureBuildSummary(output_dir=output_directory, figure_paths=tuple(figure_paths))


def _plot_prevalence_top10(
    *,
    value_model_summary: pd.DataFrame,
    model_order: list[str],
    color_map: dict[str, str],
    output_dir: Path,
) -> tuple[Path, ...]:
    # Select the 10 values with the highest mean prevalence
    top_values = (
        value_model_summary.groupby("value")["prevalence"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .index.tolist()
    )
    if len(top_values) != 10:
        raise ValueError(f"Expected 10 values in Figure 1, found {len(top_values)}.")

    data = value_model_summary.loc[value_model_summary["value"].isin(top_values)].copy()
    return _grouped_horizontal_bar_chart(
        data=data,
        values=top_values,
        model_order=model_order,
        color_map=color_map,
        rate_column="prevalence",
        x_label="Prävalenz (%)",
        output_stem=output_dir / "figure_1_value_prevalence_top10",
        figure_size=(7.2, 5.2),
    )


def _plot_prevalence_eligible(
    *,
    value_model_summary: pd.DataFrame,
    model_order: list[str],
    color_map: dict[str, str],
    output_dir: Path,
) -> tuple[Path, ...]:
    data = value_model_summary.loc[_true_mask(value_model_summary["eligible_for_glmm"])].copy()
    values = data.groupby("value")["prevalence"].mean().sort_values(ascending=False).index.tolist()
    if not values:
        raise ValueError("No GLMM-eligible values found for appendix prevalence figure.")

    return _grouped_horizontal_bar_chart(
        data=data,
        values=values,
        model_order=model_order,
        color_map=color_map,
        rate_column="prevalence",
        x_label="Prävalenz (%)",
        output_stem=output_dir / "figure_a1_value_prevalence_eligible",
        figure_size=(7.2, 8.2),
    )


def _plot_contrastive_rates(
    *,
    value_model_summary: pd.DataFrame,
    model_order: list[str],
    color_map: dict[str, str],
    output_dir: Path,
) -> tuple[Path, ...]:
    data = value_model_summary.loc[_true_mask(value_model_summary["eligible_for_glmm"])].copy()
    required_ci = data[["contrastive_ci_lower", "contrastive_ci_upper"]].notna().all(axis=1)
    if not required_ci.all():
        raise ValueError("Eligible contrastive rows must contain bootstrap confidence intervals.")

    values = (
        data.groupby("value")["contrastive_selection_rate"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    if not values:
        raise ValueError("No GLMM-eligible values found for contrastive-rate figure.")

    return _dot_whisker_by_value(
        data=data,
        values=values,
        model_order=model_order,
        color_map=color_map,
        output_stem=output_dir / "figure_2_contrastive_rates",
        figure_size=(7.4, 8.4),
    )


def _plot_significant_value_focus(
    *,
    value_model_summary: pd.DataFrame,
    global_glmm_results: pd.DataFrame,
    model_order: list[str],
    color_map: dict[str, str],
    output_dir: Path,
) -> tuple[Path, ...]:
    significant_values = (
        global_glmm_results.loc[_true_mask(global_glmm_results["significant"]), "value"]
        .sort_values()
        .tolist()
    )
    figure_paths: list[Path] = []

    for value in significant_values:
        value_data = value_model_summary.loc[value_model_summary["value"] == value].copy()
        if value_data.empty:
            raise ValueError(f"Significant value is missing from value_model_summary: {value}")

        figure_paths.extend(
            _dot_whisker_for_single_value(
                data=value_data,
                value=value,
                model_order=model_order,
                color_map=color_map,
                output_stem=output_dir / f"figure_3_{_slugify(value)}",
                figure_size=(6.8, 3.1),
            )
        )

    return tuple(figure_paths)


def _grouped_horizontal_bar_chart(
    *,
    data: pd.DataFrame,
    values: list[str],
    model_order: list[str],
    color_map: dict[str, str],
    rate_column: str,
    x_label: str,
    output_stem: Path,
    figure_size: tuple[float, float],
) -> tuple[Path, ...]:
    values_for_axis = list(reversed(values))
    y_base = np.arange(len(values_for_axis))
    bar_height = 0.18
    offsets = np.linspace(
        -bar_height * (len(model_order) - 1) / 2,
        bar_height * (len(model_order) - 1) / 2,
        len(model_order),
    )

    fig, ax = plt.subplots(figsize=figure_size)
    max_rate = float(data[rate_column].max())
    for model, offset in zip(model_order, offsets, strict=True):
        model_data = data.loc[data["model"] == model].set_index("value")
        rates = model_data.loc[values_for_axis, rate_column].to_numpy()
        ax.barh(
            y_base + offset,
            rates,
            height=bar_height,
            color=color_map[model],
            label=model,
        )

    _style_quantitative_axis(ax, x_label=x_label, x_max=_rounded_axis_max(max_rate))
    ax.set_yticks(y_base)
    ax.set_yticklabels(values_for_axis)
    ax.set_ylabel("Wert")
    ax.legend(
        title="Modell",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
    )
    _finish_figure(fig)
    return _save_figure(fig, output_stem)


def _dot_whisker_by_value(
    *,
    data: pd.DataFrame,
    values: list[str],
    model_order: list[str],
    color_map: dict[str, str],
    output_stem: Path,
    figure_size: tuple[float, float],
) -> tuple[Path, ...]:
    values_for_axis = list(reversed(values))
    y_base = np.arange(len(values_for_axis))
    offsets = np.linspace(-0.27, 0.27, len(model_order))

    fig, ax = plt.subplots(figsize=figure_size)
    ax.axvline(0.5, color="#777777", linestyle="--", linewidth=0.8, zorder=1)

    for model, offset in zip(model_order, offsets, strict=True):
        model_data = data.loc[data["model"] == model].set_index("value")
        rates = model_data.loc[values_for_axis, "contrastive_selection_rate"].to_numpy()
        lower = model_data.loc[values_for_axis, "contrastive_ci_lower"].to_numpy()
        upper = model_data.loc[values_for_axis, "contrastive_ci_upper"].to_numpy()
        xerr = np.vstack([rates - lower, upper - rates])

        ax.errorbar(
            rates,
            y_base + offset,
            xerr=xerr,
            fmt="o",
            color=color_map[model],
            ecolor=color_map[model],
            elinewidth=1.1,
            capsize=2.5,
            markersize=4.5,
            label=model,
            zorder=2,
        )

    _style_quantitative_axis(ax, x_label="Kontrastive Auswahlrate (%)")
    ax.set_yticks(y_base)
    ax.set_yticklabels(values_for_axis)
    ax.set_ylabel("Wert")
    ax.legend(
        title="Modell",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
    )
    _finish_figure(fig)
    return _save_figure(fig, output_stem)


def _dot_whisker_for_single_value(
    *,
    data: pd.DataFrame,
    value: str,
    model_order: list[str],
    color_map: dict[str, str],
    output_stem: Path,
    figure_size: tuple[float, float],
) -> tuple[Path, ...]:
    model_order_for_axis = list(reversed(model_order))
    data = data.set_index("model").loc[model_order_for_axis].reset_index()
    y_positions = np.arange(len(model_order_for_axis))

    fig, ax = plt.subplots(figsize=figure_size)
    ax.axvline(0.5, color="#777777", linestyle="--", linewidth=0.8, zorder=1)

    for row, y_position in zip(data.itertuples(index=False), y_positions, strict=True):
        rate = row.contrastive_selection_rate
        xerr = np.array([[rate - row.contrastive_ci_lower], [row.contrastive_ci_upper - rate]])
        ax.errorbar(
            rate,
            y_position,
            xerr=xerr,
            fmt="o",
            color=color_map[row.model],
            ecolor=color_map[row.model],
            elinewidth=1.2,
            capsize=3,
            markersize=5,
            zorder=2,
        )

    _style_quantitative_axis(ax, x_label="Kontrastive Auswahlrate (%)")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(model_order_for_axis)
    ax.set_ylabel("Modell")
    ax.set_title(value, fontweight="normal", pad=8)
    _finish_figure(fig)
    return _save_figure(fig, output_stem)


def _style_quantitative_axis(ax, *, x_label: str, x_max: float = 1.0) -> None:
    ax.set_xlim(0, x_max)
    ax.set_xlabel(x_label)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B0B0B0")
    ax.spines["bottom"].set_color("#B0B0B0")


def _rounded_axis_max(max_rate: float) -> float:
    return min(1.0, max(0.05, np.ceil(max_rate * 1.15 / 0.05) * 0.05))


def _true_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).eq(1)

    return series.astype(str).str.casefold().eq("true")


def _finish_figure(fig) -> None:
    fig.patch.set_facecolor("white")
    fig.tight_layout()


def _save_figure(fig, output_stem: Path) -> tuple[Path, Path]:
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def _validate_model_order(model_order: list[str]) -> None:
    if len(model_order) != 4:
        raise ValueError(f"Expected exactly four models, found {len(model_order)}.")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "value"
