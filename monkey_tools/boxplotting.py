from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _order_present_values(
    data: pd.DataFrame,
    column: str,
    order: Sequence[str] | None = None,
) -> list[str]:
    """Return plotting order restricted to values that exist in the dataframe."""
    present_values = list(pd.unique(data[column]))

    if order is None:
        return sorted(present_values)

    return [value for value in order if value in present_values]


def _clear_axis_legend(ax) -> tuple[list, list]:
    """Remove duplicated seaborn legends and keep handles for optional reuse."""
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    return handles, labels


def _add_paired_lines(
    ax,
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    pair_id_col: str,
    paired_line_order: Sequence[str],
    color: str = "red",
    alpha: float = 1.0,
    linewidth: float = 1.5,
):
    """Overlay paired lines linking repeated observations across x categories."""
    pivot = data.pivot(index=pair_id_col, columns=x, values=y).dropna()
    valid_order = [value for value in paired_line_order if value in pivot.columns]

    if len(valid_order) < 2:
        return

    x_positions = list(range(len(valid_order)))
    for pair_id in pivot.index:
        ax.plot(
            x_positions,
            pivot.loc[pair_id, valid_order].to_list(),
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )


def _align_strip_points_to_box_width(collections, *, box_width: float, strip_dodge_width: float):
    """Scale stripplot x offsets so dodged points sit over matching box centers."""
    if strip_dodge_width <= 0:
        return

    scale = box_width / strip_dodge_width
    if scale == 1:
        return

    for collection in collections:
        offsets = collection.get_offsets()
        if len(offsets) == 0:
            continue

        category_centers = offsets[:, 0].round()
        offsets[:, 0] = category_centers + (offsets[:, 0] - category_centers) * scale
        collection.set_offsets(offsets)


def _center_strip_point_clouds(collections):
    """Shift each stripplot point cloud so its mean x-position is centered."""
    for collection in collections:
        offsets = collection.get_offsets()
        if len(offsets) == 0:
            continue

        category_center = round(float(offsets[:, 0].mean()))
        offsets[:, 0] += category_center - offsets[:, 0].mean()
        collection.set_offsets(offsets)


def plot_box_strip(
    data: pd.DataFrame,
    x: str, 
    y: str,
    *,
    title: str,
    hue: str,
    order: Sequence[str] | None = None,
    hue_order: Sequence[str] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = (5, 4),
    palette: str = "tab10",
    strip_palette: str | None = "dark",
    showfliers: bool = False,
    box_width: float = 0.6,
    strip_alpha: float = 0.8,
    strip_size: float = 5,
    strip_jitter: float | bool = 0.08,
    strip_dodge_width: float = 0.8,
    center_strip_points: bool = True,
    dodge: bool = True,
    tick_fontsize: float = 16,
    title_fontsize: float = 18,
    label_fontsize: float = 16,
    ylim: tuple[float, float] | list[float] | None = None,
    show_paired_lines: bool = False,
    pair_id_col: str = "subject",
    paired_line_order: Sequence[str] | None = None,
    paired_line_color: str = "red",
    paired_line_alpha: float = 1.0,
    paired_line_width: float = 1.5,
    ax=None,
    show_legend: bool = False,
    legend_title: str = "Parameter setting",
    legend_bbox_to_anchor: tuple[float, float] = (1.05, 1),
    legend_loc: str = "upper left",
    tight_layout_rect: Sequence[float] | None = None,
    show: bool = True,
):
    """Create the boxplot + stripplot combination used across the notebooks."""
    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    plot_order = _order_present_values(data, x, order)

    sns.boxplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        order=plot_order,
        hue_order=hue_order,
        palette=palette,
        showfliers=showfliers,
        width=box_width,
        ax=ax,
    )

    existing_collection_count = len(ax.collections)
    sns.stripplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        order=plot_order,
        hue_order=hue_order,
        palette=strip_palette if strip_palette is not None else palette,
        dodge=dodge,
        jitter=strip_jitter,
        alpha=strip_alpha,
        size=strip_size,
        ax=ax,
    )

    strip_collections = ax.collections[existing_collection_count:]
    if center_strip_points and not dodge:
        _center_strip_point_clouds(strip_collections)
    elif dodge:
        _align_strip_points_to_box_width(
            strip_collections,
            box_width=box_width,
            strip_dodge_width=strip_dodge_width,
        )

    handles, labels = _clear_axis_legend(ax)

    if show_paired_lines:
        if paired_line_order is None:
            if order is None:
                raise ValueError("paired_line_order must be provided when order is not set.")
            paired_line_order = plot_order

        _add_paired_lines(
            ax,
            data,
            x=x,
            y=y,
            pair_id_col=pair_id_col,
            paired_line_order=paired_line_order,
            color=paired_line_color,
            alpha=paired_line_alpha,
            linewidth=paired_line_width,
        )

    if show_legend:
        legend_count = len(hue_order) if hue_order is not None else len(pd.unique(data[hue]))
        ax.legend(
            handles[:legend_count],
            labels[:legend_count],
            title=legend_title,
            bbox_to_anchor=legend_bbox_to_anchor,
            loc=legend_loc,
        )

    ax.set_title(title, fontsize=title_fontsize)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_fontsize)

    ax.tick_params(axis="both", labelsize=tick_fontsize)

    if ylim is not None:
        ax.set_ylim(ylim)

    if created_figure:
        if tight_layout_rect is None:
            plt.tight_layout()
        else:
            plt.tight_layout(rect=tight_layout_rect)

        if show:
            plt.show()

    return fig, ax, handles, labels


def plot_box_strip_by_group(

    data: pd.DataFrame,
    *,
    group_col: str = "algorithm",
    x: str, 
    y: str,
    hue: str, # This will select inside each algorithm e.g. config selects between diff paramater settings
    group_order: Iterable[str] | None = None, # If you want to specify the order of the algo displayed
    order: Sequence[str] | None = None, # Order for the x axis
    hue_order: Sequence[str] | None = None, # Order for the hue categories (e.g. order parameter settings)
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = (5, 4), # Come back here if you want to make figures bigger or smaller
    ylim: tuple[float, float] | list[float] | None = None, 
    show_paired_lines: bool = False,
    pair_id_col: str = "subject",
    paired_line_order: Sequence[str] | None = None,
    show_legend: bool = False,
    tight_layout_rect: Sequence[float] | None = None,
    show: bool = True,
    **plot_kwargs,
) -> list[tuple]:
    
    """
    Plot one boxplot/stripplot figure per group and return the created axes
    This code assumes that the group colum is categorical and that the group values are discrete, 
    which is the case for our use but may not be universally true. 
    If needed, this could be made more flexible by allowing a custom grouping function 
    or by adding an option to treat the group column as continuous and create bins.
    And for the purpose of our work we are going through different algorithms.
    """

    if group_order is None:
        group_values = pd.unique(data[group_col]) # Following the example, this would sort through all the provided algorithms
    else:
        group_values = [value for value in group_order if value in pd.unique(data[group_col])]

    figures = []
    for group_value in group_values:
        subset = data[data[group_col] == group_value]
        figures.append(
            plot_box_strip(
                subset,
                x=x,
                y=y,
                title=str(group_value),
                hue=hue,
                order=order,
                hue_order=hue_order,
                xlabel=xlabel,
                ylabel=ylabel,
                figsize=figsize,
                ylim=ylim,
                show_paired_lines=show_paired_lines,
                pair_id_col=pair_id_col,
                paired_line_order=paired_line_order,
                show_legend=show_legend,
                tight_layout_rect=tight_layout_rect,
                show=show,
                **plot_kwargs,
            )
        )

    return figures
