#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from matplotlib.widgets import Slider


def make_label(da, fallback):
    long_name = da.attrs.get("long_name", fallback)
    units = da.attrs.get("units", "")

    if units:
        return f"{long_name} [{units}]"

    return str(long_name)


def metadata_text(ds):
    lines = []

    for key, value in ds.attrs.items():
        lines.append(f"{key}: {value}")

    return "\n".join(lines)


def choose_variable(ds):
    data_vars = list(ds.data_vars)

    if len(data_vars) == 0:
        raise KeyError("The NetCDF file does not contain any data variables.")

    if len(data_vars) == 1:
        return data_vars[0]

    available = ", ".join(data_vars)

    raise KeyError(
        "The NetCDF file contains multiple data variables. "
        f"Available variables: {available}. "
        "Use --variable VARIABLE_NAME."
    )


def reduce_to_time_series(da, reduce_mode):
    if "time" not in da.dims:
        raise ValueError(
            f"Variable '{da.name}' does not have a 'time' dimension. "
            f"Dimensions are: {da.dims}"
        )

    other_dims = [dim for dim in da.dims if dim != "time"]

    if not other_dims:
        return da

    if reduce_mode == "mean":
        return da.mean(dim=other_dims)

    if reduce_mode == "sum":
        return da.sum(dim=other_dims)

    if reduce_mode == "first":
        indexers = {dim: 0 for dim in other_dims}
        return da.isel(indexers)

    raise ValueError(f"Unknown reduction mode: {reduce_mode}")


def get_spatial_dim(da):
    spatial_dims = [dim for dim in da.dims if dim != "time"]

    if len(spatial_dims) == 0:
        raise ValueError(
            f"Variable '{da.name}' is one-dimensional. "
            "A heat plot needs a variable with dimensions like (time, site)."
        )

    if len(spatial_dims) > 1:
        raise ValueError(
            f"Variable '{da.name}' has more than one non-time dimension: {spatial_dims}. "
            "This script expects something like (time, site)."
        )

    return spatial_dims[0]


def plot_line(ds, da, show_metadata=False, reduce_mode="mean", output=None):
    y = reduce_to_time_series(da, reduce_mode)

    fig, ax = plt.subplots(figsize=(8, 5))

    y.plot(ax=ax)

    ax.set_xlabel(make_label(ds["time"], "time"))
    ax.set_ylabel(make_label(da, da.name))
    ax.set_title(str(da.attrs.get("long_name", da.name)))
    ax.grid(True)

    if show_metadata:
        add_metadata_box(ax, ds)

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=300)
        print(f"Saved plot to {output}")
    else:
        plt.show()

def is_correlation_variable(da):
    name = (da.name or "").lower()
    long_name = str(da.attrs.get("long_name", "")).lower()

    return (
        "correlation" in name
        or "correlation" in long_name
        or "pearson" in name
        or "pearson" in long_name
    )

def plot_heatmap(
    ds,
    da,
    show_metadata=False,
    output=None,
    overlay_moments=False,
    threshold_slider=False,
    initial_threshold=None,
):
    if "time" not in da.dims:
        raise ValueError(
            f"Variable '{da.name}' does not have a 'time' dimension. "
            f"Dimensions are: {da.dims}"
        )

    spatial_dim = get_spatial_dim(da)

    # Shape after transpose:
    # rows    -> spatial coordinate
    # columns -> time
    heat = da.transpose(spatial_dim, "time")

    time_values = np.asarray(ds["time"].values)
    spatial_values = np.asarray(ds[spatial_dim].values)
    heat_values = np.asarray(heat.values)

    finite_values = heat_values[np.isfinite(heat_values)]

    if finite_values.size == 0:
        raise ValueError(
            f"Variable '{da.name}' does not contain finite values."
        )

    value_min = float(np.min(finite_values))
    value_max = float(np.max(finite_values))

    if initial_threshold is None:
        initial_threshold = 0.5 * (value_min + value_max)

    initial_threshold = float(
        np.clip(initial_threshold, value_min, value_max)
    )

    if threshold_slider and output:
        raise ValueError(
            "An interactive threshold slider cannot be used together with --output."
        )

    if threshold_slider:
        # Reserve space beneath the plot for the slider.
        fig, ax = plt.subplots(figsize=(9, 6))
        fig.subplots_adjust(bottom=0.18)
    else:
        fig, ax = plt.subplots(figsize=(9, 5))

    plot_kwargs = {
        "ax": ax,
        "x": "time",
        "y": spatial_dim,
        "add_colorbar": True,
        "cbar_kwargs": {
            "label": make_label(da, da.name),
        },
    }

    if is_correlation_variable(da):
        plot_kwargs["vmin"] = -1.0
        plot_kwargs["vmax"] = 1.0
        plot_kwargs["cmap"] = "RdBu_r"

    heat.plot.imshow(**plot_kwargs)

    threshold_contour = None

    def draw_threshold(threshold):
        nonlocal threshold_contour

        # Remove the previous contour.
        if threshold_contour is not None:
            threshold_contour.remove()
            threshold_contour = None

        # True where the selected threshold is reached or exceeded.
        threshold_mask = (
            np.isfinite(heat_values)
            & (heat_values >= threshold)
        )

        if np.any(threshold_mask) and not np.all(threshold_mask):
            threshold_contour = ax.contour(
                time_values,
                spatial_values,
                threshold_mask.astype(float),
                levels=[0.5],
                linewidths=1.5,
            )
        else:
            threshold_contour = None

        ax.set_title(
            f"{da.attrs.get('long_name', da.name)} "
            f"— threshold ≥ {threshold:.6g}"
        )

        fig.canvas.draw_idle()

    if threshold_slider:
        slider_ax = fig.add_axes([0.18, 0.06, 0.64, 0.035])

        threshold_control = Slider(
            ax=slider_ax,
            label="Energy threshold",
            valmin=value_min,
            valmax=value_max,
            valinit=initial_threshold,
        )

        draw_threshold(initial_threshold)

        threshold_control.on_changed(draw_threshold)

        # Keep a reference alive for the lifetime of the figure.
        fig._threshold_slider = threshold_control

    elif initial_threshold is not None:
        draw_threshold(initial_threshold)

    if overlay_moments:
        if "first_moment_total_energy" not in ds:
            raise KeyError(
                "Overlay requires variable "
                "'first_moment_total_energy' in the dataset."
            )

        centroid = ds["first_moment_total_energy"]

        ax.plot(
            ds["time"].values,
            centroid.values,
            linewidth=2.0,
            label="centroid",
        )

        if "total_energy_spread" in ds:
            spread = ds["total_energy_spread"]

            lower = centroid.values - spread.values
            upper = centroid.values + spread.values

            ax.fill_between(
                ds["time"].values,
                lower,
                upper,
                alpha=0.25,
                label="centroid ± spread",
            )

        ax.legend(loc="upper right")

    ax.set_xlabel(make_label(ds["time"], "time"))
    ax.set_ylabel(spatial_dim)

    if not threshold_slider:
        ax.set_title(str(da.attrs.get("long_name", da.name)))

    if show_metadata:
        add_metadata_box(ax, ds)

    if not threshold_slider:
        fig.tight_layout()

    if output:
        fig.savefig(output, dpi=300)
        print(f"Saved plot to {output}")
    else:
        plt.show()


def add_metadata_box(ax, ds):
    text = metadata_text(ds)

    if not text:
        return

    ax.text(
        0.98,
        0.98,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.85,
        },
    )


def plot_netcdf(
    path,
    variable=None,
    kind="heat",
    show_metadata=False,
    reduce_mode="mean",
    output=None,
    overlay_moments=False,
    threshold_slider=False,
    initial_threshold=None,
):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with xr.open_dataset(path, decode_times=False) as ds:
        if "time" not in ds:
            raise KeyError("NetCDF file does not contain a variable or coordinate named 'time'.")

        if variable is None:
            variable = choose_variable(ds)

        if variable not in ds:
            available = ", ".join(ds.data_vars)
            raise KeyError(
                f"Variable '{variable}' not found. "
                f"Available variables: {available}"
            )

        # Info output for min, max, NaN and inf
        da = ds[variable]
        values = da.values
        print(f"{variable}: min={np.nanmin(values)}, max={np.nanmax(values)}")
        print(f"{variable}: has_nan={np.isnan(values).any()}, has_inf={np.isinf(values).any()}")

        if kind == "line":
            plot_line(
                ds=ds,
                da=da,
                show_metadata=show_metadata,
                reduce_mode=reduce_mode,
                output=output,
            )
        elif kind == "heat":
            plot_heatmap(
                ds=ds,
                da=da,
                show_metadata=show_metadata,
                output=output,
                overlay_moments=overlay_moments,
                threshold_slider=threshold_slider,
                initial_threshold=initial_threshold,
            )
        else:
            raise ValueError(f"Unknown plot kind: {kind}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot NetCDF simulation data."
    )

    parser.add_argument(
        "path",
        help="Path to the NetCDF file.",
    )

    parser.add_argument(
        "--variable",
        "-v",
        default=None,
        help=(
            "Variable to plot. If omitted, the script automatically uses the only "
            "data variable in the file."
        ),
    )

    parser.add_argument(
        "--kind",
        choices=["heat", "line"],
        default="heat",
        help="Plot type. Default: heat.",
    )

    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Show global NetCDF metadata inside the plot.",
    )

    parser.add_argument(
        "--reduce",
        choices=["mean", "sum", "first"],
        default="mean",
        help=(
            "For line plots only: how to reduce non-time dimensions such as 'site'. "
            "Default: mean."
        ),
    )

    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional output image path. If omitted, the plot is shown interactively.",
    )

    parser.add_argument(
        "--overlay-moments",
        action="store_true",
        help="For heat plots, overlay centroid ± spread if available.",
    )

    parser.add_argument(
        "--threshold-slider",
        action="store_true",
        help=(
            "For interactive heat plots, show a slider and outline all "
            "values greater than or equal to the selected threshold."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Initial energy threshold. With --threshold-slider this sets "
            "the slider's initial value."
        ),
    )

    args = parser.parse_args()

    plot_netcdf(
        path=args.path,
        variable=args.variable,
        kind=args.kind,
        show_metadata=args.metadata,
        reduce_mode=args.reduce,
        output=args.output,
        overlay_moments=args.overlay_moments,
        threshold_slider=args.threshold_slider,
        initial_threshold=args.threshold,
    )


if __name__ == "__main__":
    main()