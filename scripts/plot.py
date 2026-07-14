#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np


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

def plot_heatmap(ds, da, show_metadata=False, output=None):
    if "time" not in da.dims:
        raise ValueError(
            f"Variable '{da.name}' does not have a 'time' dimension. "
            f"Dimensions are: {da.dims}"
        )

    spatial_dim = get_spatial_dim(da)

    heat = da.transpose(spatial_dim, "time")

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

    ax.set_xlabel(make_label(ds["time"], "time"))
    ax.set_ylabel(spatial_dim)
    ax.set_title(str(da.attrs.get("long_name", da.name)))

    if show_metadata:
        add_metadata_box(ax, ds)

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

    args = parser.parse_args()

    plot_netcdf(
        path=args.path,
        variable=args.variable,
        kind=args.kind,
        show_metadata=args.metadata,
        reduce_mode=args.reduce,
        output=args.output,
    )


if __name__ == "__main__":
    main()