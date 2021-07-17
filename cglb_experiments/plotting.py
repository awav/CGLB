import re
from numpy.ma.extras import flatten_inplace
from scipy.signal import gaussian
from scipy.ndimage import filters
from scipy import interpolate as interp
from collections import namedtuple
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional, Union

from typing_extensions import Literal
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import rcParams
import matplotlib as mpl

import json_tricks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray


# mpl.rcParams["text.usetex"] = True  # Let TeX do the typsetting
# mpl.rcParams["text.latex.preamble"] = [
#     r"\usepackage{sansmath}",
#     r"\sansmath",
# ]  # Force sans-serif math mode (for axes labels)
# mpl.rcParams["font.family"] = "sans-serif"  # ... for regular text
# mpl.rcParams[
#     "font.sans-serif"
# ] = "Helvetica, Avant Garde, Computer Modern Sans serif"  # Choose a nice font here


ExpData = namedtuple("ExpData", "name, values, dataset, seeds")
XAxis = Literal["elapsed_time", "iteration"]

_re_cglb_sgpr = re.compile(
    "^(cglb|cglbn2m|cglbnm2|cglb_vjoint|cglb_vzero|sgpr|sgprn2m)-.*-([0-9]+)-fp(32|64)(-scipy)?$"
)
_re_cglb_sgpr_v = re.compile("^(cglb|sgpr)-.*-([0-9]+)-fp.*-(vjoint|vzero)$")
_re_cglb_sgpr_adam = re.compile("^(cglb)-.*-([0-9]+)-fp.*-(adam_0.1|adam_0.01)$")
_re_exactgp = re.compile("^(exactgp|exactgp1e2adam|exactgp1e6pos|exactgp1e2adam1e6pos)-.*")


def is_allowed_file_to_read(filename: str) -> bool:
    allowed = [
        "logs.npy",
        "logs.json",
        "results.json",
        "cgsteps.json",
        "gpr_metric.npy",
        "results.npy",
    ]
    return filename in allowed


def model_properties() -> Dict[str, Tuple[str, str]]:
    indpts_fn = lambda head, m: f"{head}-{m}"
    colors1 = ["slategrey", "tab:blue", "tab:orange", "tab:green", "tab:red"]
    colors2 = ["steelblue", "tab:gray", "tab:pink", "tab:olive", "tab:brown"]
    colors3 = ["slategrey", "tab:green", "tab:blue", "tab:red", "tab:orange"]
    colors4 = ["slategrey", "tab:green", "tab:green", "tab:red", "tab:orange"]
    ms = [100, 512, 1024, 2048, 4096, 8192]
    cglbs_vjoint = {indpts_fn("cglb_vjoint", m): (c, "-.") for m, c in zip(ms, colors3)}
    cglbs_vzero = {indpts_fn("cglb_vzero", m): (c, "--") for m, c in zip(ms, colors4)}
    cglbs = {indpts_fn("cglb", m): (c, "-") for m, c in zip(ms, colors1)}
    cglbn2ms = {indpts_fn("cglbn2m", m): (c, ":") for m, c in zip(ms, colors1)}
    cglbnm2s = {indpts_fn("cglbnm2", m): (c, "-.") for m, c in zip(ms, colors1)}
    sgprs = {indpts_fn("sgpr", m): (c, "--") for m, c in zip(ms, colors2)}
    sgprsn2m = {indpts_fn("sgprn2m", m): (c, ":") for m, c in zip(ms, colors2)}

    colors_01 = ["slategrey", "navy", "coral", "darkcyan", "indianred"]
    colors_001 = [
        "slategrey",
        "midnightblue",
        "darksalmon",
        "darkslategray",
        "brown",
    ]
    cglbs_adam_01 = {indpts_fn("cglb_adam_0.1", m): (c, "--") for m, c in zip(ms, colors_01)}
    cglbs_adam_001 = {indpts_fn("cglb_adam_0.01", m): (c, "-.") for m, c in zip(ms, colors_001)}

    exactgp = {
        "exactgp": ("tab:purple", "-."),
        "exactgp1e6pos": ("navy", "--"),
        "exactgp1e2adam": ("violet", "-"),
        "exactgp1e2adam1e6pos": ("royalblue", (0, (5, 5))),
    }
    return {
        **cglbn2ms,
        **cglbnm2s,
        **cglbs_vzero,
        **cglbs_vjoint,
        **cglbs,
        **sgprs,
        **exactgp,
        **sgprsn2m,
        **cglbs_adam_01,
        **cglbs_adam_001,
    }


def model_color(key: str) -> str:
    prop = model_properties()[key]
    return prop[0]


def model_linestyle(key: str) -> str:
    prop = model_properties()[key]
    return prop[-1]


def model_label(key: str) -> str:
    if key == "exactgp":
        return "Iterative GP with $\sigma^2_{min}=1\mathrm{e}{-4}}$, $lr=0.1$"
        # return "Iterative GP, \n$\sigma^2_{min}=1\mathrm{e}{-4}}$, $lr=0.1$"
        # return "Iterative GP"

    if key == "exactgp1e6pos":
        return r"Iterative GP with $\sigma^2_{min}=1\mathrm{e}{-6}}$, $lr=0.1$"

    if key == "exactgp1e2adam":
        return r"Iterative GP with $\sigma^2_{min}=1\mathrm{e}{-4}}$, $lr=0.01$"

    if key == "exactgp1e2adam1e6pos":
        return r"Iterative GP with $\sigma^2_{min}=1\mathrm{e}{-6}}$, $lr=0.01$"

    if key.startswith("cglb_vjoint"):
        return key.replace("cglb_vjoint", "CGLB-$v_{opt}$")

    if key.startswith("cglb_vzero"):
        return key.replace("cglb_vzero", "CGLB-$v_0$")

    if key.startswith("cglb_adam_0.1"):
        ip = key.split("-")[-1]
        return f"CGLB-{ip} Adam lr=$0.1$"

    if key.startswith("cglb_adam_0.01"):
        ip = key.split("-")[-1]
        return f"CGLB-{ip} Adam lr=$0.01$"

    if key.startswith("cglbn2m"):
        return key.replace("cglbn2m", "CGLB-logdet-$\mathcal{O}(N^2M)$")

    if key.startswith("cglbnm2"):
        return key.replace("cglbnm2", "CGLB-logdet-$\mathcal{O}(NM^2)$")

    if key.startswith("sgprn2m"):
        return key.replace("sgprn2m", "SGPR-logdet-$\mathcal{O}(N^2M)$")

    if key.startswith("cglb"):
        return key.replace("cglb", "CGLB")

    if key.startswith("sgpr"):
        return key.replace("sgpr", "SGPR")

    raise RuntimeError("Unknown key passed")


def _convert_uid_to_name(uid: str) -> str:
    re1 = _re_cglb_sgpr
    sub1 = r"\1-\2"
    re2 = _re_exactgp
    sub2 = r"\1"
    re3 = _re_cglb_sgpr_v
    sub3 = r"\1_\3-\2"

    re4 = _re_cglb_sgpr_adam
    sub4 = r"\1_\3-\2"
    regexps = [(re1, sub1), (re2, sub2), (re3, sub3), (re4, sub4)]

    name = None
    for reg, sub in regexps:
        if reg.match(uid) is not None:
            name = reg.sub(sub, uid)
            break

    if name is None:
        raise RuntimeError(f"Unknown uid format: {uid}")

    return name


def _parse_experiment_path(filepath: str) -> Tuple[str, str, str]:
    parts = filepath.split("/")
    if not is_allowed_file_to_read(parts[-1]):
        raise RuntimeError(f"Unknown file {parts[-1]}")

    seed = parts[-2]
    uid = parts[-3]
    dataset = parts[-4].replace("Wilson_", "")

    name = _convert_uid_to_name(uid)

    return name, dataset, seed


def _sorted_model_names_fn(name: str):
    parts = name.split("-", maxsplit=1)
    if len(parts) == 2:
        head, num = parts
        return (head, int(num))
    elif len(parts) == 1:
        large_int = int(1e10)
        return (parts[0], large_int)


def _load_experiment_values(path: str) -> Dict:
    _path = Path(path)
    if _path.suffix == ".npy":
        arr: ndarray = np.load(path, allow_pickle=True)
        return arr.item()
    elif _path.suffix == ".json":
        return json_tricks.load(path)
    raise RuntimeError(f"Unknown file format '{_path}'")


def _load_expdata(filepaths: List[str]) -> List[ExpData]:
    groups = dict()
    for fp in filepaths:
        resolved_path = Path(fp).expanduser().resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(resolved_path)
        rfp = str(resolved_path)
        name, dataset, seed = _parse_experiment_path(rfp)
        values = _load_experiment_values(rfp)
        if name in groups:
            exp = groups[name]
            exp.values.append(values)
            exp.seeds.append(seed)
        else:
            groups[name] = ExpData(name, [values], dataset, [seed])

    return list(groups.values())


def _try_select_values_from_expdata(keys: List[str], expdatas: List[ExpData]) -> Dict:
    for k in keys:
        try:
            expdata = _select_values_from_expdata(k, expdatas)
        except KeyError:
            continue
        break
    return expdata


def _select_values_from_expdata(key: str, expdatas: List[ExpData]) -> Dict:
    output = dict()

    def select_fn(expdata: ExpData) -> Tuple[str, Sequence]:
        extracted = []
        for value in expdata.values:
            if key not in value:
                raise KeyError(f"{key} record not found in traces")
            v = value[key]
            if not isinstance(v, (float, int, ndarray, list)):
                raise TypeError(f"Unknown value type met in experiment data: {v}")
            extracted.append(np.array(v))
        return expdata.name, extracted

    return dict(map(select_fn, expdatas))


def _find_max_sequence(data: Dict[str, Sequence]) -> Sequence:
    max_seq = []
    for _, values in data.items():
        for v in values:
            max_seq = v if len(v) > len(max_seq) else max_seq
    return max_seq


def _fill_gaps(
    data: Dict[str, Sequence], max_len: int, use_nans: bool = True
) -> Dict[str, Sequence]:
    """
    Args:
        non_internal_nans: Boolean that determines whether a sequence will be filled with the last value till largest internal sequence size or not.
    """
    internal_max_lens = {}
    max_len = 0
    for k, vs in data.items():
        internal_max_len = 0
        for v in vs:
            vlen = len(v)
            max_len = vlen if vlen > max_len else max_len
            internal_max_len = vlen if vlen > internal_max_len else internal_max_len
        internal_max_lens[k] = internal_max_len

    def update_fn(kv):
        key, values = kv
        new_values = []
        for v in values:
            v_len = len(v)
            if v_len < max_len:
                if use_nans:
                    gap_len = max_len - v_len
                    gap = [np.nan] * gap_len
                else:
                    gap_len = internal_max_lens[key] - v_len
                    gap = [v[-1]] * gap_len
                new_v = np.concatenate([v, gap])
                new_values.append(new_v)
            else:
                new_values.append(v)
        return key, np.stack(new_values)

    return dict(map(update_fn, data.items()))


# Because of differences in TF/Pytorch implementations and data-flow
# graph compilations, the startup time can vary drastically. Aligning is
# required to make plots relative in time.
_align_time = False


def _plottable_metric_data(metrics, times, nan_gap: bool = False):
    output = dict()
    mins = []

    if _align_time:
        for ts in times.values():
            first = []
            for t in ts:
                first.append(t[0])
            mins.append(np.array(first))

        time_mins = np.array(mins).min(axis=0)

        for key, vs in times.items():
            for i in range(len(vs)):
                start = times[key][i][0]
                times[key][i] -= start - time_mins[i]

    for key in metrics.keys():
        ys = metrics[key]
        xs = times[key]

        assert len(ys) == len(xs)

        max_len = 0
        min_x = np.inf
        max_x = 0
        new_xs: ndarray = np.array([])
        for x in xs:
            if len(x) > max_len:
                max_len = len(x)
                new_xs = x
            min_x = x[0] if x[0] < min_x else min_x
            max_x = x[-1] if x[-1] > max_x else max_x

        # new_xs: ndarray = np.linspace(min_x, max_x, max_len)
        new_ys = []
        for i in range(len(ys)):
            x, y = xs[i], ys[i]

            y_len = len(y)
            gap_len = max_len - y_len if y_len < max_len else 0

            if nan_gap:
                tck = interp.splrep(x, y, s=0, k=1)
                new_y = interp.splev(new_xs[:y_len], tck, der=0)

                if gap_len != 0:
                    gap_y = [np.nan] * gap_len
                    new_y = np.array([*y, *gap_y])

                new_ys.append(new_y)
            else:
                if gap_len != 0:
                    gap_y = [y[-1]] * gap_len
                    y = np.array([*y, *gap_y])
                    gap_x = np.linspace(x[-1], max_x, gap_len + 1)
                    x = np.array([*x[:-1], *gap_x])

                tck = interp.splrep(x, y, s=0, k=1)
                new_y = interp.splev(new_xs, tck, der=0)
                new_ys.append(new_y)

        output[key] = (new_xs[:-5], np.stack(new_ys)[:, :-5])

    return output


@dataclass
class Plotter:
    filepaths: InitVar[List[str]]
    expdatas: List[ExpData] = field(init=False)

    def __post_init__(self, filepaths: List[str]):
        self.expdatas = _load_expdata(filepaths)

    def plot_metric_vs_time(
        self,
        yaxis: str,
        ax: Axes,
        summarize: bool = True,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        legend: bool = True,
    ):
        xaxis = "elapsed_time"
        metrics = _select_values_from_expdata(yaxis, self.expdatas)
        time = _select_values_from_expdata(xaxis, self.expdatas)

        # TODO: nan_gap=False has better characteristics
        xy_data = _plottable_metric_data(metrics, time, nan_gap=False)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        keys = sorted(xy_data.keys(), key=_sorted_model_names_fn, reverse=True)

        for k in keys:
            x, ys = xy_data[k]
            color = model_color(k)
            label = model_label(k)
            style = model_linestyle(k)

            if summarize:
                y = np.nanmedian(ys, axis=0)
                y_up, y_down = np.nanquantile(ys, [0.25, 0.75], axis=0)

                ax.plot(x, y, label=label, color=color, linestyle=style)
                ax.fill_between(x, y_up, y_down, color=color, alpha=0.2)
            else:
                for v in value:
                    ax.plot(iters, v, color=color, label=label)

        # split_legend = True
        split_legend = False
        ncol = 2 if len(keys) > 4 and split_legend else 1
        if legend:
            ax.legend(ncol=ncol, fontsize=9)

    def plot_metrics(self, xaxis: XAxis, summarize: bool):
        xname = "time (sec)" if xaxis == "elapsed_time" else "iterations"

        vs_time_fn = self.plot_metric_vs_time
        vs_iter_fn = self.plot_metric_vs_iteration
        plot_metric_fn = vs_time_fn if xaxis == "elapsed_time" else vs_iter_fn

        include_rmse_nlpd: bool = True
        include_nlml: bool = False

        opts = dict(summarize=summarize, xlabel=xname)

        if include_rmse_nlpd:
            # figsize = None  # Default option
            # figsize = (3.5, 3.9)  # Rebuttal option
            figsize = (8.5, 2.9)  # Rebuttal option
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize)
            # fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=figsize, sharex=True)
            plot_metric_fn("test/rmse", ax1, ylabel="RMSE", **opts)
            plot_metric_fn("test/nlpd", ax2, ylabel="NLPD", **opts)
            # plot_metric_fn("test/nlpd", ax2, ylabel="NLPD", legend=False, **opts)
            fig.subplots_adjust(wspace=0, hspace=0)
            # plt.tight_layout(h_pad=0)
            plt.tight_layout(w_pad=0.7)
            plt.show()

        if include_nlml:
            # figsize = (3., 2.3)  # Rebuttal option
            # figsize = None  # Default option
            figsize = (5.0, 2.6)  # Camera
            fig, axis = plt.subplots(nrows=1, ncols=1, figsize=figsize)
            axis.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useOffset=False)
            # axis.yaxis.set_major_formatter(
            #     mpl.ticker.ScalarFormatter(useMathText=True, useOffset=False)
            # )
            plot_metric_fn("loss", axis, ylabel="NLML", **opts)
            plt.tight_layout()
            plt.show()

    def plot_cgstep(self, summarize: bool):
        self.plot_metric_vs_iteration(
            "cg/steps",
            summarize=summarize,
            gen_iterations=True,
            ylable="CG steps",
            xlable="Iteration",
        )
        plt.show()

    def plot_cgstep_fval(self, summarize: bool):
        rcParams.update({"figure.autolayout": True})
        # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        figsize = (5.2, 3.2)
        fig = plt.figure(figsize=figsize)
        # ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        # ax2 = fig.add_axes([0.55, 0.65, 0.3, 0.2])

        # ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax1 = fig.add_axes([0.15, 0.15, 0.81, 0.81])
        ax1.set_ylabel("CG step")
        ax1.set_xlabel("Iteration")

        log1 = 1  # Use log(x + 1) transform
        # ax2 = fig.add_axes([0.55, 0.25, 0.3, 0.25])
        # ax2 = fig.add_axes([0.55, 0.4, 0.3, 0.25])
        ax2 = fig.add_axes([0.62, 0.53, 0.3, 0.25])

        name = "steps-per-feval"

        metrics = _select_values_from_expdata(name, self.expdatas)
        maxsize = len(_find_max_sequence(metrics))
        iters = np.array(range(maxsize))
        metrics = _fill_gaps(metrics, maxsize)
        keys = sorted(metrics.keys(), key=_sorted_model_names_fn)

        nmax = 1500
        colors, labels, vs = [], [], []
        for k in keys:
            v = metrics[k][..., :nmax]
            vflat = v.flatten()
            vs.append(vflat[np.logical_not(np.isnan(vflat))] + log1)
            v_mean = np.nanmean(v, axis=0)
            v_iters = np.array(range(nmax))
            label = model_label(k)
            labels.append(label)

            b = gaussian(50, 5)
            ga = filters.convolve1d(v_mean, b / b.sum())
            color = model_color(k)
            ax1.plot(v_iters, ga + log1, label=label, color=color)

            colors.append(color)
            ax1.plot(v_iters, v_mean + log1, alpha=0.1, color=color, label=label)

        whis = (5, 95)

        bps = ax2.boxplot(vs, vert=False, notch=False, sym="", labels=labels, whis=whis)

        medians = bps["medians"]
        for i, m in enumerate(medians):
            m.set(color=colors[i], linestyle="-", linewidth=2.0)

        boxes = bps["boxes"]
        for i, box in enumerate(boxes):
            box.set(color=colors[i])

        whiskers = bps["whiskers"]
        whiskers = [whiskers[i : i + 2] for i in range(0, len(whiskers), 2)]
        for i, (wbot, wtop) in enumerate(whiskers):
            wtop.set(color=colors[i], linestyle="--")
            wbot.set(color=colors[i], linestyle="--")

        caps = bps["caps"]
        caps = [caps[i : i + 2] for i in range(0, len(caps), 2)]
        for i, (cbot, ctop) in enumerate(caps):
            ctop.set(color=colors[i])
            cbot.set(color=colors[i])

        ax1.axhline(
            y=0,
            xmin=0,
            xmax=1,
            color="grey",
            linestyle="--",
            linewidth=0.8,
        )

        if log1 == 1:
            ax1.set_yscale("log")
            ax2.set_xscale("log")

        plt.tight_layout(pad=0)
        # plt.gcf().subplots_adjust(bottom=0.5)
        plt.show()

    def plot_metric_vs_iteration(
        self,
        yaxis: str,
        summarize: bool = True,
        gen_iterations: bool = False,
        fig_ax: Optional[Tuple[Figure, Axes]] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
    ):
        fig, ax = plt.subplots(1, 1) if fig_ax is None else fig_ax

        metrics = _select_values_from_expdata(yaxis, self.expdatas)
        maxsize = len(_find_max_sequence(metrics))
        metrics = _fill_gaps(metrics, maxsize)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        keys = sorted(metrics.keys(), key=_sorted_model_names_fn, reverse=True)

        # TODO: we assume that all data collected with the same holdout steps
        if gen_iterations:
            xname = "iteration"
            xs_exaustive = _select_values_from_expdata(xname, self.expdatas)
            x = _find_max_sequence(xs_exaustive)
        else:
            x = range(maxsize)

        for k in keys:
            value = metrics[k]
            color = model_color(k)
            label = model_label(k)
            style = model_linestyle(k)

            if summarize:
                y = np.nanmedian(value, axis=0)
                y_up, y_down = np.nanquantile(value, [0.25, 0.75], axis=0)

                ax.plot(x, y, label=label, color=color, linestyle=style)
                ax.fill_between(x, y_up, y_down, color=color, alpha=0.2, linestyle=style)
            else:
                for v in value:
                    ax.plot(iters, v, color=color, label=label, linestyle=style)

        ax.legend(fontsize=9)
        return fig, ax


@dataclass
class TablePrinter:
    filepaths: InitVar[List[str]]
    metric_data: List[ExpData] = field(init=False)

    def __post_init__(self, filepaths: List[str]):
        self.metric_data = _load_expdata(filepaths)

    def print_gpr_table(self, fmt: Optional[str] = None):
        data = self.metric_data
        lml = _try_select_values_from_expdata(["lml", "loss"], data)
        nlpd = _select_values_from_expdata("test/nlpd", data)
        rmse = _select_values_from_expdata("test/rmse", data)

        keys = sorted(lml.keys(), key=_sorted_model_names_fn, reverse=True)
        table = pd.DataFrame()
        for k in keys:
            name = model_label(k)
            col_names = ["LML", "NLPD", "RMSE"]

            def metric_round(name, value):
                if name == "LML":
                    return np.round(value, 1)
                return np.round(value, 3)

            def means_row(**kwargs):
                avgfn = lambda k, v: metric_round(k, np.mean(v))
                stdfn = lambda k, v: metric_round(k, np.std(v))

                avgs = {k: avgfn(k, v) for k, v in kwargs.items()}
                stds = {k: stdfn(k, v) for k, v in kwargs.items()}
                vals = {k: f"{avgs[k]:.03f} ±{stds[k]:.03f}" for k in avgs.keys()}

                row_values = [vals[col_names[i]] for i in range(len(col_names))]
                row = pd.DataFrame([row_values], columns=col_names, index=[name])
                return row

            def medminmax_row(**kwargs):
                cols = []
                vals = []
                for col in col_names:
                    v = kwargs[col]

                    medv = metric_round(col, np.median(v))
                    minv = metric_round(col, np.min(v))
                    maxv = metric_round(col, np.max(v))

                    head = lambda x: f"{col} ({x})"
                    cols = [*cols, head("median"), head("min"), head("max")]
                    vals = [*vals, medv, minv, maxv]
                row = pd.DataFrame([vals], columns=cols, index=[name])
                return row

            def medians_row(**kwargs):
                vals = []
                for col in col_names:
                    v = kwargs[col]
                    medv = metric_round(col, np.median(v))
                    vals = [*vals, medv]
                row = pd.DataFrame([vals], columns=col_names, index=[name])
                return row

            row = medians_row(LML=lml[k], NLPD=nlpd[k], RMSE=rmse[k])
            table = pd.concat([table, row])

        def formatted(t):
            if fmt is None:
                return t.to_string()
            elif fmt.lower() == "latex":
                return t.to_latex()

        for_printing = formatted(table)
        print(for_printing)
