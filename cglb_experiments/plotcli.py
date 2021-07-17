from functools import reduce
from operator import iconcat
from pathlib import Path
from typing import List, Optional, Union
import glob
import click
import json
import pandas as pd

import numpy as np
from cglb_experiments.plotting import Plotter, TablePrinter, XAxis
from cglb_experiments.utils import short_names


@click.group()
def main():
    pass


output_format_type = click.Choice(["standard", "latex", "markdown", "html", "excel"])


@main.command()
@click.option("-f", "--output-format", type=output_format_type, default="standard")
@click.option("-o", "--output", type=click.Path())
@click.option("--paths-file/--no-paths-file", default=False)
@click.argument("files", nargs=-1, type=click.Path(resolve_path=True, dir_okay=False, exists=True))
def results_table(files: List[str], output_format: str, paths_file, output: str):
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("precision", 3)

    models = dict()
    dataset_names = set()

    if paths_file:
        # File with a list of paths to files with results
        file = Path(files[0]).expanduser().resolve()
        files = file.read_text().split()

    for file in files:
        f = Path(file)
        file_text = Path(f).expanduser().read_text()
        content = json.loads(file_text)
        _content_id = content.pop("id")
        dataset, name = short_names(f)
        dataset_names.add(dataset)
        column = pd.DataFrame(content.values(), index=content.keys(), columns=[dataset])
        columns = models.get(name, column)
        if columns is not column:
            models[name] = pd.concat([columns, column], axis=1)
        else:
            models[name] = column

    dataset_names = list(dataset_names)
    full_index = []
    full_frame = pd.DataFrame()
    for model_name, frame in models.items():
        # Add empty columns
        column_diff = set(frame.columns) - set(dataset_names)
        if column_diff:
            for c in column_diff:
                frame[c] = np.nan
        # Rearrange columns to a fixed order
        frame = frame.reindex(dataset_names, axis=1)
        full_index += [(model_name, i) for i in frame.index]
        full_frame = pd.concat([full_frame, frame])

    full_index = pd.MultiIndex.from_tuples(full_index, names=["model", "metric"])
    full_frame.index = full_index

    def to_excel(frame: pd.DataFrame, filename: Optional[str] = None):
        filename = "report.xlsx" if filename is None else filename
        new_frame = frame.reset_index(0)
        writer = pd.ExcelWriter(filename, engine="xlsxwriter")
        frame.to_excel(writer, sheet_name="main")
        for i in set(new_frame.index):
            new_frame.loc[i].to_excel(writer, sheet_name=i)
        writer.save()

    formats = {
        "standard": lambda: full_frame,
        "latex": full_frame.to_latex,
        "markdown": full_frame.to_markdown,
        "html": full_frame.to_html,
        "excel": lambda: to_excel(full_frame, output),
    }

    print(formats[output_format]())


def extract_paths(filepaths: List[str]) -> List[str]:
    full_list = [glob.glob(f) for f in filepaths]
    return list(reduce(iconcat, full_list, []))


@main.command()
@click.argument("filepaths", type=str, nargs=-1)
@click.option("--summarize/--no-summarize", default=True)
@click.option(
    "--xaxis", "-x", default="elapsed_time", type=click.Choice(["elapsed_time", "iteration"])
)
def metrics(filepaths: List[str], summarize: bool, xaxis: XAxis):
    paths = extract_paths(filepaths)
    if paths == []:
        raise RuntimeError("None paths found")
    Plotter(paths).plot_metrics(xaxis, summarize)


@main.command()
@click.argument("filepaths", type=str, nargs=-1)
@click.option("--fval/--no-fval", default=True)
@click.option("--summarize/--no-summarize", default=True)
def cgstep(filepaths: List[str], fval: bool, summarize: bool):
    paths = extract_paths(filepaths)
    if paths == []:
        raise RuntimeError("No paths found")
    plotter = Plotter(paths)
    if fval:
        plotter.plot_cgstep_fval(summarize)
    else:
        plotter.plot_cgstep(summarize)
    

@main.command()
@click.argument("filepaths", type=str, nargs=-1)
@click.option("--fmt", "-f", default="latex", type=str)
def gpr_table(filepaths: List[str], fmt: str):
    paths = extract_paths(filepaths)
    if paths == []:
        raise RuntimeError("No paths found")
    t = TablePrinter(paths)
    t.print_gpr_table(fmt)
    print()


if __name__ == "__main__":
    main()
