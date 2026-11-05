"""High-level bias search over single attributes and their combinations."""

from __future__ import annotations

import warnings
from functools import reduce
from itertools import combinations

import pandas as pd
from prettytable import PrettyTable

from callmefair.search._search_base import BaseSearch, CType, combine_attributes

warnings.simplefilter(action="ignore", category=FutureWarning)

MIN_GROUP_SIZE = 10
MIN_PER_CLASS = 2
HEADER = ["Attribute", "Raw Fairness Score", "Normalized Fairness score",
          "Balanced Accuracy", "Privileged N", "Unprivileged N"]


def pretty_print(table: list, order_key: int = 1) -> PrettyTable:
    """Render ``table`` (header row first) sorted descending by column ``order_key``."""
    printable = PrettyTable(table[0])
    printable.add_rows(sorted(table[1:], key=lambda row: row[order_key], reverse=True))
    return printable


def _group_sizes(df: pd.DataFrame, attribute: str) -> tuple[int, int]:
    return int((df[attribute] == 1).sum()), int((df[attribute] == 0).sum())


def _is_evaluable(df: pd.DataFrame, attribute: str, label: str) -> bool:
    """True if the attribute has two groups of >= MIN_GROUP_SIZE and every
    (group, label) cell has >= MIN_PER_CLASS samples (needed for stratified splits)."""
    if set(df[attribute].unique()) != {0, 1}:
        return False
    if df[attribute].value_counts().min() < MIN_GROUP_SIZE:
        return False
    cells = df.groupby([attribute, label]).size()
    return len(cells) == 4 and cells.min() >= MIN_PER_CLASS


class BiasSearch(BaseSearch):
    """Fairness evaluation of individual attributes and attribute combinations."""

    def __init__(self, df: pd.DataFrame, label_name: str, attribute_names: list[str],
                 n_threads: int | None = None, **kwargs):
        super().__init__(df, label_name, n_threads, **kwargs)
        self.attribute_names = attribute_names

    def _row(self, attribute: str, df: pd.DataFrame, scores: dict) -> list:
        privileged_n, unprivileged_n = _group_sizes(df, attribute)
        return [attribute, scores[f"{attribute}_raw"], scores[f"{attribute}_overall"],
                scores[f"{attribute}_balanced_acc"], privileged_n, unprivileged_n]

    def evaluate_average(self, treat_umbalance=False, iterate=10, model_name: str = "lr"):
        """Evaluate every attribute; returns (table, PrettyTable)."""
        table = [HEADER]
        for attribute in self.attribute_names:
            scores = self.evaluate_attribute(attribute, treat_umbalance, iterate, model_name)
            table.append(self._row(attribute, self.df, scores))
        return table, pretty_print(table)

    def evaluate_combinations(self, treat_umbalance=False, iterate=10, model_name: str = "lgbm"):
        """Evaluate the intersection of every 2- and 3-attribute combination.

        Combinations without two sufficiently populated groups are skipped.
        """
        table = [HEADER]
        skipped = []
        for size in (2, 3):
            for columns in combinations(self.attribute_names, size):
                attribute = "_".join(columns)
                df = self._intersection(columns)
                if not _is_evaluable(df, attribute, self.label_name):
                    skipped.append(attribute)
                    continue
                scores = self.evaluate_attribute(attribute, treat_umbalance, iterate,
                                                 model_name, df_new=df)
                table.append(self._row(attribute, df, scores))

        if skipped:
            print(f"\nSummary: Skipped {len(skipped)} combinations due to insufficient data: "
                  f"{', '.join(skipped)}")
        return table, pretty_print(table)

    def _intersection(self, columns: tuple[str, ...]) -> pd.DataFrame:
        """AND the columns pairwise left to right; they are replaced by '_'.join(columns)."""
        def step(df, column):
            left = df.columns[-1]  # the combined column added by the previous step
            return combine_attributes(df, left, column, CType.intersection)

        first = combine_attributes(self.df, columns[0], columns[1], CType.intersection)
        return reduce(step, columns[2:], first)

    def evaluate_combination_average(self, col_1, col_2, treat_umbalance=False, iterate=10,
                                     model_name: str = "lr"):
        """Evaluate every CType combination of two attributes; returns (table, PrettyTable)."""
        table = [["Operator", "Attribute", "Raw Fairness Score", "Normalized Fairness score",
                  "Privileged N", "Unprivileged N"]]
        attribute = f"{col_1}_{col_2}"
        for operator in CType:
            df = combine_attributes(self.df, col_1, col_2, operator)
            scores = self.evaluate_attribute(attribute, treat_umbalance, iterate, model_name,
                                             df_new=df)
            privileged_n, unprivileged_n = _group_sizes(df, attribute)
            table.append([str(operator), attribute, scores[f"{attribute}_raw"],
                          scores[f"{attribute}_overall"], privileged_n, unprivileged_n])
        return table, pretty_print(table)
