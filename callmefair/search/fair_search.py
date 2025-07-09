"""
Fair Search Module

This module provides high-level bias search functionality for the CallMeFair framework.
It implements comprehensive bias evaluation across individual attributes and their
combinations, with support for different set operations and result visualization.

The module extends BaseSearch to provide:
- Individual attribute bias evaluation
- Attribute combination analysis (2-way and 3-way)
- Set operation comparison (union, intersection, differences)
- Pretty table output for results
- Comprehensive bias summarization

Classes:
    BiasSearch: Main class for bias search and evaluation

Functions:
    pretty_print: Format results into pretty tables

Example:
    >>> from callmefair.search.fair_search import BiasSearch
    >>> searcher = BiasSearch(df, 'target', ['gender', 'race'])
    >>> table, printable = searcher.evaluate_average()
"""

from callmefair.search._search_base import BaseSearch, combine_attributes, CType
import pandas as pd
# Fancy table print
from prettytable import PrettyTable
from multiprocessing import Pool
from itertools import combinations as combine
# Suppress FutureWarning messages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Make output table more pleasant 
def pretty_print(table: list, order_key: int = 1) -> PrettyTable:
    """
    Format a list of results into a pretty table for display.
    
    This function takes a list of results and formats them into a PrettyTable
    object for better visualization. The results are sorted by the specified
    column in descending order.

    Parameters:
        table (list): List of results where first element is header row
        order_key (int): Column index to sort by (default: 1 for fairness score)

    Returns:
        PrettyTable: Formatted table ready for display

    Example:
        >>> results = [['Attribute', 'Score'], ['gender', 0.85], ['race', 0.72]]
        >>> table = pretty_print(results)
        >>> print(table)
    """
    tab = PrettyTable(table[0])
    sorted_table = sorted(table[1:], key = lambda x: x[order_key], reverse=True)
    tab.add_rows(sorted_table)
    return tab


class BiasSearch(BaseSearch):
    """
    Main class for bias search and evaluation in the CallMeFair framework.
    
    This class extends BaseSearch to provide comprehensive bias evaluation
    across individual attributes and their combinations. It supports both
    individual attribute analysis and complex attribute combination evaluation
    using different set operations.

    The class provides methods for:
    - Individual attribute bias evaluation
    - 2-way and 3-way attribute combinations
    - Set operation comparison (union, intersection, differences)
    - Pretty table output for results
    - Comprehensive bias summarization

    Attributes:
        df (pd.DataFrame): Input dataset with features and target
        label_name (str): Name of the target variable
        attribute_names (list[str]): List of sensitive attributes to evaluate

    Example:
        >>> searcher = BiasSearch(df, 'target', ['gender', 'race', 'age'])
        >>> table, printable = searcher.evaluate_average()
        >>> print(printable)
    """

    def __init__(self, df: pd.DataFrame, label_name: str, attribute_names: list[str]):
        """
        Initialize the BiasSearch object.

        Parameters:
            df (pd.DataFrame): Input dataset containing features and target variable
            label_name (str): Name of the target variable column
            attribute_names (list[str]): List of sensitive attributes to evaluate
        """
        super(BiasSearch, self).__init__(df, label_name)

        self.attribute_names = attribute_names

    def evaluate_average(self, treat_umbalance=False, iterate=10, model_name:str = 'lr'):
        """
        Evaluate bias for all individual attributes and return averaged results.
        
        This method evaluates bias for each individual sensitive attribute
        and returns both raw and normalized fairness scores. The results
        are formatted into a pretty table for easy visualization.

        Parameters:
            treat_umbalance (bool): Whether to apply NearMiss undersampling
            iterate (int): Number of iterations for robust evaluation
            model_name (str): Type of model to use ('lr', 'cat', 'xgb', 'mlp')

        Returns:
            tuple: (table_data, pretty_table) - Raw data and formatted table

        Example:
            >>> table, printable = searcher.evaluate_average(iterate=5)
            >>> print(printable)
        """
        att_dict_list = []
        for attribute in self.attribute_names:
            att = self.evaluate_attribute(attribute, treat_umbalance, iterate, model_name)
            att_dict_list.append(att)
        
        table = [['Attribute','Raw Fairness Score','Normalized Fairness score']]

        for att_, list_ in zip(self.attribute_names, att_dict_list):
            name_raw = f'{att_}_raw'
            name_overall = f'{att_}_overall'
            table.append([att_, list_[name_raw], list_[name_overall]])

        printable = pretty_print(table)
        return table, printable
    

    def evaluate_combinations(self, treat_umbalance=False, iterate=10, model_name:str = 'lr'):
        """
        Evaluate bias for all 2-way and 3-way attribute combinations.
        
        This method creates combinations of sensitive attributes using intersection
        operations and evaluates bias for each combination. It generates both
        2-way combinations (e.g., gender_race) and 3-way combinations 
        (e.g., gender_race_age).

        Parameters:
            treat_umbalance (bool): Whether to apply NearMiss undersampling
            iterate (int): Number of iterations for robust evaluation
            model_name (str): Type of model to use ('lr', 'cat', 'xgb', 'mlp')

        Returns:
            tuple: (table_data, pretty_table) - Raw data and formatted table

        Example:
            >>> table, printable = searcher.evaluate_combinations()
            >>> print(printable)
        """
        combinations_2 = list(combine(self.attribute_names, 2))
        combinations_3 = list(combine(self.attribute_names, 3))

        table = [['Attribute','Raw Fairness Score','Normalized Fairness score']] 

        for col_1, col_2 in combinations_2:
            tmp_df = combine_attributes(self.df.copy(), col1=col_1, col2=col_2,
                                        operation=CType.intersection)
            attribute = f'{col_1}_{col_2}'
            
            att_dic = self.evaluate_attribute(attribute, treat_umbalance, iterate, model_name, df_new = tmp_df)

            name_raw = f'{attribute}_raw'
            name_overall = f'{attribute}_overall'
            table.append([attribute, att_dic[name_raw], att_dic[name_overall]])

        for col_1, col_2, col_3 in combinations_3:
            tmp_df = combine_attributes(self.df.copy(), col1=col_1, col2=col_2,
                                        operation=CType.intersection)
            tmp_df = combine_attributes(tmp_df, col1=f'{col_1}_{col_2}', col2=col_3,
                                        operation=CType.intersection)
            attribute = f'{col_1}_{col_2}_{col_3}'

            att_dic = self.evaluate_attribute(attribute, treat_umbalance, iterate, model_name, df_new = tmp_df)

            name_raw = f'{attribute}_raw'
            name_overall = f'{attribute}_overall'
            table.append([attribute, att_dic[name_raw], att_dic[name_overall]]) 
        
        printable = pretty_print(table, order_key=1)
        return table, printable


    def evaluate_combination_average(self, col_1, col_2, treat_umbalance=False, iterate=10, model_name:str = 'lr'):
        """
        Evaluate bias for all set operations between two attributes.
        
        This method compares all possible set operations (union, intersection,
        differences, symmetric difference) between two attributes and evaluates
        bias for each combination. This helps understand how different ways of
        combining attributes affect bias.

        Parameters:
            col_1 (str): Name of the first attribute
            col_2 (str): Name of the second attribute
            treat_umbalance (bool): Whether to apply NearMiss undersampling
            iterate (int): Number of iterations for robust evaluation
            model_name (str): Type of model to use ('lr', 'cat', 'xgb', 'mlp')

        Returns:
            tuple: (table_data, pretty_table) - Raw data and formatted table

        Example:
            >>> table, printable = searcher.evaluate_combination_average('gender', 'race')
            >>> print(printable)
        """
        # Output labels
        table = [['Operator','Attribute','Raw Fairness Score','Normalized Fairness score']] 

        for operator in CType:
            tmp_df = combine_attributes(self.df.copy(), col1=col_1, col2=col_2, operation=operator)
            attribute = f'{col_1}_{col_2}'
            
            att_dic = self.evaluate_attribute(attribute, treat_umbalance, iterate, model_name, df_new = tmp_df)

            name_raw = f'{attribute}_raw'
            name_overall = f'{attribute}_overall'
            table.append([str(operator), attribute, att_dic[name_raw]/iterate, att_dic[name_overall]/iterate])

        printable = pretty_print(table, order_key=1)
        return table, printable

