"""
    This helpers.py contains various functions for data science tasks
    in Jupyter notebooks, including plotting, data cleaning,
    preprocessing, and statistical analysis.

    The plotting functions provide customizable visualizations
    such as line plots, scatter plots, histograms, and box plots.

    The data cleaning and preprocessing functions handle tasks
    such as missing data, scaling, normalization, and transformation.

    The statistical analysis functions include measures of central
    tendency, variance, correlation, and regression analysis, aiding
    in drawing conclusions from data.

    and more...
"""
import copy
import logging

import lightgbm as lgb
import matplotlib.pylab as plt
import numpy as np
import optuna
import pandas as pd
import requests
import scipy
import scipy.stats as stats
import seaborn as sns
import squarify
from imblearn.pipeline import Pipeline
from IPython.display import display
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from palettable.colorbrewer import qualitative
from scipy.sparse import csr_matrix
from scipy.stats import f_oneway
from sklearn import metrics
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import IncrementalPCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              ExtraTreesClassifier, GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import (ElasticNet, Lasso, LinearRegression,
                                  LogisticRegression, Ridge)
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier

import geopandas as gpd


def count_and_plot_mape_vals(predictions: dict, y_val: np.ndarray) -> None:
    """
    Compute mean absolute percentage error (MAPE) for each model in predictions and display the results in a bar plot.

    Parameters:
    -----------
    predictions : dict
        A dictionary of predictions returned by different models.
    y_val : numpy.ndarray
        True values of the target variable.

    Returns:
    --------
    None
    """
    # Compute MAPE for each model and store the results in a dictionary
    mape_res = {}
    for model in list(predictions.items()):
        y_pred = model[1]
        mape = metrics.mean_absolute_percentage_error(y_val, y_pred)
        mape_res[type(model[0].named_steps["model"]).__name__] = mape

    # Convert the dictionary to a DataFrame and sort by the MAPE values
    df = pd.DataFrame(list(mape_res.items()), columns=["model", "mape"])
    df = df.sort_values(by="mape", ascending=False)
    display(df)

    # Display the bar plot of model vs. MAPE values
    _, _ = plt.subplots(1, 1, figsize=(15, 8))
    plt.bar(df["model"], df["mape"], color="#695280")
    plt.xlabel("model")
    plt.ylabel("mape")
    plt.title("Barplot of model vs. mape vals")
    plt.show()

 
def perform_ANOVA_test(df: pd.DataFrame, all_common_cols: list) -> (dict, dict):
    """
    This function performs an ANOVA test on categorical columns in a pandas DataFrame against the "int_rate" column.
    It returns two dictionaries:
    - rejected_cols: containing the columns whose null hypothesis is rejected with a p-value less than 0.05
    - not_rejected_cols: containing the columns whose null hypothesis is not rejected with a p-value greater than or equal to 0.05

    Args:
    - df: The pandas DataFrame to perform the ANOVA test on
    - all_common_cols: A list of column names that are common across all the dataframes that will be compared

    Returns:
    - A tuple containing two dictionaries

    Example usage:
    rejected_cols, not_rejected_cols = perform_ANOVA_test(df, ['col1', 'col2'])
    """
    data = df.copy()

    cat_cols = [
        col
        for col in data.columns
        if data[col].dtype == "object" and col not in all_common_cols
    ]

    cat_cols.append("int_rate")

    data = data[cat_cols].dropna()

    cat_cols.remove("int_rate")

    rejected_cols = {}
    not_rejected_cols = {}

    for col in cat_cols:
        groups = [
            data[data[col] == category]["int_rate"] for category in data[col].unique()
        ]

        _, p_val = f_oneway(*groups)

        if p_val < 0.05:
            rejected_cols[col] = p_val
        else:
            not_rejected_cols[col] = p_val

    return rejected_cols, not_rejected_cols


def plot_roc(X, Y, data_type, models):
    """
    Plot the Receiver Operating Characteristic (ROC) curves for multiple models using a given dataset.

    Parameters:
    X (numpy.ndarray or pandas.DataFrame): The feature matrix for the dataset.
    Y (numpy.ndarray or pandas.Series): The target variable for the dataset.
    data_type (str): A string indicating whether the data is training, validation, or testing data.
    models (list): A list of Scikit-learn Pipeline objects, where each Pipeline contains a model object
    with a "predict_proba" method and has been fitted on the input dataset.

    Returns:
    None, but a ROC curve plot is generated.
    """ 
    _, ax = plt.subplots(1, 1, figsize=(15, 8))

    model_names = []
    for model in models:
        model_name = type(model.named_steps["model"]).__name__
        model_names.append(model_name)

        y_pred_proba = model.predict_proba(X)[::, 1]
        fpr, tpr, _ = metrics.roc_curve(Y, y_pred_proba)
        plt.plot(fpr, tpr, label=model_name)
        print(f"{model_name} AUC for {data_type} data {metrics.auc(fpr, tpr)}")

    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")

    ax.set_title(f"ROC Curve for {model_names}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="upper right")
    plt.show()


def plot_scatter_plots(df, col1, col2):
    """
    Plot a scatter plot with a regression line between two columns of a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data to be plotted.
    col1 : str
        The name of the column to be plotted on the x-axis.
    col2 : str
        The name of the column to be plotted on the y-axis.

    Returns
    -------
    None
        The function displays the plot but does not return anything.
    """
    fig, axes = plt.subplots(1, 1, figsize=(16, 10))
    sns.regplot(
        x=col1,
        y=col2,
        data=df,
        marker="o",
        color=qualitative.Set2_6.hex_colors[1],
        ax=axes,
    )
    fig.suptitle(
        f"Relationship of {col1} and {col2}",
        fontsize=14,
    )
    plt.show()


def make_mi_scores(X, y):
    """
    Calculate the mutual information scores between features and target variable.

    Parameters
    ----------
    X : pandas.DataFrame
        The input data containing the features to be evaluated.
    y : pandas.Series
        The target variable used to calculate the mutual information scores.

    Returns
    -------
    pandas.Series
        A series containing the mutual information scores for each feature in X.
        The series is sorted in descending order based on the scores.
    """ 
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()

    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_classif(
        X, y, discrete_features=discrete_features, random_state=0
    )
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    """
    Plot a horizontal bar chart of mutual information scores.

    Parameters
    ----------
    scores : pandas.Series
        A series containing the mutual information scores to be plotted.

    Returns
    -------
    None
        The function displays the plot but does not return anything.
    """
    plt.figure(dpi=100, figsize=(15, 8))
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores, color=qualitative.Set2_6.hex_colors[2])
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
    plt.show()


def plot_histograms_accepted_rejected(dfs, col, col_name):
    """
    Plots histograms for a given column (col) of the dataframes (dfs) with the specified column name (col_name).
    The function generates a plot for each dataframe in dfs, representing the histograms for the accepted and 
    rejected loans respectively.
    
    Args:
    - dfs (list of dataframes): The list of dataframes to plot the histograms for.
    - col (str): The name of the column to plot the histogram for.
    - col_name (str): The name of the column to be used in the plot title and legend.
    
    Returns:
    - None
    """
    _, axes = plt.subplots(1, len(dfs), figsize=(15, 6))
    df_names = ["Accepted", "Rejected"]
    colors = ["#398053", "#80363e"]

    for i in range(0, len(dfs)):
        df = dfs[i]

        ax = axes
        if len(dfs) > 1:
            ax = axes[i]

        ax.hist(df[col], bins=40, color=colors[i])
        ax.set_xlim((df[col].min(), df[col].max()))
        ax.set_title(f"{col_name} - {df_names[i]} loans", fontsize=10)
        ax.axvline(
            np.mean(df[col]),
            color="black",
            linestyle="dashed",
            linewidth=1.3,
            label=f"mean {col_name} {round(np.mean(df[col]), 2)}",
        )
        ax.axvline(
            np.median(df[col]),
            color="blue",
            linestyle="dashed",
            linewidth=1.3,
            label=f"median {col_name} {round(np.median(df[col]), 2)}",
        )
        ax.yaxis.set_ticks([])
        ax.legend(loc=2, prop={"size": 9})


def remove_outliers_col(df, col, threshold=0.9):
    """
    Removes outliers from a specified column (col) of a dataframe (df) using the specified threshold.
    
    Args:
    - df (dataframe): The dataframe to remove outliers from.
    - col (str): The name of the column to remove outliers from.
    - threshold (float, optional): The threshold for filtering outliers. Default is 0.9.
    
    Returns:
    - df_filtered (dataframe): The filtered dataframe with outliers removed.
    """
    val1 = 1 - threshold
    val2 = threshold

    q_low = df[col].quantile(val1)
    q_hi = df[col].quantile(val2)
    df_filtered = df[(df[col] < q_hi) & (df[col] > q_low)]
    return df_filtered

def plot_pie_and_tree_plot(df, col, type):
    """
    Plots a pie chart and a tree map of the distribution of a specified column (col) in a dataframe (df).
    
    Args:
    - df (dataframe): The dataframe to plot the pie chart and tree map for.
    - col (str): The name of the column to plot the distribution for.
    - type (str): The type of dataset the dataframe represents.
    
    Returns:
    - None
    """
    df_plot = df[df[col].isnull() is False]
    vals = df_plot[col].value_counts()

    f, ax = plt.subplots(1, 2, figsize=(18, 7))

    vals.plot.pie(
        ax=ax[0], colors=qualitative.Set2_4.hex_colors, textprops={"fontsize": 7}
    )

    plt.figure(figsize=(15, 8))
    squarify.plot(
        sizes=list(dict(vals).values()),
        label=list(dict(vals).keys()),
        alpha=0.8,
        ax=ax[1],
    )
    f.suptitle(f"Distribution of employment length in {type} dataset", fontsize=14)
    plt.show()


def plot_box_plot_by_col(df1, df2, col, col_name):
    """
    Plots box plots of a specified column (col) in two dataframes (df1 and df2) side by side, with the specified 
    column name (col_name) in the plot title.
    
    Args:
    - df1 (dataframe): The first dataframe to plot the box plot for.
    - df2 (dataframe): The second dataframe to plot the box plot for.
    - col (str): The name of the column to plot the box plot for.
    - col_name (str): The name of the column to be used in the plot title.
    
    Returns:
    - None
    """
    _, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.boxplot(
        data=df1,
        x=col,
        ax=axes[0],
        color="#398053",
        notch=True,
        showcaps=False,
        flierprops={"marker": "x"},
        medianprops={"color": "coral"},
    )

    sns.boxplot(
        data=df2,
        x=col,
        ax=axes[1],
        color="#80363e",
        notch=True,
        showcaps=False,
        flierprops={"marker": "x"},
        medianprops={"color": "white"},
    )

    plt.suptitle(f"{col_name} distribution for accepted and rejected loans")


def plot_addr_state_map(df, color="Greens"):
    """
    Plots a choropleth map of the United States based on the counts of a given feature in a pandas DataFrame.

    Parameters:
    -----------
    df : pandas DataFrame
        The input DataFrame that contains a column named "addr_state" which is used to count the number of entries per state.
    color : str, optional (default="Greens")
        The color scheme to use for the choropleth map. Valid values are any of the color schemes supported by matplotlib's cm module.

    Returns:
    --------
    None

    """
    data = dict(df.addr_state.value_counts())

    url = "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip"

    response = requests.get(url)
    with open("cb_2018_us_state_20m.zip", "wb") as f:
        f.write(response.content)

    us_map = gpd.read_file("zip://./cb_2018_us_state_20m.zip!cb_2018_us_state_20m.shp")

    # Set the state abbreviations as the index of the geodataframe
    us_map.set_index("STUSPS", inplace=True)

    # Create a new column in the geodataframe for the associated numbers
    us_map["data"] = us_map.index.map(data)

    cmap = plt.cm.get_cmap(color)
    normalize = plt.Normalize(vmin=min(data.values()), vmax=max(data.values()))

    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(25, 10))

    # Plot the choropleth map
    us_map.plot(
        column="data", cmap=cmap, norm=normalize, linewidth=0.5, edgecolor="gray", ax=ax
    )

    # Add a colorbar to the plot
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
    sm.set_array([])
    fig.colorbar(sm)

    # Set the title and axis labels for the plot
    ax.set_title("Accepted Loans Map of the United States")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Set the x and y limits of the plot to the bounding box of the United States
    ax.set_xlim([-130, -60])
    ax.set_ylim([20, 55])

    # Add labels with number of data values
    for state, row in us_map.iterrows():
        value = row["data"]
        color = "white" if value > 150000 else "black"
        plt.annotate(
            text=f"{round(value/1000, 2)}k",
            xy=row["geometry"].centroid.coords[0],
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=8,
            color=color,
        )

    # Show the plot
    plt.show()


def plot_addr_state_bar_plots(df1, df2):
    """
    A function that takes two dataframes df1 and df2, each containing information about loans, and plots horizontal bar charts for the distribution of loan applications by address state for each dataframe.

    Args:
    df1 (pandas.DataFrame): A dataframe containing information about accepted loans.
    df2 (pandas.DataFrame): A dataframe containing information about rejected loans.

    Returns:
    None. The function displays the resulting bar plots in a Matplotlib figure.
    """
    f, ax = plt.subplots(1, 2, figsize=(18, 8))
    vals1 = df1["addr_state"].value_counts()
    vals2 = df2["addr_state"].value_counts()

    x = list(vals1.keys())
    y = list(vals1)
    ax[0].barh(x, y, color="#398053")
    ax[0].set_xlabel("Num. of loans")
    ax[0].set_ylabel("State Abbreviation")

    x = list(vals2.keys())
    y = list(vals2)
    ax[1].barh(x, y, height=0.8, color="#80363e")
    ax[1].set_xlabel("Num. of loans")
    ax[1].set_ylabel("State Abbreviation")

    f.suptitle(
        "Address States Distribution between accepted and rejected loans", fontsize=15
    )

    plt.show()


def plot_purpose_acc_rej(df1, df2):
    """
    Plot the distribution of loan purposes in two DataFrames representing
    accepted and rejected loan applications, respectively.

    Parameters:
    -----------
    df1 : pandas.DataFrame
        DataFrame representing accepted loan applications, with a "purpose"
        column containing strings describing the purpose of the loan.

    df2 : pandas.DataFrame
        DataFrame representing rejected loan applications, with a "purpose"
        column containing strings describing the purpose of the loan.

    Returns:
    --------
    None
    """
    accepted_purposes = df1["purpose"].value_counts()
    rejected_purposes = df2["purpose"].value_counts()

    fig, ax = plt.subplots(1, 2, figsize=(18, 7))
    accepted_purposes.plot.pie(
        ax=ax[0], colors=qualitative.Set2_4.hex_colors, textprops={"fontsize": 7}
    )
    ax[0].set_title("Accepted Loans Purpose Distribution")
    rejected_purposes.plot.pie(
        ax=ax[1], colors=qualitative.Set2_4.hex_colors, textprops={"fontsize": 7}
    )
    ax[1].set_title("Rejected Loans Purpose Distribution")
    fig.suptitle(
        "Distribution of purpose values in accepted and rejected loans", fontsize=13
    )


def get_cols_with_large_missing_vals(df) -> list: 
    """
    Get a list of columns in a DataFrame that have more than half of their values
    missing (NaN).

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to check for missing values.

    Returns:
    --------
    A dictionary mapping column names to the number of non-missing values in that
    column.
    """
    columns = {}
    for col, val in df.isnull().sum().items():
        if val > len(df) / 2:
            num_good_entries = len(df) - val
            if num_good_entries < 90000 and "sec" not in col and "joint" not in col:
                columns[col] = num_good_entries
    return columns


tokenizer = RegexpTokenizer(r"\w+")
stemmer = SnowballStemmer("english")

def stem(row):
    """
    Apply stemming to a string by splitting it into tokens, stemming each token,
    and then reassembling the stemmed tokens into a single string.

    Parameters:
    -----------
    row : str
        The input string to be stemmed.

    Returns:
    --------
    A string with each word in the input string replaced by its stemmed form.
    """
    tokens = tokenizer.tokenize(row)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)

def plot_conf_matrices(results, y, labels="", annot=True, roc_thres=False):
    """
    Plot confusion matrices for a dictionary of classification model predictions
    and true labels.

    Parameters:
    -----------
    results : dict
        A dictionary mapping pipeline names to predictions for the corresponding
        pipeline applied to the input data.

    y : array-like of shape (n_samples,)
        True labels for the input data.

    labels : array-like of shape (n_classes,), default=""
        List of labels to be used on the x and y axis of the confusion matrix.
        If not provided, integer labels will be used.

    annot : bool, default=True
        Whether to annotate the heatmap with the count in each cell.

    roc_thres : bool, default=False
        Whether to calculate the best threshold for each model based on its ROC
        curve, and use this threshold to binarize the predictions.

    Returns:
    --------
    None
    """
    f, ax = plt.subplots(1, 3, figsize=(18, 4.5))

    for i in range(0, len(results)):
        model_name = type(list(results.keys())[i].named_steps["model"]).__name__

        predictions = list(results.items())[i][1]

        if roc_thres:
            fpr, tpr, thresholds = metrics.roc_curve(predictions, y)
            j_scores = tpr - fpr
            best_threshold_idx = np.argmax(j_scores)
            best_threshold = thresholds[best_threshold_idx]
            predictions = (y >= best_threshold).astype(int)

        cm = metrics.confusion_matrix(y, list(predictions.values())[0])
        cmn = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(
            cmn,
            annot=annot,
            fmt=".2f",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax[i],
        )

        ax[i].set_title(f"Matrix for {model_name}")
        ax[i].set_xlabel("Predicted labels")
        ax[i].set_ylabel("True labels")

    thrs = ""
    if roc_thres:
        thrs = "with best threshold from ROC"

    f.suptitle(
        "Confusion matrices of validation data for chosen models " + thrs, fontsize=16
    )
    plt.show()


def objective(trial, pipeline, model_name, x_train, y_train, x_val, y_val):
    if isinstance(model_name, lgb.LGBMClassifier):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 1.0),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_child_samples": trial.suggest_int("min_child_samples", 1, 100),
            "subsample": trial.suggest_uniform("subsample", 0.1, 1.0),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.1, 1.0),
        }
        model = lgb.LGBMClassifier(**params)

    elif isinstance(model_name, XGBClassifier):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
            "subsample": trial.suggest_uniform("subsample", 0.1, 1.0),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.1, 1.0),
            "gamma": trial.suggest_loguniform("gamma", 0.001, 1.0),
            "alpha": trial.suggest_loguniform("alpha", 0.001, 1.0),
            "lambda": trial.suggest_loguniform("lambda", 0.001, 1.0),
        }
        model = XGBClassifier(**params)

    elif isinstance(model_name, KNeighborsClassifier):
        params = {
            "n_neighbors": 5,
            "weights": "uniform",
            "p": 2,
            "leaf_size": 30,
            "algorithm": "auto",
        }
        model = KNeighborsClassifier(**params)

    elif isinstance(model_name, XGBClassifier):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
            "subsample": trial.suggest_uniform("subsample", 0.1, 1.0),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.1, 1.0),
            "gamma": trial.suggest_loguniform("gamma", 0.001, 1.0),
            "alpha": trial.suggest_loguniform("alpha", 0.001, 1.0),
            "lambda": trial.suggest_loguniform("lambda", 0.001, 1.0),
        }
        model = XGBClassifier(**params)

    elif isinstance(model_name, GradientBoostingClassifier):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 1.0),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 100),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
            "subsample": trial.suggest_uniform("subsample", 0.1, 1.0),
            "max_features": trial.suggest_uniform("max_features", 0.1, 1.0),
        }
        model = GradientBoostingClassifier(**params)

    elif isinstance(model_name, AdaBoostClassifier):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 1.0),
            "algorithm": trial.suggest_categorical("algorithm", ["SAMME", "SAMME.R"]),
        }
        model = AdaBoostClassifier(**params)

    elif isinstance(model_name, RandomForestClassifier):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 100),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
            "max_features": trial.suggest_uniform("max_features", 0.1, 1.0),
        }
        model = RandomForestClassifier(**params)

    elif isinstance(model_name, ExtraTreesClassifier):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 100),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
            "max_features": trial.suggest_uniform("max_features", 0.1, 1.0),
        }
        model = ExtraTreesClassifier(**params)

    elif isinstance(model_name, LogisticRegression):
        params = {
            "penalty": trial.suggest_categorical("penalty", ["l2"]),
            "C": trial.suggest_loguniform("C", 1e-4, 1e4),
            "solver": trial.suggest_categorical(
                "solver", ["liblinear", "lbfgs", "saga"]
            ),
        }
        model = LogisticRegression(**params)

    elif isinstance(model_name, BaggingClassifier):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_samples": trial.suggest_float("max_samples", 0.1, 1.0),
            "max_features": trial.suggest_float("max_features", 0.1, 1.0),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "bootstrap_features": trial.suggest_categorical(
                "bootstrap_features", [True, False]
            ),
        }
        model = BaggingClassifier(**params)

    elif isinstance(model_name, GaussianNB):
        params = {
            "var_smoothing": trial.suggest_loguniform("var_smoothing", 1e-10, 1e-5),
            "fit_prior": trial.suggest_categorical("fit_prior", [True, False]),
        }

        model = GaussianNB(**params)

    elif isinstance(model_name, QuadraticDiscriminantAnalysis):
        params = {
            "reg_param": trial.suggest_float("reg_param", 0.0, 1.0),
            "tol": trial.suggest_float("tol", 1e-5, 1e-1, log=True),
            "store_covariance": trial.suggest_categorical(
                "store_covariance", [True, False]
            ),
        }

        model = QuadraticDiscriminantAnalysis(**params)

    elif isinstance(model_name, BernoulliNB):
        params = {
            "alpha": trial.suggest_loguniform("alpha", 1e-4, 1e-1),
            "fit_prior": trial.suggest_categorical("fit_prior", [True, False]),
            "binarize": trial.suggest_uniform("binarize", 0.0, 1.0),
        }

        model = BernoulliNB(**params)

    elif isinstance(model_name, DecisionTreeClassifier):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
        model = DecisionTreeClassifier(**params)

    new_pipeline = Pipeline(
        [
            ("preprocessor", clone(pipeline.named_steps["preprocessor"])),
            ("IPCA", clone(pipeline.named_steps["IPCA"])),
            ("model", model),
        ]
    )

    new_pipeline.fit(x_train, y_train)
    y_pred = new_pipeline.predict(x_val)
    trial.set_user_attr(key="best_pipe", value=new_pipeline)
    f1 = metrics.f1_score(y_val, y_pred, average="weighted")

    return f1


def objective_reg(trial, pipeline, model_name, x_train, y_train, x_val, y_val):
    if isinstance(model_name, RandomForestRegressor):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 5, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["auto", "sqrt", "log2"]
            ),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }
        model = RandomForestRegressor(**params)

    elif isinstance(model_name, GradientBoostingRegressor):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["auto", "sqrt", "log2"]
            ),
            "learning_rate": trial.suggest_uniform("learning_rate", 0.001, 1.0),
        }
        model = GradientBoostingRegressor(**params)

    elif isinstance(model_name, DecisionTreeRegressor):
        params = {
            "max_depth": trial.suggest_int("max_depth", 1, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["auto", "sqrt", "log2"]
            ),
        }
        model = DecisionTreeRegressor(**params)

    elif isinstance(model_name, LinearRegression):
        params = {}
        model = LinearRegression(**params)

    elif isinstance(model_name, Ridge):
        params = {
            "alpha": trial.suggest_uniform("alpha", 0.01, 10),
            "solver": trial.suggest_categorical(
                "solver",
                ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
            ),
            "max_iter": trial.suggest_int("max_iter", 1000, 5000),
        }
        model = Ridge(**params)

    elif isinstance(model_name, Lasso):
        params = {
            "alpha": trial.suggest_uniform("alpha", 0.01, 10),
            "tol": trial.suggest_uniform("tol", 1e-6, 1e-3),
            "selection": trial.suggest_categorical("selection", ["cyclic", "random"]),
            "max_iter": trial.suggest_int("max_iter", 1000, 5000),
        }
        model = Lasso(**params)

    elif isinstance(model_name, ElasticNet):
        params = {
            "alpha": trial.suggest_uniform("alpha", 0.01, 10),
            "l1_ratio": trial.suggest_uniform("l1_ratio", 0, 1),
            "tol": trial.suggest_uniform("tol", 1e-6, 1e-3),
            "selection": trial.suggest_categorical("selection", ["cyclic", "random"]),
            "max_iter": trial.suggest_int("max_iter", 1000, 5000),
        }
        model = ElasticNet(**params)

    elif isinstance(model_name, SVR):
        params = {
            "kernel": trial.suggest_categorical(
                "kernel", ["linear", "poly", "rbf", "sigmoid"]
            ),
            "C": trial.suggest_loguniform("C", 1e-4, 1e4),
            "epsilon": trial.suggest_loguniform("epsilon", 1e-4, 1),
            "gamma": trial.suggest_categorical(
                "gamma",
                ["scale", "auto"]
                + ["scale" for i in range(10)]
                + ["auto" for i in range(10)],
            ),
            "degree": trial.suggest_int("degree", 2, 5),
        }
        model = SVR(**params)

    elif isinstance(model_name, KNeighborsRegressor):
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 50),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "algorithm": trial.suggest_categorical(
                "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
            ),
            "leaf_size": trial.suggest_int("leaf_size", 10, 100),
            "p": trial.suggest_int("p", 1, 2),
        }
        model = KNeighborsRegressor(**params)

    new_pipeline = Pipeline(
        [
            ("preprocessor", clone(pipeline.named_steps["preprocessor"])),
            ("IPCA", clone(pipeline.named_steps["IPCA"])),
            ("model", model),
        ]
    )

    new_pipeline.fit(x_train, y_train)
    y_pred = new_pipeline.predict(x_val)
    trial.set_user_attr(key="best_pipe", value=new_pipeline)
    r2 = metrics.r2_score(y_val, y_pred)

    return r2


def try_models(models, x_train, y_train, x_val, y_val):
    """
    Trains and evaluates multiple machine learning models on the validation data and returns their performance scores.

    Parameters:
    models (list): A list of machine learning models to be trained and evaluated.
    x_train (array-like): Training data features.
    y_train (array-like): Training data labels.
    x_val (array-like): Validation data features.
    y_val (array-like): Validation data labels.

    Returns:
    pandas.DataFrame: A dataframe containing the performance scores of the trained models on the validation data, sorted in descending order of F1-score.
    """
    scores = {}
    for model in models:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        scores[model] = [
            metrics.f1_score(y_val, y_pred, average="weighted"),
            metrics.recall_score(y_val, y_pred, average="weighted"),
            metrics.precision_score(y_val, y_pred, average="weighted"),
        ]

    models_results = pd.DataFrame(index=None)
    models_results["model"] = list(scores.keys())

    models_results["model_name"] = [
        type(key.named_steps["model"]).__name__ for key in scores.keys()
    ]
    models_results["f1_score"] = [val[0] for val in list(scores.values())]
    models_results["recall_score"] = [val[1] for val in list(scores.values())]
    models_results["precision_score"] = [val[2] for val in list(scores.values())]
    models_results = models_results.sort_values(by="f1_score", ascending=False)
    return models_results


def try_reg_models(models, x_train, y_train, x_val, y_val):
    """
    Trains and evaluates multiple regression models on the validation data and returns their performance scores.

    Parameters:
    models (list): A list of regression models to be trained and evaluated.
    x_train (array-like): Training data features.
    y_train (array-like): Training data labels.
    x_val (array-like): Validation data features.
    y_val (array-like): Validation data labels.

    Returns:
    pandas.DataFrame: A dataframe containing the performance scores of the trained regression models on the validation data, sorted in descending order of R2-score.
    """
    scores = {}
    for model in models:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        scores[model] = [
            metrics.r2_score(y_val, y_pred),
            np.sqrt(metrics.mean_squared_error(y_val, y_pred)),
        ]

    models_results = pd.DataFrame(index=None)
    models_results["model"] = list(scores.keys())
    models_results["model_name"] = [
        type(key.named_steps["model"]).__name__ for key in scores.keys()
    ]
    models_results["r2_score"] = [val[0] for val in list(scores.values())]
    models_results["rmse"] = [val[1] for val in list(scores.values())]
    models_results = models_results.sort_values(by="r2_score", ascending=False)
    return models_results


def callback(study, trial):
    """
    Updates the best pipeline attribute in the Optuna study based on the best trial.

    Parameters:
    study (optuna.study.Study): The Optuna study.
    trial (optuna.trial.FrozenTrial): The trial being evaluated.

    Returns:
    None
    """
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_pipe", value=trial.user_attrs["best_pipe"])


def optuna_tuning(
    objective_func, chosen_models, x_train, y_train, x_val, y_val, score="f1_score"
):
    """
    Performs hyperparameter tuning using Optuna for a set of chosen models and returns the best hyperparameters and their corresponding performance scores.

    Parameters:
    objective_func (function): A function that defines the objective for the Optuna study.
    chosen_models (list): A list of scikit-learn pipelines, each containing a machine learning model to be tuned.
    x_train (array-like): Training data features.
    y_train (array-like): Training data labels.
    x_val (array-like): Validation data features.
    y_val (array-like): Validation data labels.
    score (str): The performance metric to be used for evaluating the models. Defaults to "f1_score".

    Returns:
    dict: A dictionary containing the best hyperparameters and their corresponding performance scores for each tuned model.
    """
    results = {}
    df_models = chosen_models.copy()
    optuna.logging.get_logger("optuna").setLevel(logging.WARNING)

    for pipe in list(df_models.model.unique()):
        print(f"Tuned {type(pipe.named_steps['model']).__name__}:")
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective_func(
                trial, pipe, pipe.named_steps["model"], x_train, y_train, x_val, y_val
            ),
            n_trials=25,
            callbacks=[callback],
        )

        best_pipe = study.user_attrs["best_pipe"]

        results[(best_pipe, type(best_pipe.named_steps["model"]).__name__)] = {
            "best_params": study.best_params,
            score: study.best_value,
        }

        df = pd.DataFrame(study.best_params, index=[0])
        df.insert(0, score, [study.best_value])
        display(df)

    return results


def train_val_test_split_reg(X_data, Y_data, target):
    """
    Splits the input data into training, validation, and test sets for regression tasks.

    Parameters:
    X_data (pandas.DataFrame): The input features.
    Y_data (pandas.DataFrame): The target variable.
    target (str): The name of the target variable in Y_data.

    Returns:
    tuple: A tuple containing the training, validation, and test sets for the input data as pandas.DataFrames and pandas.Series.
    The first three elements are the X_train, X_val, and X_test DataFrames, respectively.
    The last three elements are the y_train, y_val, and y_test Series, respectively.
    """
    X = X_data.copy()
    X[target] = list(Y_data[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X[[col for col in X.columns if col != target]],
        X[target],
        train_size=0.8,
        test_size=0.2,
        random_state=42,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=1
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_val_test_split(X_data, Y_data, target):
    """
    Split the data into training, validation, and test sets while ensuring
    that the target variable is properly distributed among the splits.

    Parameters:
        X_data (pandas DataFrame): The features data to be split.
        Y_data (pandas DataFrame): The target variable data to be split.
        target (str): The name of the target variable column.

    Returns:
        tuple: A tuple of six elements containing the following:
            - X_train (pandas DataFrame): The training features data.
            - X_val (pandas DataFrame): The validation features data.
            - X_test (pandas DataFrame): The test features data.
            - y_train (pandas Series): The training target variable data.
            - y_val (pandas Series): The validation target variable data.
            - y_test (pandas Series): The test target variable data.

    """
    X = X_data.copy()
    X[target] = list(Y_data[target])

    categorical_features = list(X.select_dtypes(include=["object", "category"]).columns)
    categorical_features.append(target)

    X[categorical_features] = X[categorical_features].fillna("missing")

    X_train, X_test, y_train, y_test = train_test_split(
        X[[col for col in X.columns if col != target]],
        X[target],
        train_size=0.8,
        test_size=0.2,
        random_state=42,
        stratify=X[[target]],
    )

    for col in categorical_features:
        X[col] = X[col].apply(lambda x: None if x == "missing" else x)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=1, stratify=y_train
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_best_cols_from_mi_scores(mi_scores, df, threshold):
    """
    Selects the best features based on mutual information (MI) scores and returns a set of column names.

    Parameters:
    -----------
    mi_scores : dict
        A dictionary of MI scores for each feature.
    df : pd.DataFrame
        A pandas DataFrame containing the feature data.
    threshold : float
        The MI score threshold above which features are considered important.

    Returns:
    --------
    results : set
        A set of column names representing the best features based on MI scores, joint features, and time-based features.
    """
    joint_cols = list(
        set([col for col in df.columns if "sec" in col or "joint" in col])
    )
    cols = [
        col
        for col in df.columns
        if not ("sec" in col or "joint" in col)
        and any(col in joint_col for joint_col in joint_cols)
    ]

    not_remove_cols = cols + joint_cols

    useful_cols = []
    for col, val in mi_scores.items():
        if val >= threshold:
            useful_cols.append((col))

    results = useful_cols + not_remove_cols
    return set(results)


def get_and_plot_mi_scores(df, target_df, target):
    """
    This function calculates and plots mutual information (MI) scores for features in a dataframe with respect to a target variable.

    Args:

    df: pandas DataFrame containing the features.
    target_df: pandas DataFrame containing the target variable.
    target: name of the target variable.
    Returns:

    A dictionary of MI scores for each feature.
    A plot of the MI scores.
    """
    X = df[[col for col in df.columns if col != target]].copy()
    X[target] = target_df[target]
    X = X.dropna()

    y = X[target]
    X = X[[col for col in X.columns if col != target]].copy()

    mi_scores = make_mi_scores(X, y)
    print(mi_scores[:20])
    plot_mi_scores(mi_scores)
    return mi_scores


def stratified_sample(df, target, threshold=None):
    """
    This function takes a pandas DataFrame, target column name and an optional threshold value,
    and returns a stratified sample of the DataFrame based on the target column.

    Parameters:
    - df (pandas DataFrame): Input DataFrame
    - target (str): Column name to stratify the sample based on
    - threshold (int, optional): Maximum sample size for each category. If not specified, minimum value
    of counts of each category is used as the sample size.

    Returns:
    - pandas DataFrame: Stratified sample DataFrame with the same columns as the input DataFrame.
    """
    df_clean = df.copy()
    stratified_samples = {}

    sample_size = threshold
    if not threshold:
        sample_size = df[target].value_counts().min()

    for grade in df[target].unique():
        df_cat = df_clean[df_clean[target] == grade]
        df_stratified_sample = df_cat.groupby(target, group_keys=False).apply(
            lambda x: x.sample(min(len(x), sample_size))
        )
        stratified_samples[grade] = df_stratified_sample

    return pd.concat(stratified_samples.values())


def preprocessor(
    numcols, ohcols, ordcols, joint_numcols=None, joint_catcols=None, model_type=None
):
    """
    This function returns a sklearn ColumnTransformer object that performs various preprocessing steps on the input data.

    Parameters:
    - numcols (list): List of column names to be treated as numerical features
    - ohcols (list): List of column names to be one-hot encoded
    - ordcols (list): List of column names to be treated as ordinal features
    - joint_numcols (list, optional): List of column names to be treated as numerical features in a joint transformer. Default is None.
    - joint_catcols (list, optional): List of column names to be transformed using one-hot encoding or ordinal encoding in a joint transformer. Default is None.
    - model_type (str, optional): Type of model to be used for encoding categorical variables. Default is None.

    Returns:
    - sklearn ColumnTransformer object: Performs various preprocessing steps on the input data, based on the specified parameters.
    """
    knn_imputer = KNNImputer(n_neighbors=5)

    if model_type == "tree":
        joint_cols_categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
                ("ord", OrdinalEncoder()),
            ]
        )
    else:
        joint_cols_categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
                ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
            ]
        )

    joint_cols_numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )

    numerical_transformer = Pipeline(
        steps=[("imputer", knn_imputer), ("scaler", StandardScaler())]
    )

    ordinal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OrdinalEncoder()),
        ]
    )

    if model_type != "tree":
        onehot_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
            ]
        )
    else:
        ordcols.extend(ohcols)

    if joint_numcols is not None and joint_catcols is not None:
        preprocessor = ColumnTransformer(
            transformers=[
                ("joint_num", joint_cols_numerical_transformer, joint_numcols),
                ("joint_cat", joint_cols_categorical_transformer, joint_catcols),
                ("num", numerical_transformer, numcols),
                ("ord", ordinal_transformer, ordcols),
            ],
            remainder="passthrough",
        )
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numcols),
                ("ord", ordinal_transformer, ordcols),
            ],
            remainder="passthrough",
        )

    if model_type != "tree":
        ohec_transformer = ("ohe", onehot_transformer, ohcols)
        new_pipeline = ColumnTransformer(
            transformers=[*preprocessor.transformers, ohec_transformer],
            remainder=preprocessor.remainder,
        )
        preprocessor = new_pipeline

    return preprocessor


def get_useless_joint_cols(
    mi_scores, common_not_joint_cols, joint_cols, threshold=0.004
):
    """
    Given a dictionary of mutual information scores for columns, a list of columns that are common but not joint,
    a list of joint columns, and a threshold value, returns a set of columns that are identified as useless
    joint columns based on their mutual information scores falling below the threshold.

    Parameters:
    mi_scores (dict): A dictionary of mutual information scores for columns.
    common_not_joint_cols (list): A list of columns that are common but not joint.
    joint_cols (list): A list of joint columns.
    threshold (float): A threshold value for the mutual information scores.

    Returns:
    set: A set of columns that are identified as useless joint columns based on their mutual information scores
    falling below the threshold.
    """
    remove_cols = []
    mi_scores_dict = dict(mi_scores)
    for col, val in mi_scores_dict.items():
        if val < threshold:
            if col in common_not_joint_cols:
                pair = []
                if col not in remove_cols:
                    pair.append(col)

                for j_col in joint_cols:
                    if col in j_col and j_col not in remove_cols:
                        pair.append(j_col)

                remove_cols.extend(pair)

            elif col in common_not_joint_cols:
                pair = []
                if col not in remove_cols:
                    pair.append(col)

                for nj_col in joint_cols:
                    if col in nj_col and nj_col not in remove_cols:
                        pair.append(nj_col)

                remove_cols.extend(pair)

    return set(remove_cols)


def plot_ipca_elbow_plot_and_get_n_components(X_train, threshold=0.85):
    """
    Given a training dataset X_train and a variance threshold, plots the elbow plot for incremental principal component analysis (IPCA)
    and returns the number of components to keep based on the given threshold.

    Parameters:
    X_train (array-like): The training dataset to perform IPCA on.
    threshold (float): A float value between 0 and 1 indicating the variance threshold to be used for determining
    the number of principal components to keep.

    Returns:
    int: The number of components to keep based on the variance threshold.

    Raises:
    ValueError: If the length of the input dataset X_train is less than or equal to 0.
    """
    X = X_train.copy()

    if scipy.sparse.issparse(X):
        X = X.toarray()

    if len(X) > 20000:
        batch_size = 10000
    elif len(X) > 10000:
        batch_size = 5000
    elif len(X) > 2000:
        batch_size = 1000
    elif len(X) > 300:
        batch_size = 100
    elif len(X) > 200:
        batch_size = 10
    else:
        batch_size = 1

    ipca = IncrementalPCA(n_components=None, batch_size=batch_size)

    for batch_X in np.array_split(X, len(X) // batch_size):
        ipca.partial_fit(batch_X)

    cumulative_var_ratio = np.cumsum(ipca.explained_variance_ratio_)

    n_components = np.argmax(cumulative_var_ratio >= threshold) + 1

    print("Number of components to keep:", n_components)
    plt.plot(cumulative_var_ratio)
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.show()

    return n_components


def perform_ipca(n_components, X_train):
    """
    Given the number of principal components and a training dataset, performs incremental principal component analysis (IPCA)
    and returns the transformed dataset as a pandas dataframe and the IPCA object.

    Parameters:
    n_components (int): The number of principal components to keep.
    X_train (array-like): The training dataset to perform IPCA on.

    Returns:
    tuple: A tuple of two values. The first value is a pandas dataframe of the transformed dataset with columns
    named as "component_i" where i is the index of the component. The second value is the IPCA object.

    Raises:
    ValueError: If the length of the input dataset X_train is less than or equal to 0.
    """
    X = X_train.copy()

    if scipy.sparse.issparse(X):
        X = X.toarray()

    if len(X) > 20000:
        batch_size = 10000
    elif len(X) > 10000:
        batch_size = 5000
    elif len(X) > 2000:
        batch_size = 1000
    elif len(X) > 300:
        batch_size = 100
    elif len(X) > 200:
        batch_size = 10
    else:
        batch_size = 1

    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    for batch_X in np.array_split(X, len(X) // batch_size):
        ipca.partial_fit(batch_X)

    X_ipca = ipca.transform(X)
    df_ipca = pd.DataFrame(
        data=X_ipca, columns=[f"component_{i+1}" for i in range(n_components)]
    )
    return (df_ipca, ipca)


def plot_3_pcas(ipca, X, y_train):
    """
    Given an IPCA object, a dataset, and its corresponding labels, performs IPCA to extract the top 3 principal components,
    transforms the dataset using the top 3 components, and plots the dataset in 3D using the transformed data.

    Parameters:
    ipca (object): The IPCA object.
    X (array-like): The dataset to transform.
    y_train (array-like): The corresponding labels for the dataset.

    Returns:
    None
    """
    explained_variances = ipca.explained_variance_ratio_
    print("Explained variance ratios:", explained_variances)
    top_components = ipca.components_[:3, :]
    X_3d = csr_matrix.dot(X, top_components.T)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=y_train)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")

    plt.show()


def get_top_n_models_preds(final_model_results, X_train, y_train, X_val, num_models=3):
    """
    Fits the top n best models on the training data and generates predictions for the validation data.

    Args:
        final_model_results (dict): A dictionary containing the results of fitting all the models.
        X_train (array-like of shape (n_samples, n_features)): Training input samples.
        y_train (array-like of shape (n_samples,)): Target values for training set.
        X_val (array-like of shape (n_samples, n_features)): Validation input samples.
        num_models (int, optional): Number of top models to select. Defaults to 3.

    Returns:
        dict: A dictionary containing the predictions for the top n models.
    """
    predictions = {}

    top3_models = copy.deepcopy(final_model_results)
    top3_models = list(final_model_results.items())[:num_models]

    for pipeline, res in top3_models:
        new_pipeline = copy.deepcopy(pipeline[0])

        if "model" not in new_pipeline.named_steps:
            raise ValueError("Pipeline does not contain a 'model' step.")

        best_params = res["best_params"]
        model = new_pipeline.named_steps["model"]

        model.set_params(**best_params)
        new_pipeline.model = model
        new_pipeline.fit(X_train, y_train)
        preds = new_pipeline.predict(X_val)
        predictions[new_pipeline] = preds

    return predictions


def remove_num_outliers(
    df, nunique=100000, thres=0.9, targets=["grade", "sub_grade", "int_rate"]
):
    """
    Removes outliers from numeric columns in a Pandas DataFrame.

    Parameters
    ----------
    df : pandas DataFrame
        The input DataFrame to remove outliers from.
    nunique : int, optional (default=100000)
        The maximum number of unique values allowed for a column to be considered numeric.
    thres : float, optional (default=0.9)
        The threshold value for removing outliers. Rows with values outside of (Q1-thres*IQR, Q3+thres*IQR)
        are removed, where Q1 and Q3 are the first and third quartiles and IQR is the interquartile range.
    targets : list of str, optional (default=["grade", "sub_grade", "int_rate"])
        A list of columns that should not have outliers removed.

    Returns
    -------
    pandas DataFrame
        The input DataFrame with outliers removed from numeric columns.
    """
    check_outliers_cols = df.select_dtypes(
        include=["float64", "int64"]
    ).columns.tolist()
    real_num = [col for col in check_outliers_cols if df[col].nunique() > nunique]
    data = df.copy()

    for col in real_num:
        if col not in targets:
            data = remove_outliers_col(data, col, thres)
    return data


def plot_kde_plot(df, col, hue_col):
    plt.figure(figsize=(15, 8))

    for val in sorted(df[hue_col].unique()):
        if val is not None:
            data = df[df[hue_col] == val]
            sns.kdeplot(
                data=data,
                x=col,
                fill=True,
                common_norm=False,
                alpha=0.5,
                linewidth=0,
                label=val,
            )

    plt.legend(loc="upper right")
    plt.title(f"Distribution of {col} between different {hue_col} values", fontsize=13)
    plt.show()


def plot_scatter_plots_with_target(df, target, numcols):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    i = 0
    j = 0

    for k in range(0, len(numcols)):
        sns.regplot(
            x=target,
            y=numcols[k],
            data=df,
            color="#5d95a3",
            ax=axes[i][j],
            scatter_kws={"s": 10},
        )

        j += 1
        if j == 4:
            i += 1
            j = 0

    fig.suptitle(
        f"Distribution of {target} and \n {numcols}",
        fontsize=14,
    )
    plt.show()


def plot_hist_distributions(df, cols_to_plot, color="#8DA0CB", desc=""):
    fig, axes = plt.subplots(1, len(cols_to_plot), figsize=(20, 5))

    for i in range(0, len(cols_to_plot)):
        axes[i].hist(df[cols_to_plot[i]], bins=20, color=color)
        axes[i].set_title(f"{cols_to_plot[i]}", fontsize=9)

    fig.suptitle(
        f"Distributions of {cols_to_plot}{desc}",
        fontsize=14,
    )
    plt.show()


def plot_cat_col_target_corr(corr_matrix, cat_col, target):
    plt.figure(figsize=(15, 7))
    corr_matrix.plot(kind="bar")
    plt.xlabel("Features")
    plt.ylabel("Correlation")
    plt.title(
        f'Correlation of different {cat_col} values with target - "{target}"',
        fontsize=15,
    )
    plt.show()


def plot_pearson_corr_map(df):
    corr_df = df.corr()
    plt.figure(figsize=(13, 9))
    sns.heatmap(corr_df, mask=np.triu(corr_df))
    plt.title("Feature Correlation Heatmap")
    plt.show()


def get_too_correlated_cols_and_print_pairs(df, all_common_cols, scores):
    data = df.copy()
    corr_matrix = (
        data[[col for col in data.columns if col not in all_common_cols]].corr().abs()
    )
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)
    )

    threshold = 0.9
    corr_pairs = {}
    for column in upper_triangle.columns:
        correlated = upper_triangle[column][
            upper_triangle[column] > threshold
        ].index.tolist()
        if len(correlated) > 0:
            corr_pairs[column] = correlated

    to_drop = []
    for col, correlated in corr_pairs.items():
        for c in correlated:
            col_to_drop = col
            if scores[col] > scores[c]:
                col_to_drop = c

            to_drop.append(col_to_drop)

            print(
                f"{col} and {c} are highly correlated: {round(corr_matrix.loc[col, c], 3)}. The column dropped will be {col_to_drop}."
            )
        
    if len(to_drop) == 0:
        print("There are no too correlated")

    return to_drop


def perform_cross_validation(
    pipelines, X_train, y_train, X_val, y_val, score="f1_score"
):
    predictions = {}
    scores = {}

    for pipeline in pipelines:
        val_predictions = pipeline.predict(X_val)
        X_transformed = pipeline.named_steps["preprocessor"].transform(X_train)
        train_predictions = cross_val_predict(
            pipeline.named_steps["model"], X_transformed, y_train, cv=5
        )

        if score == "f1_score":
            train_score = metrics.f1_score(
                y_train, train_predictions, average="weighted"
            )
            val_score = metrics.f1_score(y_val, val_predictions, average="weighted")
        else:
            train_score = metrics.r2_score(
                y_train, train_predictions, multioutput="variance_weighted"
            )
            val_score = metrics.r2_score(y_val, val_predictions)

        model_name = type(pipeline.named_steps["model"]).__name__
        test_str = f"VALIDATION DATA - {score} for the {model_name} model is: {round(val_score, 3)}\n"
        cross_val_str = (
            f"TRAIN DATA (cross validated) {score} is {round(train_score, 3)}\n"
        )

        header_sep = "-" * int(
            round((max(len(test_str), len(cross_val_str)) - len(model_name) - 2) / 2)
        )
        print(f"{header_sep} {model_name} {header_sep}\n{cross_val_str}{test_str}")

        predictions[pipeline] = val_predictions
        scores[pipeline] = val_score

    return scores, predictions


def plot_qq(predictions, y_val):
    fig, axes = plt.subplots(1, 1, figsize=(20, 9))

    for model in list(predictions.items()):
        y_pred = model[1]
        residuals = y_val - y_pred
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
        empirical_quantiles = np.quantile(
            np.sort(residuals), np.linspace(0.01, 0.99, len(residuals))
        )
        plt.scatter(
            theoretical_quantiles,
            empirical_quantiles,
            label=type(model[0].named_steps["model"]).__name__,
        )

    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Empirical Quantiles")
    plt.title("QQ Plot of Residuals", fontsize=16)
    plt.plot([-3, 3], [-3, 3], color="red")
    plt.legend()

    plt.show()


def getting_pipelines(
    tree_models,
    tree_preprocessor,
    tree_ipca,
    other_models,
    preproc,
    ipca,
    train=None,
    labels=None,
):
    pipelines = []
    for model in tree_models:
        pipeline = Pipeline(
            [
                ("preprocessor", tree_preprocessor),
                ("IPCA", tree_ipca),
                ("model", model),
            ]
        )

        if train is not None and labels is not None:
            pipeline.named_steps["preprocessor"].fit_transform(train, labels)

        pipelines.append(pipeline)

    for model in other_models:
        pipeline = Pipeline(
            [
                ("preprocessor", preproc),
                ("IPCA", ipca),
                ("model", model),
            ]
        )
        pipelines.append(pipeline)

    return pipelines


def drop_nan_small_missing_cols(data, num=200):
    df = data.copy()
    results = df.isnull().sum()
    for i in range(0, len(results)):
        if results[i] <= num:
            col = results.index[i]
            df = df.dropna(subset=col)
    return df
