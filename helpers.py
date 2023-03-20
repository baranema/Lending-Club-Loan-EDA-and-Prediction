import seaborn as sns
import matplotlib.pylab as plt
import numpy as np
import squarify   
from palettable.colorbrewer import qualitative
import plotly.express as px
import scipy.stats as stats
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split 
import pandas as pd
import scipy
import geopandas as gpd 
import requests 
from scipy.sparse import csr_matrix
import scipy.stats as stats
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import cross_val_predict
from nltk.tokenize import RegexpTokenizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import logging
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score


from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis 
import optuna 
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.naive_bayes import GaussianNB, BernoulliNB 
from nltk.stem import SnowballStemmer 
from sklearn.feature_selection import mutual_info_classif
from palettable.colorbrewer import qualitative
from sklearn import metrics
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
import copy 

from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline 
from sklearn.impute import KNNImputer 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder  
from sklearn.preprocessing import OneHotEncoder  
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer
from scipy.stats import f_oneway
from sklearn.metrics import mean_absolute_percentage_error
 
def count_and_plot_mape_vals(predictions, y_val):
    mape_res = {}
    for model in list(predictions.items()):  
        y_pred = model[1] 
        mape = mean_absolute_percentage_error(y_val, y_pred)
        mape_res[type(model[0].named_steps['model']).__name__] = mape 
    df = pd.DataFrame(list(mape_res.items()), columns=['model', 'mape'])
    df = df.sort_values(by='mape', ascending=False)
    display(df)
    
    _, _ = plt.subplots(1, 1, figsize=(15, 8))
    plt.bar(df['model'], df['mape'], color="#695280")
    plt.xlabel('model')
    plt.ylabel('mape')
    plt.title('Barplot of model vs. mape vals')
    plt.show()

def perform_ANOVA_test(df, all_common_cols): 
    data = df.copy()
    
    cat_cols = [col for col in data.columns if data[col].dtype == 'object' and col not in all_common_cols]
    cat_cols.append("int_rate")

    data = data[cat_cols].dropna()
    cat_cols.remove("int_rate")

    rejected_cols = {}
    not_rejected_cols = {}

    for col in cat_cols:
        groups = [data[data[col] == category]['int_rate'] for category in data[col].unique()]
        _, p_val = f_oneway(*groups)
        if p_val < 0.05:
            rejected_cols[col] = p_val 
        else:
            not_rejected_cols[col] = p_val

    return rejected_cols, not_rejected_cols


def plot_roc(X, Y, data_type, models):
    _, ax = plt.subplots(1, 1, figsize=(15, 8))

    model_names = []
    for model in models:
        model_name = type(model.named_steps['model']).__name__
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
    plt.figure(dpi=100, figsize=(15, 8))
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores, color=qualitative.Set2_6.hex_colors[2])
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
    plt.show()

def plot_histograms_accepted_rejected(dfs, col, col_name):
    _, axes = plt.subplots(1, len(dfs), figsize=(15, 6))
    df_names = ["Accepted", "Rejected"]
    colors = ['#398053', '#80363e']

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
            label=f"mean {col_name} {round(np.mean(df[col]), 2)}"
        )
        ax.axvline(
            np.median(df[col]),
            color="blue",
            linestyle="dashed",
            linewidth=1.3,
            label=f"median {col_name} {round(np.median(df[col]), 2)}"
        )
        ax.yaxis.set_ticks([])
        ax.legend(loc=2, prop={"size": 9})

def remove_outliers_col(df, col, threshold=0.9): 
    val1 = 1 - threshold
    val2 = threshold

    q_low = df[col].quantile(val1)
    q_hi  = df[col].quantile(val2)
    df_filtered = df[(df[col] < q_hi) & (df[col] > q_low)]
    return df_filtered 

def is_keyword_in_text(keyword, text):
    """Check if a keyword is present in a given text."""
    return ((len(keyword) < 3 and (keyword.lower() in text or f"{keyword.lower()} " in text or f" {keyword.lower()}" in text)) or
            (len(keyword) >= 3 and (keyword.lower()[1:] in text or keyword.lower()[:-1] in text)))

def find_matching_keyword(text, keywords):
    """Find the first keyword that matches the given text."""
    for keyword in keywords:
        if is_keyword_in_text(keyword, text):
            return keyword
    return None

def update_col(text, keywords, default_val):
    """Update col if it contains a keyword, otherwise return default_val."""
    keyword = find_matching_keyword(text.lower(), keywords)
    return default_val if keyword is not None else text
 
def plot_pie_and_tree_plot(df, col, type):
    df_plot = df[df[col].isnull() == False]
    vals = df_plot[col].value_counts()

    f, ax = plt.subplots(1, 2, figsize=(18, 7))
    
    vals.plot.pie(ax=ax[0], colors=qualitative.Set2_4.hex_colors,
                  textprops={'fontsize': 7})
                  
    plt.figure(figsize=(15, 8))
    squarify.plot(sizes=list(dict(vals).values()), label=list(dict(vals).keys()),
                  alpha=.8, ax=ax[1])
    f.suptitle(f"Distribution of employment length in {type} dataset", fontsize=14)
    plt.show()

def plot_box_plot_by_col(df1, df2, col, col_name):
    _, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.boxplot(data=df1, x=col, ax=axes[0], color="#398053", notch=True, showcaps=False,
            flierprops={"marker": "x"},
            medianprops={"color": "coral"}) 
            
    sns.boxplot(data=df2, x=col, ax=axes[1], color="#80363e", notch=True, showcaps=False,
            flierprops={"marker": "x"},
            medianprops={"color": "white"}) 

    plt.suptitle(
        f"{col_name} distribution for accepted and rejected loans"
    )  

def plot_addr_state_map(df, color='Greens'):
    data = dict(df.addr_state.value_counts())

    url = 'https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip'
    
    response = requests.get(url)
    with open('cb_2018_us_state_20m.zip', 'wb') as f:
        f.write(response.content)
    
    us_map = gpd.read_file('zip://./cb_2018_us_state_20m.zip!cb_2018_us_state_20m.shp')
     
    # Set the state abbreviations as the index of the geodataframe
    us_map.set_index('STUSPS', inplace=True)

    # Create a new column in the geodataframe for the associated numbers
    us_map['data'] = us_map.index.map(data)
   
    cmap = plt.cm.get_cmap(color)
    normalize = plt.Normalize(vmin=min(data.values()), vmax=max(data.values()))

    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(25, 10))

    # Plot the choropleth map
    us_map.plot(column='data', cmap=cmap, norm=normalize,
                linewidth=0.5, edgecolor='gray', ax=ax)

    # Add a colorbar to the plot
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
    sm.set_array([])
    fig.colorbar(sm)

    # Set the title and axis labels for the plot
    ax.set_title('Accepted Loans Map of the United States')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # Set the x and y limits of the plot to the bounding box of the United States
    ax.set_xlim([-130, -60])
    ax.set_ylim([20, 55])

    # Add labels with number of data values
    for state, row in us_map.iterrows():
        value = row['data']
        color = 'white' if value > 150000 else 'black'
        plt.annotate(text=f"{round(value/1000, 2)}k", xy=row['geometry'].centroid.coords[0], 
                    horizontalalignment='center', verticalalignment='center', 
                    fontsize=8, color=color)

    # Show the plot
    plt.show()


def plot_addr_state_bar_plots(df1, df2):
    f, ax = plt.subplots(1, 2, figsize=(18, 8))
    vals1 = df1['addr_state'].value_counts()
    vals2 = df2['addr_state'].value_counts()
    
    x = list(vals1.keys())
    y = list(vals1)
    ax[0].barh(x, y, color='#398053')
    ax[0].set_xlabel("Num. of loans")
    ax[0].set_ylabel("State Abbreviation")

    x = list(vals2.keys())
    y = list(vals2)
    ax[1].barh(x, y, height=0.8, color="#80363e")
    ax[1].set_xlabel("Num. of loans")
    ax[1].set_ylabel("State Abbreviation")

    f.suptitle( 
        "Address States Distribution between accepted and rejected loans",
        fontsize=15)

    plt.show()

def plot_purpose_acc_rej(df1, df2):
    accepted_purposes = df1['purpose'].value_counts()
    rejected_purposes = df2['purpose'].value_counts()

    fig, ax = plt.subplots(1, 2, figsize=(18, 7))
    accepted_purposes.plot.pie(ax=ax[0], colors=qualitative.Set2_4.hex_colors, textprops={'fontsize': 7})
    ax[0].set_title(f"Accepted Loans Purpose Distribution")
    rejected_purposes.plot.pie(ax=ax[1], colors=qualitative.Set2_4.hex_colors, textprops={'fontsize': 7})
    ax[1].set_title(f"Rejected Loans Purpose Distribution")
    fig.suptitle(f"Distribution of purpose values in accepted and rejected loans", fontsize=13)

def plot_kdes_col(dfs, col):
    plt.figure(figsize=(15, 8))
    colors = qualitative.Set2_6.hex_colors  

    for i in range(0, len(dfs)): 
        sns.kdeplot(
            data=dfs[i],
            x=col,
            fill=True,
            common_norm=False,
            color=colors[i],
            alpha=0.5,
            linewidth=0
        )

def get_cols_with_large_missing_vals(df) -> list:
    '''  '''
    columns = {}
    for col, val in df.isnull().sum().items():
        if val > len(df) / 2:
            num_good_entries = len(df) - val
            if num_good_entries < 90000 and "sec" not in col and "joint" not in col:
                columns[col] = num_good_entries
    return columns
 

tokenizer = RegexpTokenizer(r'\w+')
stemmer = SnowballStemmer('english') 

def stem(row):
    tokens = tokenizer.tokenize(row)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens) 
    
def fit_resample(pipeline, X_train, y_train):
    return pipeline.fit_resample(X_train, y_train)

def plot_conf_matrices(results, y, labels="", annot=True, roc_thres=False):
    f, ax = plt.subplots(1, 3, figsize=(18, 4.5))
    
    for i in range(0, len(results)):
        model_name = type(list(results.keys())[i].named_steps['model']).__name__
             
        predictions = list(results.items())[i][1]

        if roc_thres:
            fpr, tpr, thresholds = metrics.roc_curve(predictions, y)
            j_scores = tpr - fpr
            best_threshold_idx = np.argmax(j_scores)
            best_threshold = thresholds[best_threshold_idx]
            predictions = (y >= best_threshold).astype(int)
    
        cm = confusion_matrix(y, predictions)
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
        "Confusion matrices of validation data for chosen models " + thrs,
        fontsize=16)
    plt.show()
    
def objective(trial, pipeline, model_name, x_train, y_train, x_val, y_val):

    if isinstance(model_name, lgb.LGBMClassifier): 
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 1.0),
            'max_depth': trial.suggest_int('max_depth', 2, 20),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
            'subsample': trial.suggest_uniform('subsample', 0.1, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1.0)
        }
        model = lgb.LGBMClassifier(**params)

    elif isinstance(model_name, XGBClassifier):
        params = {
            'max_depth': trial.suggest_int('max_depth', 2, 20),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
            'subsample': trial.suggest_uniform('subsample', 0.1, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1.0),
            'gamma': trial.suggest_loguniform('gamma', 0.001, 1.0),
            'alpha': trial.suggest_loguniform('alpha', 0.001, 1.0),
            'lambda': trial.suggest_loguniform('lambda', 0.001, 1.0)
        }
        model = XGBClassifier(**params)

    elif isinstance(model_name, KNeighborsClassifier):
        params = {
            'n_neighbors': 5,
            'weights': 'uniform',
            'p': 2,
            'leaf_size': 30,
            'algorithm': 'auto'
        }
        model = KNeighborsClassifier(**params)

        
    elif isinstance(model_name, XGBClassifier):
        params = {
            'max_depth': trial.suggest_int('max_depth', 2, 20),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
            'subsample': trial.suggest_uniform('subsample', 0.1, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1.0),
            'gamma': trial.suggest_loguniform('gamma', 0.001, 1.0),
            'alpha': trial.suggest_loguniform('alpha', 0.001, 1.0),
            'lambda': trial.suggest_loguniform('lambda', 0.001, 1.0)
        }
        model = XGBClassifier(**params)

    elif isinstance(model_name, GradientBoostingClassifier): 
        params = {
            'max_depth': trial.suggest_int('max_depth', 2, 20),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 1.0),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 100),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50),
            'subsample': trial.suggest_uniform('subsample', 0.1, 1.0),
            'max_features': trial.suggest_uniform('max_features', 0.1, 1.0)
        }
        model = GradientBoostingClassifier(**params)

    elif isinstance(model_name, AdaBoostClassifier):  
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 1.0),
            'algorithm': trial.suggest_categorical('algorithm', ['SAMME', 'SAMME.R'])
        }
        model = AdaBoostClassifier(**params)

    elif isinstance(model_name, RandomForestClassifier):   
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
            'max_depth': trial.suggest_int('max_depth', 2, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 100),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50),
            'max_features': trial.suggest_uniform('max_features', 0.1, 1.0)
        }
        model = RandomForestClassifier(**params)

    elif isinstance(model_name, ExtraTreesClassifier):    
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
            'max_depth': trial.suggest_int('max_depth', 2, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 100),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50),
            'max_features': trial.suggest_uniform('max_features', 0.1, 1.0)
        }
        model = ExtraTreesClassifier(**params)

    elif isinstance(model_name, LogisticRegression): 
        params = {
            'penalty': trial.suggest_categorical('penalty', ['l2']),
            'C': trial.suggest_loguniform('C', 1e-4, 1e4),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs', 'saga'])
        }
        model = LogisticRegression(**params)

    elif isinstance(model_name, BaggingClassifier):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_samples': trial.suggest_float('max_samples', 0.1, 1.0),
            'max_features': trial.suggest_float('max_features', 0.1, 1.0),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'bootstrap_features': trial.suggest_categorical('bootstrap_features', [True, False])
        }
        model = BaggingClassifier(**params)

    elif isinstance(model_name, GaussianNB):
        params = {
            "var_smoothing": trial.suggest_loguniform('var_smoothing', 1e-12, 1e-5),
            "priors": trial.suggest_uniform('priors', 0, 1)
        }
        model = GaussianNB(**params)

    elif isinstance(model_name, QuadraticDiscriminantAnalysis): 
        params = {
            'reg_param': trial.suggest_float('reg_param', 0.0, 1.0),
            'tol': trial.suggest_float('tol', 1e-5, 1e-1, log=True),
            'store_covariance': trial.suggest_categorical('store_covariance', [True, False])
        }

        model = QuadraticDiscriminantAnalysis(**params)

    elif isinstance(model_name, BernoulliNB): 
        params = {
            "alpha": trial.suggest_uniform('alpha', 0.0, 1.0),
            "fit_prior": trial.suggest_categorical('fit_prior', [True, False]),
            "binarize": trial.suggest_uniform('binarize', 0.0, 1.0),
            "class_prior": trial.suggest_categorical('class_prior', [None, [0.2, 0.3, 0.5]])
        }
        model = BernoulliNB(**params)

    elif isinstance(model_name, DecisionTreeClassifier): 
        params = {
            'max_depth': trial.suggest_int("max_depth", 2, 10),
            'min_samples_split': trial.suggest_int("min_samples_split", 2, 10),
            'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 10)
        }
        model = DecisionTreeClassifier(**params) 
    
    pipeline.named_steps['model'] = model 
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_val) 
     
    f1 = f1_score(y_val, y_pred, average='weighted')
    return f1


def objective_reg(trial, pipeline, model_name, x_train, y_train, x_val, y_val):

    if isinstance(model_name, RandomForestRegressor):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        }
        model = RandomForestRegressor(**params)

    elif isinstance(model_name, GradientBoostingRegressor):
        params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
        'learning_rate': trial.suggest_uniform('learning_rate', 0.001, 1.0)
        }
        model = GradientBoostingRegressor(**params)

    elif isinstance(model_name, DecisionTreeRegressor):
        params = {
            'max_depth': trial.suggest_int('max_depth', 1, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
        }
        model = DecisionTreeRegressor(**params)

    elif isinstance(model_name, LinearRegression):
        params = {}
        model = LinearRegression(**params)
    
    elif isinstance(model_name, Ridge):
        params = {
            'alpha': trial.suggest_uniform('alpha', 0.01, 10),
            'solver': trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']),
            'max_iter': trial.suggest_int('max_iter', 1000, 5000)
        }
        model = Ridge(**params)
  
    elif isinstance(model_name, Lasso):
        params = {
            'alpha': trial.suggest_uniform('alpha', 0.01, 10),
            'tol': trial.suggest_uniform('tol', 1e-6, 1e-3),
            'selection': trial.suggest_categorical('selection', ['cyclic', 'random']),
            'max_iter': trial.suggest_int('max_iter', 1000, 5000)
        }
        model = Lasso(**params)

    elif isinstance(model_name, ElasticNet):
        params = {
            'alpha': trial.suggest_uniform('alpha', 0.01, 10),
            'l1_ratio': trial.suggest_uniform('l1_ratio', 0, 1),
            'tol': trial.suggest_uniform('tol', 1e-6, 1e-3),
            'selection': trial.suggest_categorical('selection', ['cyclic', 'random']),
            'max_iter': trial.suggest_int('max_iter', 1000, 5000)
        }
        model = ElasticNet(**params)
 
    elif isinstance(model_name, SVR):
        params = {
            'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
            'C': trial.suggest_loguniform('C', 1e-4, 1e4),
            'epsilon': trial.suggest_loguniform('epsilon', 1e-4, 1),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'] + ['scale' for i in range(10)] + ['auto' for i in range(10)]),
            'degree': trial.suggest_int('degree', 2, 5)
        }
        model = SVR(**params)

    elif isinstance(model_name, KNeighborsRegressor):
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 1, 50),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
            'leaf_size': trial.suggest_int('leaf_size', 10, 100),
            'p': trial.suggest_int('p', 1, 2)
        }
        model = KNeighborsRegressor(**params)
 
 
    pipeline.named_steps['model'] = model 
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_val) 

    r2 = r2_score(y_val, y_pred)
    return r2



def try_models(models, x_train, y_train, x_val, y_val):
    scores = {} 
    for model in models:         
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        scores[model] = [f1_score(y_val, y_pred, average='weighted'),
                         recall_score(y_val, y_pred, average='weighted'),
                         precision_score(y_val, y_pred, average='weighted')]

    models_results = pd.DataFrame(index=None)
    models_results['model'] = list(scores.keys())
    
    models_results['model_name'] = [type(key.named_steps['model']).__name__ for key in scores.keys()]
    models_results['f1_score'] = [val[0] for val in list(scores.values())]
    models_results['recall_score'] = [val[1] for val in list(scores.values())]
    models_results['precision_score'] = [val[2] for val in list(scores.values())]
    models_results = models_results.sort_values(by="f1_score", ascending=False)
    return models_results

def try_reg_models(models, x_train, y_train, x_val, y_val): 
    scores = {}
    for model in models:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        scores[model] = [r2_score(y_val, y_pred), np.sqrt(mean_squared_error(y_val, y_pred))]

    models_results = pd.DataFrame(index=None)
    models_results['model'] = list(scores.keys())
    models_results['model_name'] = [type(key.named_steps['model']).__name__ for key in scores.keys()]
    models_results['r2_score'] = [val[0] for val in list(scores.values())]
    models_results['rmse'] = [val[1] for val in list(scores.values())]
    models_results = models_results.sort_values(by="r2_score", ascending=False)
    return models_results


def optuna_tuning(objective_func, chosen_models, x_train, y_train, x_val, y_val, score='f1_score'):
    results = {}
    df_models = chosen_models.copy()
    optuna.logging.get_logger("optuna").setLevel(logging.WARNING)
    for model in list(df_models.model.unique()): 
        print(f"Tuned {type(model.named_steps['model']).__name__}:")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective_func(trial, model, model.named_steps['model'], x_train, y_train, x_val, y_val), n_trials=25) 
        results[(model, type(model.named_steps['model']).__name__)] = {"best_params": study.best_params, score: study.best_value}
        df = pd.DataFrame(study.best_params, index=[0])
        df.insert(0, score, [study.best_value]) 
        display(df) 

    return results
 

def train_val_test_split_reg(X_data, Y_data, target):
    X = X_data.copy()
    X[target] = list(Y_data[target])
    
    X_train, X_test, y_train, y_test= train_test_split(
        X[[col for col in X.columns if col != target]],
        X[target], 
        train_size=0.7,
        test_size=0.3,
        random_state=42
    )
     
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
     
    return  X_train, X_val, X_test, y_train, y_val, y_test

def train_val_test_split(X_data, Y_data, target):
    X = X_data.copy()
    X[target] = list(Y_data[target])

    categorical_features = list(X.select_dtypes(include=['object', 'category']).columns)
    categorical_features.append(target) 

    X[categorical_features] = X[categorical_features].fillna('missing')
    
    X_train, X_test, y_train, y_test= train_test_split(
        X[[col for col in X.columns if col != target]],
        X[target], 
        train_size=0.7,
        test_size=0.3,
        random_state=42, 
        stratify=X[[target]],
    )
 
    for col in categorical_features:
        X[col] = X[col].apply(lambda x: None if x == "missing" else x) 
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1, stratify=y_train)
     
    return  X_train, X_val, X_test, y_train, y_val, y_test

def get_best_cols_from_mi_scores(mi_scores, df, threshold):
    joint_cols = list( set([col for col in df.columns if "sec" in col or "joint" in col]))
    cols = [col for col in df.columns if not ("sec" in col or "joint" in col) and any(col in joint_col for joint_col in joint_cols)]

    not_remove_cols = cols + joint_cols

    useful_cols = [] 
    for col, val in mi_scores.items():
        if val >= threshold: 
            useful_cols.append((col)) 
    
    results = useful_cols + not_remove_cols
    return set(results) 
 
def get_and_plot_mi_scores(df, target_df, target):
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
    df_clean = df.copy()
    stratified_samples = {}
    
    sample_size = threshold
    if not threshold:
        sample_size = df[target].value_counts().min()
    
    for grade in df[target].unique():
        df_cat = df_clean[df_clean[target] == grade] 
        df_stratified_sample = df_cat.groupby(target, group_keys=False).apply(lambda x: x.sample(min(len(x), sample_size)))
        stratified_samples[grade] = df_stratified_sample

    return pd.concat(stratified_samples.values())

def preprocessor(numcols, ohcols, ordcols, joint_numcols=None, joint_catcols=None, model_type=None):
    emp_title = False 
    if "emp_title" in ohcols:
        ohcols.remove('emp_title')
        emp_title = True

    knn_imputer = KNNImputer(n_neighbors=5)

    if model_type == "tree":
        joint_cols_categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value="None")),
            ('ord', OrdinalEncoder())
        ])
    else:
        joint_cols_categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value="None")),
            ('onehot', OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
        ])

    joint_cols_numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', knn_imputer),
        ('scaler', StandardScaler())
    ])

    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OrdinalEncoder())
    ])

    if model_type != "tree":
        onehot_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
        ])
    else:
        ordcols.extend(ohcols)
    
    if joint_numcols is not None and joint_catcols is not None:
        preprocessor = ColumnTransformer(
            transformers=[
                ('joint_num', joint_cols_numerical_transformer, joint_numcols),
                ('joint_cat', joint_cols_categorical_transformer, joint_catcols),
                ('num', numerical_transformer, numcols),
                ('ord', ordinal_transformer, ordcols),
            ],
            remainder='passthrough'
        ) 
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numcols),
                ('ord', ordinal_transformer, ordcols),
            ],
            remainder='passthrough'
        ) 


    if model_type != "tree": 
        ohec_transformer = ('ohe', onehot_transformer, ohcols)
        new_pipeline = ColumnTransformer(
            transformers=[
                *preprocessor.transformers,
                ohec_transformer
            ],
            remainder=preprocessor.remainder
        )
        preprocessor = new_pipeline

    if emp_title:
        print("yas")
        text_pipeline = Pipeline([
            ('vect', TfidfVectorizer())
        ])
        text_transformer = ('text', text_pipeline, 'emp_title')
         
        new_pipeline = ColumnTransformer(
            transformers=[
                *preprocessor.transformers,
                text_transformer
            ],
            remainder=preprocessor.remainder
        )
        preprocessor = new_pipeline
     
    return preprocessor
    
def get_useless_joint_cols(mi_scores, common_not_joint_cols, joint_cols, threshold=0.004):
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
    elif len(X)  > 200:
        batch_size = 10
    else:
        batch_size = 1

    ipca = IncrementalPCA(n_components=None, batch_size=batch_size)
    
    for batch_X in np.array_split(X, len(X) // batch_size):
        ipca.partial_fit(batch_X)
    
    cumulative_var_ratio = np.cumsum(ipca.explained_variance_ratio_)

    n_components = np.argmax(cumulative_var_ratio >= threshold) + 1

    print('Number of components to keep:', n_components)
    plt.plot(cumulative_var_ratio)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.show()

    return n_components

def perform_ipca(n_components, X_train):
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
    elif len(X)  > 200:
        batch_size = 10
    else:
        batch_size = 1
 
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    
    for batch_X in np.array_split(X, len(X) // batch_size):
        ipca.partial_fit(batch_X)

    X_ipca = ipca.transform(X)
    df_ipca = pd.DataFrame(data=X_ipca, columns=[f'component_{i+1}' for i in range(n_components)])
    return (df_ipca, ipca)


def plot_3_pcas(ipca, X, y_train):
    explained_variances = ipca.explained_variance_ratio_
    print('Explained variance ratios:', explained_variances)
    top_components = ipca.components_[:3, :]
    X_3d = csr_matrix.dot(X, top_components.T)
 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=y_train)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')

    plt.show()

def get_top_n_models_preds(final_model_results, X_train, y_train, X_val, num_models=3):
    predictions = {}

    top3_models = copy.deepcopy(final_model_results)
    top3_models = list(final_model_results.items())[:num_models]

    for pipeline, res in top3_models:
        new_pipeline = copy.deepcopy(pipeline[0])
        
        if 'model' not in new_pipeline.named_steps:
            raise ValueError("Pipeline does not contain a 'model' step.")
        
        best_params = res['best_params']
        model = new_pipeline.named_steps['model']
        
        model.set_params(**best_params)  
        new_pipeline.model = model
        new_pipeline.fit(X_train, y_train) 
        preds = new_pipeline.predict(X_val)
        predictions[new_pipeline] = preds
        
    return predictions
 
def remove_num_outliers(df, nunique=100000, thres=0.9, targets=['grade', 'sub_grade', 'int_rate']):
    check_outliers_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    real_num = [col for col in check_outliers_cols if df[col].nunique() > nunique]
    data = df.copy()
    
    for col in real_num:
        if col not in targets:
            data = remove_outliers_col(data, col, thres)
    return data

def plot_kde_plot(df, col, hue_col):
    plt.figure(figsize=(15, 8))

    for val in sorted(df[hue_col].unique()):
        if val != None:
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
            color='#5d95a3',
            ax=axes[i][j],
            scatter_kws={'s':10}
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
        axes[i].hist(
            df[cols_to_plot[i]], bins=20, color=color
        )
        axes[i].set_title(f"{cols_to_plot[i]}", fontsize=9)

    fig.suptitle(
        f"Distributions of {cols_to_plot}{desc}",
        fontsize=14,
    )
    plt.show()

def plot_cat_col_target_corr(corr_matrix, cat_col, target): 
    plt.figure(figsize=(15, 7))
    corr_matrix.plot(kind='bar')
    plt.xlabel("Features")
    plt.ylabel('Correlation')
    plt.title(f'Correlation of different {cat_col} values with target - "{target}"', fontsize=15)
    plt.show()

def plot_pearson_corr_map(df):
    corr_df = df.corr()
    plt.figure(figsize=(13, 9))
    sns.heatmap(corr_df, mask=np.triu(corr_df))
    plt.title("Feature Correlation Heatmap")
    plt.show()

def get_too_correlated_cols_and_print_pairs(df, all_common_cols, scores):
    data = df.copy()
    corr_matrix = data[[col for col in data.columns if col not in all_common_cols]].corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    
    threshold = 0.9
    corr_pairs = {}
    for column in upper_triangle.columns:
        correlated = upper_triangle[column][upper_triangle[column] > threshold].index.tolist()
        if len(correlated) > 0:
            corr_pairs[column] = correlated

    to_drop = []
    for col, correlated in corr_pairs.items():
        for c in correlated:
            col_to_drop = col
            if scores[col] > scores[c]:
                col_to_drop = c
            
            to_drop.append(col_to_drop)

            print(f"{col} and {c} are highly correlated: {round(corr_matrix.loc[col, c], 3)}. The column dropped will be {col_to_drop}.")

    return to_drop

 
def perform_cross_validation(predictions, X_train, y_train, y_val, score='f1_score'):
    for pipeline, preds in list(predictions.items()):  
        train_predictions = cross_val_predict(pipeline, X_train, y_train, cv=5) 
 
        if score == 'f1_score':
            train_score = f1_score(y_train, train_predictions, average='weighted') 
            val_score = f1_score(y_val, preds, average='weighted')
        else:
            train_score = r2_score(y_train, train_predictions, multioutput='variance_weighted')
            val_score = r2_score(y_val, preds)
          
        model_name = type(pipeline.named_steps['model']).__name__
        test_str = f'VALIDATION DATA - {score} for the {model_name} model is: {round(val_score, 3)}\n'
        cross_val_str = f'TRAIN DATA (cross validated) {score} is {round(train_score, 3)}\n'

        header_sep = '-' * int(round((max(len(test_str), len(cross_val_str)) - len(model_name) - 2)/2)) 
        print(f"{header_sep} {model_name} {header_sep}\n{cross_val_str}{test_str}")


def plot_qq(predictions, y_val):
    fig, axes = plt.subplots(1, 1, figsize=(20, 9))

    for model in list(predictions.items()):  
        y_pred = model[1]
        residuals = y_val - y_pred
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals))) 
        empirical_quantiles = np.quantile(np.sort(residuals), np.linspace(0.01, 0.99, len(residuals)))
        plt.scatter(theoretical_quantiles, empirical_quantiles, label=type(model[0].named_steps['model']).__name__ )

    plt.xlabel("Theoretical Quantiles") 
    plt.ylabel("Empirical Quantiles")
    plt.title("QQ Plot of Residuals", fontsize=16)
    plt.plot([-3, 3], [-3, 3], color="red")
    plt.legend()

    plt.show()

def getting_pipelines(tree_models, tree_preprocessor, tree_ipca, other_models, preproc, ipca):
    pipelines = []
    for model in tree_models:
        pipeline = Pipeline([
            ('preprocessor', tree_preprocessor),
            ('IPCA', tree_ipca),
            ('model', model),])
        pipelines.append(pipeline)

    for model in other_models: 
        pipeline = Pipeline([
            ('preprocessor', preproc),
            ('IPCA', ipca),
            ('model', model),])
        pipelines.append(pipeline)

    return pipelines