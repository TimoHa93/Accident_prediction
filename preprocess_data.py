import pandas as pd
import numpy as np
import re
import datetime
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, Imputer
from sklearn.base import TransformerMixin

from imblearn.over_sampling import SMOTE

class DataFrameImputer(TransformerMixin):

        def __init__(self):
            """Impute missing values.

            Columns of dtype object are imputed with the most frequent value
            in column.

            Columns of other types are imputed with mean of column.

            """
        def fit(self, X, y=None):

            self.fill = pd.Series([X[c].value_counts().index[0]
                if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
                index=X.columns)

            return self

        def transform(self, X, y=None):
            return X.fillna(self.fill)

def _replace_dates(accidents):
    #accidents['Jahr'] = ''
    #use regex to get the dates into the same format
    pattern_1 = re.compile('\d*.\s[A-Z]{1}[a-z]{2}.')
    pattern_2 = re.compile('\d*-[A-Z]{1}[a-z]{2}-\d{2}')
    """dates = {'Apr': 'Apr', 'Aug': 'Aug','Dez': 'slaDec','Feb': 'Feb', 'Jan': 'Jan',
             'Jul': 'Jul','Jun': 'Jun','Mai': 'May', 'Mrz': 'Mar','Nov': 'Nov',
             'Okt': 'Oct','Sep': 'Sep'}"""
    dates = {'Apr': 4, 'Aug': 8,'Dez': 8,'Feb': 8, 'Jan': 1,
             'Jul': 7,'Jun': 6,'Mai': 5, 'Mrz': 3,'Nov': 11,
             'Okt': 10,'Sep': 9, 'Mar': 3, 'May':5, 'Oct':10, 'Dec':12}


    for index, row in accidents.iterrows():
        if pattern_1.match(row['Unfalldatum']):
            #tag = row['Unfalldatum'][:-1].split('.',1)[0]
            monat = dates[row['Unfalldatum'][:-1].split('.', 1)[1][1:]]
            accidents.loc[index, 'Unfalldatum'] = monat
            #accidents.loc[index,'Jahr'] = np.nan

            if len(str(row['Zeit (24h)'])) > 2:
                accidents.loc[index, 'Zeit (24h)'] = int(str(row['Zeit (24h)'])[:-2])
        elif pattern_2.match(row['Unfalldatum']):
            #tag = row['Unfalldatum'].split('-', 2)[0]
            monat = dates[row['Unfalldatum'].split('-', 2)[1]]
            accidents.loc[index, 'Unfalldatum'] = monat
            #accidents.loc[index, 'Jahr'] = row['Unfalldatum'].split('-',2)[2]

            if len(str(row['Zeit (24h)'])) > 2:
                accidents.loc[index, 'Zeit (24h)'] = int(str(row['Zeit (24h)'])[:-2])
    #since months and time of day are cyclic, store them as cyclic data
    #accidents['Zeit (24h)'] = accidents['Zeit (24h)'].astype(str).str[:-2].astype(np.int64)

    accidents['sin_month'] = np.sin(2*np.pi*accidents.Unfalldatum/12)
    accidents['cos_month'] = np.cos(2*np.pi*accidents.Unfalldatum/12)

    accidents['sin_time'] = np.sin(2*np.pi*accidents['Zeit (24h)']/24)
    accidents['cos_time'] = np.cos(2*np.pi*accidents['Zeit (24h)']/24)

    #drop the old date and time cols
    accidents.drop(['Unfalldatum', 'Zeit (24h)'], axis=1, inplace=True)
    #accidents['Unfalldatum'] = pd.to_datetime(accidents['Unfalldatum'], format='%d. %b', errors='coerce')

    return accidents

def preprocess_data(accidents, accidents_labels):
     #set unknown values to np.nan
    accidents['Strassenklasse'] = accidents.Strassenklasse.replace('nicht klassifiziert', np.nan)
    #boden = {'Frost/ Ice': 'Frost / Eis', 'Überflutung':4, 9:np.nan}
    accidents.Bodenbeschaffenheit.replace('Frost/ Ice', 'Frost / Eis', inplace=True)
    accidents.Bodenbeschaffenheit.replace('9', np.nan, inplace=True)

    #accidents['Bodenbeschaffenheit'] = accidents.Bodenbeschaffenheit.map(boden)
    fahrzeugtyp_repl = lambda x: np.nan if x == 'Unbekannt' or x == '97' else x
    accidents['Fahrzeugtyp'] = accidents.Fahrzeugtyp.apply(fahrzeugtyp_repl)
    accidents.Wetterlage.replace('Unbekannt', np.nan, inplace=True)

    accidents = _replace_dates(accidents)

    #handling missing data
    from sklearn.preprocessing import Imputer
    imputer = Imputer(missing_values=np.nan, strategy="mean")
    num_cols = accidents[accidents.dtypes[(accidents.dtypes=='Int64')].index.values].columns
    accidents[num_cols] = pd.DataFrame(imputer.fit_transform(accidents[num_cols]))
    #for col in num_cols:
     #   accidents[col] = imputer.fit_transform(accidents[col])


    #Label encoding
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    cat_cols = accidents[accidents.dtypes[(accidents.dtypes=='object')].index.values].columns
    le=LabelEncoder()
    enc=OneHotEncoder(sparse=False)
    #iterating over all the common columns in the train test set
    #for the moment, drop all rows with missing categorical data
    #accidents[cat_cols].dropna(inplace=True)

    accidents[cat_cols] = DataFrameImputer().fit_transform(accidents[cat_cols])
    for col in cat_cols:
        accidents[col] = le.fit_transform(accidents[col].astype(str))
        accidents[col] = enc.fit_transform(accidents[col].values.reshape(-1,1))

    # standardizing the data
    #from sklearn.preprocessing import scale
    #accidents_scale = scale(accidents[accidents.dtypes[(accidents.dtypes=='Int64')].index.values])
    from sklearn.preprocessing import StandardScaler
    std_scaler = StandardScaler()
    accidents = std_scaler.fit_transform(accidents)

    X_train, X_test, y_train, y_test = train_test_split(accidents, accidents_labels, test_size=.2, random_state=42)

    #feature selection
    clf_sel = ExtraTreesClassifier(n_estimators=50)
    clf_sel = clf_sel.fit(X_train, y_train)
    model_sel = SelectFromModel(clf_sel, prefit=True)
    X_train_new = model_sel.transform(X_train)
    X_test_new = model_sel.transform(X_test)

    #oversampling
    sm = SMOTE(random_state=12, ratio = 1.0)
    X_train_res, y_train_res = sm.fit_sample(X_train_new, y_train)



    return model_sel, X_train_res, X_test_new, y_train_res, y_test

def preprocess_data_to_predict(df, model_sel):
     #set unknown values to np.nan
    df['Strassenklasse'] = df.Strassenklasse.replace('nicht klassifiziert', np.nan)
    #boden = {'Frost/ Ice': 'Frost / Eis', 'Überflutung':4, 9:np.nan}
    df.Bodenbeschaffenheit.replace('Frost/ Ice', 'Frost / Eis', inplace=True)
    df.Bodenbeschaffenheit.replace('9', np.nan, inplace=True)

    #df['Bodenbeschaffenheit'] = df.Bodenbeschaffenheit.map(boden)
    fahrzeugtyp_repl = lambda x: np.nan if x == 'Unbekannt' or x == '97' else x
    df['Fahrzeugtyp'] = df.Fahrzeugtyp.apply(fahrzeugtyp_repl)
    df.Wetterlage.replace('Unbekannt', np.nan, inplace=True)

    df = _replace_dates(df)

    #handling missing data
    #from sklearn.preprocessing import Imputer
    #imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
    num_cols = df[df.dtypes[(df.dtypes=='Int64')].index.values].columns

    #df[num_cols] = pd.DataFrame(imputer.fit_transform(df[num_cols]))
    df[num_cols].fillna(df[num_cols].mean(), inplace=True)

    #for col in num_cols:
     #   df[col] = imputer.fit_transform(df[col])
    #df['Alter'] = imputer.fit_transform(df['Alter'])
    #Label encoding
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    cat_cols = df[df.dtypes[(df.dtypes=='object')].index.values].columns
    le=LabelEncoder()
    enc=OneHotEncoder(sparse=False)
    #iterating over all the common columns in the train test set
    #for the moment, drop all rows with missing categorical data
    #df[cat_cols].dropna(inplace=True)

    df[cat_cols] = DataFrameImputer().fit_transform(df[cat_cols])
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        df[col] = enc.fit_transform(df[col].values.reshape(-1,1))

    # standardizing the data
    #from sklearn.preprocessing import scale
    #df_scale = scale(df[df.dtypes[(df.dtypes=='Int64')].index.values])
    from sklearn.preprocessing import StandardScaler
    std_scaler = StandardScaler()
    df = std_scaler.fit_transform(df)

    df_new = model_sel.transform(df)
    return df_new
