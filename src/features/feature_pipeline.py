import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.pipeline import Pipeline
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.median_ages_by_title = {}
        self.median_fare_by_group = {}
        self.title_mapping = {
                'Mlle': 'Miss',
                'Major': 'Mr',
                'Col': 'Mr',
                'Sir': 'Mr',
                'Don': 'Mr',
                'Mme': 'Miss',
                'Jonkheer': 'Mr',
                'Lady': 'Mrs',
                'Capt': 'Mr',
                'Countess': 'Mrs',
                'Ms': 'Miss',
                'Dona': 'Mrs',
                'Rev': 'Mr',
                'the Countess': 'Miss'
        }

    def fit(self, x: pd.DataFrame, y=None):
        x_titles = self._extract_titles(x)

        if 'Age' in x_titles.columns and 'Title' in x_titles.columns:
            titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs']
            for title in titles:
                if title in x_titles['Title'].values:
                    median_age = x_titles.groupby('Title')['Age'].median().get(title)
                    if median_age is not None:
                        self.median_ages_by_title[title] = median_age

        if 'Fare' in x.columns:
            group_cols = ['Pclass']
            if 'Parch' in x.columns:
                group_cols.append('Parch')
            if 'SibSp' in x.columns:
                group_cols.append('SibSp')
            self.median_fare_by_group = x.groupby(group_cols)['Fare'].median().to_dict()

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x = x.copy()

        """Deal with missing values except 'Age' feature"""

        if 'Cabin' in x.columns:
            x['Cabin'] = x['Cabin'].fillna('Unknown')

        if 'Embarked' in x.columns:
            if 'Name' in x.columns:
                x.loc[x['Name'] == 'Icard, Miss. Amelie', 'Embarked'] = 'S'
                x.loc[x['Name'] == 'Stone, Mrs. George Nelson (Martha Evelyn)', 'Embarked'] = 'S'
            x['Embarked'] = x['Embarked'].fillna('S')

        if 'Fare' in x.columns and x['Fare'].isnull().any():
            x['Fare'] = x.groupby(['Pclass', 'Parch', 'SibSp'])['Fare'].transform(lambda x: x.fillna(x.median()))

        """Implement Feature Engineering"""

        if 'Name' in x.columns:
            x['Title'] = x['Name'].str.extract('([A-Za-z]+)\.', expand=False)
            x['Title'] = x['Title'].replace(self.title_mapping)

        if 'Age' in x.columns and 'Title' in x.columns:
            # Use the same fixed title list as original
            titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs']
            for title in titles:
                if title in self.median_ages_by_title:
                    age_to_impute = self.median_ages_by_title[title]
                    mask = (x['Age'].isnull()) & (x['Title'] == title)
                    x.loc[mask, 'Age'] = age_to_impute

        if 'Age' in x.columns:
            x['Age_bins'] = pd.qcut(x['Age'], 5, labels=False, duplicates='drop')

        if 'Fare' in x.columns:
            x['Fare_bins'] = pd.qcut(x['Fare'], 4, labels=False, duplicates='drop')

        if 'SibSp' in x.columns and 'Parch' in x.columns:
            x['Family_size'] = x['SibSp'] + x['Parch']

        if 'Survived' in x.columns:
            x['Family_Survival'] = self._calculate_family_survival(x)
        else:
            x['Family_Survival'] = 0.5

        if 'Sex' in x.columns:
            x['Sex'] = x['Sex'].map({'male': 0, 'female': 1})

        return x

    def _extract_titles(self, x: pd.DataFrame) -> pd.DataFrame:
        x = x.copy()

        if 'Name' in x.columns:
            x['Title'] = x['Name'].str.extract('([A-Za-z]+)\.', expand=False)
            x['Title'] = x['Title'].replace(self.title_mapping)

        return x

    def _calculate_family_survival(self, df: pd.DataFrame) -> pd.Series:
        family_survival = pd.Series(0.5, index=df.index)

        if 'Name' in df.columns and 'Fare' in df.columns and 'Survived' in df.columns:
            df = df.copy()
            df['Last_Name'] = df['Name'].apply(lambda x: str.split(x, ",")[0])

            for grp, grp_df in df[['Survived', 'Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                                   'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):
                if len(grp_df) != 1:
                    # A Family group exists
                    for ind, row in grp_df.iterrows():
                        smax = grp_df.drop(ind)['Survived'].max()
                        smin = grp_df.drop(ind)['Survived'].min()
                        if smax == 1.0:
                            family_survival.loc[ind] = 1.0
                        elif smin == 0.0:
                            family_survival.loc[ind] = 0.0

            for _, grp_df in df.groupby('Ticket'):
                if len(grp_df) != 1:
                    for ind, row in grp_df.iterrows():
                        if (family_survival.loc[ind] == 0.0) or (family_survival.loc[ind] == 0.5):
                            smax = grp_df.drop(ind)['Survived'].max()
                            smin = grp_df.drop(ind)['Survived'].min()
                            if smax == 1.0:
                                family_survival.loc[ind] = 1.0
                            elif smin == 0.0:
                                family_survival.loc[ind] = 0.0

        return family_survival


class FeatureCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop: Optional[List[str]] = None):
        self.columns_to_drop = columns_to_drop or [
            'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin',
            'Embarked', 'Title', 'Age', 'Fare', 'Last_Name'
        ]

    def fit(self, X: pd.DataFrame, y=None):
        self.existing_columns_to_drop = [col for col in self.columns_to_drop if col in X.columns]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.existing_columns_to_drop, errors='ignore')


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_encoded = X.copy()

        non_numeric_features = X_encoded.select_dtypes(include=['object', 'category']).columns.tolist()

        if 'Pclass' in X_encoded.columns:
            non_numeric_features.append('Pclass')

        X_encoded = pd.get_dummies(
            X_encoded,
            columns=non_numeric_features,
            drop_first=True
        )

        return X_encoded


def create_feature_pipeline() -> Pipeline:
    return Pipeline([
        ('preprocessor', Preprocessor()),
        ('cleaner', FeatureCleaner()),
        ('encoder', CategoricalEncoder()),
    ])


def create_preprocessing_pipeline(config: 'FeatureConfig') -> Pipeline:
    steps = []

    if config.scale_features:
        steps.append(('scaler', StandardScaler()))

    steps.extend([
        ('variance_threshold', VarianceThreshold(threshold=config.variance_threshold)),
        ('feature_selection', SelectKBest(score_func=f_classif, k=config.k_best_features))
    ])

    return Pipeline(steps)
