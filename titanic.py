import dtale
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

# constantes
PATH_TRAIN_DATA = "train.csv"
PATH_TEST_DATA = "test.csv"
RANDOM_SEED = 42
NUM_STRATEGY = "mean"
CAT_STRATEGY = "most_frequent"
FEATURE_COLS = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Cabin",
    "Embarked"
]


def preproc_data(data: pd.DataFrame) -> pd.DataFrame:
    single_women_filter = data["Name"].str.contains("(Miss|Ms)")
    married_women_filter = data["Name"].str.contains("Mrs")
    all_w_filter = data["Sex"] == "female"

    single_women = data.loc[single_women_filter]
    married_women = data.loc[married_women_filter]

    s_women_age_median = married_women.dropna()["Age"].median()
    m_women_age_median = single_women.dropna()["Age"].median()

    data.loc[all_w_filter & (data.Age.isna()) & single_women_filter, ["Age"]] = s_women_age_median
    data.loc[all_w_filter & (data.Age.isna()) & married_women_filter, ["Age"]] = m_women_age_median

    all_m_filter = data["Sex"] == "male"
    p1_men_filter = (data["Pclass"] == 1) & (all_m_filter)
    p2_men_filter = (data["Pclass"] == 2) & (all_m_filter)
    p3_men_filter = (data["Pclass"] == 3) & (all_m_filter)

    pclass1_men = data.loc[p1_men_filter].dropna()
    pclass2_men = data.loc[p2_men_filter].dropna()
    pclass3_men = data.loc[p3_men_filter].dropna()

    p1_men_median_age = pclass1_men["Age"].median()
    p2_men_median_age = pclass2_men["Age"].median()
    p3_men_median_age = pclass3_men["Age"].median()

    data.loc[(p1_men_filter) & (data["Age"].isna()), ["Age"]] = p1_men_median_age
    data.loc[(p2_men_filter) & (data["Age"].isna()), ["Age"]] = p2_men_median_age
    data.loc[(p3_men_filter) & (data["Age"].isna()), ["Age"]] = p3_men_median_age

    data["Cabin"] = data["Cabin"].fillna("Z").apply(lambda x: x[0])
    data["Embarked"] = data["Embarked"].fillna("Z")

    return data


train_data = pd.read_csv(PATH_TRAIN_DATA)
train_data = preproc_data(train_data)
X_train = train_data.loc[:, FEATURE_COLS]
y_train = train_data.loc[:, "Survived"]

test_data = pd.read_csv(PATH_TEST_DATA)
test_data = preproc_data(test_data)
X_test = test_data.loc[:, FEATURE_COLS]

ohe = OneHotEncoder()
cat_imputer = SimpleImputer(strategy=CAT_STRATEGY)
num_imputer = SimpleImputer(strategy=NUM_STRATEGY)
scaler = StandardScaler()
gbc = GradientBoostingClassifier(random_state=RANDOM_SEED)

# pipe numeric
num_feat = ["Age", "Pclass", "SibSp", "Parch", ]
num_transf = Pipeline([("Num_Imputer", num_imputer), ("Scaler", scaler)])


# pipe categoric
cat_feat = ["Sex", "Embarked", "Cabin"]
cat_transf = Pipeline([("Cat_imputer", cat_imputer), ("OneHot", ohe)])

# preprocessador
preprocessor = ColumnTransformer(
    transformers=[("Numeric", num_transf, num_feat), ("Categoric", cat_transf, cat_feat)]
)


pipe = Pipeline([("Preprocessor", preprocessor), ("Estimator", gbc)])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

test_data["Survived"] = y_pred
test_data.loc[:, ["PassengerId", "Survived"]].to_csv("submission_02.csv", index=False)

dtale.show(train_data)