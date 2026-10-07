
import pandas as pd
import json
import os
import time

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# LOAD THE TITANIC DATASET
# ============================================================

train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')


# ============================================================
# SELECT FEATURES
# ============================================================

features = [
    'Pclass',
    'Sex',
    'Age',
    'SibSp',
    'Parch',
    'Fare',
    'Embarked'
]

X = train_data[features]
y = train_data['Survived']

test_X = test_data[features]


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    random_state=0,
    test_size=0.2
)


# ============================================================
# PREPROCESSING
# ============================================================

categorical_cols = [
    'Sex',
    'Embarked'
]

numerical_cols = [
    'Pclass',
    'Age',
    'SibSp',
    'Parch',
    'Fare'
]


# Numerical preprocessing
num_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)


# Categorical preprocessing
cat_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]
)


# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numerical_cols),
        ('cat', cat_transformer, categorical_cols)
    ]
)


# ============================================================
# LOGISTIC REGRESSION
# FIXED BEST HYPERPARAMETERS
# ============================================================

model = LogisticRegression(
    C=0.1,
    
    solver='lbfgs',
    max_iter=2000,
    random_state=0
)


# ============================================================
# PREPROCESSING + LOGISTIC REGRESSION PIPELINE
# ============================================================

model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ]
)


# ============================================================
# CROSS-VALIDATION
# ============================================================

start_time = time.time()

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)


print("Cross-validation scores:")

for i, score in enumerate(cv_scores, 1):
    print(f"Fold {i}: {score:.4f}")


print("\nMean CV Accuracy:", cv_scores.mean())
print("CV Standard Deviation:", cv_scores.std())


# ============================================================
# TRAIN MODEL
# ============================================================

model.fit(
    X_train,
    y_train
)

train_time = time.time() - start_time


# ============================================================
# VALIDATION SET
# ============================================================

preds_val = model.predict(X_valid)

accuracy = accuracy_score(
    y_valid,
    preds_val
)


print("\nValidation Accuracy:", accuracy)

print("\nClassification Report:")

print(
    classification_report(
        y_valid,
        preds_val
    )
)


# ============================================================
# SAVE RESULTS FOR THE INTERFACE
# results/logisticregressionclassification.json
# ============================================================

train_accuracy = accuracy_score(
    y_train,
    model.predict(X_train)
)

lr = model.named_steps['model']

summary = {
    'name': 'Logistic Regression',
    'cv_score': float(cv_scores.mean()),
    'cv_std': float(cv_scores.std()),
    'val_acc': float(accuracy),
    'val_f1': float(f1_score(y_valid, preds_val)),
    'train_acc': float(train_accuracy),
    'overfit_gap': float(train_accuracy - accuracy),
    'time': float(train_time),
    'params': {
        'C': lr.C,
        'penalty': lr.penalty,
        'solver': lr.solver
    }
}


results_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'results'
)

os.makedirs(
    results_dir,
    exist_ok=True
)


with open(
    os.path.join(
        results_dir,
        'logisticregressionclassification.json'
    ),
    'w',
    encoding='utf-8'
) as f:
    json.dump(
        summary,
        f,
        indent=2,
        default=str
    )


print(
    "\nResults saved to results/logisticregressionclassification.json"
)


# ============================================================
# TRAIN FINAL MODEL ON ALL TRAINING DATA
# ============================================================

print("\nTraining final Logistic Regression model on full dataset...")

model.fit(
    X,
    y
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

preds_test = model.predict(
    test_X
)


# ============================================================
# CREATE SUBMISSION
# ============================================================

output = pd.DataFrame({
'PassengerId': test_data['PassengerId'],
'Survived': preds_test
})

output.to_csv(
'TitanicLogisticRegressionClassifier.csv',
     index=False
)

# print("\nSubmission saved to TitanicLogisticRegressionClassifier.csv")
