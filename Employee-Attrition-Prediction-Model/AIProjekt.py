import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# def explore_data(df):
#     print("\nUnique Columns per Feature:")
#     for column in df.columns: 
#         print(f'{column}: Number of Unique Values: {df[column].nunique()}')

#     print('\nMissing Values:\n', df.isnull().sum())
#     print('Duplicated Rows:', df.duplicated().sum())
#     print('Data Shape:', df.shape)
#     print("\nDescriptive Statistics:\n", df.describe())

def encode_categorical(df):
    label = LabelEncoder()
    df["Attrition"] = label.fit_transform(df.Attrition)

    categorical_cols = df.select_dtypes(include='object').columns
    df[categorical_cols] = df[categorical_cols].apply(LabelEncoder().fit_transform)
    return df

def plot_initial_visualizations(df):
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x='DistanceFromHome', hue='Attrition', multiple='stack', bins=30)
    plt.title('Distance From Home by Attrition')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x='Age', hue='Attrition', multiple='stack', bins=30)
    plt.title('Age Distribution by Attrition')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x='MonthlyIncome', hue='Attrition', multiple='stack', bins=30)
    plt.title('Monthly Income by Attrition')
    plt.tight_layout()
    plt.show()

    sns.histplot(data=df, x='Department')
    plt.title('Department Distribution')
    plt.xticks(rotation=45)
    plt.show()

    sns.histplot(data=df, x='JobRole')
    plt.title('JobRole Distribution')
    plt.xticks(rotation=45)
    plt.show()

    plt.figure(figsize=(12, 10))
    sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap="RdYlGn", annot_kws={"size":10})
    plt.show()

def feature_engineering(df):
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, np.inf], labels=['<30', '30-40', '40-50', '50+'])
    df['IsFarFromHome'] = (df['DistanceFromHome'] > 10).astype(int)
    df['IncomePerYearWorked'] = df['MonthlyIncome'] * 12 / df['TotalWorkingYears'].replace(0, np.nan)
    df['FrequentJobChanger'] = (df['NumCompaniesWorked'] > (df['TotalWorkingYears'] / 2)).astype(int)
    df['LoyaltyIndex'] = df['YearsAtCompany'] / df['TotalWorkingYears'].replace(0, np.nan)
    df['ExperienceLevel'] = pd.cut(df['TotalWorkingYears'], bins=[-1, 5, 15, np.inf], labels=['Junior', 'Mid', 'Senior'])
    df['IncomeCategory'] = pd.cut(df['IncomePerYearWorked'], bins=[-np.inf, 5000, 10000, np.inf], labels=['Low', 'Medium', 'High'])
    df['LoyaltyCategory'] = pd.cut(df['LoyaltyIndex'], bins=[-np.inf, 0.3, 0.7, np.inf], labels=['Low', 'Medium', 'High'])
    df['CompanyTenure'] = pd.cut(df['YearsAtCompany'], bins=[-1, 2, 7, np.inf], labels=['New', 'Moderate', 'Long'])
    df['MonthlyIncomeCategory'] = pd.qcut(df['MonthlyIncome'], q=3, labels=['Low', 'Medium', 'High'])

    # print("\nFeature Engineered Columns:\n", df[['AgeGroup', 'IsFarFromHome', 'IncomeCategory', 'LoyaltyCategory',
    #                                           'CompanyTenure', 'MonthlyIncomeCategory', 'ExperienceLevel']].head())
    return df


def clean_data(df):
    df.drop(['Age','DistanceFromHome', 'MonthlyIncome', 'NumCompaniesWorked', 'TotalWorkingYears', 'YearsAtCompany'],
            axis='columns', inplace=True, errors='ignore')
    # print("\nColumns dropped successfully!")

    df = df.fillna({
        'AgeGroup': df['AgeGroup'].mode()[0],
        'IsFarFromHome': df['IsFarFromHome'].median(),
        'IncomePerYearWorked': df['IncomePerYearWorked'].median(),
        'FrequentJobChanger': df['FrequentJobChanger'].median(),
        'LoyaltyIndex': df['LoyaltyIndex'].median(),
        'ExperienceLevel': df['ExperienceLevel'].mode()[0],
        'IncomeCategory': df['IncomeCategory'].mode()[0],
        'LoyaltyCategory': df['LoyaltyCategory'].mode()[0],
        'CompanyTenure': df['CompanyTenure'].mode()[0],
        'MonthlyIncomeCategory': df['MonthlyIncomeCategory'].mode()[0]
    })

    label = LabelEncoder()
    cols_to_encode = [
        "BusinessTravel", "JobRole", "MaritalStatus", "OverTime", 
        "Department", "AgeGroup", "ExperienceLevel", "IncomeCategory", 
        "LoyaltyCategory", "CompanyTenure", "MonthlyIncomeCategory"
    ]

    for col in cols_to_encode:
        df[col] = label.fit_transform(df[col])

    df.dropna(inplace=True)
    return df

def train_model(df):
    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    k = 5  # not used in model, just printed for reference

    # print("\nClassification Report:\n", classification_report(y_test, y_pred))
    # print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    # print(f'Accuracy for rfc = {k} : {accuracy * 100:.2f}%')

    return model, y_test, y_pred, accuracy

def save_artifacts(model, df, y_test, y_pred, accuracy):
    joblib.dump(model, 'attrition_model.pkl')
    joblib.dump(df, 'processed_employee_attrition.pkl')
    joblib.dump({
        'y_test': y_test,
        'y_pred': y_pred,
        'accuracy': accuracy
    }, 'model_metrics.pkl')
    # print("\nArtifacts saved successfully (model + processed dataframe).")


