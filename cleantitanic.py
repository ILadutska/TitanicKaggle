import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import OneHotEncoder


def substrings_in_string(big_string, substrings):
    if big_string == None:
        return "Unknown"
    for substring in substrings:
        if substring in big_string:
            return substring
    return np.nan


def replace_titles(x):
    title = x["Title"]
    if title in ["Don", "Major", "Capt", "Jonkheer", "Rev", "Col"]:
        return "Mr"
    elif title in ["Countess", "Mme"]:
        return "Mrs"
    elif title in ["Mlle", "Ms"]:
        return "Miss"
    elif title == "Dr":
        if x["Sex"] == "Male":
            return "Mr"
        else:
            return "Mrs"
    else:
        return title


def guessAge(df):
    i = 0
    for row in df["Age"]:
        if row == "Unknown":
            if df["Parch"][i] > 1:
                df["Age"][i] = (float)(random.randint(1, 10))
            else:
                df["Age"][i] = (float)(random.randint(10, 40))
        i += 1

    return df


def cleanTitanic(df):
    if "PassengerId" and "Ticket" in df.columns:
        df = df.drop(["PassengerId", "Ticket"], axis=1)
    df = df.replace(np.nan, "Unknown")

    cabin_list = ["A", "B", "C", "D", "E", "F", "T", "G", "Unknown"]
    title_list = [
        "Mrs",
        "Mr",
        "Master",
        "Miss",
        "Major",
        "Rev",
        "Dr",
        "Ms",
        "Mlle",
        "Col",
        "Capt",
        "Mme",
        "Countess",
        "Don",
        "Jonkheer",
    ]

    # replacing all titles with mr, mrs, miss, master
    if "Name" in df.columns:
        df["Title"] = df["Name"].map(lambda x: substrings_in_string(x, title_list))
        df["Title"] = df.apply(replace_titles, axis=1)
        df = df.drop(["Name"], axis=1)

    if "Cabin" in df.columns:
        df["Deck"] = df["Cabin"].map(lambda x: substrings_in_string(x, cabin_list))
        df = df.drop(["Cabin"], axis=1)

    if "FamilySize" not in df.columns:
        df["FamilySize"] = df["SibSp"] + df["Parch"]

    # df = guessAge(df)

    # df["Age"] = df["Age"].astype(float)

    return df


def encodeCategories(categories, df):
    for category in categories:
        if category in df.columns:
            category_col = df[category]

            # Create an instance of the OneHotEncoder
            encoder = OneHotEncoder(sparse=False)

            # Fit and transform the categorical column using one-hot encoding
            encoded_data = encoder.fit_transform(category_col.to_numpy().reshape(-1, 1))

            # Convert the encoded data into a DataFrame
            encoded_df = pd.DataFrame(
                encoded_data, columns=encoder.get_feature_names_out([category])
            )

            # Concatenate the encoded data with the original DataFrame (if needed)
            df = pd.concat([df, encoded_df], axis=1)
            df = df.drop([category], axis=1)

    return df
