from pprint import pprint

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import explore as e
import prepare as p 

import nltk
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier






def get_original_df():
    df = pd.read_json('data.json')
    df = p.create_df(df)
    return df



def get_mvp(df):
    rows = ['C#', 'PHP', 'Shell', 'C\+\+']
    
    for row in rows:
        df = df[df["language"].str.contains(row) == False]
    return df


    


