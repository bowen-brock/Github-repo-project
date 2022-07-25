import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import prepare as p

import nltk
import nltk.sentiment

from wordcloud import WordCloud


def top_6_languages():
    # read json
    df = pd.read_json('data.json')
    df = p.create_df(df)
    df = df.drop(columns={'original'})

    # Swift programming language
    swift = df[df.language == 'Swift']

    # JavaScript programming language
    javascript = df[df.language == 'JavaScript']

    # Objective-C programming language
    objective_c = df[df.language == 'Objective-C']
    
    # python programming language
    python = df[df.language == 'Python']

    # Java programming language
    java = df[df.language == 'Java']

    # C programming language
    c = df[df.language == 'C']
    
    return swift, javascript, objective_c, python, java, c


def create_trigram(df):
    words = ' '.join(df.lemmatized)        
    trigrams = pd.Series(nltk.ngrams(words.split(),3))
    top_25_trigrams = trigrams.value_counts().head(25).sort_values(ascending=False)
    return top_25_trigrams


def word_cloud(trigram, string):
    #Generate a wordcloud
    #First create a dictionary for the fequencies of the bigrams
    data = {k[0] + ' ' + k[1] + ' ' + k[2]: count for k, count in trigram.to_dict().items()}
    #Create the wordcloud
    #Change the figsize before creating the wordcloud
    plt.figure(figsize=(10, 5))
    image = WordCloud(background_color = 'white', width = 800, height = 400).generate_from_frequencies(data)
    plt.title(f'Top 25 {string} WordCloud')
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    return

def get_graph(top_25_trigram, string):
    # Visualize the top 20 spam bigrams
    top_25_trigram.sort_values().plot.barh(width = .9, figsize = (12,6))
    plt.xlabel('Trigram Count')
    plt.ylabel('Trigram')
    plt.title(f'25 Most Common {string} Trigrams')
    plt.show()
    return
    
   