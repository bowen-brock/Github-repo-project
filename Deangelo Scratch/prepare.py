import pandas as pd
import json
from pandas import json_normalize

import unicodedata
import re

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords


# creating the function
def basic_clean(string):
    
    # lowercase everything
    string = string.lower()
    
    # remove inconsistenceis
    # encode into ascii byte strings
    # decode back into UTF-8
    # (This process will normalize the unicode characters)
    
    string = unicodedata.normalize('NFKD', string).encode('ascii', 'ignore').decode('UTF-8')
    
    # replace anything that is not a letter, number, whitespace, etc
    # use regex to perform this operation
    string = re.sub(r"[^a-z0-9\s]", ' ', string)
    
    return string


def tokenize(string):
    """
    This function will take in a string, tokenize the string and 
    return the tokenize string
    """
    
    #create the token
    token = nltk.tokenize.ToktokTokenizer()
    
    #Use the token
    string = token.tokenize(string,  return_str=True)
    
    return string


def lemmatize(string):
    """This function takes in a string and returns a lmeeatized 
    version of the string"""
    
    # create the lemmatizer
    wnl = nltk.stem.WordNetLemmatizer()
    
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    
    string_lemmatize = ' '.join(lemmas)
    
    return string_lemmatize

def remove_stopwords(string, extra_words=[], exclude_words=[]):
    """
    This function will take in a string, filter out stop words from the nltk standard english list 
    as well as any other extra words, and return a version of the text without these stopwords.
    It includes optional paramaters allowing the user to add extra words to remove 
    or to exclude words from the stopword list.
    """
    
    #get english stopwords from nltk
    stop_words = stopwords.words('english')
    new_stopwords = ["github", "com", "www", "http", "the", "a", "it", "they", "apple", "youtube",
                     "build", "status", 'sup', "introduction", "intro", "developer", "samplecode",
                    "user", "experience", "400", "200", "100", "&#9;", "kbd", "td", "tr", "org", 
                    "wiki", "wikipedia", "3rdparty", "license", "0", "version", "maximkulkin", "22", 
                    "33","44", "p", "e", "printername", "nsbp", "touch", "id", "error", "working", 
                    "green", "githubusercontent", "45159366", ]
    stop_words.extend(new_stopwords)
    
    #Add extra words to be removed to the stop word list
    for word in extra_words:
        stop_words.append(word)
    
    #Remove words to be excluded from the stop word list
    for word in exclude_words:
        stop_words.remove(word)
    
    #Create a list of words to be checked by splitting the string
    words = string.split()
    
    #Filter out all of the stop words
    filtered_words = [word for word in words if word not in stop_words]
    
    #Join the list of filtered words into a string
    filtered_string = ' '.join(filtered_words)
    
    return filtered_string

def advanced_clean(df):
    string = df.readme_contents
    
    lis = []
    i = 0
    while i <= 489:
        cleaned = basic_clean(string[i])
        token = tokenize(cleaned)
        lemmatized = lemmatize(token)
        filtered = remove_stopwords(lemmatized)
        lis.append(filtered)
        i+=1
    return lis  

def create_df(df):
    cleaned_df = advanced_clean(df)
    df1 = pd.DataFrame(cleaned_df)
    df2 = pd.concat([df, df1], axis=1, join="inner")
    
    #drop nulls
    df2 = df2.dropna()
    
    #rename columns
    df2 = df2.rename(columns={'readme_contents':'original', 0:'lemmatized'})
    
    #identify low sample size languages
    rows = ['AppleScript', 'TypeScript', 'Go','HTML', 'QML' , 'CSS', 'Dart', 'Vue', 'Starlark', 'Assembly', 'Kotlin',
        'Makefile', 'Perl','Zig', 'Eagle' , 'Dockerfile', 'CMake', 'Julia', 'ASL', 'CoffeeScript', 'Erlang',
    'Rich Text Format', 'ActionScript', 'VHDL' , 'Verilog', 'Objective-C\+\+', 'Matlab', 'R', 'ASP.NET', 'F#']
    
    # drop low sample sized languages
    for row in rows:
        df2 = df2[df2["language"].str.contains(row) == False]
    
    return df2