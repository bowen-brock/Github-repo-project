#### Group Project By: Deangelo Bowen & Kayla Brock | Codeup | July 25, 2022

# A Bite Out Of Apple

#### A project using classification and natural language processing to predict repository programming language.

<hr style="border-top: 1px groove LightCyan ; margin-top: 1px; margin-bottom: 1px"></hr>

# I. Project Overview & EXECUTIVE SUMMARY

<hr style="border-top: 1px groove LightCyan ; margin-top: 1px; margin-bottom: 1px"></hr>

#### _Project Goal_

The goal of this project is to build a model that can accurately predict the programming language of a repository using its' README file.   

#### _Description_

This project was designed to reinforce the web scraping and natural language processing skills taught at Codeup. The prompt for the project required us to: scrape a minimum of 100 readme files from Github, prepare the data, evaluate it using natural language processing techniques, and build a model that could successfully predict the repository's programming language. We chose to scrape approximately 490 repositories. After data cleaning and preparation, our total count left was 463 repositories. Before moving into exploration, we decided to further drop data languages that had less than 20 repositories. Finally we explored the data using NLP techniques and created three classification models. Please refer to the final notebook in this repository to see our final results! 

#### _Initial Thoughts & Hypotheses_

- We believed, because we isolated the subject "Apple", our bigrams and trigrams could have more topic related keywords
- We believed, because there is no standard for the creation of readme files, the content of the README files would be more subjective to the writer and therefore more challenging to predict the programming language


#### _Key Findings_

After scraping GitHub repositories  using the search engine filtering for the keyword 'Apple', we discovered the most frequent keywords used across 22 different programming languages. We then created a model designed around the top 6 most used languages in our sample data that could predict the use of those languages based on their keywords and features with 53% accuracy. 

We do believe that with more time and more sample data, we could create a more accurate model with properly conducted feature engineering. 

#### _Deliverables_

- Indepth analysis on the NLP. 
- Summary of exploration and key takeaways. 
- Creating model(s) that can accurately predict the programming language used in a repository based on the READMe content.

# II. Project Data

<hr style="border-top: 1px groove LightCyan ; margin-top: 1px; margin-bottom: 1px"></hr>

The final DataFrame used to explore the data for this project contains the following variables (columns). The variables, along with their data types, are defined below: 


```python
from collections import OrderedDict
import pandas as pd
features = OrderedDict([ ('feature', ['repo', 'language', 'original', 'lemmatized']), ('description', ['GitHub repository name', 'Programming Language', 'Original Text before Cleaning/Preparation', 'Text after Cleaning/Preparation'])])                           

df = pd.DataFrame.from_dict(features)
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>repo</td>
      <td>GitHub repository name</td>
    </tr>
    <tr>
      <th>1</th>
      <td>language</td>
      <td>Programming Language</td>
    </tr>
    <tr>
      <th>2</th>
      <td>original</td>
      <td>Original Text before Cleaning/Preparation</td>
    </tr>
    <tr>
      <th>3</th>
      <td>lemmatized</td>
      <td>Text after Cleaning/Preparation</td>
    </tr>
  </tbody>
</table>
</div>



# III. Project PLAN

<hr style="border-top: 1px groove LightCyan ; margin-top: 1px; margin-bottom: 1px"></hr>

#### The following outlines the process taken through the data science pipeline to complete this project

#### _Plan_

In the planning stage We: read project expectations, created a project outline, wrote a project goal to include how we would measure success or failure, reviewed the overview of the dataset, documented all initial thoughts, questions, and hypotheses, created a plan for completing the project, created a data dictionary to define features, created a local folder and github repository.

#### _Acquire_

In the acquire stage we: created a .gitignore, searched for Apple repositories on GitHub and implemented a function that could go through each repository and return the README content.

#### _Prepare_

In the Prepare stage we: 
- Started with 490 values
    - dropped null-identified languages
    - Removed small sample sized languages identified such as, AppleScript, TypeScript, GO, HTML, CSS, etc.
    - tokenized then lemmatized the readme samples
    - removed stopwords, which involved appending `new_stopwords` for words we identified as irrelevant in the prediction of programming language (such as the excessive use of 'com' and 'www').

#### _Explore_

Before Exploration We: 
- Identified our top 6 languages, which were the languages that were used the most frequent based on our search. 
- These languages were `Swift`, `JavaScript`, `Objective-C`, `Python`,`Java`, and `C`.
- created trigrams for each languages most used words
- Represented them each in a word cloud to and bar graph to discover what the most frequent languages were. 

Through Exploration we were able to identify the most frequent words per programming language: 
- Swift
`Style`, `IMG` , `Shield IO`, and `SRV & SVC`
- JavaScript
`Open Source` , `Freeware` , `Software`
- Objective-C
`HTML`, `Audio`, `Video` , `Integration`, `Graphic & Animation`
- Python
`Python language was very case specific as none of the words defined pythonic language`
- Java
`APP`, `SVG`, `REACT`, `Shield IO`
- C
`Series`, `Trackpad`,`SRC`,`DRV`

Additional note about exploration: Throughout exploration we noticed that many 'frequent words' had more to do with the common theme of the repositories 'Apple' rather than the programming language itself. 

#### Model & Evaluate 

In the model and evaluate stage we: established baseline accuracy, trained and fit multiple models, compared evaluation metrics across models, evaluated best performing models using validate set, tested final model on out-of-sample testing dataset, and summarized performance #### bag of words 

#### _Deliver_

In the final stage we: prepared final notebook in Jupyter Notebook. We: wrote out the project description, introduction to include goals, created an executive summary which included all our key findings and recommendations, created headers and dividers to organize the flow of the notebook, and added summaries and supplementary markdown to guide the reader through the notebook.

# IV. Supplementary Files 

<hr style="border-top: 1px groove LightCyan ; margin-top: 1px; margin-bottom: 1px"></hr>

- acquire.py 
- explore.py 
- modeling.py 
- prepare.py 

# V. Steps to Reproduce

<hr style="border-top: 1px groove LightCyan ; margin-top: 1px; margin-bottom: 1px"></hr>

- Clone this repo (including acquire.py, explore.py, prepare.py and modeling.py)
- Run Final Report Jupyter notebook to view the final product
