from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
import glob,re, os, sys, random
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, matthews_corrcoef, f1_score
from nltk.corpus import stopwords
from random import shuffle
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import pickle
import ast 
import pandas as pd
import re
import math


def weighted_accuracy(y_pred, y_true):
    '''
    Custom function that calculates the accuracy of each label and then averages the result
    '''
    label_1_true = []
    label_1_preds = []
    label_2_true = []
    label_2_preds = []
    for idx, y in enumerate(y_true):
        if y == 1:
            label_1_true.append(y)
            label_1_preds.append(y_pred[idx])
        else:
            label_2_true.append(y)
            label_2_preds.append(y_pred[idx])
    accs_1 = accuracy_score(label_1_preds, label_1_true)
    accs_2 = accuracy_score(label_2_preds, label_2_true)
    return (accs_1 + accs_2) / 2    

def return_metrics(y_pred, y_true, show=False):
    acc = accuracy_score(y_pred, y_true)
    mcc = matthews_corrcoef(y_pred, y_true)
    f1 = f1_score(y_pred, y_true)
    # wac = weighted_accuracy(y_pred, y_true)
    
    if show:
        print('ACC: ', round(acc,2))
        # print('WAC: ', round(wac,2))
        print('MCC: ', round(mcc,2))
        print('F1-: ', round(f1,2))
    return acc, mcc, f1

def split_db(df_complete, method, year=None, window=None, train_size=None):
    '''
    Splits a dataset into a train and test section based on criteria
    :param df_complete: the complete pandas dataframe of data
    :param method: the method used to complete the split:
        float: use train_test_split with the number as test_size
        str: 'only': test set includes all cases of 'year', traing set includes all cases before 'year'
        str: 'after': test set includes all cases of 'year' and all years after 'year', traing set includes all cases before 'year'
        str: 'window': test set includes all cases of a given year, train set of the past 'window' years
        str: 'random': test set includes all cases of a given year, training is a random sample of train_size
    :param year: integer, used in the 'only' and 'after' method
    return: train dataset and test dataset as pandas dataframes
    '''
    if isinstance(method, float): # method is a number between 0 and 1, indicating the percentage of cases that act as the test set
        return train_test_split(df_complete, 
                                test_size=method, 
                                random_state=1995, 
                                stratify=df_complete['violation'])
    if method == 'after': # test set includes all cases after the year (including the year itself), traing set includes all cases before 'year'
        return df_complete[df_complete['year'] < year], df_complete[df_complete['year'] >= year]
    if method == 'only': # test set includes all cases of a given year, traing set includes all cases before 'year'
        return df_complete[df_complete['year'] < year], df_complete[df_complete['year'] == year]
    if method == 'window': # test set includes all cases of a given year, train set of the past 'window' years
        x = df_complete[df_complete['year'] < year]
        return x[x['year'] >= year-window], df_complete[df_complete['year'] == year]
    if method == 'random': # test set includes all cases of a given year, training is a random sample of train_size
        
        # Get all of the cases that are not in the test set and balance that set
        train_df = df_complete[df_complete['year'] != year]
        train_df = balance_dataset(train_df)
        
        # Select equal number of violation and non violation to make a dataset of size 'train_size'
        train_df_v = train_df[train_df['violation']==0]
        train_df_v = train_df_v.sample(n=min(len(train_df_v), int(train_size/2)))
        
        train_df_nv = train_df[train_df['violation']==1]
        train_df_nv = train_df_nv.sample(n=min(len(train_df_nv), int(train_size/2)))
        
        train_df = pd.concat([train_df_v, train_df_nv])       
        return train_df, df_complete[df_complete['year'] == year] 

    print('Wrong format in split_db function')
    return None

def create_dataset(path, article, part):
    if 'json' in path:
        return create_dataset_echrod(path, article, part)
    else:
        return create_dataset_med(path, article, part)
    
def json_to_text(doc, part='facts'):
    '''
    Extracts relevant parts of the case text from a json case
    :param doc: list of dictionaries (json format) representing the content of a case
    :param part: the relevant part to be extracted
    :return: string, the relevant text from the case    
    '''
    part_dict = {
        'procedure': 0,
        'facts': 1,
        'law': 2,
        'conclusion': 3,
    }
    if len(doc)-1 < part_dict[part]:
        return None
    doc = doc[part_dict[part]]
    
    def json_to_text_(doc):
        res = []
        if not len(doc['elements']):  # Remove this condition to add subsection titles 
            res.append(doc['content'])
        for e in doc['elements']:
            res.extend(json_to_text_(e))
        return res
    return '\n'.join(json_to_text_(doc))

def create_dataset_echrod(path, article, part):
    '''
    Returns the desired text and labels from a json file.
    :param doc: json file in pandas dataframe format
    :param article: string, the relevant article
    :param part: string, the relevant parts to use
    :return: Tuple(list[str], list[str]) of the relevant text, labels for all relevant cases
    '''
       
    if article == 'All': 
        return return_all_cases(path, part)
    if article == 'multi':
        return create_multilabel_dataset(path, part)

    doc = pd.read_json(path)
    X = []
    y = []
    years = []
    case_ids = []

    # Select only cases with the relevant article
    relevant_cases = doc[pd.Series([article in art for art in doc['article']])]

    # Iterate through all relevant cases
    for idx, case in relevant_cases.iterrows():
        for conclusions in case['conclusion']: # A case has a conclusion for each article
            if 'base_article' in conclusions and conclusions['base_article'] == article: # if the conclusion is related to the relevant article
                label = conclusions['type'] # Violation or non-violation
                if '+' in part:
                    text = ''
                    for p in part.split('+'):
                        t = json_to_text(list(case['content'].values())[0], p)
                        if isinstance(t, str):
                            text += t
                else:
                    text = json_to_text(list(case['content'].values())[0], part) # Extract the relevant parts (text) from the cas
                if text:
                    # y.append(label)
                    case_ids.append(case['itemid'])
                    y.append(1 if label == 'violation' else 0)
                    X.append(rmc(text))
                    years.append(int(case['judgementdate'].split('/')[2].split(' ')[0]))
    # return X, y, years
    return pd.DataFrame({
        'id': case_ids,
        'text': X,
        'year': years,
        'violation': y
    })

def get_article_index():
    return {'2':0, '3':1, '5':2, '6':3, '8':4, '10':5, '11':6, '13':7, '14':8}


def create_multilabel_dataset(path, part):
    doc = pd.read_json(path)
    X = []
    y = []
    years = []

    article_numbers = ['2', '3', '5', '6', '8', '10', '11', '13', '14']
    
    # Iterate through all relevant cases
    for idx, case in doc.iterrows():
        labels = {}
        for conclusion in case['conclusion']: # A case has a conclusion for each article
            if 'base_article' in conclusion.keys() and conclusion['base_article'] in article_numbers:
                article = conclusion['base_article']
                label = conclusion['type']
            labels[article] = label
        if '+' in part:
            text = ''
            for p in part.split('+'):
                t = json_to_text(list(case['content'].values())[0], p)
                if isinstance(t, str):
                    text += t
        else:
            text = json_to_text(list(case['content'].values())[0], part) # Extract the relevant parts (text) from the case
        if text:
            y.append(labels)
            X.append(rmc(text))
            years.append(int(case['judgementdate'].split('/')[2].split(' ')[0]))
    
    def conv_y(lbl):
        y = '000000000'
        for key, val in lbl.items():
            if val == 'violation':
                y = y[:get_article_index()[key]] + '1' + y[get_article_index()[key]+1:]
        return y
    
    y = [conv_y(lbl) for lbl in y]
    return pd.DataFrame({
        'text': X,
        'year': years,
        'violation': y
    })

def return_all_cases(path, part):
    doc = pd.read_json(path)
    X = []
    y = []
    years = []
    case_ids = []

    article_numbers = ['2', '3', '5', '6', '8', '10', '11', '13', '14']
    
    # Iterate through all relevant cases
    for idx, case in doc.iterrows():
        label = 0
        for conclusion in case['conclusion']: # A case has a conclusion for each article
            if 'base_article' in conclusion.keys() and conclusion['base_article'] in article_numbers:
                if conclusion['type'] == 'violation':
                    label = 1
        if '+' in part:
            text = ''
            for p in part.split('+'):
                t = json_to_text(list(case['content'].values())[0], p)
                if isinstance(t, str):
                    text += t
        else:
            text = json_to_text(list(case['content'].values())[0], part) # Extract the relevant parts (text) from the case
        if text:
            y.append(label)
            X.append(rmc(text))
            case_ids.append(case['itemid'])
            years.append(int(case['judgementdate'].split('/')[2].split(' ')[0]))
    
    return pd.DataFrame({
        'id': case_ids,
        'text': X,
        'year': years,
        'violation': y
    })

def rmc(text):
    '''
    Removes unnecessary characters if needed
    '''
    remove_words = ['\n', 'THE FACTS', 'THE CIRCUMSTANCES OF THE CASE', 'I.',  '\xa0', '\t', '•'] # Note that these are applied in order

    text = re.sub('\n\d.', '>', text) # Removes numbering of facts and turns the numbers into into >
    text = text.replace('\n', ' ') # Removes \n marks and replaces them with white spaces
    for word in remove_words:
        text = text.replace(word, '')
    text = " ".join(text.split()) # Removes additional white spaces
    return text

    
def create_dataset_med(path, article, part):
    if article != 'All':
        article = 'Article'+article
    v = extract_parts(path+'train/'+article+'/violation/*.txt', 'violation', part)
    nv = extract_parts(path+'train/'+article+'/non-violation/*.txt', 'non-violation', part)
    
    df = pd.DataFrame([{'text': rmc(c[0]), 'year' : c[2], 'violation': 1} for c in v] + 
                      [{'text': rmc(c[0]), 'year' : c[2], 'violation': 0} for c in nv])
    return df

def balance_dataset(df, label='violation'):
    if df['violation'].mean() < 0.5: # too many non-violation cases
        new_df = df[df['violation']==1]
        nv_df = df[df['violation']==0].sample(n=len(new_df))
        new_df = pd.concat([new_df, nv_df])
    else: # Too may violation cases
        new_df = df[df['violation']==0]
        v_df = df[df['violation']==1].sample(n=len(new_df))
        new_df = pd.concat([new_df, v_df])
    return new_df


# Functions
def extract_text(starts, ends, cases, violation):
    facts = []
    D = []
    years = []
    for case in cases:
        contline = ''
        year = 0
        with open(case, encoding="utf8") as f:
            for line in f:
                dat = re.search('^([0-9]{1,2}\s\w+\s([0-9]{4}))', line)
                if dat != None:
                    year = int(dat.group(2))
                    break
            if year>0:
                years.append(year)
                wr = 0
                for line in f:
                    if wr == 0:
                        if re.search(starts, line) != None:
                            wr = 1
                    if wr == 1 and re.search(ends, line) == None:
                        contline += line
                        contline += '\n'
                    elif re.search(ends, line) != None:
                        break
                facts.append(contline)
    for i in range(len(facts)):
        D.append((facts[i], violation, years[i])) 
    return D

def extract_parts(train_path, violation, part): #extract text from different parts
    cases = glob.glob(train_path)

    facts = []
    D = []
    years = []
    
    if part == 'relevant_law': #seprarte extraction for relevant law
        for case in cases:
            year = 0
            contline = ''
            with open(case, 'r') as f:
                for line in f:
                    dat = re.search('^([0-9]{1,2}\s\w+\s([0-9]{4}))', line)
                    if dat != None:
                        year = int(dat.group(2))
                        break
                if year> 0:
                    years.append(year)
                    wr = 0
                    for line in f:
                        if wr == 0:
                            if re.search('RELEVANT', line) != None:
                                wr = 1
                        if wr == 1 and re.search('THE LAW', line) == None and re.search('PROCEEDINGS', line) == None:
                            contline += line
                            contline += '\n'
                        elif re.search('THE LAW', line) != None or re.search('PROCEEDINGS', line) != None:
                            break
                    facts.append(contline)
        for i in range(len(facts)):
            D.append((facts[i], violation, years[i]))
        
    if part == 'facts':
        starts = 'THE FACTS'
        ends ='THE LAW'
        D = extract_text(starts, ends, cases, violation)
    if part == 'circumstances':
        starts = 'CIRCUMSTANCES'
        ends ='RELEVANT'
        D = extract_text(starts, ends, cases, violation)
    if part == 'procedure':
        starts = 'PROCEDURE'
        ends ='THE FACTS'
        D = extract_text(starts, ends, cases, violation)
    if part == 'procedure+facts':
        starts = 'PROCEDURE'
        ends ='THE LAW'
        D = extract_text(starts, ends, cases, violation)
    return D

### Functions for running individual articles
def train_model_cross_val(Xtrain, Ytrain, vec, clf, debug=False, cv=10, n_jobs=-1): #Linear SVC model cross-validation
    if debug: print('***10-fold cross-validation***')
    pipeline = Pipeline([
        ('features', FeatureUnion(
            [vec],
        )),
        ('classifier', clf)
        ])
    Ypredict = cross_val_predict(pipeline, Xtrain, Ytrain, cv=cv, n_jobs=n_jobs) #10-fold cross-validation
    return return_metrics(Ytrain, Ypredict)

def evaluate(Ytest, Ypredict, debug=False): #evaluate the model (accuracy, precision, recall, f-score, confusion matrix)
        acc = accuracy_score(Ytest, Ypredict)
        if debug:
            print('Accuracy:', acc)
            print('\nClassification report:\n', classification_report(Ytest, Ypredict))
            print('\nCR:', precision_recall_fscore_support(Ytest, Ypredict, average='macro'))
            print('\nConfusion matrix:\n', confusion_matrix(Ytest, Ypredict), '\n\n_______________________\n\n')
        return acc
def run_pipeline(df_complete, vec, clf, debug=False, cv=10, n_jobs=-1): #run tests
   
    df_train, _ = split_db(df_complete, 'after', 2015)
    df_train = balance_dataset(df_train) 
    X_train = df_train['text'].to_numpy()
    y_train = df_train['violation'].to_numpy()
       
    return train_model_cross_val(Xtrain, Ytrain, vec, clf, debug=debug, cv=cv, n_jobs=n_jobs) #use for cross-validation
    
def get_classifier(article_name, classifier_name):
    params = get_best_params(article_name, classifier_name)

    if classifier_name == 'SVM':
        return SGDClassifier(alpha=params['alpha'],
                             penalty=params['penalty'],
                             n_jobs=-1)
    if classifier_name == 'NB':
        return MultinomialNB(alpha=params['alpha'],
                             fit_prior=params['fit_prior'])
    if classifier_name == 'RF':
        if math.isnan(params['max_depth']): params['max_depth'] = None
        return RandomForestClassifier(max_depth=params['max_depth'],
                                      n_estimators=params['n_estimators'],
                                      min_samples_leaf=params['min_samples_leaf'],
                                      min_samples_split=params['min_samples_split'],
                                      max_features=params['max_features'],
                                      bootstrap=params['bootstrap'],
                                      n_jobs=-1)
    if classifier_name == 'SVM_med':
        return LinearSVC(C=params['C'])
    

def get_best_params(article_name, classifier_name):
    if 'All' in article_name:
        article_name = 'Article11'
    return pickle.load(open('results/parameter_optimization/' + classifier_name + '/best_params.pl','rb'))[article_name]



    
