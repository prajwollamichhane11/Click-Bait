import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import spacy
from spacy.tokenizer import Tokenizer

import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

nlp = spacy.load('en_core_web_sm')
tokenizer = Tokenizer(nlp.vocab)


inpDF = pd.DataFrame(np.nan, index=[0], columns=['Clickbaits'])
inp = input("Input the text you want to verify: ")

inpDF["Clickbaits"][0] = inp

def calculator(sentence):
    arr = sentence.split(" ")
    totalChar = len(sentence)-(len(arr)-1)
    avgWordLength = totalChar/len(arr)

    s = sentence
    tokens = tokenizer(s)

    stopWords = 0
    for token in tokens:
        if token.is_stop == True:
            stopWords += 1

    stopWordtoContentWord = stopWords/len(arr)

    return avgWordLength,stopWordtoContentWord,len(arr)

inpDF["Length"] = np.zeros((1,1),dtype=float)
inpDF["AvgWordLength"] = np.zeros((1,1),dtype=float)
inpDF["StoptoContent"] = np.zeros((1,1),dtype=float)
inpDF["Cardinality"] = np.zeros((1,1),dtype=float)
inpDF["WordCount"] = np.zeros((1,1),dtype=float)
inpDF["Verb"] = np.zeros((1,1),dtype=float)
inpDF["Auxiliary"] = np.zeros((1,1),dtype=float)
inpDF["CoorConj"] = np.zeros((1,1),dtype=float)



inpDF["Length"][0] = len(inpDF["Clickbaits"][0])

avgWordLength,stopWordtoContentWord,wordCount = calculator(inpDF["Clickbaits"][0])

inpDF["AvgWordLength"][0] = avgWordLength
inpDF["StoptoContent"][0] = stopWordtoContentWord
inpDF["WordCount"][0]     = wordCount


num = 0
verb = 0
aux = 0
cconj = 0

doc = nlp(inpDF["Clickbaits"][0])
for token in doc:
    if token.pos_ == "NUM":
        num += 1
    if token.pos_ == "VERB":
        verb += 1
    if token.pos_ == "AUX":
        aux += 1
    if token.pos_ == "CCONJ":
        cconj += 1

inpDF["Cardinality"][0] = num
inpDF["Verb"][0] = verb
inpDF["Auxiliary"][0] = aux
inpDF["CoorConj"][0] = cconj

X = inpDF.drop("Clickbaits",axis=1)

filename = 'finalized_model.sav'

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(X)

print("----------------------------")
if result[0] == 1:
    print("CLICKBAIT")
else:
    print("NOT a CLICKBAIT")
print("----------------------------")
