import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import spacy
from spacy.tokenizer import Tokenizer

############
nlp = spacy.load('en_core_web_sm')
tokenizer = Tokenizer(nlp.vocab)

df = pd.read_csv("./dataset.csv")
##############

#################################################
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
############################################################

#####################################################################
df["Length"] = np.zeros((len(df["Clickbaits"]),1),dtype=float)
df["AvgWordLength"] = np.zeros((len(df["Clickbaits"]),1),dtype=float)
df["StoptoContent"] = np.zeros((len(df["Clickbaits"]),1),dtype=float)
df["Cardinality"] = np.zeros((len(df["Clickbaits"]),1),dtype=float)
df["WordCount"] = np.zeros((len(df["Clickbaits"]),1),dtype=float)
df["Verb"] = np.zeros((len(df["Clickbaits"]),1),dtype=float)
df["Auxiliary"] = np.zeros((len(df["Clickbaits"]),1),dtype=float)
df["CoorConj"] = np.zeros((len(df["Clickbaits"]),1),dtype=float)

for i in range(len(df["Clickbaits"])):
    df["Length"][i] = len(df["Clickbaits"][i])

    avgWordLength,stopWordtoContentWord,wordCount = calculator(df["Clickbaits"][i])

    df["AvgWordLength"][i] = avgWordLength
    df["StoptoContent"][i] = stopWordtoContentWord
    df["WordCount"][i]     = wordCount
##########################################################################

###########################################################################
lenData = len(df)
num = 0
verb = 0
aux = 0
cconj = 0

for i in range(lenData):
    doc = nlp(df["Clickbaits"][i])
    for token in doc:
        if token.pos_ == "NUM":
            num += 1
        if token.pos_ == "VERB":
            verb += 1
        if token.pos_ == "AUX":
            aux += 1
        if token.pos_ == "CCONJ":
            cconj += 1

    df["Cardinality"][i] = num
    df["Verb"][i] = verb
    df["Auxiliary"][i] = aux
    df["CoorConj"][i] = cconj
#############################################################################

print(df.head())
