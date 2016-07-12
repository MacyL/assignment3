import os, sys
import glob
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
path = "/home/admin1/Documents/Machine_learning/txt_sentoken/neg/*.txt"
#out_path="/home/admin1/Documents/Machine_learning/txt_sentoken/lemma/pos/"
files = glob.glob(path)   
for name in files:
	f=open(name,'r')
	myfile=f.read()
	myfile_remove_num=re.sub(r'\d+','',myfile)
	tokenizer= RegexpTokenizer(r'\w+')
	st = WordNetLemmatizer()
	tokens = tokenizer.tokenize(myfile_remove_num)
	output = []
	for token in tokens :
		lemma = st.lemmatize(token,'v')
		output += [lemma]		
	out=open(name,'w',encoding='UTF-8')
	out.write(str(output)+'\n')
	out.close()

	
