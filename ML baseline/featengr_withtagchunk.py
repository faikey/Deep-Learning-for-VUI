
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import cross_val_score
from csv import DictReader, DictWriter
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from nltk import ngrams
from nltk.corpus import stopwords
from gensim import corpora, models
import gensim
get_ipython().run_line_magic('matplotlib', 'inline')


# calculate frequency of some tags in a sentence
class TagTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, examples):
        # return self and nothing else 
        return self
    
    def transform(self, examples):
        
        import numpy as np 
        from scipy.sparse import csr_matrix
        
        # can add more tags, 'SYM' represents operators
        tags = ['SYM']
        
        # Initiaize matrix 
        X = np.zeros((len(examples), 1))
        
        # Loop over examples and count letters 
        for ii, x in enumerate(examples):
            tag = nltk.pos_tag(nltk.word_tokenize(x))            
            X[ii,0] = [t[1] for t in tag].count('SYM') 

        # normalization for a feature
        X = preprocessing.normalize(X, norm='l2')
        return csr_matrix(X) 

# get name_entity
def get_continuous_chunks(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    prev = None
    continuous_chunk = []
    num_name_entity = 0
    current_chunk = []
    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                num_name_entity += 1 
                continuous_chunk.append(named_entity)
                current_chunk = []
            else:
                num_name_entity += 1 
                continue
    return num_name_entity

# get the frequency of name_entity in each sentence
class NameTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, examples):
        # return self and nothing else 
        return self
    
    def transform(self, examples):
        
        import numpy as np 
        from scipy.sparse import csr_matrix
        
        # Initiaize matrix 
        X = np.zeros((len(examples), 1))
        
        # Loop over examples and count letters 
        for ii, x in enumerate(examples):
            X[ii,:] = get_continuous_chunks(x)
            
        return csr_matrix(X) 

#calculate the length of the sentence    
class LengthTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, examples):
        # return self and nothing else 
        return self
    
    def transform(self, examples):
        
        import numpy as np 
        from scipy.sparse import csr_matrix
                 
        # Initiaize matrix 
        X = np.zeros((len(examples), 1))
        
        # Loop over examples and count letters 
        for ii, x in enumerate(examples):
            X[ii, :] = np.array([len(x)])
        return csr_matrix(X)

import math
def entropy(labels):
    freqdist = nltk.FreqDist(labels)
    probs = [freqdist.freq(l) for l in freqdist]
    return -sum(p * math.log(p,2) for p in probs)

#calculate entropy of a sentence    
class EntropyTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, examples):
        # return self and nothing else 
        return self
    
    def transform(self, examples):
        
        import numpy as np 
        from scipy.sparse import csr_matrix
           
        # Initiaize matrix 
        X = np.zeros((len(examples), 1))
        
        # Loop over examples and count letters 
        for ii, x in enumerate(examples):
            X[ii,:] = entropy(ngrams(x,2)) 
        X = preprocessing.normalize(X, norm='l2')
        return csr_matrix(X)

    
    
#LDA model    

class LDATransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, examples):
        # return self and nothing else 
        return self
    
    def transform(self, examples):
        
        import numpy as np 
        from scipy.sparse import csr_matrix
        
        tokenizer = RegexpTokenizer(r'\w+')
        
        # create English stop words list
        en_stop = set(stopwords.words('english'))
        
        # Create p_stemmer of class PorterStemmer
        p_stemmer = PorterStemmer()
        
        # list for tokenized documents in loop
        texts = []
        
        # Initiaize matrix 
        X = np.zeros((len(examples), 1))
        
        # Loop over examples and count letters 
        for ii, x in enumerate(examples):
            # clean and tokenize document string
            raw = x.lower()
            tokens = tokenizer.tokenize(raw)
            
            # remove stop words from tokens
            stopped_tokens = [i for i in tokens if not i in en_stop]
            
            # stem tokens
            stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
            
            # add tokens to list
            texts.append(stemmed_tokens)
            
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word = dictionary)
        
        for ii, x in enumerate(examples):
            # clean and tokenize document string
            raw = x.lower()
            tokens = tokenizer.tokenize(raw)
            
            # remove stop words from tokens
            stopped_tokens = [i for i in tokens if not i in en_stop]
            
            # stem tokens
            stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
            

            bow = ldamodel.id2word.doc2bow(stemmed_tokens)
            doc_topics, word_topics, phi_values = ldamodel.get_document_topics(bow,per_word_topics=True)
            
            #print(doc_topics[1][1])
            if doc_topics[0][1] > doc_topics[1][1]:
                X[ii,:] = doc_topics[0][0]
            else:
                X[ii,:] = doc_topics[1][0]
            #print(X[ii,:])
            
        #X = preprocessing.normalize(X, norm='l2')
        return csr_matrix(X)



class FeatEngr:
    def __init__(self):
        
        
        
        #self.vectorizer = CountVectorizer(stop_words='english')  # "bag-of-words" Accuracy: 0.67 (+/- 0.04)
        self.vectorizer_t = TfidfVectorizer(analyzer='word',ngram_range=(1,2), lowercase=True, norm='l2',stop_words='english')   # "bag-of-words-tfidf" Accuracy: 0.682206 (+/- 0.011113)
        
        #extract features from different columns
        self.vectorizer = FeatureUnion( 
        [       
                ('bag of words', 
                  Pipeline([('extract_field', FunctionTransformer(lambda x: x[0], validate = False)),
                            ('tfid', TfidfVectorizer(analyzer='word',ngram_range=(1,2), lowercase=True, norm='l2',stop_words='english'))])),              
                ('type of page',
                  Pipeline([('extract_field', FunctionTransformer(lambda x:  x[2], validate = False)), 
                            ('page', CountVectorizer())])),
                ('type of trope', 
                  Pipeline([('extract_field', FunctionTransformer(lambda x: x[1], validate = False)),
                            ('trope', CountVectorizer())])),
                ('tag appearance',
                  Pipeline([('extract_field', FunctionTransformer(lambda x:  x[0], validate = False)), 
                            ('tag', TagTransformer())])),
                ('name entities',
                  Pipeline([('extract_field', FunctionTransformer(lambda x:  x[0], validate = False)), 
                            ('chunk', NameTransformer())])),
                ('length of sentence',
                  Pipeline([('extract_field', FunctionTransformer(lambda x:  x[0], validate = False)), 
                            ('length', LengthTransformer())])),
                ('entropy of sentence',
                  Pipeline([('extract_field', FunctionTransformer(lambda x:  x[0], validate = False)), 
                            ('entropy', EntropyTransformer())])),    
                ('LDA model',
                  Pipeline([('extract_field', FunctionTransformer(lambda x:  x[0], validate = False)), 
                            ('lda', LDATransformer())])),  
        ])
    
    def build_train_features(self, examples):
        """
        Method to take in training text features and do further feature engineering 
        Most of the work in this homework will go here, or in similar functions  
        :param examples: currently just a list of forum posts  
        """
        self.vectorizer_t.fit_transform(examples[0])
        self.feature_names = np.asarray(self.vectorizer_t.get_feature_names())
        return self.vectorizer.fit_transform(examples)

    def get_test_features(self, examples):
        """
        Method to take in test text features and transform the same way as train features 
        :param examples: currently just a list of forum posts  
        """
        return self.vectorizer.transform(examples)

    def show_top10(self):
        """
        prints the top 10 features for the positive class and the 
        top 10 features for the negative class. 
        """
        #print(np.asarray(self.vectorizer_t.get_feature_names()))
        top10 = np.argsort(self.logreg.coef_[0])[-10:]
        bottom10 = np.argsort(self.logreg.coef_[0])[:10]
        print(self.feature_names)
        print("Pos: %s" % " ".join(self.feature_names[top10]))
        print("Neg: %s" % " ".join(self.feature_names[bottom10]))
                
    def train_model(self, random_state=1234):
        """
        Method to read in training data from file, and 
        train Logistic Regression classifier. 
        
        :param random_state: seed for random number generator 
        """
        
        from sklearn.linear_model import LogisticRegression 
        
        # load data 
        dfTrain = pd.read_csv("train.csv")
        # get training features and labels 
        self.X_train = self.build_train_features([list(dfTrain["response"])])
        self.y_train = np.array(dfTrain["label"], dtype=int)
        
        # train logistic regression model.  !!You MAY NOT CHANGE THIS!! 
        self.logreg = LogisticRegression(random_state=random_state)
        self.logreg.fit(self.X_train, self.y_train)
        
        scores = cross_val_score(self.logreg, self.X_train, self.y_train, cv =10)
        print("train CV: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))
        
        
        
        
        
        
    def model_predict(self):
        """
        Method to read in test data from file, make predictions
        using trained model, and dump results to file 
        """
        
        # read in test data 
        dfTest  = pd.read_csv("test.csv")
        
        # featurize test data 
        self.X_test = self.get_test_features([list(dfTest["response"])])
        
        # make predictions on test data 
        pred = self.logreg.predict(self.X_test)
        
        # dump predictions to file for submission to Kaggle  
        self.y_test = dfTest['label'].tolist()
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import confusion_matrix
        print(accuracy_score(pred,self.y_test))
        
        print("confusion matrix:")
        print(confusion_matrix(pred,self.y_test))



# Instantiate the FeatEngr clas 
feat = FeatEngr()

# Train your Logistic Regression classifier 
feat.train_model(random_state=1230)

# Shows the top 10 features for each class 
#feat.show_top10()

# Make prediction on test data and produce Kaggle submission file 
feat.model_predict()

