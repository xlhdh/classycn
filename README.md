classycn: Classical Chinese sentence segmenter. 
========

1. Data - Warning: the data folder is over 1G in size! 

data/sjw - cleaned data from Seungjeongwon Ilgi - memos from ancient Korean Royal Secretariat. Over 200 million characters and 16k+ uniques. 

data/24s - semi - cleaned data from the "Twenty-Four Histories" of China, except Han Shu and San Guo Zhi. Data is from Wikisource, may contain noisy tokens. 20m tokens, 12k uniques.  

data/vectors - word vectors produced using GloVe & Word2Vec. 

2. Scripts
3. 
runhmm - trains and tests an HMM tagger from NLTK

runcrf - trains and tests a CRF tagger from CRF Suite

runlstm - trains and tests a bi-directional LSTM tagger. Implemented with Theano. 

Contact: Yizhou Hu @ huyz725 at gmail.com
