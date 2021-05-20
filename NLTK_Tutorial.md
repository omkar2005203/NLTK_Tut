#  Natural Language Toolkit (NLTK)


## 01 Reading Files


```python
#Read file using standard python library

import os
#getcwd() returns current working directory of a process
with open(os.getcwd()+ "/testdata.txt","r",encoding="utf8") as fr:
    filedata=fr.read()
    
# display first few characters
print("Data in file: ",filedata[0:200])
print("\n")
# display entire file data
print("Data in file: ",filedata)
```

    Data in file:  Natural language programming (NLP) is a way of programming using natural human languages, for example, English. Actually, a complete, structured, and unambiguous sentence of human language can be seen
    
    
    Data in file:  Natural language programming (NLP) is a way of programming using natural human languages, for example, English. Actually, a complete, structured, and unambiguous sentence of human language can be seen as a piece of computer programming code. For example, this imperative sentence “Buy some milk from the Whole Food on your way home.” It’s quiet straightforward for human to comprehend that during the process of going “home”, a function “buy” is called at a place named “the Whole Food”, and an object “milk” is the target. However, it’s relatively difficult for a computer to understand the above sentence. What’s “home”? What’s “milk”? What’s “buy”? The computers have no idea about those nouns and verbs. In the field of the Artificial Intelligent(AI), in order to teach the computer to read and understand the human language, researchers have been studying on the natural language processing(communication) for natural language programming for years. 
    In this project, I bring the natural language programming into the physical world by applying it on the Arduino broads (Arduino UNO is focused in this report). This gives us another perspective to look at the natural language programming, and how it is operated in the real world.
    

### What is NLTK corpus and how to read files with it?
The NLTK corpus is a massive dump of all kinds of natural language data sets.NLTK corpus readers. The modules in this package provide functions that can be used to read corpus files in variety of formats. these functions can be used to read both the corpus files that are distributed in the NLTK corpus package,and corpus files that are part of external corpora.

References
1.  https://www.nltk.org/data.html
2.  http://www.nltk.org/nltk_data/


```python
# Before that install nltk from anaconda prompt using 'pip install nltk'

import nltk
# download punkt
nltk.download('punkt')

from nltk.corpus.reader.plaintext import PlaintextCorpusReader

corpus=PlaintextCorpusReader(os.getcwd(),"testdata.txt")


print(corpus.raw())

```

    Natural language programming (NLP) is a way of programming using natural human languages, for example, English. Actually, a complete, structured, and unambiguous sentence of human language can be seen as a piece of computer programming code. For example, this imperative sentence “Buy some milk from the Whole Food on your way home.” It’s quiet straightforward for human to comprehend that during the process of going “home”, a function “buy” is called at a place named “the Whole Food”, and an object “milk” is the target. However, it’s relatively difficult for a computer to understand the above sentence. What’s “home”? What’s “milk”? What’s “buy”? The computers have no idea about those nouns and verbs. In the field of the Artificial Intelligent(AI), in order to teach the computer to read and understand the human language, researchers have been studying on the natural language processing(communication) for natural language programming for years. 
    In this project, I bring the natural language programming into the physical world by applying it on the Arduino broads (Arduino UNO is focused in this report). This gives us another perspective to look at the natural language programming, and how it is operated in the real world.
    

    [nltk_data] Downloading package punkt to
    [nltk_data]     C:\Users\omkar\AppData\Roaming\nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    

## 02 Corpus Operations
The corpus library supports a number of functions to extract words,paragraph and sentences from the corpus



```python
# Extract file ID's from corpus

print(corpus.fileids())  # here we are using single file 

# Extract Paragraph from corpus
paragraph = corpus.paras()
print("Total number of paragraphs: ",len(paragraph))

print("\n")

# Extract Sentences from corpus
sentences=corpus.sents()
print("Total sentences in this paragraph :",len(sentences))

print("\n")

print("First sentence is :",sentences[0])

print("\n")

for i in range(len(sentences)):
    print("Sentence ",i+1,"is :",sentences[i])
    

print("\n")

#Extract words from the corpus
print("Words in corpus :",corpus.words())
print("Total Words in corpus :",len(corpus.words()))

print("\n")
word=corpus.words()

# for i in range(len(word)):
#     print("word ",i+1," is :",word[i])



```

    ['testdata.txt']
    Total number of paragraphs:  1
    
    
    Total sentences in this paragraph : 11
    
    
    First sentence is : ['Natural', 'language', 'programming', '(', 'NLP', ')', 'is', 'a', 'way', 'of', 'programming', 'using', 'natural', 'human', 'languages', ',', 'for', 'example', ',', 'English', '.']
    
    
    Sentence  1 is : ['Natural', 'language', 'programming', '(', 'NLP', ')', 'is', 'a', 'way', 'of', 'programming', 'using', 'natural', 'human', 'languages', ',', 'for', 'example', ',', 'English', '.']
    Sentence  2 is : ['Actually', ',', 'a', 'complete', ',', 'structured', ',', 'and', 'unambiguous', 'sentence', 'of', 'human', 'language', 'can', 'be', 'seen', 'as', 'a', 'piece', 'of', 'computer', 'programming', 'code', '.']
    Sentence  3 is : ['For', 'example', ',', 'this', 'imperative', 'sentence', '“', 'Buy', 'some', 'milk', 'from', 'the', 'Whole', 'Food', 'on', 'your', 'way', 'home', '.”', 'It', '’', 's', 'quiet', 'straightforward', 'for', 'human', 'to', 'comprehend', 'that', 'during', 'the', 'process', 'of', 'going', '“', 'home', '”,', 'a', 'function', '“', 'buy', '”', 'is', 'called', 'at', 'a', 'place', 'named', '“', 'the', 'Whole', 'Food', '”,', 'and', 'an', 'object', '“', 'milk', '”', 'is', 'the', 'target', '.']
    Sentence  4 is : ['However', ',', 'it', '’', 's', 'relatively', 'difficult', 'for', 'a', 'computer', 'to', 'understand', 'the', 'above', 'sentence', '.']
    Sentence  5 is : ['What', '’', 's', '“', 'home', '”?']
    Sentence  6 is : ['What', '’', 's', '“', 'milk', '”?']
    Sentence  7 is : ['What', '’', 's', '“', 'buy', '”?']
    Sentence  8 is : ['The', 'computers', 'have', 'no', 'idea', 'about', 'those', 'nouns', 'and', 'verbs', '.']
    Sentence  9 is : ['In', 'the', 'field', 'of', 'the', 'Artificial', 'Intelligent', '(', 'AI', '),', 'in', 'order', 'to', 'teach', 'the', 'computer', 'to', 'read', 'and', 'understand', 'the', 'human', 'language', ',', 'researchers', 'have', 'been', 'studying', 'on', 'the', 'natural', 'language', 'processing', '(', 'communication', ')', 'for', 'natural', 'language', 'programming', 'for', 'years', '.']
    Sentence  10 is : ['In', 'this', 'project', ',', 'I', 'bring', 'the', 'natural', 'language', 'programming', 'into', 'the', 'physical', 'world', 'by', 'applying', 'it', 'on', 'the', 'Arduino', 'broads', '(', 'Arduino', 'UNO', 'is', 'focused', 'in', 'this', 'report', ').']
    Sentence  11 is : ['This', 'gives', 'us', 'another', 'perspective', 'to', 'look', 'at', 'the', 'natural', 'language', 'programming', ',', 'and', 'how', 'it', 'is', 'operated', 'in', 'the', 'real', 'world', '.']
    
    
    Words in corpus : ['Natural', 'language', 'programming', '(', 'NLP', ')', ...]
    Total Words in corpus : 249
    
    
    

## 03 Analysis of Corpus 
The NLTK library provides a number of functions to analyze the distribution and aggregates for data in the corpus.


```python
# we use frequency distributuion method to analyze distribution of words in the corpus
freq_dist = nltk.FreqDist(corpus.words())

# top 10 popular or common words in corpus
print("Top 10 popular words in corpus :",freq_dist.most_common(10))
print("\n")

# find the specific word distribution in corpus
print("Distribution of word 'natural' in corpus :",freq_dist.get("natural"))


```

    Top 10 popular words in corpus : [('the', 15), (',', 10), ('“', 8), ('language', 7), ('.', 7), ('programming', 6), ('a', 6), ('is', 5), ('of', 5), ('natural', 5)]
    
    
    Distribution of word 'natural' in corpus : 5
    

## 04 Pre-Processing of Data (Text Cleansing and Extraction )

### Text Tokenization
The process of breaking a stream of textual content: words,terms,symbols,or other meaningful elements. Tokenization refers to converting a text string into individual tokens Tokens may be words or punctations.


```python
import nltk
import os

# reading file 
file = open(os.getcwd()+"/testdata.txt",'rt',encoding="utf8")
text=file.read()
file.close()

# extract tokens
token_list=nltk.word_tokenize(text)
print("Number of Token  : ",len(token_list))
print("\n")
print("Token list : ",token_list)
print("\n")
print("20 Token list : ",token_list[:20])
```

    Number of Token  :  256
    
    
    Token list :  ['Natural', 'language', 'programming', '(', 'NLP', ')', 'is', 'a', 'way', 'of', 'programming', 'using', 'natural', 'human', 'languages', ',', 'for', 'example', ',', 'English', '.', 'Actually', ',', 'a', 'complete', ',', 'structured', ',', 'and', 'unambiguous', 'sentence', 'of', 'human', 'language', 'can', 'be', 'seen', 'as', 'a', 'piece', 'of', 'computer', 'programming', 'code', '.', 'For', 'example', ',', 'this', 'imperative', 'sentence', '“', 'Buy', 'some', 'milk', 'from', 'the', 'Whole', 'Food', 'on', 'your', 'way', 'home.', '”', 'It', '’', 's', 'quiet', 'straightforward', 'for', 'human', 'to', 'comprehend', 'that', 'during', 'the', 'process', 'of', 'going', '“', 'home', '”', ',', 'a', 'function', '“', 'buy', '”', 'is', 'called', 'at', 'a', 'place', 'named', '“', 'the', 'Whole', 'Food', '”', ',', 'and', 'an', 'object', '“', 'milk', '”', 'is', 'the', 'target', '.', 'However', ',', 'it', '’', 's', 'relatively', 'difficult', 'for', 'a', 'computer', 'to', 'understand', 'the', 'above', 'sentence', '.', 'What', '’', 's', '“', 'home', '”', '?', 'What', '’', 's', '“', 'milk', '”', '?', 'What', '’', 's', '“', 'buy', '”', '?', 'The', 'computers', 'have', 'no', 'idea', 'about', 'those', 'nouns', 'and', 'verbs', '.', 'In', 'the', 'field', 'of', 'the', 'Artificial', 'Intelligent', '(', 'AI', ')', ',', 'in', 'order', 'to', 'teach', 'the', 'computer', 'to', 'read', 'and', 'understand', 'the', 'human', 'language', ',', 'researchers', 'have', 'been', 'studying', 'on', 'the', 'natural', 'language', 'processing', '(', 'communication', ')', 'for', 'natural', 'language', 'programming', 'for', 'years', '.', 'In', 'this', 'project', ',', 'I', 'bring', 'the', 'natural', 'language', 'programming', 'into', 'the', 'physical', 'world', 'by', 'applying', 'it', 'on', 'the', 'Arduino', 'broads', '(', 'Arduino', 'UNO', 'is', 'focused', 'in', 'this', 'report', ')', '.', 'This', 'gives', 'us', 'another', 'perspective', 'to', 'look', 'at', 'the', 'natural', 'language', 'programming', ',', 'and', 'how', 'it', 'is', 'operated', 'in', 'the', 'real', 'world', '.']
    
    
    20 Token list :  ['Natural', 'language', 'programming', '(', 'NLP', ')', 'is', 'a', 'way', 'of', 'programming', 'using', 'natural', 'human', 'languages', ',', 'for', 'example', ',', 'English']
    

### Text Cleansing
We have to remove following things while cleansing text.

1. Formatting and standardization (for eg: dates)
2. Remove punctuation
3. Remove abbrevations
4. Case conversion
5. Remove elements like hastags.
6. Remove URL
7. Convert to lower case





```python
# Removing punctuation
token_list1=list(filter(lambda token:nltk.tokenize.punkt.PunktToken(token).is_non_punct,token_list))
print("first 20 Token list after removing punctuation : ",token_list1[:20]) 
print("\n")
print("Token list after removing punctuation : ",token_list1)
print("\n")
print("Total tokens after removing punctuation : ",len(token_list1))
# we will see reduction in token size after removal of punctuation
```

    first 20 Token list after removing punctuation :  ['Natural', 'language', 'programming', 'NLP', 'is', 'a', 'way', 'of', 'programming', 'using', 'natural', 'human', 'languages', 'for', 'example', 'English', 'Actually', 'a', 'complete', 'structured']
    
    
    Token list after removing punctuation :  ['Natural', 'language', 'programming', 'NLP', 'is', 'a', 'way', 'of', 'programming', 'using', 'natural', 'human', 'languages', 'for', 'example', 'English', 'Actually', 'a', 'complete', 'structured', 'and', 'unambiguous', 'sentence', 'of', 'human', 'language', 'can', 'be', 'seen', 'as', 'a', 'piece', 'of', 'computer', 'programming', 'code', 'For', 'example', 'this', 'imperative', 'sentence', 'Buy', 'some', 'milk', 'from', 'the', 'Whole', 'Food', 'on', 'your', 'way', 'home.', 'It', 's', 'quiet', 'straightforward', 'for', 'human', 'to', 'comprehend', 'that', 'during', 'the', 'process', 'of', 'going', 'home', 'a', 'function', 'buy', 'is', 'called', 'at', 'a', 'place', 'named', 'the', 'Whole', 'Food', 'and', 'an', 'object', 'milk', 'is', 'the', 'target', 'However', 'it', 's', 'relatively', 'difficult', 'for', 'a', 'computer', 'to', 'understand', 'the', 'above', 'sentence', 'What', 's', 'home', 'What', 's', 'milk', 'What', 's', 'buy', 'The', 'computers', 'have', 'no', 'idea', 'about', 'those', 'nouns', 'and', 'verbs', 'In', 'the', 'field', 'of', 'the', 'Artificial', 'Intelligent', 'AI', 'in', 'order', 'to', 'teach', 'the', 'computer', 'to', 'read', 'and', 'understand', 'the', 'human', 'language', 'researchers', 'have', 'been', 'studying', 'on', 'the', 'natural', 'language', 'processing', 'communication', 'for', 'natural', 'language', 'programming', 'for', 'years', 'In', 'this', 'project', 'I', 'bring', 'the', 'natural', 'language', 'programming', 'into', 'the', 'physical', 'world', 'by', 'applying', 'it', 'on', 'the', 'Arduino', 'broads', 'Arduino', 'UNO', 'is', 'focused', 'in', 'this', 'report', 'This', 'gives', 'us', 'another', 'perspective', 'to', 'look', 'at', 'the', 'natural', 'language', 'programming', 'and', 'how', 'it', 'is', 'operated', 'in', 'the', 'real', 'world']
    
    
    Total tokens after removing punctuation :  203
    


```python
token_list2 = [word.lower() for word in token_list1]
print("first 20 Token list after removing punctuation : ",token_list2[:20]) 
print("\n")
print("Token list after applying lower case : ",token_list2)
print("\n")
print("Total tokens after applying lower case : ",len(token_list2))
```

    first 20 Token list after removing punctuation :  ['natural', 'language', 'programming', 'nlp', 'is', 'a', 'way', 'of', 'programming', 'using', 'natural', 'human', 'languages', 'for', 'example', 'english', 'actually', 'a', 'complete', 'structured']
    
    
    Token list after applying lower case :  ['natural', 'language', 'programming', 'nlp', 'is', 'a', 'way', 'of', 'programming', 'using', 'natural', 'human', 'languages', 'for', 'example', 'english', 'actually', 'a', 'complete', 'structured', 'and', 'unambiguous', 'sentence', 'of', 'human', 'language', 'can', 'be', 'seen', 'as', 'a', 'piece', 'of', 'computer', 'programming', 'code', 'for', 'example', 'this', 'imperative', 'sentence', 'buy', 'some', 'milk', 'from', 'the', 'whole', 'food', 'on', 'your', 'way', 'home.', 'it', 's', 'quiet', 'straightforward', 'for', 'human', 'to', 'comprehend', 'that', 'during', 'the', 'process', 'of', 'going', 'home', 'a', 'function', 'buy', 'is', 'called', 'at', 'a', 'place', 'named', 'the', 'whole', 'food', 'and', 'an', 'object', 'milk', 'is', 'the', 'target', 'however', 'it', 's', 'relatively', 'difficult', 'for', 'a', 'computer', 'to', 'understand', 'the', 'above', 'sentence', 'what', 's', 'home', 'what', 's', 'milk', 'what', 's', 'buy', 'the', 'computers', 'have', 'no', 'idea', 'about', 'those', 'nouns', 'and', 'verbs', 'in', 'the', 'field', 'of', 'the', 'artificial', 'intelligent', 'ai', 'in', 'order', 'to', 'teach', 'the', 'computer', 'to', 'read', 'and', 'understand', 'the', 'human', 'language', 'researchers', 'have', 'been', 'studying', 'on', 'the', 'natural', 'language', 'processing', 'communication', 'for', 'natural', 'language', 'programming', 'for', 'years', 'in', 'this', 'project', 'i', 'bring', 'the', 'natural', 'language', 'programming', 'into', 'the', 'physical', 'world', 'by', 'applying', 'it', 'on', 'the', 'arduino', 'broads', 'arduino', 'uno', 'is', 'focused', 'in', 'this', 'report', 'this', 'gives', 'us', 'another', 'perspective', 'to', 'look', 'at', 'the', 'natural', 'language', 'programming', 'and', 'how', 'it', 'is', 'operated', 'in', 'the', 'real', 'world']
    
    
    Total tokens after applying lower case :  203
    

### Stop Word Removal
These words do not carry any insight hence they are removed. They are group of words that carry no meaning by themselves.
* for eg: in,a,and,the,which
* they are not required for analytics so these words are removed.
* A standard or custom stop-word dictionary can be used to identify and remove stop words.


```python
#download the standard stop words list
nltk.download('stopwords')
from nltk.corpus import stopwords

# Remove stop words
token_list3 = list(filter(lambda token: token not in stopwords.words('english'),token_list2))

print("first 20 Token list after removing stop words : ",token_list3[:20]) 
print("\n")
print("Token list after after removing stop words  : ",token_list3)
print("\n")
print("Total tokens after removing stop words  : ",len(token_list3))
# we will see reduction in token size after removal of stop words
```

    first 20 Token list after removing stop words :  ['natural', 'language', 'programming', 'nlp', 'way', 'programming', 'using', 'natural', 'human', 'languages', 'example', 'english', 'actually', 'complete', 'structured', 'unambiguous', 'sentence', 'human', 'language', 'seen']
    
    
    Token list after after removing stop words  :  ['natural', 'language', 'programming', 'nlp', 'way', 'programming', 'using', 'natural', 'human', 'languages', 'example', 'english', 'actually', 'complete', 'structured', 'unambiguous', 'sentence', 'human', 'language', 'seen', 'piece', 'computer', 'programming', 'code', 'example', 'imperative', 'sentence', 'buy', 'milk', 'whole', 'food', 'way', 'home.', 'quiet', 'straightforward', 'human', 'comprehend', 'process', 'going', 'home', 'function', 'buy', 'called', 'place', 'named', 'whole', 'food', 'object', 'milk', 'target', 'however', 'relatively', 'difficult', 'computer', 'understand', 'sentence', 'home', 'milk', 'buy', 'computers', 'idea', 'nouns', 'verbs', 'field', 'artificial', 'intelligent', 'ai', 'order', 'teach', 'computer', 'read', 'understand', 'human', 'language', 'researchers', 'studying', 'natural', 'language', 'processing', 'communication', 'natural', 'language', 'programming', 'years', 'project', 'bring', 'natural', 'language', 'programming', 'physical', 'world', 'applying', 'arduino', 'broads', 'arduino', 'uno', 'focused', 'report', 'gives', 'us', 'another', 'perspective', 'look', 'natural', 'language', 'programming', 'operated', 'real', 'world']
    
    
    Total tokens after removing stop words  :  109
    

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\omkar\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    

## 05 Text Mining

Text mining, also referred to as text data mining, similar to text analytics, is the process of deriving high-quality information from text. It involves "the discovery by computer of new, previously unknown information, by automatically extracting information from different written resources

### Stemming

A stem is the base part of the word, to which affixes can be attached for derivatives.This process converts word into stem.
So stemming keeps only base word,thus reducing the total words in the corpus.
* for eg: "combin" is stem word for combine,combining,combined.
* stemming simply cuts off the affix and may not be a complete word.



```python
# i will use PorterStemmer libarary for stemming.
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
token_list4 = [stemmer.stem(word) for word in token_list3]

print("first 20 Token list after stemming : ",token_list4[:20]) 
print("\n")
print("Token list after stemming  : ",token_list4)
print("\n")
print("Total tokens after stemming  : ",len(token_list4))


```

    first 20 Token list after stemming :  ['natur', 'languag', 'program', 'nlp', 'way', 'program', 'use', 'natur', 'human', 'languag', 'exampl', 'english', 'actual', 'complet', 'structur', 'unambigu', 'sentenc', 'human', 'languag', 'seen']
    
    
    Token list after stemming  :  ['natur', 'languag', 'program', 'nlp', 'way', 'program', 'use', 'natur', 'human', 'languag', 'exampl', 'english', 'actual', 'complet', 'structur', 'unambigu', 'sentenc', 'human', 'languag', 'seen', 'piec', 'comput', 'program', 'code', 'exampl', 'imper', 'sentenc', 'buy', 'milk', 'whole', 'food', 'way', 'home.', 'quiet', 'straightforward', 'human', 'comprehend', 'process', 'go', 'home', 'function', 'buy', 'call', 'place', 'name', 'whole', 'food', 'object', 'milk', 'target', 'howev', 'rel', 'difficult', 'comput', 'understand', 'sentenc', 'home', 'milk', 'buy', 'comput', 'idea', 'noun', 'verb', 'field', 'artifici', 'intellig', 'ai', 'order', 'teach', 'comput', 'read', 'understand', 'human', 'languag', 'research', 'studi', 'natur', 'languag', 'process', 'commun', 'natur', 'languag', 'program', 'year', 'project', 'bring', 'natur', 'languag', 'program', 'physic', 'world', 'appli', 'arduino', 'broad', 'arduino', 'uno', 'focus', 'report', 'give', 'us', 'anoth', 'perspect', 'look', 'natur', 'languag', 'program', 'oper', 'real', 'world']
    
    
    Total tokens after stemming  :  109
    

### Lemmatization

It is similar to stemming,but produces a proper root word that belongs to the language.

* for eg: "combine" is the lemmetized version of combine,combined and combining
* lemmatization uses dictionary to match words to their root word.



```python
# we can use wordnet library to map words to their lemmatized form

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer() 
#note that we are using token_list3 (output list from removal of stopwords)
token_list5=[lemmatizer.lemmatize(word) for word in token_list3] 

print("first 20 Token list after lemmatization : ",token_list5[:20]) 
print("\n")
print("Token list after lemmatization  : ",token_list5)
print("\n")
print("Total tokens after lemmatization  : ",len(token_list5))
```

    first 20 Token list after lemmatization :  ['natural', 'language', 'programming', 'nlp', 'way', 'programming', 'using', 'natural', 'human', 'language', 'example', 'english', 'actually', 'complete', 'structured', 'unambiguous', 'sentence', 'human', 'language', 'seen']
    
    
    Token list after lemmatization  :  ['natural', 'language', 'programming', 'nlp', 'way', 'programming', 'using', 'natural', 'human', 'language', 'example', 'english', 'actually', 'complete', 'structured', 'unambiguous', 'sentence', 'human', 'language', 'seen', 'piece', 'computer', 'programming', 'code', 'example', 'imperative', 'sentence', 'buy', 'milk', 'whole', 'food', 'way', 'home.', 'quiet', 'straightforward', 'human', 'comprehend', 'process', 'going', 'home', 'function', 'buy', 'called', 'place', 'named', 'whole', 'food', 'object', 'milk', 'target', 'however', 'relatively', 'difficult', 'computer', 'understand', 'sentence', 'home', 'milk', 'buy', 'computer', 'idea', 'noun', 'verb', 'field', 'artificial', 'intelligent', 'ai', 'order', 'teach', 'computer', 'read', 'understand', 'human', 'language', 'researcher', 'studying', 'natural', 'language', 'processing', 'communication', 'natural', 'language', 'programming', 'year', 'project', 'bring', 'natural', 'language', 'programming', 'physical', 'world', 'applying', 'arduino', 'broad', 'arduino', 'uno', 'focused', 'report', 'give', 'u', 'another', 'perspective', 'look', 'natural', 'language', 'programming', 'operated', 'real', 'world']
    
    
    Total tokens after lemmatization  :  109
    

    [nltk_data] Downloading package wordnet to
    [nltk_data]     C:\Users\omkar\AppData\Roaming\nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    

## Comparsion between original word,stemmed and lemmatized word


```python
print("Original word : ",token_list3[0]," ,Stemmed Word : ",token_list4[0]," ,Lemmatized Word : ",token_list5[0])
```

    Original word :  natural  ,Stemmed Word :  natur  ,Lemmatized Word :  natural
    

# 06 Text Processing Technique's (Text Mining)

## N-Grams

N-grams is a sequence of n items in a sample of text . it can called as Bigrams,Trigrams,Four Grams and so on.
n grams are used for buidling predective text systems that predict next squence of words.

* eg : Cats are favorite pets
* Bigrams: (Cats , are),(are,favorite),(favorite,pets)
* Trigrams : (Cats,are,favorite),(are,favorite,pets)


```python
from nltk.util import ngrams
from collections import Counter


# find bigrams and first 5 most common 

bigrams=ngrams(token_list5,2)
print("Most 5 common bigrams :")
print(Counter(bigrams).most_common(5))

# find trigrams and first 5 most common 

bigrams=ngrams(token_list5,3)
print("Most 5 common trigrams :")
print(Counter(bigrams).most_common(5))

```

    Most 5 common bigrams :
    [(('natural', 'language'), 5), (('language', 'programming'), 4), (('human', 'language'), 3), (('whole', 'food'), 2), (('programming', 'nlp'), 1)]
    Most 5 common trigrams :
    [(('natural', 'language', 'programming'), 4), (('language', 'programming', 'nlp'), 1), (('programming', 'nlp', 'way'), 1), (('nlp', 'way', 'programming'), 1), (('way', 'programming', 'using'), 1)]
    

## Parts Of Speech Tagging (POS)

* POS tagging involves identifying the part of speech for each word in a corpus.
* It used for entity recognition,filtering and sentiment analysis.
* eg : Word - Man , POS- NN , for Noun
* eg : Word - Engage , POS - VBP,for Verb
* eg : Word - Top,POS-JJ , Adjective

References:
* Tagging List https://www.guru99.com/pos-tagging-chunking-nltk.html


```python
#download the tagger package
import nltk
nltk.download('averaged_perceptron_tagger')
#first 10 
nltk.pos_tag(token_list5)[:10]
# nltk.pos_tag(token_list5)
```

    [nltk_data] Downloading package averaged_perceptron_tagger to
    [nltk_data]     C:\Users\omkar\AppData\Roaming\nltk_data...
    [nltk_data]   Package averaged_perceptron_tagger is already up-to-
    [nltk_data]       date!
    




    [('natural', 'JJ'),
     ('language', 'NN'),
     ('programming', 'VBG'),
     ('nlp', 'JJ'),
     ('way', 'NN'),
     ('programming', 'VBG'),
     ('using', 'VBG'),
     ('natural', 'JJ'),
     ('human', 'JJ'),
     ('language', 'NN')]



## Term frequency - Inverse document frequency (TF-IDF)


* most of ML models need a numeric representation of text.
* TF-IDF is used to convert text into a numeric table representation.
* TF-IDF output is a table where rows represent documents and columns represent words.
* Each cell provides a count/value that indicates the strength of word with respect to the document.



## Algorithm:
1. Orginal documents
* Doc 1 = "This is a sampling of good words"
* Doc 2 = "He said again and again the same word after word"
* Doc 3 = "Words can really hurt"
2. After Text cleansing
* Doc 1 = "sample good words"
* Doc 2 = "again again same word word"
* Doc 3 = "word really hurt"
3. create count table (Number times word appeared). 
![image-2.png](attachment:image-2.png)
4. Create Text frequency table (TF). divide each count in table by total number of words eg: 1/3=0.33
![image-3.png](attachment:image-3.png)
5. We find inverse document frequency table. (IDF)
log e (Total docs/docs with words). This find unique and prevalent word in document.
![image-5.png](attachment:image-5.png)
6. TF-IDF = TF * IDF (multiply step 4 and step 5)
![image-6.png](attachment:image-6.png)


References:
1. https://www.geeksforgeeks.org/understanding-tf-idf-term-frequency-inverse-document-frequency/
2. https://www.onely.com/blog/what-is-tf-idf/
3. https://towardsdatascience.com/text-summarization-using-tf-idf-e64a0644ace3


```python
# TF-IDF fuctionality is not supported by NLTK library so we have to use scikit learn libarry
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


#corpus declaration
vector_corpus = [
    'ICC is a Cricket Board.',
    'Cricket is popular in India.',
    'TV in India telecast Cricket.'
]

# vector_corpus = [
#     'This is a sampling of good words.',
#     'He said again and again the same word after word.',
#     'Words can really hurt.'
# ]


# creating vectorizer for english language
vectorizer = TfidfVectorizer(stop_words='english')

# create the vector 
tfidf=vectorizer.fit_transform(vector_corpus)

#tokens
print("Tokens used as features are :")
print(vectorizer.get_feature_names())

print("\n size of array. Each row represents a document. Each column represents a feature/token")
print(tfidf.shape)

print("\n TF-IDF array")
tfidf.toarray()
```

    Tokens used as features are :
    ['board', 'cricket', 'icc', 'india', 'popular', 'telecast', 'tv']
    
     size of array. Each row represents a document. Each column represents a feature/token
    (3, 7)
    
     TF-IDF array
    




    array([[0.65249088, 0.38537163, 0.65249088, 0.        , 0.        ,
            0.        , 0.        ],
           [0.        , 0.42544054, 0.        , 0.54783215, 0.72033345,
            0.        , 0.        ],
           [0.        , 0.34520502, 0.        , 0.44451431, 0.        ,
            0.5844829 , 0.5844829 ]])




 ## 07 Storing Text Data
* We must use suitable free-format big data storage for text.
* eg : HDFS,S3 or Google Cloud
* Create indexes on key data elements for easy access.
* MongoDB
* Elasticsearch
* store processed text like tokens and TF-IDF
 ## Processing Text Data
* Filter text as early as possible in the processing cycle.
* use an exhaustive and context-specific stop word list
* identify domain specific data for special use.
* Eliminate data with low frequency.
* Build a clean and indexed corpus.
 ## Scalable Processing
* Use technologies that allow paralle access and storage
* Kafka,HDFS,MongoDB and so on.
* process each document independtly with map() functions(in hadoop or apache spark)
* use reduce() functions later in the processing cycle.

### - Made By Omkar Rane
