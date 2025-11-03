def remove_stopwords(lemmas):
    from nltk.corpus import stopwords
    english_stop_words = set(stopwords.words('english'))
    return [lemma for lemma in lemmas if lemma not in english_stop_words] 
    
def tokenize_and_lemmatize(text):
    from nltk import word_tokenize
    import string
    tokens = [t for t in word_tokenize(text) if t.isalpha() and t not in string.punctuation]

    from nltk import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]

    return ' '.join(remove_stopwords(lemmas))