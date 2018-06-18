# coding:utf-8
import re
import csv
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import jieba

#-----清洗twitter文本代码-----
emoticon_string = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth      
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
    )"""

# The components of the tokenizer:
regex_strings = (
    # Phone numbers:
    r"""
    (?:
      (?:            # (international)
        \+?[01]
        [\-\s.]*
      )?            
      (?:            # (area code)
        [\(]?
        \d{3}
        [\-\s.\)]*
      )?    
      \d{3}          # exchange
      [\-\s.]*   
      \d{4}          # base
    )"""
    ,
    # Emoticons:
    emoticon_string
    ,
    # HTML tags:
    r"""<[^>]+>"""
    ,
    # Twitter username:
    r"""(?:@[\w_]+)"""
    ,
    # Twitter hashtags:
    r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)"""
    ,
    # Remaining word types:
    r"""
    (?:[a-z][a-z'\-_]+[a-z])       # Words with apostrophes or dashes.
    |
    (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
    |
    (?:[\w_]+)                     # Words without apostrophes or dashes.
    |
    (?:\.(?:\s*\.){1,})            # Ellipsis dots. 
    |
    (?:\S)                         # Everything else that isn't whitespace.
    """
)

######################################################################
# This is the core tokenizing regex:

word_re = re.compile(r"""(%s)""" % "|".join(regex_strings), re.VERBOSE | re.I | re.UNICODE)

# The emoticon string gets its own regex so that we can preserve case for them as needed:
emoticon_re = re.compile(regex_strings[1], re.VERBOSE | re.I | re.UNICODE)

# These are for regularizing HTML entities to Unicode:
html_entity_digit_re = re.compile(r"&#\d+;")
html_entity_alpha_re = re.compile(r"&\w+;")
amp = "&amp;"


######################################################################

class Tokenizer:
    def __init__(self, preserve_case=False):
        self.preserve_case = preserve_case

    def tokenize(self, s):
        """
        Argument: s -- any string or unicode object
        Value: a tokenize list of strings; conatenating this list returns the original string if preserve_case=False
        """
        # Try to ensure unicode:
        try:
            s = unicode(s)
        except UnicodeDecodeError:
            s = str(s).encode('string_escape')
            s = unicode(s)
        # Fix HTML character entitites:
        s = self.__html2unicode(s)
        # Tokenize:
        words = word_re.findall(s)
        # Possible alter the case, but avoid changing emoticons like :D into :d:
        if not self.preserve_case:
            words = map((lambda x: x if emoticon_re.search(x) else x.lower()), words)
        return words

    def tokenize_random_tweet(self):
        """
        If the twitter library is installed and a twitter connection
        can be established, then tokenize a random tweet.
        """
        try:
            import twitter
        except ImportError:
            print
            "Apologies. The random tweet functionality requires the Python twitter library: http://code.google.com/p/python-twitter/"
        from random import shuffle
        api = twitter.Api()
        tweets = api.GetPublicTimeline()
        if tweets:
            for tweet in tweets:
                if tweet.user.lang == 'en':
                    return self.tokenize(tweet.text)
        else:
            raise Exception(
                "Apologies. I couldn't get Twitter to give me a public English-language tweet. Perhaps try again")

    def __html2unicode(self, s):
        """
        Internal metod that seeks to replace all the HTML entities in
        s with their corresponding unicode characters.
        """
        # First the digits:
        ents = set(html_entity_digit_re.findall(s))
        if len(ents) > 0:
            for ent in ents:
                entnum = ent[2:-1]
                try:
                    entnum = int(entnum)
                    s = s.replace(ent, unichr(entnum))
                except:
                    pass
        # Now the alpha versions:
        ents = set(html_entity_alpha_re.findall(s))
        ents = filter((lambda x: x != amp), ents)
        for ent in ents:
            entname = ent[1:-1]
            try:
                s = s.replace(ent, unichr(htmlentitydefs.name2codepoint[entname]))
            except:
                pass
            s = s.replace(amp, " and ")
        return s
#------------------------------------


l = open("/home/info/Info/personality-detection-master/newdata/200g.csv","r")


label_name = ["C","D","E","F","G"]
result = {}
for name in label_name:
    result[name] = 0
label = []
corpus = []
tok = Tokenizer(preserve_case=False)
l_reader = csv.reader(l)

def metrics_result(actual, predict):
    accuracy = round(metrics.accuracy_score(actual, predict),3) #round() 第二个参数为精确小数点后几位
    return accuracy

ngram_vec = CountVectorizer(ngram_range=(2,2),min_df=5)
print ngram_vec.ngram_range


for line in l_reader:
    if l_reader.line_num == 1: #若csv文件第一行没有列属性，需手动写入
        continue
    else:
        content = line[1]
        content = tok.tokenize(content)
        content = " ".join(content).lower()
        content = content.replace("@user","").replace("@url","")
        corpus.append(content)
        one_text_label = line[2:]
        label.append(one_text_label)

data = ngram_vec.fit_transform(corpus).toarray()



kf = KFold(n_splits=10,shuffle=True)
data = shuffle(data)
for num in range(1,5): #取后四个标签
    key = label_name[num]
    for train_index,test_index in kf.split(data):
        data = np.array(data)
        label = np.array(label)
        train_X = data[train_index].tolist()
        train_y = [i[num] for i in label[train_index].tolist()]
        test_X = data[test_index].tolist()
        test_y = [i[num] for i in label[test_index].tolist()]

        clf = SVC(kernel="rbf")
        clf.fit(train_X,train_y)
        y_predict = clf.predict(test_X)

        score = metrics_result(y_predict,test_y)

        result[key] += 1.0*score/10

    print key+":",result[key]

l.close()
