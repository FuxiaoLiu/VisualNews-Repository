#PYTHONIOENCODING=utf-8 python raw.py

#This commend can solve "UnicodeEncodeError" error.
#Also filter html label out.
from bs4 import BeautifulSoup
html = '<p>PHOENIX — Arizona state officials have paid $3 million to settle a lawsuit from a former <a href="http://azc.cc/1OHLzX7">teacher who was raped</a> and beaten by an inmate in January 2014, records obtained Monday by <em>The Arizona Republic</em> show.</p>'
soup = BeautifulSoup(html, 'html.parser')
result = soup.get_text()
#result: PHOENIX — Arizona state officials have paid $3 million to settle a lawsuit from a former teacher who was raped and beaten by an inmate in January 2014, records obtained Monday by The Arizona Republic show.


#filter punctuations
import re
s = re.sub(r'[^\w\s]','',caption)
#result: PHOENIX  Arizona state officials have paid 3 million to settle a lawsuit from a former teacher who was raped and beaten by an inmate in January 2014 records obtained Monday by The Arizona Republic show

#named entity label recognition
import spacy
from spacy import displacy
NER = spacy.load("en_core_web_sm")
raw_text = 'The U.S. Immigration Debate is alwasys very popular'
text1= NER(raw_text)
for word in text1.ents:
    print(word.text,word.label_)
  
#Read txt file
filename = '/p/newscaptioning/data/news_caption/usa_today_articles/'
id = '317118'
#id.append(datas[0]['article_title'])
filename1 = filename + str(id) + '.txt'
f = open(filename1, "r", encoding="utf-8")
lines = f.read()
print(lines)


# Statistic of words frequency in the list, and turn into dict
from collections import Counter
a = [1,2,1,3,1,2]
result = Counter(a)
print(result)
result = dict(result)
for key,value in result.items():
    print(key,value）
          
# Read and Write json
with open('data.json', 'w') as f:
    json.dump(data, f)

# Reading data back
with open('data.json', 'r') as f:
    data = json.load(f)
