#PYTHONIOENCODING=utf-8 python raw.py
#This commend can solve "UnicodeEncodeError" error.
from bs4 import BeautifulSoup

html = '<p>PHOENIX — Arizona state officials have paid $3 million to settle a lawsuit from a former <a href="http://azc.cc/1OHLzX7">teacher who was raped</a> and beaten by an inmate in January 2014, records obtained Monday by <em>The Arizona Republic</em> show.</p>'
soup = BeautifulSoup(html, 'html.parser')
result = soup.get_text()
#result: PHOENIX — Arizona state officials have paid $3 million to settle a lawsuit from a former teacher who was raped and beaten by an inmate in January 2014, records obtained Monday by The Arizona Republic show.
