#get the author
from bs4 import *
import urllib.request
import sys
import os
sys.path.append('../')
from imgdl import dl_jpg
from from_search import thumbnails
import re

def author(href):
    return href and  re.compile("Collaborator").search(href)

#aim: GET the pictures from the thumbnail page
for link in thumbnails:
	print(link)
	#-------------- 
	#open the thumbnail page
	response = urllib.request.urlopen(link)
	html = response.read()
	thumbsoup = BeautifulSoup(html, "html.parser")
	with open("authorlist.txt", 'a') as a_li:
		for i in thumbsoup.find_all(href=author):
			a_li.write(i.get_text() + "\n")
			
	
