from bs4 import *
import urllib.request



with open("flyingmoss.html") as fmoss:
	mysoup = BeautifulSoup(fmoss, "html.parser")

#print(mysoup.prettify())
#print(mysoup.body.h2)

with open("alice.html") as alice:
	soup = BeautifulSoup(alice, "html.parser")


with open("../dbasewp/Sc_name_search_res.html") as search:
	fishsoup = BeautifulSoup(search, "html.parser")

links = []

for link in fishsoup.find_all('a'):
    links.append(link.get('href'))

#response = urllib.request.urlopen(links[7])
#ismyfish = response.read()
#fishsoup2 = BeautifulSoup(ismyfish, "html.parser")
#

for x in range(10):
	print(x, ":  ",	links[x], "\n")


#print(fishsoup.body.table.tbody.tr.prettify())


