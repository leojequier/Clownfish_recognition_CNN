from bs4 import *
import urllib.request

#the link of the search result
fishbase_search = "https://www.fishbase.de/Nomenclature/ValidNameList.php?syng=Amphiprion&syns=&vtitle=Scientific+Names+where+Genus+Equals+%3Ci%3EAmphiprion%3C%2Fi%3E&crit2=CONTAINS&crit1=EQUAL"

#---------------
#opens the link
response = urllib.request.urlopen(fishbase_search)
html = response.read()
fishsoup = BeautifulSoup(html, "html.parser")


#----------------
#creates a list of url to go directly to the thumbnails webpage,
#from where all the images for a genus can be downloaded.

thumbnails = []
parent = "https://www.fishbase.de/photos/thumbnailssummary.php?Genus=Amphiprion&Species="
for spp in fishsoup.find_all('a'):
    if "Amphiprion" in spp.get_text():
    	spp_name = spp.get_text().split()[1]
    	thumbnails.append(parent + spp_name)