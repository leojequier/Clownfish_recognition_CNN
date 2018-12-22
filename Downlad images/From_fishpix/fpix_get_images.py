from bs4 import *
import urllib.request
import sys
import os
sys.path.append('../')
from imgdl_ctr_size import dl_jpg

parent = "http://fishpix.kahaku.go.jp/"
beg = "http://fishpix.kahaku.go.jp/fishimage-e/detail?START="
end = "&FAMILY=&SPECIES=Amphiprion&LOCALITY=&FISH_Y=&FISH_M=&FISH_D=&PERSON=&PHOTO_ID=&JPN_FAMILY_OPT=1&FAMILY_OPT=1&JPN_NAME_OPT=1&SPECIES_OPT=1&LOCALITY_OPT=1&PERSON_OPT=1&PHOTO_ID_OPT=1"

##!!!!! change range to get all the image
for i in range(902): #902 images on website
	beg = "http://fishpix.kahaku.go.jp/fishimage-e/detail?START="
	end = "&FAMILY=&SPECIES=Amphiprion&LOCALITY=&FISH_Y=&FISH_M=&FISH_D=&PERSON=&PHOTO_ID=&JPN_FAMILY_OPT=1&FAMILY_OPT=1&JPN_NAME_OPT=1&SPECIES_OPT=1&LOCALITY_OPT=1&PERSON_OPT=1&PHOTO_ID_OPT=1"
	thumbnail = beg + str(i) + end
	#-------------- 
	#open the thumbnail page
	response = urllib.request.urlopen(thumbnail)
	html = response.read()
	thumbsoup = BeautifulSoup(html, "html.parser")
	link = thumbsoup.img['src'][3:]
	link = parent + link
	#extension = link[-4:]
	name = thumbsoup.b.get_text()
	name = "_".join(name.strip().split(" ")[:2])
	if not name == "Amphiprion_sp.":
		## change parent!!! $"../images/"
		path = "../images/" +name
		if not os.path.exists(path):
			os.makedirs(path)
		name = "pix" + name + str(i) + ".jpg"
		dl_jpg(link, path, name)

	if i % 10 == 0:
		print(i)