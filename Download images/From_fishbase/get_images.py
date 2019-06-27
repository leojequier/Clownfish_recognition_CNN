from bs4 import *
import urllib.request
import sys
import os
sys.path.append('../')
from imgdl import dl_jpg
from from_search import thumbnails



#aim: GET the pictures from the thumbnail page
for link in thumbnails:
	#-------------- 
	#open the thumbnail page
	response = urllib.request.urlopen(link)
	html = response.read()
	thumbsoup = BeautifulSoup(html, "html.parser")
    #get the object containing the genus name
	species_name = thumbsoup.font.a.get_text()
	species_name = species_name.replace(" ", "_")
    #create a folder for the genus
	path_create = "../images/" + species_name
	if not os.path.exists(path_create):
		os.makedirs(path_create)
	
	#extracts each image object of the thumbnail page
	all_img_objects = []
	for i in thumbsoup.find_all('img'):
		all_img_objects.append(i)
	
	#extract the src of the image from the object,
    #save the image_name, for example "amaka_01.jpg"
    #an downlod the image at "../images/species_name/image_name.jpg"
	parent = 'https://www.fishbase.de'
	for i in all_img_objects:
		if not "thumb" in i['src'] and "species" in i['src']:
			link_end = i['src'][2:]
			image_name = link_end.split("/")[-1]
			link = parent+link_end
			path = "../images/" + species_name 
			dl_jpg(link, path, image_name)



