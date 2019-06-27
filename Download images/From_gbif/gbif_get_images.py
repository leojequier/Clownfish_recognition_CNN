from bs4 import *
import urllib.request
import sys
sys.path.append('../')
import os
from imgdl_ctr_size import dl_jpg


with open("database/final_data.csv", encoding = "ISO-8859-1") as data:
	compteur = 0
	for line in data:
		pass
		if compteur >= 760:
			link = (line.split(";")[1])
			link = link[1:-1] #removes quotes
			#extension = link.split(".")[-1][:3]
			family  = line.split(";")[3][1:-1]
			path = "../images/" + family 
			name = "gbif_" + family[11:15] + "_" + str(compteur) + ".jpg"
			dl_jpg(link, path, name)

		if compteur%10 == 0:
			print(compteur)	

		compteur += 1
	



