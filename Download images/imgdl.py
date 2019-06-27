import urllib.request	

def dl_jpg(url, file_path, file_name):
	full_path = file_path + "/" + file_name 
	urllib.request.urlretrieve(url, full_path)
	


