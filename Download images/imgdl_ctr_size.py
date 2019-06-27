import urllib.request	
from skimage import io, transform
from PIL import Image

def dl_jpg(url, file_path, file_name):
	full_path = file_path + "/" + file_name 
	dwnld = urllib.request.urlretrieve(url, full_path)
	im = Image.open(full_path)
	if im.mode != "RGB":
		im = im.convert("RGB")
	width = round(im.size[1]/(im.size[0]/256)) 
	size = 256, width
	im.thumbnail(size)
	im.save(full_path, "JPEG")
	
	#image = io.imread(full_path)
	#image = transform.resize(image, (256, width))
	#io.imsave(full_path, image)



