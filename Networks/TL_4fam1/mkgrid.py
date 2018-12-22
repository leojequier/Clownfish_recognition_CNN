from PIL import Image

from os import listdir
from os.path import isfile, join

files = ["classify4/val/Amphiprion_frenatus/pixAmphiprion_frenatus726.jpg",
"classify4/val/Amphiprion_clarkii/gbif_clar_1009.jpg",
"classify4/val/Amphiprion_clarkii/pixAmphiprion_clarkii856.jpg",
"classify4/val/Amphiprion_ocellaris/gbif_ocel_124.jpg",
"classify4/val/Amphiprion_frenatus/pixAmphiprion_frenatus879.jpg",
"classify4/val/Amphiprion_perideraion/gbif_peri_1137.jpg",
"classify4/val/Amphiprion_ocellaris/gbif_ocel_59.jpg",
"classify4/val/Amphiprion_perideraion/gbif_peri_15.jpg",
"classify4/val/Amphiprion_ocellaris/gbif_ocel_400.jpg",
"classify4/val/Amphiprion_ocellaris/pixAmphiprion_ocellaris592.jpg",
"classify4/val/Amphiprion_ocellaris/gbif_ocel_117.jpg",
"classify4/val/Amphiprion_ocellaris/gbif_ocel_339.jpg",
"classify4/val/Amphiprion_ocellaris/gbif_ocel_908.jpg"]


new_im = Image.new('RGB', (200*7,200*2))

index = 0
for i in range(0,200*7,200):
    for j in range(0,200*2,200):
        print(index)
        im = Image.open(files[index])
        im.thumbnail((200,200))
        new_im.paste(im, (i,j))
        if index == len(files)-1:
            break
        index += 1

new_im.save("all_false_pred.png")

