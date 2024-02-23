import os
from PIL import Image
from IPython.display import Image as Img

def generate_gif(path):
    img_list = []
    for i, img in enumerate(sorted(os.listdir(path))):
        if i % 10 == 0:
            img_list.append(os.path.join(path, img))
    images = [Image.open(i) for i in img_list]
    
    im = images[0]
    im.save('out.gif', save_all=True, append_images=images[1:], loop=0xff, duration=500)

    return Img(url='out.gif')

generate_gif('img')