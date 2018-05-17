#conver a picture to list
from PIL import Image
def imageprepare(argv):
    raw_im = Image.open(argv)
    img = raw_im.convert('L')
    xsize, ysize = raw_im.size
    if xsize != 28 or ysize != 28:
        img = img.resize((28, 28), Image.ANTIALIAS)
    arr = []
    for i in range(28):
        for j in range(28):
            pixel = float(1.0 - float(img.getpixel((j, i))) / 255.0)
            arr.append(pixel)
    return arr