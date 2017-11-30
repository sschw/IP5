"""
"""
from PIL import Image,ImageFilter
import os
from joblib import Parallel, delayed
from colorsys import rgb_to_hsv


GREEN_RANGE_MIN_HSV = (0.2, 0.2, 0)
GREEN_RANGE_MAX_HSV = (0.5, 1, 1)

def get_bounding_box_by_hsv_range(im):
    # Go through all pixels and make a bounding box around pixels that are not 
    # in the hsv range.
    im2 = im.filter(ImageFilter.MedianFilter())
    pix = im2.load()
    
    width, height = im.size
    min_x = width
    min_y = height
    max_x = 0
    max_y = 0
    
    for x in range(width):
        for y in range(height):
            r, g, b = pix[x, y]
            h, s, v = rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            
            min_h, min_s, min_v = GREEN_RANGE_MIN_HSV
            max_h, max_s, max_v = GREEN_RANGE_MAX_HSV
            if not(min_h <= h <= max_h and min_s <= s <= max_s and min_v <= v <= max_v):
                pix[x, y] = (255, 255, 255)
            else:
                pix[x, y] = (0, 0, 0)
    return im2

def scale_on_object(im, padding):
    #print("search for object")
    im = get_bounding_box_by_hsv_range(im)
    
    #print("found object:", min_x, min_y, max_x, max_y)
    # find longer side for crop
    
    # crop the image by the object with a padding
    return im#.crop((min_x-padding, min_y-padding, max_x+padding, max_y+padding))

def scale_and_resize_object(fn, root, dir, dest, existing):
    fn = os.path.join(root, fn)
    
    if not '/'.join(fn.split('/')[-2:]).replace('.JPG', '.PNG') in existing \
            and (fn.endswith(".JPG") or fn.endswith(".PNG")):
        # Load image
        im = Image.open(fn)
        # Pre-crop - increases overall performance
        center_x, center_y = im.size[0] >> 1, im.size[1] >> 1
        center = center_x
        if center_x > center_y:
            center = center_y
        new_im = im.crop((center_x - center, center_y - center, 
                          center_x + center, center_y + center))
        
        # Crop on object
        new_im = scale_on_object(new_im, 120)
        #new_im = new_im.resize((224, 224), Image.ANTIALIAS)
        
        new_fn = fn.replace(dir, dest).replace('.JPG', '.PNG')
        # create directory in training dir if it doesn't already exist
        if not os.path.exists(os.path.dirname(new_fn)):
            try:
                os.makedirs(os.path.dirname(new_fn))
            except OSError as e:
                if e.errno != 17:
                    raise
                    # time.sleep might help here
                pass
        new_im.save(new_fn)

def scale_and_resize(dir, dest, existing):
    # scales and resizes in a folder with parallelization
    for root, dirs, files in os.walk(dir):
        Parallel(n_jobs=8)(
            delayed(scale_and_resize_object)(fn=fn, root=root, dir=dir, dest=dest, existing=existing) for fn in files)
