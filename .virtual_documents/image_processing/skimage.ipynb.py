from skimage import data, color
import matplotlib.pyplot as plt
import seaborn as sns 

# Color Setting
color_pal = sns.color_palette()
plt.style.use('seaborn')



rocket = data.rocket()
display(rocket.shape)

# Turning into gray
gray_rocket = color.rgb2gray(rocket)

display(gray_rocket.shape)


def show_image(image, title='Image', color_map_type='gray'):
    plt.imshow(image, cmap=color_map_type)
    plt.title(title)
    plt.axis('off')
    plt.show()


show_image(rocket)


import numpy as np

# Reading image
me = plt.imread('me.jpg')
plt.imshow(me)
plt.show()

# Currecting the Rotation
me = np.rot90(me, 3)

# Original Me
plt.imshow(me)
plt.show()
# Here me after rgb to gray
me_in_gray = color.rgb2gray(me)
# print(read_img.shape)

# Converting into grayscale
# sweet_gray = color.rgb2gray(read_img)
plt.imshow(me_in_gray)
plt.show()



plt.imshow(me)
plt.show()

# Creaing a Block
melo = me

melo[100:1000, 100:1000, 0] = 1
melo[100:1000, 100:1000, 1] = 0
melo[100:1000, 100:1000, 2] = 0

plt.imshow(melo)
plt.show()





red = me[:, :, 0]
green = me[:, :, 1]
blue = me[:, :, 2]


for i in [red, blue, green]:
    show_image(i)


right_flip_v = np.flipud(me) # for up/down or Vertically flip
right_flip_h = np.fliplr(me) # for left/right Horizontally

plt.imshow(me)
plt.title('The original One')
print(me.shape)
plt.show()

for i in [right_flip_v, right_flip_h]:
    plt.imshow(i)
    plt.show()


fig, ax = plt.subplots(figsize=(10, 6))

# Obtain the red channel
red_channel = me[:, :, 0]

# Plot the red histogram with bins in a range of 256
plt.hist(red_channel.ravel(),# Return a contiguous flattened array. Like: if a= [[1, 2], [3, 4]] Then a.ravel() = [1, 2, 3, 4]
                             # A 1-D array, containing the elements of the input, is returned
         bins=256,
         color=color_pal[0]) 

# Set title and show
plt.title('Red Histogram')
plt.show()


# Reading the Image
me_again = plt.imread('Amirul_Islam.jpg')
plt.imshow(me_again)
plt.show()


# Convert rgb2gray Scale
me_again_gray = color.rgb2gray(me_again)



from skimage.filters import threshold_otsu

# optinal threshold
thresh = threshold_otsu(me_again_gray)
print(thresh)

# Identify the background
back = thresh > me_again_gray

show_image(back)


from skimage.filters import threshold_local, try_all_threshold, threshold_otsu

# Reading the Image
hand_written = plt.imread('hand_writing.jpg')

# Convert rgb2gray Scale
hand_written_gray = color.rgb2gray(hand_written)
# show_image(hand_written_gray)



# Global way
thresh_glo = threshold_otsu(hand_written_gray)
back_glo = hand_written_gray > thresh_glo



# Local way
# Assigning Thresh
block_size = 35

# Obtain the optimal local threshold
local_thresh = threshold_local(hand_written_gray, block_size, offset= 10) # the offset=10 for Subtracting from mean
                                                                          # to achive optinal threshold

# Appling the thresh
back_loc = local_thresh > hand_written_gray


# Finding the best fit
# fig, ax = try_all_threshold(hand_written_gray, verbose=False)
# print(local_thresh)

# plot Designing
fig, axes = plt.subplots(nrows=3, figsize=(7, 8))
ax = axes.ravel()
plt.gray()



ax[0].imshow(hand_written)
ax[0].set_title('Original')

ax[1].imshow(back_glo)
ax[1].set_title('Global thresholding')

ax[2].imshow(back_loc)
ax[2].set_title('Local thresholding')

for a in ax:
    a.axis('off')

plt.show()



def best_one(image):
#     import library
    from skimage.filters import  try_all_threshold
    from skimage.color import rgb2gray
    
    if image.ndim == 3:
#       Converting to gray Scale
        image_gray = rgb2gray(image)
    
#       obtain all thresholds
        fig, ax = try_all_threshold(image_gray, verbose=False)
#         plt.show(fig, ax)
    elif image.ndim == 2:
#       obtain all thresholds
        fig, ax = try_all_threshold(image, verbose=False)
#         plt.show(fig, ax)
    else:
        print('image is not in right shape')
    return 

best_one(me_again)




from skimage.filters import sobel
from skimage.color import rgb2gray

random = plt.imread('random.jpg')

to_rgb = rgb2gray(random)

edge = sobel(to_rgb)

show_image(edge)


from skimage.filters import gaussian

# to_rgb = rgb2gray(random)
blurrly = gaussian(to_rgb, multichannel=True)
show_image(blurrly)
show_image(random)


from skimage.feature import canny

# Using gaussian() to remove noice
blurrly = gaussian(to_rgb, multichannel=True)

# Detecting Edge
edge_d = canny(blurrly, sigma=2) # the lower value of sigma the less effect of gaussian effect on the image Default=1
show_image(edge_d)


show_image(edge, 'soble')
show_image(edge_d, 'canny')


def show_corners(image, corners, title='Detected Corners'):
    plt.imshow(image, interpolation='nearest', cmap='gray')
    plt.title(title)
    plt.plot(corners[:, 1], corners[:, 0], '+r', markersize=15)
    plt.axis('off')
    plt.show()


# Reading image
corner = plt.imread('Image Processing with Python course exercise dataset/chapter 4/corners_building_top.jpg')

# to gray scale
to_gray = rgb2gray(corner)

from skimage.feature import corner_harris, corner_peaks

# Point of interest with sobel
# from skimage.filters import sobel
poin_of_in = sobel(to_gray)
show_image(poin_of_in, 'Point of Interest')

# corner Detect
corners = corner_harris(poin_of_in)


# This image show's here (Posible Point of Interest corners) of corners candidates
show_image(corners, 'The Images Here condidates of corners')

corner_p = corner_peaks(corners, min_distance=10)
# print(len(corner_p))


show_corners(corner, corner_p)














# Library for Enchance Contrast
from skimage import exposure

xray = plt.imread('Image Processing with Python course exercise dataset/chapter 2/chest_xray_image.png')

# improvimg
improved_xray = exposure.equalize_hist(xray)

show_image(xray, 'Original')
show_image(improved_xray, 'Improved One')



further_improved = exposure.equalize_adapthist(xray, clip_limit=0.03)

show_image(xray, 'Original')
show_image(further_improved, 'Improved One')


from skimage.transform import rescale, resize

# Reading image
random_cat = plt.imread('Image Processing with Python course exercise dataset/chapter 2/image_cat.jpg')

# Rescaling
rescaled_cat = rescale(random_cat, 1/8, anti_aliasing=True, multichannel=True)

# Rsizeing
resized_cat = resize(rescaled_cat, (random_cat.shape[0]/2, random_cat.shape[1]/2), anti_aliasing=True)

# Showing
show_image(random_cat)
show_image(rescaled_cat, 'Scaled')
show_image(resized_cat, 'Resized')



from skimage.transform import rotate

# Rotating image
rotate_cat = rotate(random_cat, -90)

show_image(random_cat)
show_image(rotate_cat, 'after Anti-Clock-wise-Rotate 90')


# Reading r
r = plt.imread('Image Processing with Python course exercise dataset/chapter 2/r5.png')   

# In gray Scale
r = rgb2gray(r)           

# Importing libray
from skimage import morphology

# Erosion
erosion = morphology.binary_erosion(r)

# Dilation
dilation = morphology.binary_dilation(r)

# Showing Image
show_image(r, 'Original')
show_image(erosion, 'Erosion')
show_image(dilation, 'Dilation')




from skimage.restoration import inpaint
inpaint.
# Reading image
damaged_image = plt.imread('damaged.jpg')

# Creating mask
# mask = get



from skimage.util import random_noise

noise_added = random_noise(rotate_cat)

show_image(rotate_cat, 'original cat')
show_image(noise_added, 'noise added')



from skimage.restoration import denoise_tv_bregman

# removde
remove_noise = denoise_tv_bregman(noise_added, weigth=0.1, multichannel=True)

show_image(remove_noise)





import tensorflow as tf


print("Num GPUs Available: ", len(tf.config.list_physical_devices('CPU')))


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# generating gray scale image
gray = tf.random.uniform([2, 2], maxval=255, dtype='int32')
print(gray)
plt.imshow(gray)
plt.show()

# Reshaping gray Scale
gray = tf.reshape(gray, [2*2, 1])
print(gray)
plt.imshow(gray)
plt.show()


gray = tf.random.uniform([2, 2, 3], maxval=255, dtype='int32')
print(gray)
plt.imshow(gray)
plt.show()

# Reshaping gray Scale
gray = tf.reshape(gray, [2*2, 3])
print(gray)
plt.imshow(gray)
plt.show()









