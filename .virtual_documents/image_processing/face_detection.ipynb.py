# Loading library
from skimage.feature import Cascade
import matplotlib.pyplot as plt
import matplotlib
from skimage import data


# Reading data
faces = plt.imread('Image Processing with Python course exercise dataset/chapter 4/face_det_friends22.jpg')

# X
train_data = data.lbp_frontal_face_cascade_filename()

# defining the detector
detector = Cascade(train_data)

# Applying Detector to image
detected_face = detector.detect_multi_scale(img=faces,
                                       scale_factor=1.2,
                                       step_ratio=1,
                                       min_size=(10, 10),
                                       max_size=(200, 200))

print(detected_face)

# face Detection function
def show_face_detection(result, detected, title='Face Image'):
    plt.imshow(result)
    img_desc = plt.gca()
    plt.set_cmap('gray')
    plt.title(title)
    plt.axis('off')
    
    for patch in detected:
        img_desc.add_patch(
                matplotlib.patches.Rectangle(
                (patch['c'], patch['r']), 
                patch['width'], 
                patch['height'],
                fill = False, color= 'r', linewidth=2)
        )
    plt.show()
        

show_face_detection(faces, detected_face)
        

