import cv2

def new_prediction(img_path):
    
    IMG_SIZE = 128
    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)

__all__ = ['new_prediction']

