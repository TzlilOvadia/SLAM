import cv2

DATA_PATH = r'...\VAN_ex\dataset\sequences\00\\'
def read_images(idx):
    img_name = '{:06d}.png'.format(idx)
    img1 = cv2.imread(DATA_PATH+'image_0\\'+img_name, 0)
    img2 = cv2.imread(DATA_PATH+'image_1\\'+img_name, 0)
    return img1, img2