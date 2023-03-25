import cv2

DATA_PATH = "/VAN_ex/dataset/sequences/05/"
# DATA_PATH = '...\VAN_ex\dataset\\2023_dataset\sequences\\00\\'
# noinspection PyUnresolvedReferences
def read_images(idx):
    img_name = '{:06d}.png'.format(idx)
    img1 = cv2.imread(DATA_PATH+'image_0/'+img_name, 0)
    img2 = cv2.imread(DATA_PATH+'image_1/'+img_name, 0)
    return img1, img2

if __name__ == '__main__':
    im1, im2 = read_images(0)
    im1_features = cv2.cornerHarris(im1, 2, 3, 0.04)
    im2_features = cv2.cornerHarris(im2, 2, 3, 0.04)

