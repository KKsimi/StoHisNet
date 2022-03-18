import cv2
import os

def rotate(self, degrees):
    # see http://stackoverflow.com/a/23990392
    if degrees == 90:
        self.image = cv2.transpose(self.image)
        cv2.flip(self.image, 0, self.image)
    elif degrees == 180:
        cv2.flip(self.image, -1, self.image)
    elif degrees == 270:
        self.image = cv2.transpose(self.image)
        cv2.flip(self.image, 1, self.image)

def augu(name):
    imgpath = new + name
    img = cv2.imread(imgpath)
    img1 = cv2.transpose(img)
    cv2.flip(img1, 0, img1)
    cv2.imwrite(outpath + name[:-4] + '-1.png', img1)
    img1 = cv2.transpose(img)
    cv2.flip(img, -1, img1)
    cv2.imwrite(outpath + name[:-4] + '-2.png', img1)
    img1 = cv2.transpose(img)
    cv2.flip(img1, 1, img1)
    cv2.imwrite(outpath + name[:-4] + '-3.png', img1)
    img1 = cv2.transpose(img1)
    cv2.imwrite(outpath + name[:-4] + '-4.png', img1)
    cv2.flip(img1, 1, img1)
    cv2.imwrite(outpath + name[:-4] + '-5.png', img1)


path = 'Data/wei_data/'
outt = 'Data/wei_data/'
a = ['test/', 'train/', 'val/']
for i in range(0, 3):
    new = path + a[i] + '3' + '/'
    namelist = os.listdir(new)
    outpath = outt + a[i] + '3' + '/'
    for name in namelist:
        imgpath = new + name
        img = cv2.imread(imgpath)
        print(imgpath)
        print(outpath + name[:-4] + '-1.png')
        img1 = cv2.transpose(img)
        cv2.flip(img1, 0, img1)
        cv2.imwrite(outpath + name[:-4] + '-1.png', img1)