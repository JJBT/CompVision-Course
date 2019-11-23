import cv2
# import cv2.cv as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.functional import _is_pil_image
from torchvision import transforms

filename = 'Kaggle/plates/plates/train/cleaned/0004.jpg'


class Segmentate:
    center_crop = transforms.CenterCrop(224)
    initThresh = 105

    def __init__(self):
        pass

    def __call__(self, img):
        if not _is_pil_image(img):
            raise TypeError('Img should be PIL Image. Got {}'.format(type(img)))

        cimg = img.copy()
        img = np.array(img)

        # Convert to gray-scale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        p1 = self.initThresh
        p2 = self.initThresh * 0.4

        # Detect circles using HoughCircles transform
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 500, param1=p1, param2=p2, minRadius=10,
                                   maxRadius=170)

        t = 400
        if circles is None:
            return self.center_crop(cimg)

        c = np.uint16(np.around(circles))[0, 0]

        # Draw the outer circle
        cv2.circle(img, (c[0], c[1]), c[2] + t // 2 - 15, (0, 0, 0), t)

        thr = -10
        # Centering ad cropping
        try:
            img = img[c[1] - c[2] - thr:c[1] + c[2] + thr, c[0] - c[2] - thr:c[0] + c[2] + thr]
            pil_img = Image.fromarray(img)
        except ValueError:
            return cimg

        return pil_img


if __name__ == "__main__":
    import tqdm
    dirname = 'Kaggle/val/'
    a = Segmentate()
    # s = Image.open(filename)
    # s = a(s)
    p = transforms.Pad(padding=100)

    import os
    os.chdir('Kaggle/')
    for i in tqdm.tqdm(os.listdir('val/dirty')):
        print(i)
        s = Image.open('val/dirty/' + i)
        s = p(s)
        s = a(s)
        plt.imshow(s)
        plt.title(i)
        plt.show()
        s.save('val_segm/dirty/' + i)
