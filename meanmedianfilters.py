import numpy as np
import matplotlib.pyplot as plt
import cv2

noise_img = cv2.imread('noise_image.png')
cv2.imshow("noise_img",noise_img)
plt.hist(noise_img.ravel(), 255, [0,255])
plt.title('NOISE IMAGE HISTOGRAM')
plt.savefig("noise_image_his.png")
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
def meanfilter():
    # Read the image
    img = cv2.imread('noise_image.png', 0)
    cv2.imshow('image',img)

    # get the number of rows and columns
    m, n = img.shape

    # create mean filter(3, 3) mask
    mask = np.ones([3, 3], dtype=int)
    mask = mask / 9

    # now Convolve the 3X3 mask over the image
    img_new = np.zeros([m, n])

    for i in range(1, m - 1):
        for j in range(1, n - 1):
            temp = img[i - 1, j - 1] * mask[0, 0] + img[i - 1, j] * mask[0, 1] + img[i - 1, j + 1] * mask[0, 2] + img[
                i, j - 1] * mask[1, 0] + img[i, j] * mask[1, 1] + img[i, j + 1] * mask[1, 2] + img[i + 1, j - 1] * mask[
                       2, 0] + img[i + 1, j] * mask[2, 1] + img[i + 1, j + 1] * mask[2, 2]

            img_new[i, j] = temp

    img_new = img_new.astype(np.uint8)

    # plt.axis('tight')
    # plt.title('MEAN FILTER')
    cv2.imwrite('output1.png', img_new)
    cv2.imshow('mean filter image', img_new)
meanfilter()


output1histo = cv2.imread('output1.png')
vals = output1histo.mean(axis=2).flatten()
counts, bins = np.histogram(vals, range(257))
# plot histogram centered on values 0..255
plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
plt.xlim([-0.5, 255.5])
plt.title('OUTPUT1 HISTOGRAM')
plt.savefig("output1_his.png")
plt.show()

def medianfilter():
    # Read the image
    img = cv2.imread('noise_image.png', 0)


    m, n = img.shape

    # Traverse the image. For every 3X3 area,
    # median of the pixels then replace the center pixel by the median
    img_new1 = np.zeros([m, n])

    for i in range(1, m - 1):
        for j in range(1, n - 1):
            temp = [img[i - 1, j - 1],
                    img[i - 1, j],
                    img[i - 1, j + 1],
                    img[i, j - 1],
                    img[i, j],
                    img[i, j + 1],
                    img[i + 1, j - 1],
                    img[i + 1, j],
                    img[i + 1, j + 1]]

            temp = sorted(temp)
            img_new1[i, j] = temp[4]

    img_new = img_new1.astype(np.uint8)
    cv2.imwrite('output2.png', img_new)
    cv2.imshow('median filter image', img_new)
medianfilter()
cv2.waitKey(0)
cv2.destroyAllWindows()

output2histo = cv2.imread('output2.png')
vals = output1histo.mean(axis=2).flatten()
counts, bins = np.histogram(vals, range(257))
# plot histogram centered on values 0..255
plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
plt.xlim([-0.5, 255.5])
plt.title('OUTPUT2 HISTOGRAM')
plt.savefig("output2_his.png")
plt.show()