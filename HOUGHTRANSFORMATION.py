import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import sys
import os
from PIL import Image
import time

# Canny
def Canny(img):

    # Gray scale
    def BGR2GRAY(img):
        b = img[:, :, 0].copy()
        g = img[:, :, 1].copy()
        r = img[:, :, 2].copy()

        # Gray scale
        out = 0.2126 * r + 0.7152 * g + 0.0722 * b
        out = out.astype(np.uint8)

        return out


    # Gaussian filter for grayscale
    def gaussian_filter(img, K_size=3, sigma=1.3):

        if len(img.shape) == 3:
            H, W, C = img.shape
            gray = False
        else:
            img = np.expand_dims(img, axis=-1)
            H, W, C = img.shape
            gray = True

        ## Zero padding
        pad = K_size // 2
        out = np.zeros([H + pad * 2, W + pad * 2, C], dtype=float)
        out[pad : pad + H, pad : pad + W] = img.copy().astype(float)

        ## prepare Kernel
        K = np.zeros((K_size, K_size), dtype=float)
        for x in range(-pad, -pad + K_size):
            for y in range(-pad, -pad + K_size):
                K[y + pad, x + pad] = np.exp( - (x ** 2 + y ** 2) / (2 * sigma * sigma))
        #K /= (sigma * np.sqrt(2 * np.pi))
        K /= (2 * np.pi * sigma * sigma)
        K /= K.sum()

        tmp = out.copy()

        # filtering
        for y in range(H):
            for x in range(W):
                for c in range(C):
                    out[pad + y, pad + x, c] = np.sum(K * tmp[y : y + K_size, x : x + K_size, c])

        out = np.clip(out, 0, 255)
        out = out[pad : pad + H, pad : pad + W]
        out = out.astype(np.uint8)

        if gray:
            out = out[..., 0]

        return out


    # sobel filter
    def sobel_filter(img, K_size=3):
        if len(img.shape) == 3:
            H, W, C = img.shape
        else:
            H, W = img.shape

        # Zero padding
        pad = K_size // 2
        out = np.zeros((H + pad * 2, W + pad * 2), dtype=float)
        out[pad : pad + H, pad : pad + W] = img.copy().astype(float)
        tmp = out.copy()

        out_v = out.copy()
        out_h = out.copy()

        ## Sobel vertical
        Kv = [[1., 2., 1.],[0., 0., 0.], [-1., -2., -1.]]
        ## Sobel horizontal
        Kh = [[1., 0., -1.],[2., 0., -2.],[1., 0., -1.]]

        # filtering
        for y in range(H):
            for x in range(W):
                out_v[pad + y, pad + x] = np.sum(Kv * (tmp[y : y + K_size, x : x + K_size]))
                out_h[pad + y, pad + x] = np.sum(Kh * (tmp[y : y + K_size, x : x + K_size]))

        out_v = np.clip(out_v, 0, 255)
        out_h = np.clip(out_h, 0, 255)

        out_v = out_v[pad : pad + H, pad : pad + W]
        out_v = out_v.astype(np.uint8)
        out_h = out_h[pad : pad + H, pad : pad + W]
        out_h = out_h.astype(np.uint8)

        return out_v, out_h


    def get_edge_angle(fx, fy):
        # get edge strength
        edge = np.sqrt(np.power(fx.astype(np.float32), 2) + np.power(fy.astype(np.float32), 2))
        edge = np.clip(edge, 0, 255)

        fx = np.maximum(fx, 1e-10)
        #fx[np.abs(fx) <= 1e-5] = 1e-5

        # get edge angle
        angle = np.arctan(fy / fx)

        return edge, angle


    def angle_quantization(angle):
        angle = angle / np.pi * 180
        angle[angle < -22.5] = 180 + angle[angle < -22.5]
        _angle = np.zeros_like(angle, dtype=np.uint8)
        _angle[np.where(angle <= 22.5)] = 0
        _angle[np.where((angle > 22.5) & (angle <= 67.5))] = 45
        _angle[np.where((angle > 67.5) & (angle <= 112.5))] = 90
        _angle[np.where((angle > 112.5) & (angle <= 157.5))] = 135

        return _angle


    def non_maximum_suppression(angle, edge):
        H, W = angle.shape
        _edge = edge.copy()

        for y in range(H):
            for x in range(W):
                    if angle[y, x] == 0:
                            dx1, dy1, dx2, dy2 = -1, 0, 1, 0
                    elif angle[y, x] == 45:
                            dx1, dy1, dx2, dy2 = -1, 1, 1, -1
                    elif angle[y, x] == 90:
                            dx1, dy1, dx2, dy2 = 0, -1, 0, 1
                    elif angle[y, x] == 135:
                            dx1, dy1, dx2, dy2 = -1, -1, 1, 1
                    if x == 0:
                            dx1 = max(dx1, 0)
                            dx2 = max(dx2, 0)
                    if x == W-1:
                            dx1 = min(dx1, 0)
                            dx2 = min(dx2, 0)
                    if y == 0:
                            dy1 = max(dy1, 0)
                            dy2 = max(dy2, 0)
                    if y == H-1:
                            dy1 = min(dy1, 0)
                            dy2 = min(dy2, 0)
                    if max(max(edge[y, x], edge[y + dy1, x + dx1]), edge[y + dy2, x + dx2]) != edge[y, x]:
                            _edge[y, x] = 0

        return _edge
    def hysteresis(edge, weak=100, strong=255):
        M, N = edge.shape
        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if (edge[i, j] == weak):
                    try:
                        if ((edge[i + 1, j - 1] == strong) or (edge[i + 1, j] == strong) or (edge[i + 1, j + 1] == strong)
                                or (edge[i, j - 1] == strong) or (edge[i, j + 1] == strong)
                                or (edge[i - 1, j - 1] == strong) or (edge[i - 1, j] == strong) or (
                                        edge[i - 1, j + 1] == strong)):
                            edge[i, j] = strong
                        else:
                            edge[i, j] = 0
                    except IndexError as e:
                        pass
        return edge
    # grayscale
    gray = BGR2GRAY(img)

    # gaussian filtering
    gaussian = gaussian_filter(gray, K_size=5, sigma=1.4)

    # sobel filtering
    fy, fx = sobel_filter(gaussian, K_size=3)

    # get edge strength, angle
    edge, angle = get_edge_angle(fx, fy)

    # angle quantization
    angle = angle_quantization(angle)

    # non maximum suppression
    edge = non_maximum_suppression(angle, edge)

    # hysterisis threshold
    out = hysteresis(edge, 100, 30)

    return out

# Hough transform to detect 20 straight lines in an image
def Hough_Line(edge, img):
    ## Voting
    def voting(edge):
        H, W = edge.shape

        drho = 1
        dtheta = 1

        # get rho max length
        rho_max = np.ceil(np.sqrt(H ** 2 + W ** 2)).astype(int)

        # hough table
        hough = np.zeros((rho_max, 180), dtype=int)

        # get index of edge
        # ind[0] is the conditional vertical coordinate, ind[1] is the conditional horizontal coordinate
        ind = np.where(edge == 255)

        ## hough transformation
        # The zip function returns a tuple
        for y, x in zip(ind[0], ind[1]):
                for theta in range(0, 180, dtheta):
                        # get polar coordinat4s
                        t = np.pi / 180 * theta
                        rho = int(x * np.cos(t) + y * np.sin(t))

                        # vote
                        hough[rho, theta] += 1

        out = hough.astype(np.uint8)

        return out

    # non maximum suppression
    def non_maximum_suppression(hough):
        rho_max, _ = hough.shape

        ## non maximum suppression
        for y in range(rho_max):
            for x in range(180):
                # get 8 nearest neighbor
                x1 = max(x-1, 0)
                x2 = min(x+2, 180)
                y1 = max(y-1, 0)
                y2 = min(y+2, rho_max-1)
                if np.max(hough[y1:y2, x1:x2]) == hough[y,x] and hough[y, x] != 0:
                    pass
                    #hough[y,x] = 255
                else:
                    hough[y,x] = 0

        return hough

    def inverse_hough(hough, img):
        H, W, _= img.shape
        rho_max, _ = hough.shape

        out = img.copy()

        # get x, y index of hough table
        # np.ravel reduces a multidimensional array to 1 dimension
        # argsort sorts array elements from smallest to largest, returns index
        # [::-1] reverse order -> from largest to smallest
        # [:20] first 20
        ind_x = np.argsort(hough.ravel())[::-1][:20]
        ind_y = ind_x.copy()
        thetas = ind_x % 180
        rhos = ind_y // 180

        # each theta and rho
        for theta, rho in zip(thetas, rhos):
            # theta[radian] -> angle[degree]
            t = np.pi / 180. * theta

            # hough -> (x,y)
            for x in range(W):
                if np.sin(t) != 0:
                    y = - (np.cos(t) / np.sin(t)) * x + (rho) / np.sin(t)
                    y = int(y)
                    if y >= H or y < 0:
                        continue
                    out[y, x] = [0,255,255]
            for y in range(H):
                if np.cos(t) != 0:
                    x = - (np.sin(t) / np.cos(t)) * y + (rho) / np.cos(t)
                    x = int(x)
                    if x >= W or x < 0:
                        continue
                    out[y, x] = [0,0,255]

        out = out.astype(np.uint8)

        return out


    # voting
    hough = voting(edge)

    # non maximum suppression
    hough = non_maximum_suppression(hough)

    # inverse hough
    out = inverse_hough(hough, img)

    return out


# Read image
img = cv2.imread("images/4.jpg")
    # .astype(np.float32)

# Canny
edge = Canny(img)

# Hough
out = Hough_Line(edge, img)

out = out.astype(np.uint8)

# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()



# img1 = cv2.imread('images/1.jpg', cv2.IMREAD_GRAYSCALE)
# img = cv2.imread('images/1.jpg').astype(np.float32)
# sigma1 = sigma2 = 1
# sum = 0
# gaussian = np.zeros([5, 5])
# for i in range(5):
#     for j in range(5):
#         gaussian[i, j] = math.exp(
#             -1 / 2 * (np.square(i - 3) / np.square(sigma1)  # Generate a 2-D Gaussian distribution matrix
#                       + (np.square(j - 3) / np.square(sigma2)))) / (2 * math.pi * sigma1 * sigma2)
#         sum = sum + gaussian[i, j]
#
# # step1. Gaussian filtering
# W, H = img1.shape[:2]
# new_img = np.zeros([W - 5, H - 5])
# for i in range(W - 5):
#     for j in range(H - 5):
#         new_img[i, j] = np.sum(img1[i:i + 5, j:j + 5] * gaussian)  # Filtering with Gaussian matrix convolution
#
# # step2. enhancement By finding the gradient amplitude
# W1, H1 = new_img.shape
# dx = np.zeros([W1 - 1, H1 - 1])
# dy = np.zeros([W1 - 1, H1 - 1])
# d = np.zeros([W1 - 1, H1 - 1])
# for i in range(W1 - 1):
#     for j in range(H1 - 1):
#         dx[i, j] = new_img[i, j + 1] - new_img[i, j]
#         dy[i, j] = new_img[i + 1, j] - new_img[i, j]
#         d[i, j] = np.sqrt(np.square(dx[i, j]) + np.square(dy[i, j]))  # 图像梯度幅值作为图像强度值
#
# # setp3. Non-extreme value suppression NMS
# W2, H2 = d.shape
# NMS = np.copy(d)
# NMS[0, :] = NMS[W2 - 1, :] = NMS[:, 0] = NMS[:, H2 - 1] = 0
# for i in range(1, W2 - 1):
#     for j in range(1, H2 - 1):
#         if d[i, j] == 0:
#             NMS[i, j] = 0
#         else:
#             gradX = dx[i, j]
#             gradY = dy[i, j]
#             gradTemp = d[i, j]
#
#             # If the amplitude value in Y direction is larger
#             if np.abs(gradY) > np.abs(gradX):
#                 weight = np.abs(gradX) / np.abs(gradY)
#                 grad2 = d[i - 1, j]
#                 grad4 = d[i + 1, j]
#                 # If the x,y direction gradients have the same sign
#                 if gradX * gradY > 0:
#                     grad1 = d[i - 1, j - 1]
#                     grad3 = d[i + 1, j + 1]
#                 # If the gradients in x,y direction have opposite signs
#                 else:
#                     grad1 = d[i - 1, j + 1]
#                     grad3 = d[i + 1, j - 1]
#
#             # If the x,y direction gradients have the same sign
#             else:
#                 weight = np.abs(gradY) / np.abs(gradX)
#                 grad2 = d[i, j - 1]
#                 grad4 = d[i, j + 1]
#                 # If the x,y direction gradients have the same sign
#                 if gradX * gradY > 0:
#                     grad1 = d[i + 1, j - 1]
#                     grad3 = d[i - 1, j + 1]
#                 # If the gradients in x,y direction have opposite signs
#                 else:
#                     grad1 = d[i - 1, j - 1]
#                     grad3 = d[i + 1, j + 1]
#
#             gradTemp1 = weight * grad1 + (1 - weight) * grad2
#             gradTemp2 = weight * grad3 + (1 - weight) * grad4
#             if gradTemp >= gradTemp1 and gradTemp >= gradTemp2:
#                 NMS[i, j] = gradTemp
#             else:
#                 NMS[i, j] = 0
#
# # step4. double thresholding algorithm to detect, connect edges
# W3, H3 = NMS.shape
# DT = np.zeros([W3, H3])
# # Define high and low thresholds
# TL = 0.2 * np.max(NMS)
# TH = 0.3 * np.max(NMS)
# for i in range(1, W3 - 1):
#     for j in range(1, H3 - 1):
#         if (NMS[i, j] < TL):
#             DT[i, j] = 0
#         elif (NMS[i, j] > TH):
#             DT[i, j] = 1
#         elif ((NMS[i - 1, j - 1:j + 1] < TH).any() or (NMS[i + 1, j - 1:j + 1]).any()
#               or (NMS[i, [j - 1, j + 1]] < TH).any()):
#             DT[i, j] = 1
#
# cv2.imshow("dt th", DT)
# # plt.savefig('results/Canny_result_img1.jpg')
#
# edge = DT
# weak = np.int32(25)
# strong = 255
#
#
# # hysteresis
# def hysterisis(edge, HT=100, LT=30):
#     H, W = edge.shape
#
#     # Histeresis threshold
#     edge[edge >= HT] = 255
#     edge[edge <= LT] = 0
#
#     _edge = np.zeros((H + 2, W + 2), dtype=np.float32)
#     _edge[1: H + 1, 1: W + 1] = edge
#     ## 8 - Nearest neighbor
#     nn = np.array(((1., 1., 1.), (1., 0., 1.), (1., 1., 1.)), dtype=np.float32)
#     for y in range(1, H + 2):
#         for x in range(1, W + 2):
#             if _edge[y, x] < LT or _edge[y, x] > HT:
#                 continue
#             if np.max(_edge[y - 1:y + 2, x - 1:x + 2] * nn) >= HT:
#                 _edge[y, x] = 255
#             else:
#                 _edge[y, x] = 0
#
#     edge = _edge[1:H + 1, 1:W + 1]
#
#     return edge
#
#
# cv2.imshow("edgelinkinh", edge)
#
#
# # cv2.imwrite('results/Canny_result_img2.jpg', edge)
#
#
# def Hough_Line(edge, img):
#     ## Voting
#     def voting(edge):
#         H, W = edge.shape
#
#         drho = 1
#         dtheta = 1
#
#         # get rho max length
#         rho_max = np.ceil(np.sqrt(H ** 2 + W ** 2)).astype(int)
#
#         # hough table
#         hough = np.zeros((rho_max, 180), dtype=int)
#
#         # get index of edge
#
#         ind = np.where(edge == 255)
#
#         ## hough transformation
#
#         for y, x in zip(ind[0], ind[1]):
#             for theta in range(0, 180, dtheta):
#                 # get polar coordinat4s
#                 t = np.pi / 180 * theta
#                 rho = int(x * np.cos(t) + y * np.sin(t))
#
#                 # vote
#                 hough[rho, theta] += 1
#
#         out = hough.astype(np.uint8)
#
#         return out
#
#     # non maximum suppression
#     def non_maximum_suppression(hough):
#         rho_max, _ = hough.shape
#
#         ## non maximum suppression
#         for y in range(rho_max):
#             for x in range(180):
#                 # get 8 nearest neighbor
#                 x1 = max(x - 1, 0)
#                 x2 = min(x + 2, 180)
#                 y1 = max(y - 1, 0)
#                 y2 = min(y + 2, rho_max - 1)
#                 if np.max(hough[y1:y2, x1:x2]) == hough[y, x] and hough[y, x] != 0:
#                     pass
#                     # hough[y,x] = 255
#                 else:
#                     hough[y, x] = 0
#
#         return hough
#
#     def inverse_hough(hough, img):
#         H, W, _ = img.shape
#         rho_max, _ = hough.shape
#
#         out = img.copy()
#
#         ind_x = np.argsort(hough.ravel())[::-1][:20]
#         ind_y = ind_x.copy()
#         thetas = ind_x % 180
#         rhos = ind_y // 180
#
#         # each theta and rho
#         for theta, rho in zip(thetas, rhos):  # theta[radian] -> angle[degree]
#             t = np.pi / 180. * theta
#
#             # hough -> (x,y)
#             for x in range(W):
#                 if np.sin(t) != 0:
#                     y = - (np.cos(t) / np.sin(t)) * x + (rho) / np.sin(t)
#                     y = int(y)
#                     if y >= H or y < 0:
#                         continue
#                     out[y, x] = [0, 255, 255]
#             for y in range(H):
#                 if np.cos(t) != 0:
#                     x = - (np.sin(t) / np.cos(t)) * y + (rho) / np.cos(t)
#                     x = int(x)
#                     if x >= W or x < 0:
#                         continue
#                     out[y, x] = [0, 0, 255]
#
#         out = out.astype(np.uint8)
#
#         return out
#
#     # voting
#     hough = voting(edge)
#
#     # non maximum suppression
#     hough = non_maximum_suppression(hough)
#
#     # inverse hough
#     out = inverse_hough(hough, img)
#
#     return out
#
#
# out = Hough_Line(edge, img)
# out = out.astype(np.uint8)
# cv2.imshow("result", out)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
