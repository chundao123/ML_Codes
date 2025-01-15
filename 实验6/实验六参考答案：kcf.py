import cv2
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from numpy import conj, real

class HOG():
    def __init__(self, winSize):
        self.winSize = winSize
        self.blockSize = (8, 8)
        self.blockStride = (4, 4)
        self.cellSize = (4, 4)
        self.nbins = 9
        self.hog = cv2.HOGDescriptor(winSize, self.blockSize, self.blockStride,
                                     self.cellSize, self.nbins)

    def get_feature(self, image):
        winStride = self.winSize
        hist = self.hog.compute(image, winStride, padding = (0, 0))
        w, h = self.winSize
        sw, sh = self.blockStride
        w = w // sw - 1
        h = h // sh - 1
        return hist.reshape(w, h, 36).transpose(2, 1, 0)



    def show_hog(self, hog_feature):
        c, h, w = hog_feature.shape
        feature = hog_feature.reshape(2, 2, 9, h, w).sum(axis=(0, 1))
        grid = 16
        hgrid = grid // 2
        img = np.zeros((h * grid, w * grid))
        for i in range(h):
            for j in range(w): 
                for k in range(9):
                    x = int(10 * feature[k, i, j] * np.cos(np.pi / 9 * k))
                    y = int(10 * feature[k, i, j] * np.sin(np.pi / 9 * k))
                    cv2.rectangle(img, (j * grid, i * grid), ((j+1) * grid, (i+1) * grid), (255, 255, 255))
                    x1 = j * grid + hgrid - x
                    y1 = i * grid + hgrid - y
                    x2 = j * grid + hgrid + x
                    y2 = i * grid + hgrid + y
                    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1) 
        cv2.imshow("img", img)
        cv2.waitKey(0)

class Tracker():
    def __init__(self):
        self.max_patch_size = 256
        self.padding = 2.5
        self.sigma = 0.6
        self.lambdar = 0.0001
        self.update_rate = 0.012
        self.gray_feature = True
        self.debug = False

    def get_feature(self, image, roi):
        cx, cy, w, h = roi
        w = int(w * self.padding) // 2 * 2
        h = int(h * self.padding) // 2 * 2
        x = int(cx - w // 2)
        y = int(cy - h // 2)

        sub_image = image[y:y+h, x:x+w, :]
        try:
            resized_image = cv2.resize(sub_image, (self.pw, self.ph))
            print("sub_image:", sub_image)
            print("(self.pw, self.ph):", (self.pw, self.ph))
        except Exception as e:
            print("发生异常：", e)
            print("sub_image:", sub_image)
            print("(self.pw, self.ph):", (self.pw, self.ph))

        if self.gray_feature:
            feature = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            feature = feature.reshape(1, self.ph, self.pw) / 255.0 - 0.5
        else:
            feature = self.hog.get_feature(resized_image)
            if self.debug:
                self.hog.show_hog(feature)

        fc, fh, fw = feature.shape
        self.scale_h = float(fh) / h
        self.scale_w = float(fw) / w

        # 针对提取得到的特征，采用余弦窗进行相乘平滑计算
        # 因为移动样本的边缘比较突兀，会干扰训练的结果
        # 如果加了余弦窗，图像边缘像素值就都接近0了，循环移位过程中只要目标保持完整那这个样本就是合理的
        hann2t, hann1t = np.ogrid[0:fh, 0:fw]
        hann1t = 0.5 * (1 - np.cos(2 * np.pi * hann1t / (fw - 1)))
        hann2t = 0.5 * (1 - np.cos(2 * np.pi * hann2t / (fh - 1)))
        hann2d = hann2t * hann1t
        feature = feature * hann2d

        feature = np.sum(feature, axis=0)
        return feature

    def gaussian_peak(self, w, h):
        """
        使用高斯函数制作样本标签y
        """
        output_sigma = 0.125
        sigma = np.sqrt(w * h) / self.padding * output_sigma
        syh, sxh = h // 2, w // 2  # 目标框中心点
        y, x = np.mgrid[-syh:-syh + h, -sxh:-sxh + w]
        x = x + (1 - w % 2) / 2.
        y = y + (1 - h % 2) / 2.
        # 生成标签(h,w)，越靠近中心点值越大
        g = 1. / (2. * np.pi * sigma ** 2) * np.exp(-((x ** 2 + y ** 2) / (2. * sigma ** 2)))
        return g

    def train(self, x, y, sigma, lambdar):
        #返回w_hat
        c_xx = self.l_correlation(x, x)
        c_xy = self.l_correlation(x, y)
        return c_xy / (c_xx + lambdar)#岭回归

        #k = self.kernel_correlation(x, x, sigma)
        #return fft2(y) / (fft2(k) + lambdar)#核岭回归

    def detect(self, alphaf, x, z, sigma):
        #返回f(z)
        c = fft2(z)
        return real(ifft2(self.alphaf * c))#岭回归

        #k = self.kernel_correlation(x, z, sigma)
        #return real(ifft2(self.alphaf * fft2(k)))#核岭回归

    def kernel_correlation(self, x1, x2, sigma):
        c = ifft2(conj(fft2(x1)) * fft2(x2))
        c = fftshift(c)
        d = np.sum(x1 ** 2) + np.sum(x2 ** 2) - 2.0 * c
        k = np.exp(-1 / sigma ** 2 * np.abs(d) / d.size)
        return k

    def l_correlation(self, x1, x2):
        c = conj(fft2(x1)) * fft2(x2)
        return c

    def init(self, image, roi):
        x1, y1, w, h = roi
        # 目标区域的中心坐标
        cx = x1 + w // 2
        cy = y1 + h // 2
        roi = (cx, cy, w, h)

        scale = self.max_patch_size / float(max(w, h))
        self.ph = int(h * scale) // 4 * 4 + 4
        self.pw = int(w * scale) // 4 * 4 + 4
        self.hog = HOG((self.pw, self.ph))
        x = self.get_feature(image, roi)
        y = self.gaussian_peak(x.shape[1], x.shape[0])
        self.alphaf = self.train(x, y, self.sigma, self.lambdar) # 求解参数
        self.x = x
        self.roi = roi

    def update(self, image):
        """
        对给定的图像，重新计算其目标的位置
        """
        cx, cy, w, h = self.roi
        max_response = -1
        # 尝试多个尺度，也就是在目标跟踪的过程中，适应目标大小的变化
        for scale in [0.85, 1.0, 1.02]:
            roi = map(int, (cx, cy, w * scale, h * scale))
            z = self.get_feature(image, roi)
            responses = self.detect(self.alphaf, self.x, z, self.sigma)#检测目标
            height, width = responses.shape
            print("height, width:",height, width)
            #input("Next?")
            idx = np.argmax(responses)
            res = np.max(responses)
            #选取检测预测值最大的位置作为目标的新位置
            if res > max_response:
                max_response = res
                dx = int((idx % width - width / 2) / self.scale_w)
                dy = int((idx / width - height / 2) / self.scale_h)
                best_w = int(w * scale)
                best_h = int(h * scale)
                best_z = z
        # 更新,每次训练得到的参数受以往训练得到的参数的影响，有一个加权的过程
        self.roi = (cx + dx, cy + dy, best_w, best_h)
        self.x = self.x * (1 - self.update_rate) + best_z * self.update_rate
        y = self.gaussian_peak(best_z.shape[1], best_z.shape[0])
        new_alphaf = self.train(best_z, y, self.sigma, self.lambdar)
        self.alphaf = self.alphaf * (1 - self.update_rate) + new_alphaf * self.update_rate
        cx, cy, w, h = self.roi
        # 返回目标区域的中心坐标和大小
        return (cx - w // 2, cy - h // 2, w, h)
