import numpy as np
from scipy.special import comb


class AppearanceTransform(object):

    def __init__(
            self,
            local_rate=0.8,
            nonlinear_rate=0.9,
            paint_rate=0.9,
            inpaint_rate=0.2,
            is_local=True,
            is_nonlinear=True,
            is_in_painting=True):

        self.is_local = is_local
        self.is_nonlinear = is_nonlinear
        self.is_in_painting = is_in_painting
        self.local_rate = local_rate
        self.nonlinear_rate = nonlinear_rate
        self.paint_rate = paint_rate
        self.inpaint_rate = inpaint_rate

    def rand_aug(self, data):
        if self.is_local:
            data = self.local_pixel_shuffling(data, prob=self.local_rate)
        if self.is_nonlinear:
            data = self.nonlinear_transformation(data, self.nonlinear_rate)
        if self.is_in_painting:
            data = self.image_in_painting(data)
        data = data.astype(np.float32)
        return data

    def bernstein_poly(self, i, n, t):
        return comb(n, i) * (t ** (n - i)) * (1 - t) ** i

    def bezier_curve(self, points, nTimes=1000):
        nPoints = len(points)
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])

        t = np.linspace(0.0, 1.0, nTimes)

        polynomial_array = np.array([self.bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

        xvals = np.dot(xPoints, polynomial_array)
        yvals = np.dot(yPoints, polynomial_array)
        return xvals, yvals

    def nonlinear_transformation(self, x, prob=0.5):
        if np.random.random() >= prob:
            return x
        points = [[0, 0], [np.random.random(), np.random.random()], [np.random.random(), np.random.random()], [1, 1]]

        xvals, yvals = self.bezier_curve(points, nTimes=100000)

        xvals, yvals = np.sort(xvals), np.sort(yvals)
        nonlinear_x = np.interp(x, xvals, yvals)
        return nonlinear_x

    def local_pixel_shuffling(self, x, prob=0.5):
        if np.random.random() >= prob:
            return x
        image_temp = x.copy()
        orig_image = x.copy()
        img_rows, img_cols, img_deps = x.shape
        num_block = 5000
        block_noise_size_x = int(img_rows // 20)
        block_noise_size_y = int(img_cols // 20)
        block_noise_size_z = int(img_deps // 20)
        noise_x = np.random.randint(low=img_rows - block_noise_size_x, size=num_block)
        noise_y = np.random.randint(low=img_cols - block_noise_size_y, size=num_block)
        noise_z = np.random.randint(low=img_deps - block_noise_size_z, size=num_block)
        window=[orig_image[noise_x[i]:noise_x[i] + block_noise_size_x, noise_y[i]:noise_y[i] + block_noise_size_y,
                     noise_z[i]:noise_z[i] + block_noise_size_z,] for i in range(num_block)]
        window = np.concatenate(window, axis=0)
        window = window.reshape(num_block, -1)
        np.random.shuffle(window.T)
        window = window.reshape((num_block, block_noise_size_x,
                                 block_noise_size_y,
                                 block_noise_size_z))
        for i in range(num_block):
            image_temp[noise_x[i]:noise_x[i] + block_noise_size_x,
            noise_y[i]:noise_y[i] + block_noise_size_y,
            noise_z[i]:noise_z[i] + block_noise_size_z] = window[i]
        local_shuffling_x = image_temp

        return local_shuffling_x

    def image_in_painting(self, x):
        img_rows, img_cols, img_deps = x.shape
        cnt = 30
        while cnt > 0 and np.random.random() < 0.95:
            block_noise_size_x = np.random.randint(img_rows // 10, img_rows // 5)
            block_noise_size_y = np.random.randint(img_cols // 10, img_cols // 5)
            block_noise_size_z = np.random.randint(img_deps // 10, img_deps // 5)
            noise_x = np.random.randint(3, img_rows - block_noise_size_x - 3)
            noise_y = np.random.randint(3, img_cols - block_noise_size_y - 3)
            noise_z = np.random.randint(3, img_deps - block_noise_size_z - 3)
            x[
            noise_x:noise_x + block_noise_size_x,
            noise_y:noise_y + block_noise_size_y,
            noise_z:noise_z + block_noise_size_z] = np.random.rand(block_noise_size_x,
                                                                   block_noise_size_y,
                                                                   block_noise_size_z, ) * 1.0
            cnt -= 1
        return x
