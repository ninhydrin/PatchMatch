import numpy as np
import itertools
from PIL import Image

from tqdm import tqdm


class PatchMatch:
    def __init__(self, img_a, img_b, patch_radius=2, save_path="result.png"):
        self.img_a = img_a
        self.img_b = img_b
        self.img_b_ymax = img_b.shape[0]
        self.img_b_xmax = img_b.shape[1]
        self.p_r = patch_radius
        self.r = min(self.img_a.shape[:2]) // (4/3)
        self.nnf = None
        self.save_path = save_path

    def init_nnf(self):
        self.nnf = [np.random.randint(0, self.img_b.shape[i], self.img_a.shape[:2]) for i in range(2)]
        self.loss_map = self.calc_loss(self.nnf)

    def calc_loss(self, nnf):
        r = self.p_r
        loss_map = np.zeros(self.img_a.shape[:2], dtype="f")
        for i, j in itertools.product(range(-r, r + 1), range(-r, r + 1)):
            y = np.clip(nnf[0] + i, 0, self.img_b_ymax - 1)
            x = np.clip(nnf[1] + j, 0, self.img_b_xmax - 1)
            loss_map += ((self.img_b[y, x] - self.img_a) ** 2).sum(2)
        return loss_map

    def new_nnf(self, it=10):
        nnf = self.nnf.copy()
        now_loss = self.loss_map.copy()
        for i in range(it):
            new_nnf = [np.clip(self.nnf[j] + np.random.randint(-self.r, self.r, self.img_a.shape[:2]), 0, self.img_b.shape[j] - 1) for j in range(2)]
            loss = self.calc_loss(new_nnf)
            update_pix = loss < now_loss
            for j in range(2):
                nnf[j][update_pix] = new_nnf[j][update_pix]
            now_loss[update_pix] = loss[update_pix]
        return nnf, now_loss

    def reconstruction(self):
        return self.img_b[self.nnf]

    def __call__(self, iteration=10):
        self.init_nnf()
        for i in tqdm(range(iteration)):
            nnf, loss = self.new_nnf()
            update = loss < self.loss_map
            if not update.any():
                break
            self.nnf = nnf
            self.loss_map = loss
            if i % 2:
                self.r //= 4/3
            if self.r == 0:
                break
        print(f"save {self.save_path}")
        Image.fromarray(self.reconstruction()).save(self.save_path)


if __name__ == '__main__':
    img_a = np.array(Image.open("img_a.png"))
    img_b = np.array(Image.open("img_b.png"))
    pm = PatchMatch(img_a, img_b, 2)
    pm(40)
