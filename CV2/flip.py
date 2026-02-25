class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, boxes):
        if random.random() < self.p:
            img = cv2.flip(img, 1)
            w = img.shape[1]
            boxes[:, [0,2]] = w - boxes[:, [2,0]]
        return img, boxes
