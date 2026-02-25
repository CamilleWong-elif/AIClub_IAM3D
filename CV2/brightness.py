class RandomBrightnessContrast:
    def __init__(self, p=0.4):
        self.p = p

    def __call__(self, img, boxes):
        if random.random() < self.p:
            alpha = 1 + random.uniform(-0.2,0.2)
            beta = random.randint(-25,25)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        return img, boxes
