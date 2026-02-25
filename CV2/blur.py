class RandomBlur:
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img, boxes):
        if random.random() < self.p:
            k = random.choice([3,5])
            img = cv2.GaussianBlur(img,(k,k),0)
        return img, boxes


#blur for if too many images in dataset are high quality