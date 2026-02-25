class RandomZoom:
    def __init__(self, p=0.25, scale=(0.85,1.15)):
        self.p = p
        self.scale = scale

    def __call__(self, img, boxes):
        if random.random() < self.p:
            s = random.uniform(*self.scale)

            h,w = img.shape[:2]
            img = cv2.resize(img,None,fx=s,fy=s)

            boxes[:,:4] *= s

            img = cv2.resize(img,(w,h))
w
        return img, boxes
