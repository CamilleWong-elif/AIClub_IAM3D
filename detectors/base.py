class BaseDetector:
    def detect(self, image):
        """
        Must return List[BoundingBox]
        """
        raise NotImplementedError
