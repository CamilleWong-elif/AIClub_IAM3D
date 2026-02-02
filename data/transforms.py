import torchvision.transforms as T

def default_transforms():
    return T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
