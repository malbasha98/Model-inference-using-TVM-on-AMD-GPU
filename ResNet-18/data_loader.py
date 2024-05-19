import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
warnings.filterwarnings("ignore")

DATA_PATH = os.path.join(os.getcwd(), 'imagenet-mini')
STEP = 35
NUM_OF_IMAGES = 100

PATHS = []
LABELS = []

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def get_step():
    return STEP


def load_paths():
    for dirname, _, filenames in os.walk(DATA_PATH + '/train'):
        for filename in filenames:
            if filename[-4:] == 'JPEG':
                PATHS.append((os.path.join(dirname, filename)))
                label = dirname.split('/')[-1]
                LABELS.append(label)


def plot_image(image):
    plt.imshow(np.transpose(image, (1, 2, 0)))
    plt.show()


def load_data(num_of_images=100, step=STEP):
    load_paths()
    images = []
    for i in range(0, num_of_images * step, step):
        image_data = Image.open(PATHS[i]).convert('RGB')
        image = transform(image_data)
        # image = np.expand_dims(image.numpy(), axis=0)
        images.append(image)

    return torch.stack(images, dim=0).numpy()


if __name__ == "__main__":
    load_paths()
    print(len(LABELS), "image paths loaded.")
