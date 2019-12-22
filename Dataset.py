# TODO: scrivere una classe per l'import del dataset;
# TODO: definire gli split in TRAINING, VALIDATION e TEST SET;
# TODO: creare i dataloader;
# TODO: creare una funzione per il caricamento dei pesi per la rete (vedi codice deepfake);
# TODO: creare una funzione per salvare lo stato della rete;
# TODO: creare una funzione per ripristinare lo stato della rete;
# TODO: creare funzione per la selezione degli hyperparameters -> grid o random search;
# TODO: creare funzione per il training;
# TODO: creare la funzione di testing;

from torchvision.datasets import VisionDataset
import os
import os.path
import sys
import json
import cv2
import dlib
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool, current_process


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Dataset(VisionDataset):
    def __init__(self, pathFrames, type="train", transform=None, target_transform=None):
        self.samples = []
        self.labels = []
        self.type = type

        for dir in os.listdir(pathFrames):
            for file in os.listdir(pathFrames + "/" + dir):
                if dir == "real":
                    self.labels.append(1)
                else:
                    self.labels.append(0)
                self.samples.append(pil_loader(pathFrames + "/" + dir + "/" + file))

    def __getitem__(self, index):
        image = self.samples[index]
        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        length = len(self.samples)
        return length

    def setTransformantion(self, transform):
        self.transform = transform

def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def extractFrames(video_path, output_path, start_frame=0, end_frame=None, count=None, total=None):
    # Read and write
    current = current_process()
    reader = cv2.VideoCapture(video_path)
    video_filename = video_path.split('/')[-1].split('.')[0]
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Face detector
    face_detector = dlib.get_frontal_face_detector()

    # Frame numbers and length of output video
    frame_num = 0
    assert start_frame < num_frames - 1
    end_frame = end_frame if end_frame else num_frames
    if count is not None:
        if total is not None:
            text = "{} - Processing video {}/{}: ".format(str(current.name), count, total)
        else:
            text = "{} - Processing video {}: ".format(str(current.name), count)
        pbar = tqdm(total=end_frame - start_frame, desc=text)
    else:
        pbar = tqdm(total=end_frame - start_frame)

    while reader.isOpened():
        _, image = reader.read()
        if image is None:
            break
        frame_num += 1

        if frame_num < start_frame:
            continue
        pbar.update(1)

        # Image size
        height, width = image.shape[:2]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces):
            # For now only take biggest face
            face = faces[0]

            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y + size, x:x + size]
            # store image
            im = Image.fromarray(cropped_face)
            im.save(output_path + "/" + video_filename + str(frame_num) + ".jpg")

        if frame_num >= end_frame:
            break

    pbar.close()


def storeFrame(pathROOT, pathJSONInput, pathOutput):
    # estrapolazione di frame dove ci sono i volti da ogni video
    with open(pathJSONInput) as f:
        data = json.load(f)

    p = Pool(os.cpu_count())
    args = []

    for count, key in enumerate(data, 1):
        args.append((pathROOT + "/" + key, pathOutput + "/" + data[key]["set"] + "/" + data[key]["label"], 0, None, count, len(data)))

    p.starmap(extractFrames, args)


# storeFrame("/aiml/project/DFDC/Datasets/v01a/fb_dfd_release_0.1_final", "/aiml/project/DFDC/Datasets/v01a/fb_dfd_release_0.1_final/dataset.json", "/aiml/project/DFDC/FramesDataset_prova")

train = Dataset("/aiml/project/DFDC/FramesDataset/train")
print(f"Len trian: {len(train)}")