# TODO: scrivere una classe per l'import del dataset -> da completare;
# TODO: definire gli split in TRAINING, VALIDATION;
# TODO: creare i dataloader;
# TODO: creare una funzione per salvare lo stato della rete;
# TODO: creare una funzione per ripristinare lo stato della rete;
import random
from torchvision.datasets import VisionDataset
import os
import os.path
import sys
import json
import cv2
import dlib
import random
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool, current_process


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def train_valid_split(dataset, num_targets, train_size):
    '''
    The train_valid_split function splits a training set in training and validation sets and returns them.
    Args:
        dataset (VisionDataset): dataset to be splitted
        num_targets (int): number of targets present in given dataset
        train_size (float): ratio between training and original set size
    Returns:
        tuple : (train_idx, valid_idx)
    '''
    classes = [[] for i in range(num_targets)]

    for i in range(len(dataset)):
        target = dataset.getlabel(i)
        classes[target].append(i)

    train_idx = []
    valid_idx = []

    for c in classes:
        random.shuffle(c)
        assert 0 < train_size < 1
        split = int(len(c)*train_size)
        [train_idx.append(idx) for idx in c[:split]]
        [valid_idx.append(idx) for idx in c[split:]]

    return train_idx, valid_idx


class Dataset(VisionDataset):
    def __init__(self, pathFrames, type="train", transform=None, target_transform=None, max_real=None, max_fake=None):
        super(Dataset, self).__init__(pathFrames, transform=transform, target_transform=target_transform)
        self.type = type
        self.dic = {}       # contiene: [nome del video] = indice nella dizionario frames
        self.frames = {}    # contiene: [indice] = lista dei frame del video
        self.labels = []    # coniene: [indice] = 0(FAKE)/1(REAL)
        index_video = 0     # contatore per assegnare un codice al video
        fake_cnt = 0        # contatore per il numero di video fake
        real_cnt = 0        # contatore per il numero di video real

        for dir in os.listdir(pathFrames):
            print(f"Direcotory: {dir}")
            for file in tqdm(os.listdir(pathFrames + "/" + dir)):
                nome_video = file.split("_")[0]
                if nome_video in self.dic:
                    index = self.dic[nome_video]                                    # reperisco l'indice a cui accedere
                else:
                    if dir == "REAL" and max_real is not None:
                        if real_cnt < max_real:
                            self.dic[nome_video] = index_video
                            index = index_video
                            index_video += 1
                            self.frames[index] = []
                            self.labels.append(1)
                            real_cnt += 1
                        else:
                            continue

                    elif dir == "FAKE" and max_fake is not None:
                        if fake_cnt < max_fake:
                            self.dic[nome_video] = index_video
                            index = index_video
                            index_video += 1
                            self.frames[index] = []
                            self.labels.append(0)
                            fake_cnt += 1
                        else:
                            continue

                    else:
                        self.dic[nome_video] = index_video
                        index = index_video
                        index_video += 1
                        self.frames[index] = []                                         # creazione di una nuova lista per il video
                        if dir == "REAL":                                               # assegnazione della label
                            self.labels.append(1)
                            real_cnt += 1
                        else:
                            self.labels.append(0)
                            fake_cnt += 1

                self.frames[index].append(pathFrames + "/" + dir + "/" + file)  # memorizzazione del path
        print(f"Real videos: {real_cnt}")
        print(f"Fake videos: {fake_cnt}")


    def __getitem__(self, index):
        image = None
        while image is None:
            try:
                image = pil_loader(random.choice(self.frames[index]))
            except Exception as e:
                print(e)
        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        length = len(self.labels)
        return length

    def setTransformantion(self, transform):
        self.transform = transform

    def getlabel(self, index):
        return self.labels[index]

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

        # Sampling
        if frame_num % 10 is not 0:
            continue

        # Check if frame was already computed
        if os.path.exists(output_path + "/" + video_filename + "_" + str(frame_num) + ".jpg"):
            print("{} already exists!".format(output_path + "/" + video_filename + "_" + str(frame_num) + ".jpg"))
            continue

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
            im.save(output_path + "/" + video_filename + "_" + str(frame_num) + ".jpg")

        if frame_num >= end_frame:
            break

    pbar.close()


def storeFrame(pathROOT, pathJSONInput, pathOutput):
    # estrapolazione di frame dove ci sono i volti da ogni video
    print("Loaded JSON: {}".format(pathJSONInput))
    with open(pathJSONInput) as f:
        data = json.load(f)

    p = Pool(4)
    args = []

    for count, key in enumerate(data, 1):
        set_dir = os.path.join(pathOutput, data[key]["split"])
        if not os.path.isdir(set_dir):
            os.mkdir(set_dir)

        label_dir = os.path.join(set_dir, data[key]["label"])
        if not os.path.isdir(label_dir):
            os.mkdir(label_dir)

        args.append((os.path.join(pathROOT, key), label_dir, 0, None, count, len(data)))

    p.starmap_async(extractFrames, args)
    p.close()
    p.join()


def storeFrame_noJSON(pathROOT, pathOutput):
    if not os.path.isdir(pathOutput):
        os.mkdir(pathOutput)

    p = Pool(4)
    args = []

    for dir_path, dir_names, file_names in os.walk(pathROOT):
        [args.append((os.path.join(dir_path, file), pathOutput)) for file in file_names if file.endswith('.mp4')]

    p.starmap_async(extractFrames, args)
    p.close()
    p.join()


    p.starmap(extractFrames, args)
    
def loadJSONs(pathJSONs, pathOutput):
    # apertura dei JSON con ricerca in profondità

    if not os.path.isdir(pathOutput):
        os.mkdir(pathOutput)

    for dir_path, dir_names, file_names in os.walk(pathJSONs):
        [storeFrame(dir_path, os.path.join(dir_path, file), pathOutput) for file in file_names if file.endswith('.json')]


# storeFrame("/aiml/project/DFDC/Datasets/v01a/fb_dfd_release_0.1_final", "/aiml/project/DFDC/Datasets/v01a/fb_dfd_release_0.1_final/dataset.json", "/aiml/project/DFDC/FramesDataset_prova")

# loadJSONs("/aiml/project/DFDC/Datasets/dfdc_train", "/aiml/project/DFDC/FramesDataset_full")

# storeFrame_noJSON("/aiml/project/DFDC/Datasets/test_videos", "/aiml/project/DFDC/FramesDataset_test_videos")

# train = Dataset("/aiml/project/DFDC/FramesDataset/train")
# print(f"Len trian: {len(train)}")
