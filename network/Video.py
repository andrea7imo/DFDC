import os
import os.path


class Video:
    def __init__(self, pathFrames):
        dim=0
        frames = [] #lista frames
        nome = [] #liasta nomi video
        self.dic = {} #dizionario per ogni video -> lista frames
        self.labels = {} # dizionario per ogni video -> lista lables
        nome = ListaNomi('/aiml/project/DFDC/Datasets/dfdc_train')
        for n in nome:
            print("Video: "+n)
            video = n.split(".")[0] #elimino .mp4
            for dir in os.listdir(pathFrames):
                for file in os.listdir(pathFrames + "/" + dir):
                        nomeframe = file.split("_")[0]
                        if video == nomeframe:
                            frames.append(pathFrames + "/" + dir + "/" + file)
                            print("Frames: "+pathFrames + "/" + dir + "/" + file)
            if len(frames) != 0:
                print("Add dic")
                self.dic[n] = frames
                dim = dim +1
                frames = []
            if dim == 4:
                break

    def __getitem__(self, index):
        #per ogni indice restituisce il nome del video e la sua lista di frames
        nome = list(self.dic)[index]
        frames = self.dic[nome]
        return nome, frames

    def __len__(self):
        length = len(list(self.dic))
        return length




def ListaNomi(pathFrames):
    nomi = []
    for dir in os.listdir(pathFrames):
        for file in os.listdir(pathFrames + "/" + dir):
            if file not in nomi:
                nomi.append(file)
                print("Trovato: "+file)

    print("--------------------------------------------------------------------")
    return nomi













