import numpy as np
import torch
import cv2
from torch.utils.data import Dataset


# define datasetModelClassifywithnpy class wiht npy
class datasetModelClassifywithnpy(Dataset):
    def __init__(self, images, labels, targetsize=(1, 64, 128, 128)):
        super(datasetModelClassifywithnpy).__init__()

        self.labels = labels
        self.images = images
        self.targetsize = targetsize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """
        image should normalization,and npy type
        :param index:
        :return:
        """
        imagepath = self.images[index]
        image = np.load(imagepath)
        # transpose (D,H,W,C) order to (C,D,H,W) order
        D, H, W = np.shape(image)[0], np.shape(image)[1], np.shape(image)[2]
        image = np.reshape(image, (D, H, W, 1))
        image = np.transpose(image, (3, 0, 1, 2))
        assert image.shape[0] == self.targetsize[0] and image.shape[1] == self.targetsize[1] and image.shape[2] == \
               self.targetsize[2] and image.shape[3] == self.targetsize[3]
        images_tensor = torch.as_tensor(image).float()  # transform ndarray to tensor
        # torch.set_printoptions(8)
        label = self.labels[index]
        label = int(label)
        label_tensor = torch.as_tensor(label).long()
        return {'image': images_tensor, 'label': label_tensor}


# define datasetModelClassifywithnpy class wiht npy
class datasetModelClassifywithmutilnpy(Dataset):
    def __init__(self, images, labels, targetsize=(1, 64, 128, 128)):
        super(datasetModelClassifywithnpy).__init__()

        self.labels = labels
        self.images = images
        self.targetsize = targetsize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """
        image should normalization,and npy type
        :param index:
        :return:
        """
        imagepath = self.images[index]
        imagesrc = np.load(imagepath + '/image.npy')
        mask = np.load(imagepath + '/mask.npy')
        # transpose (D,H,W,C) order to (C,D,H,W) order
        image = np.array([imagesrc, mask])
        assert image.shape[0] == self.targetsize[0] and image.shape[1] == self.targetsize[1] and image.shape[2] == \
               self.targetsize[2] and image.shape[3] == self.targetsize[3]
        images_tensor = torch.as_tensor(image).float()  # transform ndarray to tensor
        # torch.set_printoptions(8)
        label = self.labels[index]
        label = int(label)
        label_tensor = torch.as_tensor(label).long()
        return {'image': images_tensor, 'label': label_tensor}


# define datasetModelClassifywithopencv class with npy
class datasetModelClassifywithopencv(Dataset):
    def __init__(self, images, labels, targetsize=(1, 512, 512)):
        super(datasetModelClassifywithopencv).__init__()

        self.labels = labels
        self.images = images
        self.targetsize = targetsize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """
        image should normalization,and npy type
        :param index:
        :return:
        """
        imagepath = self.images[index]
        # load image
        image = cv2.imread(imagepath, 0)
        # resize image to fixed size
        image = cv2.resize(image, (self.targetsize[1], self.targetsize[2]), interpolation=cv2.INTER_LINEAR)
        # normalization image to zscore
        image = (image - image.mean()) / image.std()
        # transpose (H,W,C) order to (C,H,W) order
        H, W = np.shape(image)[0], np.shape(image)[1]
        image = np.reshape(image, (H, W, 1))
        image = np.transpose(image, (2, 0, 1))
        assert image.shape[0] == self.targetsize[0] and image.shape[1] == self.targetsize[1] and image.shape[2] == \
               self.targetsize[2]
        # convert numpy to tensor
        images_tensor = torch.as_tensor(image).float()  # transform ndarray to tensor
        # torch.set_printoptions(8)
        label = self.labels[index]
        label = int(label)
        label_tensor = torch.as_tensor(label).long()
        return {'image': images_tensor, 'label': label_tensor}


# define datasetModelSegwithnpy class wiht npy
class datasetModelSegwithnpy(Dataset):
    def __init__(self, images, labels, targetsize=(16, 64, 128, 128)):
        super(datasetModelSegwithnpy).__init__()

        self.labels = labels
        self.images = images
        self.targetsize = targetsize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """
        image should normalization,and npy type
        :param index:
        :return:
        """
        imagepath = self.images[index]
        image = np.load(imagepath)
        # transpose (D,H,W,C) order to (C,D,H,W) order
        D, H, W = np.shape(image)[0], np.shape(image)[1], np.shape(image)[2]
        image = np.reshape(image, (D, H, W, 1))
        image = np.transpose(image, (3, 0, 1, 2))
        assert image.shape[0] == self.targetsize[0] and image.shape[1] == self.targetsize[1] and image.shape[2] == \
               self.targetsize[2] and image.shape[3] == self.targetsize[3]
        images_tensor = torch.as_tensor(image).float()  # transform ndarray to tensor
        # torch.set_printoptions(8)
        labelpath = self.labels[index]
        label = np.load(labelpath)
        # transpose (D,H,W,C) order to (C,D,H,W) order
        D, H, W = np.shape(label)[0], np.shape(label)[1], np.shape(label)[2]
        label = np.reshape(label, (D, H, W)).astype(np.uint8)
        label_tensor = torch.as_tensor(label).long()
        return {'image': images_tensor, 'label': label_tensor}


# define datasetModelSegwithnpy class wiht npy
class datasetModelSegwithnpy4mutil(Dataset):
    def __init__(self, images, labels, targetsize=(16, 64, 128, 128)):
        super(datasetModelSegwithnpy).__init__()

        self.labels = labels
        self.images = images
        self.targetsize = targetsize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """
        image should normalization,and npy type
        :param index:
        :return:
        """
        imagepath = self.images[index]
        imaget1c = np.load(imagepath + '/t1c.npy')
        imaget1n = np.load(imagepath + '/t1n.npy')
        imaget2f = np.load(imagepath + '/t2f.npy')
        imaget2w = np.load(imagepath + '/t2w.npy')
        # transpose (D,H,W,C) order to (C,D,H,W) order
        image = np.array([imaget1c, imaget1n, imaget2f, imaget2w])
        assert image.shape[0] == self.targetsize[0] and image.shape[1] == self.targetsize[1] and image.shape[2] == \
               self.targetsize[2] and image.shape[3] == self.targetsize[3]
        images_tensor = torch.as_tensor(image).float()  # transform ndarray to tensor
        # torch.set_printoptions(8)
        labelpath = self.labels[index]
        label = np.load(labelpath)
        # transpose (D,H,W,C) order to (C,D,H,W) order
        D, H, W = np.shape(label)[0], np.shape(label)[1], np.shape(label)[2]
        label = np.reshape(label, (D, H, W))
        label_tensor = torch.as_tensor(label).long()
        return {'image': images_tensor, 'label': label_tensor}


# define datasetModelSegwithnpy class wiht npy
class datasetModelSegwithnpymutil(Dataset):
    def __init__(self, images, labels, targetsize=(16, 64, 128, 128)):
        super(datasetModelSegwithnpy).__init__()

        self.labels = labels
        self.images = images
        self.targetsize = targetsize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """
        image should normalization,and npy type
        :param index:
        :return:
        """
        imagepath = self.images[index]
        imaget1 = np.load(imagepath + '/t1.npy')
        imaget2 = np.load(imagepath + '/t2.npy')
        # transpose (D,H,W,C) order to (C,D,H,W) order
        image = np.array([imaget1, imaget2])
        assert image.shape[0] == self.targetsize[0] and image.shape[1] == self.targetsize[1] and image.shape[2] == \
               self.targetsize[2] and image.shape[3] == self.targetsize[3]
        images_tensor = torch.as_tensor(image).float()  # transform ndarray to tensor
        # torch.set_printoptions(8)
        labelpath = self.labels[index]
        label = np.load(labelpath)
        # transpose (D,H,W,C) order to (C,D,H,W) order
        D, H, W = np.shape(label)[0], np.shape(label)[1], np.shape(label)[2]
        label = np.reshape(label, (D, H, W))
        label_tensor = torch.as_tensor(label).long()
        return {'image': images_tensor, 'label': label_tensor}


# define datasetModelSegwithopencv class with npy
class datasetModelSegwithopencv(Dataset):
    def __init__(self, images, labels, targetsize=(1, 512, 512)):
        super(datasetModelSegwithopencv).__init__()

        self.labels = labels
        self.images = images
        self.targetsize = targetsize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """
        image should normalization,and npy type
        :param index:
        :return:
        """
        imagepath = self.images[index]
        # load image
        image = cv2.imread(imagepath, 0)
        # resize image to fixed size
        image = cv2.resize(image, (self.targetsize[1], self.targetsize[2]), interpolation=cv2.INTER_LINEAR)
        # normalization image to zscore
        image = (image - image.mean()) / image.std()
        # transpose (H,W,C) order to (C,H,W) order
        H, W = np.shape(image)[0], np.shape(image)[1]
        image = np.reshape(image, (H, W, 1))
        image = np.transpose(image, (2, 0, 1))
        assert image.shape[0] == self.targetsize[0] and image.shape[1] == self.targetsize[1] and image.shape[2] == \
               self.targetsize[2]
        # convert numpy to tensor
        images_tensor = torch.as_tensor(image).float()  # transform ndarray to tensor
        # torch.set_printoptions(8)
        labelpath = self.labels[index]
        label = cv2.imread(labelpath, 0)
        label = cv2.resize(label, (self.targetsize[1], self.targetsize[2]), interpolation=cv2.INTER_NEAREST)
        # transpose (H,W,C) order to (C,H,W) order
        label = np.reshape(label, (H, W))
        label_tensor = torch.as_tensor(label).long()
        return {'image': images_tensor, 'label': label_tensor}


# define datasetModelRegressionwithnpy class with npy
class datasetModelRegressionwithmutilnpy(Dataset):
    def __init__(self, images, labels, targetsize=(16, 64, 128, 128)):
        super(datasetModelRegressionwithmutilnpy).__init__()

        self.labels = labels
        self.images = images
        self.targetsize = targetsize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """
        image should normalization,and npy type
        :param index:
        :return:
        """
        imagepath = self.images[index]
        image = np.load(imagepath + '/' + "image.npy")
        mask = np.load(imagepath + '/' + "mask.npy")
        # normalization image to zscore
        mean = image.mean()
        std = image.std()
        eps = 1e-5
        image = (image - mean) / (std + eps)
        # transpose (D,H,W,C) order to (C,D,H,W) order
        image_ = np.array([image, mask])
        images_tensor = torch.as_tensor(image_).float()  # transform ndarray to tensor
        # torch.set_printoptions(8)
        labelpath = self.labels[index]
        label = np.load(labelpath)
        label[mask == 0] = 0
        label = (label - mean) / (std + eps)
        # transpose (D,H,W,C) order to (C,D,H,W) order
        D, H, W = np.shape(label)[0], np.shape(label)[1], np.shape(label)[2]
        label = np.reshape(label, (D, H, W))
        label_tensor = torch.as_tensor(label).float()

        mean_tensor = torch.as_tensor(mean).float()
        std_tensor = torch.as_tensor(std + eps).float()
        return {'image': images_tensor, 'label': label_tensor, 'mean': mean_tensor, 'std': std_tensor}


class datasetModelRegressionwithnpy(Dataset):
    def __init__(self, images, labels, targetsize=(1, 64, 128, 128)):
        super(datasetModelRegressionwithnpy).__init__()

        self.labels = labels
        self.images = images
        self.targetsize = targetsize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """
        image should normalization,and npy type
        :param index:
        :return:
        """
        imagepath = self.images[index]
        imagesrc = np.load(imagepath + '/image.npy')
        # normalization image to zscore
        mean = imagesrc.mean()
        std = imagesrc.std()
        eps = 1e-5
        imagesrc = (imagesrc - mean) / (std + eps)
        D, H, W = np.shape(imagesrc)[0], np.shape(imagesrc)[1], np.shape(imagesrc)[2]
        image = np.reshape(imagesrc, (D, H, W, 1))
        image = np.transpose(image, (3, 0, 1, 2))
        assert image.shape[0] == self.targetsize[0] and image.shape[1] == self.targetsize[1] and image.shape[2] == \
               self.targetsize[2] and image.shape[3] == self.targetsize[3]
        images_tensor = torch.as_tensor(image).float()  # transform ndarray to tensor

        labelpath = self.labels[index]
        label = np.load(labelpath)
        label = (label - mean) / (std + eps)
        # transpose (D,H,W,C) order to (C,D,H,W) order
        D, H, W = np.shape(label)[0], np.shape(label)[1], np.shape(label)[2]
        label = np.reshape(label, (D, H, W))
        label_tensor = torch.as_tensor(label).float()

        mean_tensor = torch.as_tensor(mean).float()
        std_tensor = torch.as_tensor(std + eps).float()
        return {'image': images_tensor, 'label': label_tensor, 'mean': mean_tensor, 'std': std_tensor}


# define datasetModelRegressionwithopencv class with npy
class datasetModelRegressionwithopencv(Dataset):
    def __init__(self, images, labels, targetsize=(1, 512, 512)):
        super(datasetModelRegressionwithopencv).__init__()

        self.labels = labels
        self.images = images
        self.targetsize = targetsize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """
        image should normalization,and npy type
        :param index:
        :return:
        """
        imagepath = self.images[index]
        # load image
        image = cv2.imread(imagepath, 0)
        # resize image to fixed size
        image = cv2.resize(image, (self.targetsize[1], self.targetsize[2]), interpolation=cv2.INTER_LINEAR)
        # normalization image to zscore
        mean = image.mean()
        std = image.std()
        eps = 1e-5
        image = (image - mean) / (std + eps)
        # transpose (H,W,C) order to (C,H,W) order
        H, W = np.shape(image)[0], np.shape(image)[1]
        image = np.reshape(image, (H, W, 1))
        image = np.transpose(image, (2, 0, 1))
        assert image.shape[0] == self.targetsize[0] and image.shape[1] == self.targetsize[1] and image.shape[2] == \
               self.targetsize[2]
        # convert numpy to tensor
        images_tensor = torch.as_tensor(image).float()  # transform ndarray to tensor
        # torch.set_printoptions(8)
        labelpath = self.labels[index]
        label = cv2.imread(labelpath, 0)
        label = cv2.resize(label, (self.targetsize[1], self.targetsize[2]), interpolation=cv2.INTER_LINEAR)
        # transpose (H,W,C) order to (C,H,W) order
        label = np.reshape(label, (H, W))
        label = (label - mean) / (std + eps)
        label_tensor = torch.as_tensor(label).float()

        mean_tensor = torch.as_tensor(mean).float()
        std_tensor = torch.as_tensor(std + eps).float()
        return {'image': images_tensor, 'label': label_tensor, 'mean': mean_tensor, 'std': std_tensor}
