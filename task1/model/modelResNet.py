import torch
import torch.nn.functional as F
from networks.ResNet2d import ResNet2d
from networks.ResNet3d import ResNet3d
from networks import initialize_weights
from .dataset import datasetModelClassifywithopencv, datasetModelClassifywithnpy, datasetModelClassifywithmutilnpy, \
    datasetModelRegressionwithnpy, datasetModelRegressionwithmutilnpy
from torch.utils.data import DataLoader
from .losses import BinaryFocalLoss, BinaryCrossEntropyLoss, MutilFocalLoss, MutilCrossEntropyLoss, MSELoss, L1Loss
import torch.optim as optim
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
import numpy as np
from dataprocess.utils import ConvertitkTrunctedValue, resize_image_itkwithsize, normalize
from tqdm import tqdm
from .metric import calc_accuracy, calc_mse, calc_abs
from .visualization import plot_result
from pathlib import Path
import time
import os
import cv2
import multiprocessing
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import SimpleITK as sitk


class MutilResNet2dModel(object):
    """
    ResNet2d with mutil class,should rewrite the dataset class and inference fucntion
    """

    def __init__(self, image_height, image_width, image_channel, numclass, batch_size, loss_name='MutilFocalLoss',
                 inference=False, model_path=None, amp=True, accum_iter=1, num_cpu=8, use_cuda=True):
        self.batch_size = batch_size
        self.loss_name = loss_name
        self.accuracyname = 'accu'
        self.image_height = image_height
        self.image_width = image_width
        self.image_channel = image_channel
        self.numclass = numclass
        self.amp = amp
        self.accum_iter = accum_iter
        self.num_cpu = num_cpu

        self.alpha = [1.] * self.numclass
        self.gamma = 2

        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = ResNet2d(self.image_channel, self.numclass)
        self.model.to(device=self.device)
        self.alpha = torch.as_tensor(self.alpha).contiguous().to(self.device)

        self.scaler = None
        self.opt = None
        self.lossFunc = None
        self.model_dir = None

        if inference:
            print(f'Loading model {model_path}')
            print(f'Using device {self.device}')
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print('Model loaded!')

    def _dataloder(self, images, labels, batch_size, shuffle=False):
        """"
        train dataset shuffle is true,validation is false
        """
        # Number of workers
        dataset = datasetModelClassifywithopencv(images, labels,
                                                 targetsize=(self.image_channel, self.image_height, self.image_width))
        # fow window num_workers is only zero,for linux num_workers can not zero
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=self.num_cpu,
                                pin_memory=True)
        return dataloader

    def _loss_function(self, lossname):
        if lossname == 'MutilCrossEntropyLoss':
            return MutilCrossEntropyLoss(self.alpha)
        if lossname == 'MutilFocalLoss':
            return MutilFocalLoss(self.alpha, self.gamma)

    def _accuracy_function(self, accuracyname, input, target):
        if accuracyname == 'accu':
            if self.numclass == 1:
                input = (input > 0.5).float()
                target = (target > 0.5).float()
                return calc_accuracy(input, target)
            else:
                input = torch.argmax(input, 1)
                return calc_accuracy(input, target)

    def _train_loop(self, train_loader, totalTrainLoss, totalTrainAccu, e):
        for batch_idx, batch in enumerate(train_loader):
            # x should tensor with shape (N,C,W,H)
            x = batch['image']
            # y should tensor with shape (N,C,W,H),
            # if have mutil label y should one-hot,if only one label,the C is one
            y = batch['label']
            # send the input to the device
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            # 利用with语句，在autocast实例的上下文范围内，进行模型的前向推理和loss计算
            with autocast(enabled=self.amp):
                # perform a forward pass and calculate the training loss and accu
                pred_logits = self.model(x)
                loss = self.lossFunc(pred_logits, y)
                # normalize loss to account for batch accumulation
                loss = loss / self.accum_iter
            # then perform backpropagation,
            if self.amp:
                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                # Backward passes under autocast are not recommended.
                # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            # then update model parameters with Gradient Accumulation
            if (batch_idx + 1) % self.accum_iter == 0:
                if self.amp:
                    # scaler.step() first unscales the gradients of the optimizer's assigned params.
                    # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                    # otherwise, optimizer.step() is skipped.
                    self.scaler.step(self.opt)
                    # Updates the scale for next iteration.
                    self.scaler.update()
                else:
                    self.opt.step()
                # zero out any previously accumulated gradients,
                # Don't perform zero_grad() on each step, only on steps where you call optimizer.step().
                self.opt.zero_grad()
            pred = F.softmax(pred_logits, dim=1)
            accu = self._accuracy_function(self.accuracyname, pred, y)
            # add the loss to the total training loss so far
            totalTrainLoss.append(loss.cpu().detach().numpy() * self.accum_iter)
            totalTrainAccu.append(accu.cpu().detach().numpy())

    def _validation_loop(self, val_loader, totalValidationLoss, totalValiadtionAccu, e):
        with torch.no_grad():
            # loop over the validation set
            for batch_idx, batch in enumerate(val_loader):
                # x should tensor with shape (N,C,W,H)
                x = batch['image']
                # y should tensor with shape (N,)
                y = batch['label']
                # send the input to the device
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                # make the predictions and calculate the validation loss
                pred_logits = self.model(x)
                loss = self.lossFunc(pred_logits, y)
                # save_images
                pred = F.softmax(pred_logits, dim=1)
                accu = self._accuracy_function(self.accuracyname, pred, y)
                totalValidationLoss.append(loss.cpu().detach().numpy())
                totalValiadtionAccu.append(accu.cpu().detach().numpy())

    def trainprocess(self, trainimage, trainmask, validationimage, validationmask, model_dir, epochs=50, lr=1e-3):
        print("[INFO] training the network...")
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        MODEL_PATH = os.path.join(model_dir, "MutilResNet2d.pth")
        summary(self.model, input_size=(1, 64, 64))
        print(self.model)
        self.model_dir = model_dir
        # 1、initialize loss function and optimizer
        self.model.apply(initialize_weights)
        self.lossFunc = self._loss_function(self.loss_name)
        self.opt = optim.AdamW(self.model.parameters(), lr=lr)
        self.scaler = GradScaler()
        # 2、load data train and validation dataset
        train_loader = self._dataloder(trainimage, trainmask, batch_size=self.batch_size, shuffle=True)
        val_loader = self._dataloder(validationimage, validationmask, batch_size=self.batch_size * 2, shuffle=True)
        # 3、initialize a dictionary to store training history
        H = {"train_loss": [], "train_accuracy": [], "valdation_loss": [], "valdation_accuracy": []}
        # 4、start loop training wiht epochs times
        startTime = time.time()
        best_validation_dsc = 0.0
        # Tensorboard summary
        writer = SummaryWriter(log_dir=model_dir)
        for e in tqdm(range(epochs)):
            # 4.1、set the model in training mode
            self.model.train()
            # 4.2、initialize the total training and validation loss
            totalTrainLoss = []
            totalTrainAccu = []
            totalValidationLoss = []
            totalValiadtionAccu = []
            # 4.3、loop over the training set
            self._train_loop(train_loader, totalTrainLoss, totalTrainAccu, e)
            # 4.4、switch off autograd and loop over the validation set
            # set the model in evaluation mode
            self.model.eval()
            self._validation_loop(val_loader, totalValidationLoss, totalValiadtionAccu, e)
            # 4.5、calculate the average training and validation loss
            avgTrainLoss = np.mean(np.stack(totalTrainLoss))
            avgValidationLoss = np.mean(np.stack(totalValidationLoss))
            avgTrainAccu = np.mean(np.stack(totalTrainAccu))
            avgValidationAccu = np.mean(np.stack(totalValiadtionAccu))
            # 4.6、update our training history
            H["train_loss"].append(avgTrainLoss)
            H["valdation_loss"].append(avgValidationLoss)
            H["train_accuracy"].append(avgTrainAccu)
            H["valdation_accuracy"].append(avgValidationAccu)
            # 4.7、print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
            print("Train loss: {:.5f}, Train accu: {:.5f}，validation loss: {:.5f}, validation accu: {:.5f}".format(
                avgTrainLoss, avgTrainAccu, avgValidationLoss, avgValidationAccu))
            # Record training loss and accuracy for each phase
            writer.add_scalar('Train/Loss', avgTrainLoss, e + 1)
            writer.add_scalar('Train/accu', avgTrainAccu, e + 1)
            writer.add_scalar('Valid/loss', avgValidationLoss, e + 1)
            writer.add_scalar('Valid/accu', avgValidationAccu, e + 1)
            writer.flush()
            # 4.8、save best_validation_dsc model params
            if avgValidationAccu > best_validation_dsc:
                best_validation_dsc = avgValidationAccu
                # best_model_params = self.model.state_dict()
                # serialize best model to disk
                torch.save(self.model.state_dict(), MODEL_PATH)
            # 4.9、clear cache memory
            self.clear_GPU_cache()
        # display the total time needed to perform the training
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
        # 5、plot the training loss
        plot_result(model_dir, H["train_loss"], H["valdation_loss"], "train_loss", "valdation_loss", "loss")
        plot_result(model_dir, H["train_accuracy"], H["valdation_accuracy"], "train_accuracy", "valdation_accuracy",
                    "accuracy")
        self.clear_GPU_cache()

    def predict(self, full_img, out_threshold=0.5):
        # 1、clear cache
        self.clear_GPU_cache()
        # 2、set model eval
        self.model.eval()
        # 3、convet numpy image to tensor
        img = torch.as_tensor(full_img).float().contiguous()
        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)
        # 4、predict result
        with torch.no_grad():
            output = self.model(img)
            if self.numclass == 1:
                probs = torch.sigmoid(output[0])
            else:
                probs = torch.softmax(output[0], dim=0)
            full_mask_np = probs.detach().cpu().squeeze().numpy()
        # 5、get numpy result
        if self.numclass == 1:
            out_mask = (full_mask_np > out_threshold)
            out_mask = out_mask * 1
        else:
            out_mask = np.argmax(full_mask_np, axis=0)
            out_mask = np.squeeze(out_mask)
        return out_mask

    def inference(self, image):
        # resize image and normalization
        imageresize = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)
        imageresize = (imageresize - imageresize.mean()) / imageresize.std()
        # transpose (H,W,C) order to (C,H,W) order
        H, W = np.shape(imageresize)[0], np.shape(imageresize)[1]
        imageresize = np.reshape(imageresize, (H, W, 1))
        imageresize = np.transpose(imageresize, (2, 0, 1))
        out_mask = self.predict(imageresize)
        return out_mask

    def clear_GPU_cache(self):
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())


class MutilResNet3dModel(object):
    """
    ResNet3d with mutil class,should rewrite the dataset class
    """

    def __init__(self, image_depth, image_height, image_width, image_channel, numclass, batch_size,
                 loss_name='MutilFocalLoss', inference=False, model_path=None, amp=True, accum_iter=1, num_cpu=4,
                 use_cuda=True):
        self.batch_size = batch_size
        self.loss_name = loss_name
        self.accuracyname = 'accu'
        self.image_height = image_height
        self.image_width = image_width
        self.image_depth = image_depth
        self.image_channel = image_channel
        self.numclass = numclass
        self.amp = amp
        self.accum_iter = accum_iter
        self.num_cpu = num_cpu

        self.alpha = [1.] * self.numclass
        self.gamma = 2

        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = ResNet3d(self.image_channel, self.numclass, 0.3)
        self.model.to(device=self.device)
        self.alpha = torch.as_tensor(self.alpha).contiguous().to(self.device)

        self.scaler = None
        self.opt = None
        self.lossFunc = None
        self.model_dir = None

        if inference:
            print(f'Loading model {model_path}')
            print(f'Using device {self.device}')
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print('Model loaded!')

    def _dataloder(self, images, labels, batch_size, shuffle=False):
        """"
        train dataset shuffle is true,validation is false
        """
        dataset = datasetModelClassifywithmutilnpy(images, labels,
                                                   targetsize=(
                                                       self.image_channel, self.image_depth, self.image_height,
                                                       self.image_width))
        # fow window num_workers is only zero,for linux num_workers can not zero
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=self.num_cpu,
                                pin_memory=True)
        return dataloader

    def _loss_function(self, lossname):
        if lossname == 'MutilCrossEntropyLoss':
            return MutilCrossEntropyLoss(self.alpha)
        if lossname == 'MutilFocalLoss':
            return MutilFocalLoss(self.alpha, self.gamma)

    def _accuracy_function(self, accuracyname, input, target):
        if accuracyname == 'accu':
            if self.numclass == 1:
                input = (input > 0.5).float()
                target = (target > 0.5).float()
                return calc_accuracy(input, target)
            else:
                input = torch.argmax(input, 1)
                return calc_accuracy(input, target)

    def _train_loop(self, train_loader, totalTrainLoss, totalTrainAccu, e):
        for batch_idx, batch in enumerate(train_loader):
            # x should tensor with shape (N,C,W,H)
            x = batch['image']
            # y should tensor with shape (N,C,W,H),
            # if have mutil label y should one-hot,if only one label,the C is one
            y = batch['label']
            # send the input to the device
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            # 利用with语句，在autocast实例的上下文范围内，进行模型的前向推理和loss计算
            with autocast(enabled=self.amp):
                # perform a forward pass and calculate the training loss and accu
                pred_logit = self.model(x)
                loss = self.lossFunc(pred_logit, y)
                # normalize loss to account for batch accumulation
                loss = loss / self.accum_iter
            # then perform backpropagation,
            if self.amp:
                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                # Backward passes under autocast are not recommended.
                # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            # then update model parameters with Gradient Accumulation
            if (batch_idx + 1) % self.accum_iter == 0:
                if self.amp:
                    # scaler.step() first unscales the gradients of the optimizer's assigned params.
                    # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                    # otherwise, optimizer.step() is skipped.
                    self.scaler.step(self.opt)
                    # Updates the scale for next iteration.
                    self.scaler.update()
                else:
                    self.opt.step()
                # zero out any previously accumulated gradients,
                # Don't perform zero_grad() on each step, only on steps where you call optimizer.step().
                self.opt.zero_grad()
            pred = F.softmax(pred_logit, dim=1)
            accu = self._accuracy_function(self.accuracyname, pred, y)
            # add the loss to the total training loss so far
            totalTrainLoss.append(loss.cpu().detach().numpy() * self.accum_iter)
            totalTrainAccu.append(accu.cpu().detach().numpy())

    def _validation_loop(self, val_loader, totalValidationLoss, totalValiadtionAccu, e):
        with torch.no_grad():
            # loop over the validation set
            for batch_idx, batch in enumerate(val_loader):
                # x should tensor with shape (N,C,W,H)
                x = batch['image']
                # y should tensor with shape (N,C,W,H)
                y = batch['label']
                # send the input to the device
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                # make the predictions and calculate the validation loss
                pred_logit = self.model(x)
                loss = self.lossFunc(pred_logit, y)
                pred = F.softmax(pred_logit, dim=1)
                accu = self._accuracy_function(self.accuracyname, pred, y)
                totalValidationLoss.append(loss.cpu().detach().numpy())
                totalValiadtionAccu.append(accu.cpu().detach().numpy())

    def trainprocess(self, trainimage, trainmask, validationimage, validationmask, model_dir, epochs=50, lr=1e-3):
        print("[INFO] training the network...")
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        MODEL_PATH = os.path.join(model_dir, "MutilResNet3d.pth")
        summary(self.model, input_size=(self.image_channel, self.image_depth, self.image_height, self.image_width))
        print(self.model)
        self.model_dir = model_dir
        # 1、initialize loss function and optimizer
        self.model.apply(initialize_weights)
        self.lossFunc = self._loss_function(self.loss_name)
        self.opt = optim.AdamW(self.model.parameters(), lr=lr)
        self.scaler = GradScaler()
        # 2、load data train and validation dataset
        train_loader = self._dataloder(trainimage, trainmask, batch_size=self.batch_size, shuffle=True)
        val_loader = self._dataloder(validationimage, validationmask, batch_size=self.batch_size * 2, shuffle=True)
        # 3、initialize a dictionary to store training history
        H = {"train_loss": [], "train_accuracy": [], "valdation_loss": [], "valdation_accuracy": []}
        # 4、start loop training wiht epochs times
        startTime = time.time()
        best_validation_dsc = 1000000
        # Tensorboard summary
        writer = SummaryWriter(log_dir=model_dir)
        for e in tqdm(range(epochs)):
            # 4.1、set the model in training mode
            self.model.train()
            # 4.2、initialize the total training and validation loss
            totalTrainLoss = []
            totalTrainAccu = []
            totalValidationLoss = []
            totalValiadtionAccu = []
            # 4.3、loop over the training set
            self._train_loop(train_loader, totalTrainLoss, totalTrainAccu, e)
            # 4.4、switch off autograd and loop over the validation set
            # set the model in evaluation mode
            self.model.eval()
            self._validation_loop(val_loader, totalValidationLoss, totalValiadtionAccu, e)
            # 4.5、calculate the average training and validation loss
            avgTrainLoss = np.mean(np.stack(totalTrainLoss))
            avgValidationLoss = np.mean(np.stack(totalValidationLoss))
            avgTrainAccu = np.mean(np.stack(totalTrainAccu))
            avgValidationAccu = np.mean(np.stack(totalValiadtionAccu))
            # 4.6、update our training history
            H["train_loss"].append(avgTrainLoss)
            H["valdation_loss"].append(avgValidationLoss)
            H["train_accuracy"].append(avgTrainAccu)
            H["valdation_accuracy"].append(avgValidationAccu)
            # 4.7、print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
            print("Train loss: {:.5f}, Train accu: {:.5f}，validation loss: {:.5f}, validation loss: {:.5f}".format(
                avgTrainLoss, avgTrainAccu, avgValidationLoss, avgValidationAccu))
            # Record training loss and accuracy for each phase
            writer.add_scalar('Train/Loss', avgTrainLoss, e + 1)
            writer.add_scalar('Train/accu', avgTrainAccu, e + 1)
            writer.add_scalar('Valid/loss', avgValidationLoss, e + 1)
            writer.add_scalar('Valid/accu', avgValidationAccu, e + 1)
            writer.flush()
            # 4.8、save best_validation_dsc model params
            if avgValidationLoss < best_validation_dsc:
                best_validation_dsc = avgValidationLoss
                # best_model_params = self.model.state_dict()
                # serialize best model to disk
                torch.save(self.model.state_dict(), MODEL_PATH)
            # 4.9、clear cache memory
            self.clear_GPU_cache()
        # display the total time needed to perform the training
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
        # 5、plot the training loss
        plot_result(model_dir, H["train_loss"], H["valdation_loss"], "train_loss", "valdation_loss", "loss")
        plot_result(model_dir, H["train_accuracy"], H["valdation_accuracy"], "train_accuracy", "valdation_accuracy",
                    "accuracy")
        self.clear_GPU_cache()

    def predict(self, full_img, out_threshold=0.5):
        # 1、clear cache
        self.clear_GPU_cache()
        # 2、set model eval
        self.model.eval()
        # 3、convet numpy image to tensor
        img = torch.as_tensor(full_img).float().contiguous()
        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)
        # 4、predict result
        with torch.no_grad():
            output = self.model(img)
            if self.numclass == 1:
                probs = torch.sigmoid(output[0])
            else:
                probs = torch.softmax(output[0], dim=0)
            full_mask_np = probs.detach().cpu().squeeze().numpy()
        # 5、get numpy result
        if self.numclass == 1:
            out_mask = (full_mask_np > out_threshold)
            out_mask = out_mask * 1
        else:
            out_mask = np.argmax(full_mask_np, axis=0)
            out_mask = np.squeeze(out_mask)
        return out_mask, full_mask_np

    def inference(self, imageresize, maskresize):
        # resize image and normalization,should rewrite
        # newSize = (self.image_width, self.image_height, self.image_depth)
        # _, resizesitkimage = resize_image_itkwithsize(sitkimage, newSize, sitkimage.GetSize(), sitk.sitkLinear)
        # _, resizesitkmask = resize_image_itkwithsize(sitkmask, newSize, sitkmask.GetSize(), sitk.sitkNearestNeighbor)
        # resizesitkimage = ConvertitkTrunctedValue(resizesitkimage, 800, -1000, 'meanstd')
        # imageresize = sitk.GetArrayFromImage(resizesitkimage)
        # maskresize = sitk.GetArrayFromImage(resizesitkmask)
        imageresize = normalize(imageresize)
        # transpose (D,H,W,C) order to (C,D,H,W) order
        imageresize = np.array([imageresize, maskresize])
        out_mask, out_mask_prob = self.predict(imageresize)
        # resize mask to src image size,shou rewrite
        return out_mask, out_mask_prob

    def inferencevalid(self, image, mask):
        imageresize = np.array([image, mask])
        out_mask, out_mask_prob = self.predict(imageresize)
        # resize mask to src image size,shou rewrite
        return out_mask, out_mask_prob

    def clear_GPU_cache(self):
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())


class RegressionMutilResNet3dModel(object):
    """
    ResNet3d with mutil class,should rewrite the dataset class
    """

    def __init__(self, image_depth, image_height, image_width, image_channel, numclass, batch_size,
                 loss_name='MSE', inference=False, model_path=None, amp=True, accum_iter=1, num_cpu=4,
                 use_cuda=True):
        self.batch_size = batch_size
        self.loss_name = loss_name
        self.accuracyname = 'accu'
        self.image_height = image_height
        self.image_width = image_width
        self.image_depth = image_depth
        self.image_channel = image_channel
        self.numclass = numclass
        self.amp = amp
        self.accum_iter = accum_iter
        self.num_cpu = num_cpu

        self.alpha = [1.] * self.numclass
        self.gamma = 2

        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = ResNet3d(self.image_channel, self.numclass, 0.3)
        self.model.to(device=self.device)
        self.alpha = torch.as_tensor(self.alpha).contiguous().to(self.device)

        self.scaler = None
        self.opt = None
        self.lossFunc = None
        self.model_dir = None

        if inference:
            print(f'Loading model {model_path}')
            print(f'Using device {self.device}')
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print('Model loaded!')

    def _dataloder(self, images, labels, batch_size, shuffle=False):
        """"
        train dataset shuffle is true,validation is false
        """
        dataset = datasetModelRegressionwithmutilnpy(images, labels,
                                                     targetsize=(
                                                         self.image_channel, self.image_depth, self.image_height,
                                                         self.image_width))
        # fow window num_workers is only zero,for linux num_workers can not zero
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=self.num_cpu,
                                pin_memory=True)
        return dataloader

    def _loss_function(self, lossname):
        if lossname == 'L1':
            return L1Loss()
        if lossname == 'MSE':
            return MSELoss()

    def _accuracy_function(self, accuracyname, input, target):
        if accuracyname == 'accu':
            if self.numclass == 1:
                return calc_abs(input, target)

    def _train_loop(self, train_loader, totalTrainLoss, totalTrainAccu, e):
        for batch_idx, batch in enumerate(train_loader):
            # x should tensor with shape (N,C,W,H)
            x = batch['image']
            # y should tensor with shape (N,C,W,H),
            # if have mutil label y should one-hot,if only one label,the C is one
            y = batch['label']
            # send the input to the device
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            # 利用with语句，在autocast实例的上下文范围内，进行模型的前向推理和loss计算
            with autocast(enabled=self.amp):
                # perform a forward pass and calculate the training loss and accu
                pred_logit = self.model(x)
                loss = self.lossFunc(pred_logit, y)
                # normalize loss to account for batch accumulation
                loss = loss / self.accum_iter
            # then perform backpropagation,
            if self.amp:
                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                # Backward passes under autocast are not recommended.
                # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            # then update model parameters with Gradient Accumulation
            if (batch_idx + 1) % self.accum_iter == 0:
                if self.amp:
                    # scaler.step() first unscales the gradients of the optimizer's assigned params.
                    # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                    # otherwise, optimizer.step() is skipped.
                    self.scaler.step(self.opt)
                    # Updates the scale for next iteration.
                    self.scaler.update()
                else:
                    self.opt.step()
                # zero out any previously accumulated gradients,
                # Don't perform zero_grad() on each step, only on steps where you call optimizer.step().
                self.opt.zero_grad()
            accu = self._accuracy_function(self.accuracyname, pred_logit, y)
            # add the loss to the total training loss so far
            totalTrainLoss.append(loss.cpu().detach().numpy() * self.accum_iter)
            totalTrainAccu.append(accu.cpu().detach().numpy())

    def _validation_loop(self, val_loader, totalValidationLoss, totalValiadtionAccu, e):
        with torch.no_grad():
            # loop over the validation set
            for batch_idx, batch in enumerate(val_loader):
                # x should tensor with shape (N,C,W,H)
                x = batch['image']
                # y should tensor with shape (N,C,W,H)
                y = batch['label']
                # send the input to the device
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                # make the predictions and calculate the validation loss
                pred_logit = self.model(x)
                loss = self.lossFunc(pred_logit, y)
                accu = self._accuracy_function(self.accuracyname, pred_logit, y)
                totalValidationLoss.append(loss.cpu().detach().numpy())
                totalValiadtionAccu.append(accu.cpu().detach().numpy())

    def trainprocess(self, trainimage, trainmask, validationimage, validationmask, model_dir, epochs=50, lr=1e-3):
        print("[INFO] training the network...")
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        MODEL_PATH = os.path.join(model_dir, "MutilResNet3d.pth")
        print(self.model)
        self.model_dir = model_dir
        # 1、initialize loss function and optimizer
        self.model.apply(initialize_weights)
        self.lossFunc = self._loss_function(self.loss_name)
        self.opt = optim.AdamW(self.model.parameters(), lr=lr)
        self.scaler = GradScaler()
        # 2、load data train and validation dataset
        train_loader = self._dataloder(trainimage, trainmask, batch_size=self.batch_size, shuffle=True)
        val_loader = self._dataloder(validationimage, validationmask, batch_size=self.batch_size, shuffle=True)
        # 3、initialize a dictionary to store training history
        H = {"train_loss": [], "train_accuracy": [], "valdation_loss": [], "valdation_accuracy": []}
        # 4、start loop training wiht epochs times
        startTime = time.time()
        best_validation_dsc = 1000000
        # Tensorboard summary
        writer = SummaryWriter(log_dir=model_dir)
        for e in tqdm(range(epochs)):
            # 4.1、set the model in training mode
            self.model.train()
            # 4.2、initialize the total training and validation loss
            totalTrainLoss = []
            totalTrainAccu = []
            totalValidationLoss = []
            totalValiadtionAccu = []
            # 4.3、loop over the training set
            self._train_loop(train_loader, totalTrainLoss, totalTrainAccu, e)
            # 4.4、switch off autograd and loop over the validation set
            # set the model in evaluation mode
            self.model.eval()
            self._validation_loop(val_loader, totalValidationLoss, totalValiadtionAccu, e)
            # 4.5、calculate the average training and validation loss
            avgTrainLoss = np.mean(np.stack(totalTrainLoss))
            avgValidationLoss = np.mean(np.stack(totalValidationLoss))
            avgTrainAccu = np.mean(np.stack(totalTrainAccu))
            avgValidationAccu = np.mean(np.stack(totalValiadtionAccu))
            # 4.6、update our training history
            H["train_loss"].append(avgTrainLoss)
            H["valdation_loss"].append(avgValidationLoss)
            H["train_accuracy"].append(avgTrainAccu)
            H["valdation_accuracy"].append(avgValidationAccu)
            # 4.7、print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
            print("Train loss: {:.5f}, Train accu: {:.5f}，validation loss: {:.5f}, validation loss: {:.5f}".format(
                avgTrainLoss, avgTrainAccu, avgValidationLoss, avgValidationAccu))
            # Record training loss and accuracy for each phase
            writer.add_scalar('Train/Loss', avgTrainLoss, e + 1)
            writer.add_scalar('Train/accu', avgTrainAccu, e + 1)
            writer.add_scalar('Valid/loss', avgValidationLoss, e + 1)
            writer.add_scalar('Valid/accu', avgValidationAccu, e + 1)
            writer.flush()
            # 4.8、save best_validation_dsc model params
            if avgValidationLoss < best_validation_dsc:
                best_validation_dsc = avgValidationLoss
                # best_model_params = self.model.state_dict()
                # serialize best model to disk
                torch.save(self.model.state_dict(), MODEL_PATH)
            # 4.9、clear cache memory
            self.clear_GPU_cache()
        # display the total time needed to perform the training
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
        # 5、plot the training loss
        plot_result(model_dir, H["train_loss"], H["valdation_loss"], "train_loss", "valdation_loss", "loss")
        plot_result(model_dir, H["train_accuracy"], H["valdation_accuracy"], "train_accuracy", "valdation_accuracy",
                    "accuracy")
        self.clear_GPU_cache()

    def predict(self, full_img, out_threshold=0.5):
        # 1、clear cache
        self.clear_GPU_cache()
        # 2、set model eval
        self.model.eval()
        # 3、convet numpy image to tensor
        img = torch.as_tensor(full_img).float().contiguous()
        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)
        # 4、predict result
        with torch.no_grad():
            output = self.model(img)
            full_mask_np = output.detach().cpu().squeeze().numpy()
        # 5、get numpy result
        return full_mask_np

    def inference(self, imageresize, maskresize):
        # resize image and normalization,should rewrite
        # newSize = (self.image_width, self.image_height, self.image_depth)
        # _, resizesitkimage = resize_image_itkwithsize(sitkimage, newSize, sitkimage.GetSize(), sitk.sitkLinear)
        # _, resizesitkmask = resize_image_itkwithsize(sitkmask, newSize, sitkmask.GetSize(), sitk.sitkNearestNeighbor)
        # resizesitkimage = ConvertitkTrunctedValue(resizesitkimage, 800, -1000, 'meanstd')
        # imageresize = sitk.GetArrayFromImage(resizesitkimage)
        # maskresize = sitk.GetArrayFromImage(resizesitkmask)
        # transpose (D,H,W,C) order to (C,D,H,W) order
        imageresize = np.array([imageresize, maskresize])
        out_mask = self.predict(imageresize)
        # resize mask to src image size,shou rewrite
        return out_mask

    def inferencevalid(self, image, mask):
        imageresize = np.array([image, mask])
        out_mask = self.predict(imageresize)
        # resize mask to src image size,shou rewrite
        return out_mask

    def clear_GPU_cache(self):
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())
