import torch
from networks.VNet2d import VNet2d
from networks.VNet3d import VNet3d
from networks.VNet3dthin import VNet3dthin
from networks import initialize_weights
from .dataset import datasetModelSegwithopencv, datasetModelSegwithnpy, datasetModelRegressionwithnpy, \
    datasetModelRegressionwithmutilnpy, datasetModelSegwithnpy4mutil, datasetModelSegwithnpymutil
from torch.utils.data import DataLoader
from .losses import BinaryDiceLoss, BinaryFocalLoss, BinaryCrossEntropyLoss, BinaryCrossEntropyDiceLoss, \
    MutilDiceLoss, MutilFocalLoss, MutilCrossEntropyLoss, MutilCrossEntropyDiceLoss, MSELoss, L1Loss, SSIMLoss, \
    L1SSIMLoss
import torch.optim as optim
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
import numpy as np
from tqdm import tqdm
from .metric import dice_coeff, iou_coeff, multiclass_dice_coeff, multiclass_iou_coeff, multiclass_dice_coeffv2, \
    calc_ssim, calc_psnr
from .visualization import plot_result, save_images2d, save_images3d, save_images3dregression
from pathlib import Path
import time
import os
import cv2
from dataprocess.utils import resize_image_itkwithsize, ConvertitkTrunctedValue, normalize, resize_image_itk
import SimpleITK as sitk
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter


class BinaryVNet2dModel(object):
    """
    Vnet2d with binary class,should rewrite the dataset class and inference fucntion
    """

    def __init__(self, image_height, image_width, image_channel, numclass, batch_size, loss_name='BinaryDiceLoss',
                 inference=False, model_path=None, amp=True, accum_gradient_iter=1, num_cpu=4, use_cuda=True):
        self.batch_size = batch_size
        self.loss_name = loss_name
        self.accuracyname = 'dice'
        self.image_height = image_height
        self.image_width = image_width
        self.image_channel = image_channel
        self.numclass = numclass
        self.amp = amp
        self.accum_iter = accum_gradient_iter
        self.num_cpu = num_cpu

        self.alpha = 0.25
        self.gamma = 2

        self.lossFunc = None
        self.model_dir = None
        self.showpixelvalue = None
        self.scaler = None
        self.opt = None

        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = VNet2d(self.image_channel, self.numclass)
        self.model.to(device=self.device)

        if inference:
            print(f'Loading model {model_path}')
            print(f'Using device {self.device}')
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print('Model loaded!')

    def _dataloder(self, images, labels, batch_size, shuffle=False):
        """"
        train dataset shuffle is true,validation is false
        """
        dataset = datasetModelSegwithopencv(images, labels,
                                            targetsize=(self.image_channel, self.image_height, self.image_width))
        # fow window num_workers is only zero,for linux num_workers can not zero
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=self.num_cpu,
                                pin_memory=True)
        return dataloader

    def _loss_function(self, lossname):
        if lossname == 'BinaryCrossEntropyLoss':
            return BinaryCrossEntropyLoss()
        if lossname == 'BinaryDiceLoss':
            return BinaryDiceLoss()
        if lossname == 'BinaryCrossEntropyDiceLoss':
            return BinaryCrossEntropyDiceLoss()
        if lossname == 'BinaryFocalLoss':
            return BinaryFocalLoss()

    def _accuracy_function(self, accuracyname, input, target):
        if accuracyname == 'dice':
            if self.numclass == 1:
                return dice_coeff(input, target)
            else:
                return multiclass_dice_coeff(input, target)
        if accuracyname == 'iou':
            if self.numclass == 1:
                return iou_coeff(input, target)
            else:
                return multiclass_iou_coeff(input, target)

    def _train_loop(self, train_loader, totalTrainLoss, totalTrainAccu, trainshow, e):
        self.clear_GPU_cache()
        for batch_idx, batch in enumerate(train_loader):
            # x should tensor with shape (N,C,W,H)
            x = batch['image']
            # y should tensor with shape (N,C,W,H),
            # if have mutil label y should one-hot,if only one label,the C is one
            y = batch['label']
            y[y != 0] = 1
            # send the input to the device
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            # 利用with语句，在autocast实例的上下文范围内，进行模型的前向推理和loss计算
            with autocast(enabled=self.amp):
                # perform a forward pass and calculate the training loss and accu
                pred_logit, pred = self.model(x)
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
            accu = self._accuracy_function(self.accuracyname, pred, y)
            if trainshow:
                # save_images
                savepath = self.model_dir + '/' + str(e + 1) + "_Train_EPOCH_"
                save_images2d(pred[0], y[0], savepath, pixelvalue=self.showpixelvalue)
                trainshow = False
            # add the loss to the total training loss so far
            totalTrainLoss.append(loss.cpu().detach().numpy() * self.accum_iter)
            totalTrainAccu.append(accu.cpu().detach().numpy())

    def _validation_loop(self, val_loader, totalValidationLoss, totalValiadtionAccu, trainshow, e):
        self.clear_GPU_cache()
        with torch.no_grad():
            # loop over the validation set
            for batch_idx, batch in enumerate(val_loader):
                # x should tensor with shape (N,C,W,H)
                x = batch['image']
                # y should tensor with shape (N,C,W,H)
                y = batch['label']
                y[y != 0] = 1
                # send the input to the device
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                # make the predictions and calculate the validation loss
                pred_logit, pred = self.model(x)
                loss = self.lossFunc(pred_logit, y)
                accu = self._accuracy_function(self.accuracyname, pred, y)
                if trainshow:
                    # save_images
                    savepath = self.model_dir + '/' + str(e + 1) + "_Val_EPOCH_"
                    save_images2d(pred[0], y[0], savepath, pixelvalue=self.showpixelvalue)
                    trainshow = False
                totalValidationLoss.append(loss.cpu().detach().numpy())
                totalValiadtionAccu.append(accu.cpu().detach().numpy())

    def trainprocess(self, trainimage, trainmask, validationimage, validationmask, model_dir, epochs=50, lr=1e-3):
        print("[INFO] training the network...")
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        MODEL_PATH = os.path.join(model_dir, "BinaryVNet2dModel.pth")
        MODEL_PATH_best = os.path.join(model_dir, "BinaryVNet2dModelbest.pth")
        print(self.model)
        self.model_dir = model_dir
        self.showpixelvalue = 255.
        if self.numclass > 1:
            self.showpixelvalue = self.showpixelvalue // (self.numclass - 1)
        # 1、initialize loss function and optimizer
        self.model.apply(initialize_weights)
        self.lossFunc = self._loss_function(self.loss_name)
        self.opt = optim.AdamW(self.model.parameters(), lr=lr)
        self.scaler = GradScaler()
        # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=2, verbose=True)
        # 2、load data train and validation dataset
        train_loader = self._dataloder(trainimage, trainmask, self.batch_size, True)
        val_loader = self._dataloder(validationimage, validationmask, self.batch_size * 2, True)
        # 3、initialize a dictionary to store training history
        H = {"train_loss": [], "train_accuracy": [], "valdation_loss": [], "valdation_accuracy": []}
        # 4、start loop training wiht epochs times
        startTime = time.time()
        best_validation_dsc = 50000000
        best_e = 0
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
            trainshow = True
            self._train_loop(train_loader, totalTrainLoss, totalTrainAccu, trainshow, e)
            # 4.4、switch off autograd and loop over the validation set
            # set the model in evaluation mode
            self.model.eval()
            trainshow = True
            self._validation_loop(val_loader, totalValidationLoss, totalValiadtionAccu, trainshow, e)
            # 4.5、calculate the average training and validation loss
            avgTrainLoss = np.mean(np.stack(totalTrainLoss))
            avgValidationLoss = np.mean(np.stack(totalValidationLoss))
            avgTrainAccu = np.mean(np.stack(totalTrainAccu))
            avgValidationAccu = np.mean(np.stack(totalValiadtionAccu))
            # lr_scheduler.step(avgValidationLoss)
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
            if avgValidationLoss < best_validation_dsc:
                best_validation_dsc = avgValidationLoss
                # best_model_params = self.model.state_dict()
                # serialize best model to disk
                torch.save(self.model.state_dict(), MODEL_PATH_best)
                best_e = e
            torch.save(self.model.state_dict(), MODEL_PATH)
            # 4.9、clear cache memory
            self.clear_GPU_cache()
            # 4.10、early stopping
            if abs(best_e - e) > epochs // 3:
                break
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
            _, output = self.model(img)
            probs = output[0]
            full_mask_np = probs.detach().cpu().squeeze().numpy()
        # 5、get numpy result
        if self.numclass == 1:
            out_mask = (full_mask_np > out_threshold)
            out_mask = out_mask * 1
        else:
            out_mask = np.argmax(full_mask_np, axis=0)
            out_mask = np.squeeze(out_mask)
        return out_mask.astype(np.uint8)

    def inference(self, image):
        # resize image and normalization
        imageresize = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)
        imageresize = (imageresize - imageresize.mean()) / imageresize.std()
        # transpose (H,W,C) order to (C,H,W) order
        H, W = np.shape(imageresize)[0], np.shape(imageresize)[1]
        imageresize = np.reshape(imageresize, (H, W, self.image_channel))
        imageresize = np.transpose(imageresize, (2, 0, 1))
        out_mask = self.predict(imageresize)
        # resize mask to src image size
        out_mask = cv2.resize(out_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        return out_mask

    def clear_GPU_cache(self):
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())


class MutilVNet2dModel(object):
    """
    Vnet2d with mutil class,should rewrite the dataset class and inference fucntion
    """

    def __init__(self, image_height, image_width, image_channel, numclass, batch_size, loss_name='MutilFocalLoss',
                 inference=False, model_path=None, amp=True, accum_gradient_iter=1, num_cpu=2, use_cuda=True):
        self.batch_size = batch_size
        self.loss_name = loss_name
        self.accuracyname = 'dice'
        self.image_height = image_height
        self.image_width = image_width
        self.image_channel = image_channel
        self.numclass = numclass
        self.amp = amp
        self.accum_iter = accum_gradient_iter
        self.num_cpu = num_cpu

        self.alpha = [1.] * self.numclass
        self.gamma = 2

        self.lossFunc = None
        self.model_dir = None
        self.showpixelvalue = None
        self.scaler = None
        self.opt = None

        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = VNet2d(self.image_channel, self.numclass)
        self.model.to(device=self.device)
        self.alpha = torch.as_tensor(self.alpha).contiguous().to(self.device)
        if inference:
            print(f'Loading model {model_path}')
            print(f'Using device {self.device}')
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print('Model loaded!')

    def _dataloder(self, images, labels, batch_size, shuffle=False):
        """"
        train dataset shuffle is true,validation is false
        """
        dataset = datasetModelSegwithopencv(images, labels,
                                            targetsize=(self.image_channel, self.image_height, self.image_width))
        # fow window num_workers is only zero,for linux num_workers can not zero
        # num_cpu = multiprocessing.cpu_count()
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=self.num_cpu,
                                pin_memory=True)
        return dataloader

    def _loss_function(self, lossname):
        if lossname == 'MutilCrossEntropyLoss':
            return MutilCrossEntropyLoss(alpha=self.alpha)
        if lossname == 'MutilDiceLoss':
            return MutilDiceLoss(alpha=self.alpha)
        if lossname == 'MutilFocalLoss':
            return MutilFocalLoss(alpha=self.alpha, gamma=self.gamma)
        if lossname == 'MutilCrossEntropyDiceLoss':
            return MutilCrossEntropyDiceLoss(alpha=self.alpha)

    def _accuracy_function(self, accuracyname, input, target):
        if accuracyname == 'dice':
            if self.numclass == 1:
                return dice_coeff(input, target)
            else:
                return multiclass_dice_coeffv2(input, target)
        if accuracyname == 'iou':
            if self.numclass == 1:
                return iou_coeff(input, target)
            else:
                return multiclass_iou_coeff(input, target)

    def _train_loop(self, train_loader, totalTrainLoss, totalTrainAccu, trainshow, e):
        self.clear_GPU_cache()
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
                pred_logit, pred = self.model(x)
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
            if trainshow:
                savepath = self.model_dir + '/' + str(e + 1) + "_Train_EPOCH_"
                save_images2d(pred[0], y[0], savepath, pixelvalue=self.showpixelvalue)
                trainshow = False
            # add the loss to the total training loss so far
            totalTrainLoss.append(loss.cpu().detach().numpy() * self.accum_iter)
            totalTrainAccu.append(accu.cpu().detach().numpy())

    def _validation_loop(self, val_loader, totalValidationLoss, totalValiadtionAccu, trainshow, e):
        self.clear_GPU_cache()
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
                pred_logit, pred = self.model(x)
                loss = self.lossFunc(pred_logit, y)
                accu = self._accuracy_function(self.accuracyname, pred_logit, y)
                if trainshow:
                    # save_images
                    savepath = self.model_dir + '/' + str(e + 1) + "_Val_EPOCH_"
                    save_images2d(pred[0], y[0], savepath, pixelvalue=self.showpixelvalue)
                    trainshow = False
                totalValidationLoss.append(loss.cpu().detach().numpy())
                totalValiadtionAccu.append(accu.cpu().detach().numpy())

    def trainprocess(self, trainimage, trainmask, validationimage, validationmask, model_dir, epochs=50, lr=1e-3):
        print("[INFO] training the network...")
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        MODEL_PATH = os.path.join(model_dir, "MutilVNet2dModel.pth")
        MODEL_PATH_best = os.path.join(model_dir, "MutilVNet2dModelbest.pth")
        print(self.model)
        self.model_dir = model_dir
        self.showpixelvalue = 255.
        if self.numclass > 1:
            self.showpixelvalue = self.showpixelvalue // (self.numclass - 1)
        # 1、initialize net weight init loss function and optimizer
        self.model.apply(initialize_weights)
        self.lossFunc = self._loss_function(self.loss_name)
        self.opt = optim.AdamW(self.model.parameters(), lr=lr)
        self.scaler = GradScaler()
        # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=2, verbose=True)
        # 2、load data train and validation dataset
        train_loader = self._dataloder(trainimage, trainmask, self.batch_size, True)
        val_loader = self._dataloder(validationimage, validationmask, self.batch_size, True)
        # 3、initialize a dictionary to store training history
        H = {"train_loss": [], "train_accuracy": [], "valdation_loss": [], "valdation_accuracy": []}
        # 4、start loop training wiht epochs times
        startTime = time.time()
        best_validation_dsc = 1000000000
        best_epoch = 0
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
            trainshow = True
            self._train_loop(train_loader, totalTrainLoss, totalTrainAccu, trainshow, e)
            # 4.4、switch off autograd and loop over the validation set
            # set the model in evaluation mode
            self.model.eval()
            trainshow = True
            self._validation_loop(val_loader, totalValidationLoss, totalValiadtionAccu, trainshow, e)
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
            if avgValidationLoss < best_validation_dsc:
                best_validation_dsc = avgValidationLoss
                # best_model_params = self.model.state_dict()
                # serialize best model to disk
                torch.save(self.model.state_dict(), MODEL_PATH_best)
                best_epoch = e
            torch.save(self.model.state_dict(), MODEL_PATH)
            # 4.9、clear cache memory
            self.clear_GPU_cache()
            # 4.10、early stopping
            if abs(best_epoch - e) > epochs // 3:
                break
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
            _, output = self.model(img)
            probs = output[0]
            full_mask_np = probs.detach().cpu().squeeze().numpy()
        # 5、get numpy result
        if self.numclass == 1:
            out_mask = full_mask_np * 1
        else:
            out_mask = np.squeeze(full_mask_np)
        return out_mask.astype(np.uint8)

    def inference(self, image):
        # resize image and normalization
        imageresize = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)
        imageresize = (imageresize - imageresize.mean()) / imageresize.std()
        # transpose (H,W,C) order to (C,H,W) order
        H, W = np.shape(imageresize)[0], np.shape(imageresize)[1]
        imageresize = np.reshape(imageresize, (H, W, 1))
        imageresize = np.transpose(imageresize, (2, 0, 1))
        out_mask = self.predict(imageresize)
        # resize mask to src image size
        out_mask = cv2.resize(out_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        return out_mask

    def clear_GPU_cache(self):
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())


class BinaryVNet3dModel(object):
    """
    Vnet3d with binary class,should rewrite the dataset class and inference fucntion
    """

    def __init__(self, image_depth, image_height, image_width, image_channel, numclass, batch_size,
                 loss_name='BinaryDiceLoss', inference=False, model_path=None, amp=True, accum_gradient_iter=1,
                 num_cpu=4, use_cuda=True):
        self.batch_size = batch_size
        self.loss_name = loss_name
        self.accuracyname = 'dice'
        self.image_height = image_height
        self.image_width = image_width
        self.image_depth = image_depth
        self.image_channel = image_channel
        self.numclass = numclass
        self.amp = amp
        self.accum_iter = accum_gradient_iter
        self.num_cpu = num_cpu

        self.alpha = 0.25
        self.gamma = 2

        self.lossFunc = None
        self.showwind = None
        self.model_dir = None
        self.showpixelvalue = None
        self.scaler = None
        self.opt = None

        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.model = VNet3d(self.image_channel, self.numclass)
        self.model.to(device=self.device)

        if not inference:
            self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))

        if inference:
            print(f'Loading model {model_path}')
            print(f'Using device {self.device}')
            self.model.load_state_dict(
                {k.replace('module.', ''): v for k, v in torch.load(model_path, map_location=self.device).items()})
            print('Model loaded!')

    def _dataloder(self, images, labels, batch_size, shuffle=False):
        """"
        train dataset shuffle is true,validation is false
        """
        dataset = datasetModelSegwithnpy(images, labels,
                                         targetsize=(
                                             self.image_channel, self.image_depth, self.image_height,
                                             self.image_width))
        # fow window num_workers is only zero,for linux num_workers can not zero
        # num_cpu = multiprocessing.cpu_count()
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=self.num_cpu,
                                pin_memory=True)
        return dataloader

    def _loss_function(self, lossname):
        if lossname == 'BinaryCrossEntropyLoss':
            return BinaryCrossEntropyLoss()
        if lossname == 'BinaryDiceLoss':
            return BinaryDiceLoss()
        if lossname == 'BinaryCrossEntropyDiceLoss':
            return BinaryCrossEntropyDiceLoss()
        if lossname == 'BinaryFocalLoss':
            return BinaryFocalLoss()

    def _accuracy_function(self, accuracyname, input, target):
        if accuracyname == 'dice':
            if self.numclass == 1:
                return dice_coeff(input, target)
            else:
                return multiclass_dice_coeff(input, target)
        if accuracyname == 'iou':
            if self.numclass == 1:
                return iou_coeff(input, target)
            else:
                return multiclass_iou_coeff(input, target)

    def _train_loop(self, train_loader, totalTrainLoss, totalTrainAccu, trainshow, e):
        self.clear_GPU_cache()
        for batch_idx, batch in enumerate(train_loader):
            # x should tensor with shape (N,C,D,W,H)
            x = batch['image']
            # y should tensor with shape (N,C,D,W,H),
            # if have mutil label y should one-hot,if only one label,the C is one
            y = batch['label']
            y[y != 0] = 1
            # send the input to the device
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            # perform a forward pass and calculate the training loss and accu
            # 利用with语句，在autocast实例的上下文范围内，进行模型的前向推理和loss计算
            with autocast(enabled=self.amp):
                # perform a forward pass and calculate the training loss and accu
                pred_logit, pred = self.model(x)
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
            accu = self._accuracy_function(self.accuracyname, pred, y)
            if trainshow:
                savepath = self.model_dir + '/' + str(e + 1) + "_Train_EPOCH_"
                save_images3d(pred[0], y[0], self.showwind, savepath, pixelvalue=self.showpixelvalue)
                trainshow = False
            # add the loss to the total training loss so far
            totalTrainLoss.append(loss.cpu().detach().numpy() * self.accum_iter)
            totalTrainAccu.append(accu.cpu().detach().numpy())

    def _validation_loop(self, val_loader, totalValidationLoss, totalValiadtionAccu, trainshow, e):
        self.clear_GPU_cache()
        with torch.no_grad():
            # loop over the validation set
            for batch_idx, batch in enumerate(val_loader):
                # x should tensor with shape (N,C,W,H)
                x = batch['image']
                # y should tensor with shape (N,C,W,H)
                y = batch['label']
                y[y != 0] = 1
                # send the input to the device
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                # make the predictions and calculate the validation loss
                pred_logit, pred = self.model(x)
                loss = self.lossFunc(pred_logit, y)
                accu = self._accuracy_function(self.accuracyname, pred, y)
                if trainshow:
                    # save_images
                    savepath = self.model_dir + '/' + str(e + 1) + "_Val_EPOCH_"
                    save_images3d(pred[0], y[0], self.showwind, savepath, pixelvalue=self.showpixelvalue)
                    trainshow = False
                totalValidationLoss.append(loss.cpu().detach().numpy())
                totalValiadtionAccu.append(accu.cpu().detach().numpy())

    def trainprocess(self, trainimage, trainmask, validationimage, validationmask, model_dir, epochs=50, lr=1e-3,
                     showwind=[8, 8]):
        print("[INFO] training the network...")
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        MODEL_PATH = os.path.join(model_dir, "BinaryVNet3d.pth")
        MODEL_PATH_best = os.path.join(model_dir, "BinaryVNet3dbest.pth")
        print(self.model)
        self.model_dir = model_dir
        self.showpixelvalue = 255.
        self.showwind = showwind
        if self.numclass > 1:
            self.showpixelvalue = self.showpixelvalue // (self.numclass - 1)
        # 1、initialize loss function and optimizer
        self.model.apply(initialize_weights)
        self.lossFunc = self._loss_function(self.loss_name)
        self.opt = optim.AdamW(self.model.parameters(), lr=lr)
        self.scaler = GradScaler()
        # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=2, verbose=True)
        # 2、load data train and validation dataset
        train_loader = self._dataloder(trainimage, trainmask, self.batch_size, True)
        val_loader = self._dataloder(validationimage, validationmask, self.batch_size, True)
        # 3、initialize a dictionary to store training history
        H = {"train_loss": [], "train_accuracy": [], "valdation_loss": [], "valdation_accuracy": []}
        # 4、start loop training wiht epochs times
        startTime = time.time()
        best_validation_dsc = 1000000
        best_e = 0
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
            trainshow = True
            self._train_loop(train_loader, totalTrainLoss, totalTrainAccu, trainshow, e)
            # 4.4、switch off autograd and loop over the validation set
            # set the model in evaluation mode
            self.model.eval()
            trainshow = True
            self._validation_loop(val_loader, totalValidationLoss, totalValiadtionAccu, trainshow, e)
            # 4.5、calculate the average training and validation loss
            avgTrainLoss = np.mean(np.stack(totalTrainLoss))
            avgValidationLoss = np.mean(np.stack(totalValidationLoss))
            avgTrainAccu = np.mean(np.stack(totalTrainAccu))
            avgValidationAccu = np.mean(np.stack(totalValiadtionAccu))
            # lr_scheduler.step(avgValidationLoss)
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
            if avgValidationLoss < best_validation_dsc:
                best_validation_dsc = avgValidationLoss
                # best_model_params = self.model.state_dict()
                # serialize best model to disk
                torch.save(self.model.state_dict(), MODEL_PATH_best)
                best_e = e
            torch.save(self.model.state_dict(), MODEL_PATH)
            # 4.9、clear cache memory
            self.clear_GPU_cache()
            # 4.10、early stopping
            if abs(best_e - e) > epochs // 3:
                break
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
            _, output = self.model(img)
            probs = output[0]
            full_mask_np = probs.detach().cpu().squeeze().numpy()
        # 5、get numpy result
        if self.numclass == 1:
            out_mask = (full_mask_np > out_threshold)
            out_mask = out_mask * 1.
        else:
            out_mask = np.argmax(full_mask_np, axis=0)
            out_mask = np.squeeze(out_mask)
        return out_mask.astype(np.uint8)

    def inference(self, imagesitk, newSize=(96, 96, 96)):
        # resize image and normalization,should rewrite
        _, resizeimagesitk = resize_image_itkwithsize(imagesitk, newSize, imagesitk.GetSize(), sitk.sitkLinear)
        resizeimagesitk = ConvertitkTrunctedValue(resizeimagesitk, 2000, 0, 'meanstd')
        imageresize = sitk.GetArrayFromImage(resizeimagesitk)
        # transpose (D,H,W,C) order to (C,D,H,W) order
        # transpose (D,H,W,C) order to (C,D,H,W) order
        D, H, W = np.shape(imageresize)[0], np.shape(imageresize)[1], np.shape(imageresize)[2]
        imageresize = np.reshape(imageresize, (D, H, W, 1))
        imageresize = np.transpose(imageresize, (3, 0, 1, 2))

        out_mask = self.predict(imageresize)
        # resize mask to src image size,should rewrite
        out_mask_sitk = sitk.GetImageFromArray(out_mask)
        out_mask_sitk.SetOrigin(resizeimagesitk.GetOrigin())
        out_mask_sitk.SetSpacing(resizeimagesitk.GetSpacing())
        out_mask_sitk.SetDirection(resizeimagesitk.GetDirection())
        _, final_out_mask_sitk = resize_image_itkwithsize(out_mask_sitk, imagesitk.GetSize(), newSize,
                                                          sitk.sitkNearestNeighbor)
        final_out_mask_sitk.SetOrigin(imagesitk.GetOrigin())
        final_out_mask_sitk.SetSpacing(imagesitk.GetSpacing())
        final_out_mask_sitk.SetDirection(imagesitk.GetDirection())
        return final_out_mask_sitk

    def inference_patch(self, imagesitk, newSpacing=(0.5, 0.5, 0.5)):
        # resize image and normalization,should rewrite
        _, resizeimagesitk = resize_image_itk(imagesitk, newSpacing, imagesitk.GetSpacing(), sitk.sitkLinear)
        resizeimagesitk = ConvertitkTrunctedValue(resizeimagesitk, 800, -1000, 'meanstd')
        imageresize = sitk.GetArrayFromImage(resizeimagesitk)
        # transpose (D,H,W,C) order to (C,D,H,W) order
        D, H, W = np.shape(imageresize)[0], np.shape(imageresize)[1], np.shape(imageresize)[2]
        imageresize = np.reshape(imageresize, (D, H, W, 1))
        imageresize = np.transpose(imageresize, (3, 0, 1, 2))
        # predict patch
        stepx = self.image_width // 2
        stepy = self.image_height // 2
        stepz = self.image_depth // 2
        # check reize image size is or not smaller than net input size
        x_size = max(W, self.image_width)
        y_size = max(H, self.image_height)
        z_size = max(D, self.image_depth)
        newimageresize = np.zeros((1, z_size, y_size, x_size))
        newimageresize[:, 0:D, 0:H, 0:W] = imageresize[:, :, :, :]

        out_mask = np.zeros((z_size, y_size, x_size))
        for z in range(0, z_size, stepz):
            for y in range(0, y_size, stepy):
                for x in range(0, x_size, stepx):
                    x_min = x
                    x_max = x_min + self.image_width
                    if x_max > x_size:
                        x_max = x_size
                        x_min = x_size - self.image_width
                    y_min = y
                    y_max = y_min + self.image_height
                    if y_max > y_size:
                        y_max = y_size
                        y_min = y_size - self.image_height
                    z_min = z
                    z_max = z_min + self.image_depth
                    if z_max > z_size:
                        z_max = z_size
                        z_min = z_size - self.image_depth
                    patch_xs = newimageresize[:, z_min:z_max, y_min:y_max, x_min:x_max]
                    predictresult = self.predict(patch_xs)
                    out_mask[z_min:z_max, y_min:y_max, x_min:x_max] = \
                        out_mask[z_min:z_max, y_min:y_max, x_min:x_max] + predictresult.copy()
        # resize mask to src image size,should rewrite
        roi_out_mask = out_mask[0:D, 0:H, 0:W]
        roi_out_mask[roi_out_mask != 0] = 1
        out_mask_sitk = sitk.GetImageFromArray(roi_out_mask)
        out_mask_sitk.SetOrigin(resizeimagesitk.GetOrigin())
        out_mask_sitk.SetSpacing(resizeimagesitk.GetSpacing())
        out_mask_sitk.SetDirection(resizeimagesitk.GetDirection())
        _, resize_out_mask_sitk = resize_image_itk(out_mask_sitk, imagesitk.GetSpacing(), newSpacing,
                                                   sitk.sitkNearestNeighbor)
        _, final_out_mask_sitk = resize_image_itkwithsize(resize_out_mask_sitk, imagesitk.GetSize(),
                                                          resize_out_mask_sitk.GetSize(), sitk.sitkNearestNeighbor)
        final_out_mask_sitk.SetOrigin(imagesitk.GetOrigin())
        final_out_mask_sitk.SetSpacing(imagesitk.GetSpacing())
        final_out_mask_sitk.SetDirection(imagesitk.GetDirection())
        return final_out_mask_sitk

    def clear_GPU_cache(self):
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())


class BinaryVNet3dthinModel(object):
    """
    Vnet3d with binary class,should rewrite the dataset class and inference fucntion
    """

    def __init__(self, image_depth, image_height, image_width, image_channel, numclass, batch_size,
                 loss_name='BinaryDiceLoss', inference=False, model_path=None, amp=True, accum_gradient_iter=1,
                 num_cpu=8, use_cuda=True):
        self.batch_size = batch_size
        self.loss_name = loss_name
        self.accuracyname = 'dice'
        self.image_height = image_height
        self.image_width = image_width
        self.image_depth = image_depth
        self.image_channel = image_channel
        self.numclass = numclass
        self.amp = amp
        self.accum_iter = accum_gradient_iter
        self.num_cpu = num_cpu

        self.alpha = 0.25
        self.gamma = 2

        self.lossFunc = None
        self.showwind = None
        self.model_dir = None
        self.showpixelvalue = None
        self.scaler = None
        self.opt = None

        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = VNet3dthin(self.image_channel, self.numclass)
        self.model.to(device=self.device)

        if inference:
            print(f'Loading model {model_path}')
            print(f'Using device {self.device}')
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print('Model loaded!')

    def _dataloder(self, images, labels, batch_size, shuffle=False):
        """"
        train dataset shuffle is true,validation is false
        """
        dataset = datasetModelSegwithnpy(images, labels,
                                         targetsize=(
                                             self.image_channel, self.image_depth, self.image_height, self.image_width))
        # fow window num_workers is only zero,for linux num_workers can not zero
        # num_cpu = multiprocessing.cpu_count()
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=self.num_cpu,
                                pin_memory=True)
        return dataloader

    def _loss_function(self, lossname):
        if lossname == 'BinaryCrossEntropyLoss':
            return BinaryCrossEntropyLoss()
        if lossname == 'BinaryDiceLoss':
            return BinaryDiceLoss()
        if lossname == 'BinaryCrossEntropyDiceLoss':
            return BinaryCrossEntropyDiceLoss()
        if lossname == 'BinaryFocalLoss':
            return BinaryFocalLoss()

    def _accuracy_function(self, accuracyname, input, target):
        if accuracyname == 'dice':
            if self.numclass == 1:
                return dice_coeff(input, target)
            else:
                return multiclass_dice_coeff(input, target)
        if accuracyname == 'iou':
            if self.numclass == 1:
                return iou_coeff(input, target)
            else:
                return multiclass_iou_coeff(input, target)

    def _train_loop(self, train_loader, totalTrainLoss, totalTrainAccu, trainshow, e):
        for batch_idx, batch in enumerate(train_loader):
            # x should tensor with shape (N,C,D,W,H)
            x = batch['image']
            # y should tensor with shape (N,C,D,W,H),
            # if have mutil label y should one-hot,if only one label,the C is one
            y = batch['label']
            y[y != 0] = 1
            # send the input to the device
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            # perform a forward pass and calculate the training loss and accu
            # 利用with语句，在autocast实例的上下文范围内，进行模型的前向推理和loss计算
            with autocast(enabled=self.amp):
                # perform a forward pass and calculate the training loss and accu
                pred_logit, pred = self.model(x)
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
            accu = self._accuracy_function(self.accuracyname, pred, y)
            if trainshow:
                savepath = self.model_dir + '/' + str(e + 1) + "_Train_EPOCH_"
                save_images3d(pred[0], y[0], self.showwind, savepath, pixelvalue=self.showpixelvalue)
                trainshow = False
            # add the loss to the total training loss so far
            totalTrainLoss.append(loss.cpu().detach().numpy() * self.accum_iter)
            totalTrainAccu.append(accu.cpu().detach().numpy())

    def _validation_loop(self, val_loader, totalValidationLoss, totalValiadtionAccu, trainshow, e):
        with torch.no_grad():
            # loop over the validation set
            for batch_idx, batch in enumerate(val_loader):
                # x should tensor with shape (N,C,W,H)
                x = batch['image']
                # y should tensor with shape (N,C,W,H)
                y = batch['label']
                y[y != 0] = 1
                # send the input to the device
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                # make the predictions and calculate the validation loss
                pred_logit, pred = self.model(x)
                loss = self.lossFunc(pred_logit, y)
                accu = self._accuracy_function(self.accuracyname, pred, y)
                if trainshow:
                    # save_images
                    savepath = self.model_dir + '/' + str(e + 1) + "_Val_EPOCH_"
                    save_images3d(pred[0], y[0], self.showwind, savepath,
                                  pixelvalue=self.showpixelvalue)
                    trainshow = False
                totalValidationLoss.append(loss.cpu().detach().numpy())
                totalValiadtionAccu.append(accu.cpu().detach().numpy())

    def trainprocess(self, trainimage, trainmask, validationimage, validationmask, model_dir, epochs=50, lr=1e-3,
                     showwind=[8, 8]):
        print("[INFO] training the network...")
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        MODEL_PATH = os.path.join(model_dir, "BinaryVNet3d.pth")
        summary(self.model, input_size=(1, 64, 64, 64))
        print(self.model)
        self.showwind = showwind
        self.model_dir = model_dir
        self.showpixelvalue = 255.
        if self.numclass > 1:
            self.showpixelvalue = self.showpixelvalue // (self.numclass - 1)
        # 1、initialize loss function and optimizer
        self.model.apply(initialize_weights)
        self.lossFunc = self._loss_function(self.loss_name)
        self.opt = optim.AdamW(self.model.parameters(), lr=lr)
        self.scaler = GradScaler()
        # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=2, verbose=True)
        # 2、load data train and validation dataset
        train_loader = self._dataloder(trainimage, trainmask, self.batch_size, True)
        val_loader = self._dataloder(validationimage, validationmask, self.batch_size * 2, True)
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
            trainshow = True
            self._train_loop(train_loader, totalTrainLoss, totalTrainAccu, trainshow, e)
            # 4.4、switch off autograd and loop over the validation set
            # set the model in evaluation mode
            self.model.eval()
            trainshow = True
            self._validation_loop(val_loader, totalValidationLoss, totalValiadtionAccu, trainshow, e)
            # 4.5、calculate the average training and validation loss
            avgTrainLoss = np.mean(np.stack(totalTrainLoss))
            avgValidationLoss = np.mean(np.stack(totalValidationLoss))
            avgTrainAccu = np.mean(np.stack(totalTrainAccu))
            avgValidationAccu = np.mean(np.stack(totalValiadtionAccu))
            # lr_scheduler.step(avgValidationLoss)
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
            _, output = self.model(img)
            probs = output[0]
            full_mask_np = probs.detach().cpu().squeeze().numpy()
        # 5、get numpy result
        if self.numclass == 1:
            out_mask = (full_mask_np > out_threshold)
            out_mask = out_mask * 1.
        else:
            out_mask = np.argmax(full_mask_np, axis=0)
            out_mask = np.squeeze(out_mask)
        return out_mask.astype(np.uint8)

    def inference(self, imagesitk, newSize=(96, 96, 96)):
        # resize image and normalization,should rewrite
        _, resizeimagesitk = resize_image_itkwithsize(imagesitk, newSize, imagesitk.GetSize(), sitk.sitkLinear)
        # resizeimagesitk = ConvertitkTrunctedValue(resizeimagesitk, 100, 0, 'meanstd')
        imageresize = sitk.GetArrayFromImage(resizeimagesitk)
        imageresize = normalize(imageresize, 99, 1)
        # transpose (D,H,W,C) order to (C,D,H,W) order
        D, H, W = np.shape(imageresize)[0], np.shape(imageresize)[1], np.shape(imageresize)[2]
        imageresize = np.reshape(imageresize, (D, H, W, 1))
        imageresize = np.transpose(imageresize, (3, 0, 1, 2))
        out_mask = self.predict(imageresize)
        # resize mask to src image size,should rewrite
        out_mask_sitk = sitk.GetImageFromArray(out_mask)
        out_mask_sitk.SetOrigin(resizeimagesitk.GetOrigin())
        out_mask_sitk.SetSpacing(resizeimagesitk.GetSpacing())
        out_mask_sitk.SetDirection(resizeimagesitk.GetDirection())
        _, final_out_mask_sitk = resize_image_itkwithsize(out_mask_sitk, imagesitk.GetSize(), newSize,
                                                          sitk.sitkNearestNeighbor)
        final_out_mask_sitk.SetOrigin(imagesitk.GetOrigin())
        final_out_mask_sitk.SetSpacing(imagesitk.GetSpacing())
        final_out_mask_sitk.SetDirection(imagesitk.GetDirection())
        return final_out_mask_sitk

    def clear_GPU_cache(self):
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())


class MutilVNet3dModel(object):
    """
    VNet3d with mutil class,should rewrite the dataset class
    """

    def __init__(self, image_depth, image_height, image_width, image_channel, numclass, batch_size,
                 loss_name='MutilFocalLoss', inference=False, model_path=None, amp=True, accum_gradient_iter=1,
                 num_cpu=2, use_cuda=True):
        self.batch_size = batch_size
        self.loss_name = loss_name
        self.accuracyname = 'dice'
        self.image_height = image_height
        self.image_width = image_width
        self.image_depth = image_depth
        self.image_channel = image_channel
        self.numclass = numclass
        self.amp = amp
        self.accum_iter = accum_gradient_iter
        self.num_cpu = num_cpu

        self.alpha = [1.] * self.numclass
        self.gamma = 3

        self.lossFunc = None
        self.showwind = None
        self.model_dir = None
        self.showpixelvalue = None
        self.scaler = None
        self.opt = None

        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = VNet3d(self.image_channel, self.numclass)
        self.model.to(device=self.device)
        self.alpha = torch.as_tensor(self.alpha).contiguous().to(self.device)
        if inference:
            print(f'Loading model {model_path}')
            print(f'Using device {self.device}')
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print('Model loaded!')

    def _dataloder(self, images, labels, batch_size, shuffle=False):
        """"
        train dataset shuffle is true,validation is false
        """
        dataset = datasetModelSegwithnpy(images, labels,
                                         targetsize=(
                                             self.image_channel, self.image_depth, self.image_height,
                                             self.image_width))
        # fow window num_workers is only zero,for linux num_workers can not zero
        # num_cpu = multiprocessing.cpu_count()
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=self.num_cpu,
                                pin_memory=True)
        return dataloader

    def _loss_function(self, lossname):
        if lossname == 'MutilCrossEntropyLoss':
            return MutilCrossEntropyLoss(alpha=self.alpha)
        if lossname == 'MutilDiceLoss':
            return MutilDiceLoss(alpha=self.alpha)
        if lossname == 'MutilFocalLoss':
            return MutilFocalLoss(alpha=self.alpha, gamma=self.gamma)
        if lossname == 'MutilCrossEntropyDiceLoss':
            return MutilCrossEntropyDiceLoss(alpha=self.alpha)

    def _accuracy_function(self, accuracyname, input, target):
        if accuracyname == 'dice':
            if self.numclass == 1:
                return dice_coeff(input, target)
            else:
                return multiclass_dice_coeffv2(input, target)
        if accuracyname == 'iou':
            if self.numclass == 1:
                return iou_coeff(input, target)
            else:
                return multiclass_iou_coeff(input, target)

    def _train_loop(self, train_loader, totalTrainLoss, totalTrainAccu, trainshow, e):
        self.clear_GPU_cache()
        for batch_idx, batch in enumerate(train_loader):
            # x should tensor with shape (N,C,D,W,H)
            x = batch['image']
            # y should tensor with shape (N,C,D,W,H),
            # if have mutil label y should one-hot,if only one label,the C is one
            y = batch['label']
            # send the input to the device
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            # perform a forward pass and calculate the training loss and accu
            # 利用with语句，在autocast实例的上下文范围内，进行模型的前向推理和loss计算
            with autocast(enabled=self.amp):
                # perform a forward pass and calculate the training loss and accu
                pred_logit, pred = self.model(x)
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
            if trainshow:
                savepath = self.model_dir + '/' + str(e + 1) + "_Train_EPOCH_"
                save_images3d(pred[0], y[0], self.showwind, savepath, pixelvalue=self.showpixelvalue)
                trainshow = False
            # add the loss to the total training loss so far
            totalTrainLoss.append(loss.cpu().detach().numpy() * self.accum_iter)
            totalTrainAccu.append(accu.cpu().detach().numpy())

    def _validation_loop(self, val_loader, totalValidationLoss, totalValiadtionAccu, trainshow, e):
        self.clear_GPU_cache()
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
                pred_logit, pred = self.model(x)
                loss = self.lossFunc(pred_logit, y)
                accu = self._accuracy_function(self.accuracyname, pred_logit, y)
                if trainshow:
                    # save_images
                    savepath = self.model_dir + '/' + str(e + 1) + "_Val_EPOCH_"
                    save_images3d(pred[0], y[0], self.showwind, savepath, pixelvalue=self.showpixelvalue)
                    trainshow = False
                totalValidationLoss.append(loss.cpu().detach().numpy())
                totalValiadtionAccu.append(accu.cpu().detach().numpy())

    def trainprocess(self, trainimage, trainmask, validationimage, validationmask, model_dir, epochs=50, lr=1e-3,
                     showwind=None):
        if showwind is None:
            showwind = [8, 8]
        print("[INFO] training the network...")
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        MODEL_PATH = os.path.join(model_dir, "MutilVNet3dModel.pth")
        MODEL_PATH_best = os.path.join(model_dir, "MutilVNet3dModelbest.pth")
        print(self.model)
        self.model_dir = model_dir
        self.showpixelvalue = 255.
        self.showwind = showwind
        if self.numclass > 1:
            self.showpixelvalue = self.showpixelvalue // (self.numclass - 1)
        # 1、initialize loss function and optimizer
        self.model.apply(initialize_weights)
        self.lossFunc = self._loss_function(self.loss_name)
        self.opt = optim.AdamW(self.model.parameters(), lr=lr)
        self.scaler = GradScaler()
        # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=2, verbose=True)
        # 2、load data train and validation dataset
        train_loader = self._dataloder(trainimage, trainmask, self.batch_size, True)
        val_loader = self._dataloder(validationimage, validationmask, self.batch_size, True)
        # 3、initialize a dictionary to store training history
        H = {"train_loss": [], "train_accuracy": [], "valdation_loss": [], "valdation_accuracy": []}
        # 4、start loop training wiht epochs times
        startTime = time.time()
        best_validation_dsc = 1000000000
        best_epoch = 0
        # Tensorboard summary
        writer = SummaryWriter(log_dir=self.model_dir)
        for e in tqdm(range(epochs)):
            # 4.1、set the model in training mode
            self.model.train()
            # 4.2、initialize the total training and validation loss
            totalTrainLoss = []
            totalTrainAccu = []
            totalValidationLoss = []
            totalValiadtionAccu = []
            # 4.3、loop over the training set
            trainshow = True
            self._train_loop(train_loader, totalTrainLoss, totalTrainAccu, trainshow, e)
            # 4.4、switch off autograd and loop over the validation set
            # set the model in evaluation mode
            self.model.eval()
            trainshow = True
            self._validation_loop(val_loader, totalValidationLoss, totalValiadtionAccu, trainshow, e)
            # 4.5、calculate the average training and validation loss
            avgTrainLoss = np.mean(np.stack(totalTrainLoss))
            avgValidationLoss = np.mean(np.stack(totalValidationLoss))
            avgTrainAccu = np.mean(np.stack(totalTrainAccu))
            avgValidationAccu = np.mean(np.stack(totalValiadtionAccu))
            # lr_scheduler.step(avgValidationLoss)
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
            if avgValidationLoss < best_validation_dsc:
                best_validation_dsc = avgValidationLoss
                # best_model_params = self.model.state_dict()
                # serialize best model to disk
                torch.save(self.model.state_dict(), MODEL_PATH_best)
                best_epoch = e
            torch.save(self.model.state_dict(), MODEL_PATH)
            # 4.9、clear cache memory
            self.clear_GPU_cache()
            # 4.10、early stopping
            if abs(best_epoch - e) > epochs // 3:
                break
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
            _, output = self.model(img)
            probs = output[0]
            full_mask_np = probs.detach().cpu().squeeze().numpy()
        # 5、get numpy result
        if self.numclass == 1:
            out_mask = full_mask_np * 1
        else:
            out_mask = np.squeeze(full_mask_np)
        return out_mask.astype(np.uint8)

    def inference(self, imagesitkt1, newSize=(96, 96, 96)):
        # resize image and normalization,should rewrite
        _, resizeimagesitkt1 = resize_image_itkwithsize(imagesitkt1, newSize, imagesitkt1.GetSize(), sitk.sitkLinear)
        resizeimagesitkt1 = ConvertitkTrunctedValue(resizeimagesitkt1, 2000, 0, 'meanstd')
        imageresize = sitk.GetArrayFromImage(resizeimagesitkt1)
        D, H, W = np.shape(imageresize)[0], np.shape(imageresize)[1], np.shape(imageresize)[2]
        imageresize = np.reshape(imageresize, (D, H, W, 1))
        imageresize = np.transpose(imageresize, (3, 0, 1, 2))
        out_mask = self.predict(imageresize)
        # resize mask to src image size,should rewrite
        out_mask_sitk = sitk.GetImageFromArray(out_mask)
        out_mask_sitk.SetOrigin(resizeimagesitkt1.GetOrigin())
        out_mask_sitk.SetSpacing(resizeimagesitkt1.GetSpacing())
        out_mask_sitk.SetDirection(resizeimagesitkt1.GetDirection())
        _, final_out_mask_sitk = resize_image_itkwithsize(out_mask_sitk, imagesitkt1.GetSize(), newSize,
                                                          sitk.sitkNearestNeighbor)
        final_out_mask_sitk.SetOrigin(imagesitkt1.GetOrigin())
        final_out_mask_sitk.SetSpacing(imagesitkt1.GetSpacing())
        final_out_mask_sitk.SetDirection(imagesitkt1.GetDirection())
        return final_out_mask_sitk

    def inference_patch(self, imagesitk, newSpacing=(0.7, 0.7, 0.7)):
        # resize image and normalization,should rewrite
        _, resizeimagesitk = resize_image_itk(imagesitk, newSpacing, imagesitk.GetSpacing(), sitk.sitkLinear)
        resizeimagesitk = ConvertitkTrunctedValue(resizeimagesitk, 600, -1000, 'meanstd')
        imageresize = sitk.GetArrayFromImage(resizeimagesitk)
        # transpose (D,H,W,C) order to (C,D,H,W) order
        D, H, W = np.shape(imageresize)[0], np.shape(imageresize)[1], np.shape(imageresize)[2]
        imageresize = np.reshape(imageresize, (D, H, W, 1))
        imageresize = np.transpose(imageresize, (3, 0, 1, 2))
        # predict patch
        stepx = self.image_width
        stepy = self.image_height
        stepz = self.image_depth
        # check reize image size is or not smaller than net input size
        x_size = max(W, stepx)
        y_size = max(H, stepy)
        z_size = max(D, stepz)
        newimageresize = np.zeros((1, z_size, y_size, x_size))
        newimageresize[:, 0:D, 0:H, 0:W] = imageresize[:, :, :, :]

        out_mask = np.zeros((z_size, y_size, x_size))
        for z in range(0, z_size, stepz):
            for y in range(0, y_size, stepy):
                for x in range(0, x_size, stepx):
                    x_min = x
                    x_max = x_min + self.image_width
                    if x_max > x_size:
                        x_max = x_size
                        x_min = x_size - self.image_width
                    y_min = y
                    y_max = y_min + self.image_height
                    if y_max > y_size:
                        y_max = y_size
                        y_min = y_size - self.image_height
                    z_min = z
                    z_max = z_min + self.image_depth
                    if z_max > z_size:
                        z_max = z_size
                        z_min = z_size - self.image_depth
                    patch_xs = newimageresize[:, z_min:z_max, y_min:y_max, x_min:x_max]
                    predictresult = self.predict(patch_xs)
                    out_mask[z_min:z_max, y_min:y_max, x_min:x_max] = predictresult.copy()
        # resize mask to src image size,should rewrite
        roi_out_mask = out_mask[0:D, 0:H, 0:W]
        out_mask_sitk = sitk.GetImageFromArray(roi_out_mask)
        out_mask_sitk.SetOrigin(resizeimagesitk.GetOrigin())
        out_mask_sitk.SetSpacing(resizeimagesitk.GetSpacing())
        out_mask_sitk.SetDirection(resizeimagesitk.GetDirection())
        _, resize_out_mask_sitk = resize_image_itk(out_mask_sitk, imagesitk.GetSpacing(), newSpacing,
                                                   sitk.sitkNearestNeighbor)
        _, final_out_mask_sitk = resize_image_itkwithsize(resize_out_mask_sitk, imagesitk.GetSize(),
                                                          resize_out_mask_sitk.GetSize(), sitk.sitkNearestNeighbor)
        final_out_mask_sitk.SetOrigin(imagesitk.GetOrigin())
        final_out_mask_sitk.SetSpacing(imagesitk.GetSpacing())
        final_out_mask_sitk.SetDirection(imagesitk.GetDirection())
        return final_out_mask_sitk

    def clear_GPU_cache(self):
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())


class VNet3dRegressionModel(object):
    """
    Unet3d with binary class,should rewrite the dataset class and inference fucntion
    """

    def __init__(self, image_depth, image_height, image_width, image_channel, numclass, batch_size,
                 loss_name='MSE', inference=False, model_path=None, amp=False, accum_iter=4, num_cpu=4, use_cuda=True):
        self.batch_size = batch_size
        self.loss_name = loss_name
        self.accuracyname = ['PSNR', 'SSIM']
        self.image_height = image_height
        self.image_width = image_width
        self.image_depth = image_depth
        self.image_channel = image_channel
        self.numclass = numclass
        self.amp = amp
        self.accum_iter = accum_iter
        self.num_cpu = num_cpu

        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = VNet3d(self.image_channel, self.numclass)
        self.model.to(device=self.device)

        self.scaler = None
        self.opt = None
        self.lossFunc = None
        self.showpixelvalue = None
        self.model_dir = None
        self.showwind = None

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
        # num_cpu = multiprocessing.cpu_count()
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=self.num_cpu,
                                pin_memory=True)
        return dataloader

    def _loss_function(self, lossname):
        if lossname == 'MSE':
            return MSELoss()
        if lossname == 'L1':
            return L1Loss()
        if lossname == 'SSIM':
            return SSIMLoss()
        if lossname == 'L1SSIM':
            return L1SSIMLoss()

    def _accuracy_function(self, accuracyname, input, target, mean, std):
        if accuracyname[0] == 'PSNR':
            psnr = calc_psnr(input, target, mean, std)
        if accuracyname[1] == 'SSIM':
            ssim = calc_ssim(input, target, mean, std)
        return psnr, ssim

    def _train_loop(self, train_loader, totalTrainLoss, totalTrainAccu, totalTrainssim, trainshow, e):
        for batch_idx, batch in enumerate(train_loader):
            # x should tensor with shape (N,C,D,W,H)
            x = batch['image']
            # y should tensor with shape (N,C,D,W,H),
            # if have mutil label y should one-hot,if only one label,the C is one
            y = batch['label']
            mean = batch['mean']
            std = batch['std']
            # send the input to the device
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            # 利用with语句，在autocast实例的上下文范围内，进行模型的前向推理和loss计算
            with autocast(enabled=self.amp):
                # perform a forward pass and calculate the training loss and accu
                pred_logit, _ = self.model(x)
                loss = self.lossFunc(pred_logit, y)
                # normalize loss to account for batch accumulation
                loss = loss / self.accum_iter
            # then perform backpropagation,
            if self.amp:
                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                # Backward passes under autocast are not recommended.
                # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                self.scaler.scale(loss).backward()
                # clip gradient in order to Gradient explosion
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), max_norm=20)
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
            psnr, ssim = self._accuracy_function(self.accuracyname, pred_logit, y, mean, std)
            # save_images
            if trainshow:
                savepath = self.model_dir + '/' + str(e + 1) + "_Train_EPOCH_"
                x0 = x[0] * std[0] + mean[0]
                x0 = x0[0]
                images = x0.detach().cpu().squeeze().numpy()
                sitk_image = sitk.GetImageFromArray(images)
                sitk.WriteImage(sitk_image, self.model_dir + "/trainsrc.nii.gz")
                x0 = 255. * (x0 - torch.min(x0)) / (torch.max(x0) - torch.min(x0))
                pd0 = pred_logit[0] * std[0] + mean[0]
                pd0 = pd0.type_as(x0)
                pdimages = pd0.detach().cpu().squeeze().numpy()
                sitk_image = sitk.GetImageFromArray(pdimages)
                sitk.WriteImage(sitk_image, self.model_dir + "/trainpd.nii.gz")
                pd0 = 255. * (pd0 - torch.min(pd0)) / (torch.max(pd0) - torch.min(pd0))
                y0 = y[0] * std[0] + mean[0]
                yimages = y0.detach().cpu().squeeze().numpy()
                sitk_image = sitk.GetImageFromArray(yimages)
                sitk.WriteImage(sitk_image, self.model_dir + "/trainmask.nii.gz")
                y0 = 255. * (y0 - torch.min(y0)) / (torch.max(y0) - torch.min(y0))
                save_images3dregression(x0, pd0, y0, self.showwind, savepath, pixelvalue=self.showpixelvalue)
                trainshow = False
            # add the loss to the total training loss so far
            totalTrainLoss.append(loss.cpu().detach().numpy() * self.accum_iter)
            totalTrainAccu.append(psnr.cpu().detach().numpy())
            totalTrainssim.append(ssim)

    def _validation_loop(self, val_loader, totalValidationLoss, totalValiadtionAccu, totalValiadtionssim, trainshow, e):
        with torch.no_grad():
            # loop over the validation set
            for batch_idx, batch in enumerate(val_loader):
                # x should tensor with shape (N,C,W,H)
                x = batch['image']
                # y should tensor with shape (N,C,W,H)
                y = batch['label']
                mean = batch['mean']
                std = batch['std']
                # send the input to the device
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                # make the predictions and calculate the validation loss
                pred_logit, _ = self.model(x)
                loss = self.lossFunc(pred_logit, y)
                psnr, ssim = self._accuracy_function(self.accuracyname, pred_logit, y, mean, std)
                if trainshow:
                    # save_images
                    savepath = self.model_dir + '/' + str(e + 1) + "_Val_EPOCH_"
                    x0 = x[0] * std[0] + mean[0]
                    x0 = x0[0]
                    images = x0.detach().cpu().squeeze().numpy()
                    sitk_image = sitk.GetImageFromArray(images)
                    sitk.WriteImage(sitk_image, self.model_dir + "/valsrc.nii.gz")
                    x0 = 255. * (x0 - torch.min(x0)) / (torch.max(x0) - torch.min(x0))
                    pd0 = pred_logit[0] * std[0] + mean[0]
                    pdimages = pd0.detach().cpu().squeeze().numpy()
                    sitk_image = sitk.GetImageFromArray(pdimages.astype(images.dtype))
                    sitk.WriteImage(sitk_image, self.model_dir + "/valpd.nii.gz")
                    pd0 = 255. * (pd0 - torch.min(pd0)) / (torch.max(pd0) - torch.min(pd0))
                    y0 = y[0] * std[0] + mean[0]
                    yimages = y0.detach().cpu().squeeze().numpy()
                    sitk_image = sitk.GetImageFromArray(yimages)
                    sitk.WriteImage(sitk_image, self.model_dir + "/valmask.nii.gz")
                    y0 = 255. * (y0 - torch.min(y0)) / (torch.max(y0) - torch.min(y0))
                    save_images3dregression(x0, pd0, y0, self.showwind, savepath, pixelvalue=self.showpixelvalue)
                    trainshow = False
                totalValidationLoss.append(loss.cpu().detach().numpy())
                totalValiadtionAccu.append(psnr.cpu().detach().numpy())
                totalValiadtionssim.append(ssim)

    def trainprocess(self, trainimage, trainmask, validationimage, validationmask, model_dir, epochs=50, lr=1e-3,
                     showwind=[8, 8]):
        print("[INFO] training the network...")
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        MODEL_PATH = os.path.join(model_dir, "Vnet3dregression.pth")
        print(self.model)
        self.model_dir = model_dir
        self.showpixelvalue = 1.
        self.showwind = showwind
        # 1、initialize loss function and optimizer
        self.model.apply(initialize_weights)
        self.lossFunc = self._loss_function(self.loss_name)
        self.opt = optim.AdamW(self.model.parameters(), lr=lr, eps=1e-4)
        self.scaler = GradScaler()
        # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=2, verbose=True)
        # 2、load data train and validation dataset
        train_loader = self._dataloder(trainimage, trainmask, batch_size=self.batch_size, shuffle=True)
        val_loader = self._dataloder(validationimage, validationmask, batch_size=self.batch_size, shuffle=True)
        # 3、initialize a dictionary to store training history
        H = {"train_loss": [], "train_accuracy": [], "train_ssim": [], "valdation_loss": [], "valdation_accuracy": [],
             "valdation_ssim": []}
        # 4、start loop training wiht epochs times
        startTime = time.time()
        best_validation_dsc = 1000000
        # Tensorboard summary
        writer = SummaryWriter(log_dir=self.model_dir)
        for e in tqdm(range(epochs)):
            # 4.1、set the model in training mode
            self.model.train()
            # 4.2、initialize the total training and validation loss
            totalTrainLoss = []
            totalTrainAccu = []
            totalTrainssim = []
            totalValidationLoss = []
            totalValiadtionAccu = []
            totalValiadtionssim = []
            trainshow = True
            # 4.3、loop over the training set
            self._train_loop(train_loader, totalTrainLoss, totalTrainAccu, totalTrainssim, trainshow, e)
            # 4.4、switch off autograd and loop over the validation set
            # set the model in evaluation mode
            self.model.eval()
            trainshow = True
            self._validation_loop(val_loader, totalValidationLoss, totalValiadtionAccu, totalValiadtionssim, trainshow,
                                  e)
            # 4.5、calculate the average training and validation loss
            avgTrainLoss = np.mean(np.stack(totalTrainLoss))
            avgValidationLoss = np.mean(np.stack(totalValidationLoss))
            avgTrainAccu = np.mean(np.stack(totalTrainAccu))
            avgValidationAccu = np.mean(np.stack(totalValiadtionAccu))
            avgTrainssim = np.mean(np.stack(totalTrainssim))
            avgValidationssim = np.mean(np.stack(totalValiadtionssim))
            # lr_scheduler.step(avgValidationLoss)
            # 4.6、update our training history
            H["train_loss"].append(avgTrainLoss)
            H["valdation_loss"].append(avgValidationLoss)
            H["train_accuracy"].append(avgTrainAccu)
            H["valdation_accuracy"].append(avgValidationAccu)
            H["train_ssim"].append(avgTrainssim)
            H["valdation_ssim"].append(avgValidationssim)
            # 4.7、print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
            print(
                "Train loss: {:.5f}, Train accu: {:.5f}, Train ssim: {:.5f}，"
                "validation loss: {:.5f}, validation accu: {:.5f}, validation ssim: {:.5f}".format(
                    avgTrainLoss, avgTrainAccu, avgTrainssim, avgValidationLoss, avgValidationAccu, avgValidationssim))
            # Record training loss and accuracy for each phase
            writer.add_scalar('Train/Loss', avgTrainLoss, e + 1)
            writer.add_scalar('Train/accu', avgTrainAccu, e + 1)
            writer.add_scalar('Train/ssim', avgTrainssim, e + 1)
            writer.add_scalar('Valid/loss', avgValidationLoss, e + 1)
            writer.add_scalar('Valid/accu', avgValidationAccu, e + 1)
            writer.add_scalar('Valid/ssim', avgValidationssim, e + 1)
            writer.flush()
            # 4.8、save best_validation_dsc model params
            # serialize best model to disk
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
        plot_result(self.model_dir, H["train_loss"], H["valdation_loss"], "train_loss", "valdation_loss", "loss")
        plot_result(self.model_dir, H["train_accuracy"], H["valdation_accuracy"], "train_accuracy",
                    "valdation_accuracy",
                    "accuracy")
        plot_result(self.model_dir, H["train_ssim"], H["valdation_ssim"], "train_ssim", "valdation_ssim", "ssim")
        self.clear_GPU_cache()

    def predict(self, full_img):
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
            output, _ = self.model(img)
            probs = output[0]
            full_mask_np = probs.detach().cpu().squeeze().numpy()
            return full_mask_np

    def inference(self, imagesitk, masksitk, newSize=(96, 96, 96)):
        # resize image and normalization,should rewrite
        _, resizeimagesitk = resize_image_itkwithsize(imagesitk, newSize, imagesitk.GetSize(), sitk.sitkLinear)
        _, resizemasksitk = resize_image_itkwithsize(masksitk, newSize, masksitk.GetSize(), sitk.sitkLinear)
        imageresize = sitk.GetArrayFromImage(resizeimagesitk)
        maskresize = sitk.GetArrayFromImage(resizemasksitk)
        mean = imageresize.mean()
        std = imageresize.std()
        imageresize = (imageresize - mean) / std
        # transpose (D,H,W,C) order to (C,D,H,W) order
        image = np.array([imageresize, maskresize])
        out_mask = self.predict(image)
        out_mask = out_mask * std + mean
        # resize mask to src image size,should rewrite
        out_mask_sitk = sitk.GetImageFromArray(out_mask)
        out_mask_sitk.SetOrigin(resizeimagesitk.GetOrigin())
        out_mask_sitk.SetSpacing(resizeimagesitk.GetSpacing())
        out_mask_sitk.SetDirection(resizeimagesitk.GetDirection())
        _, final_out_mask_sitk = resize_image_itkwithsize(out_mask_sitk, imagesitk.GetSize(), newSize,
                                                          sitk.sitkLinear)
        final_out_mask_sitk.SetOrigin(imagesitk.GetOrigin())
        final_out_mask_sitk.SetSpacing(imagesitk.GetSpacing())
        final_out_mask_sitk.SetDirection(imagesitk.GetDirection())
        # keep roi region
        final_out_mask_array = sitk.GetArrayFromImage(final_out_mask_sitk)
        image_array = sitk.GetArrayFromImage(imagesitk)
        mask_array = sitk.GetArrayFromImage(masksitk)
        final_out_mask_array[mask_array == 0] = 0
        final_out_mask_array = final_out_mask_array + image_array
        final_out_mask_sitk = sitk.GetImageFromArray(np.around(final_out_mask_array))
        final_out_mask_sitk.SetOrigin(imagesitk.GetOrigin())
        final_out_mask_sitk.SetSpacing(imagesitk.GetSpacing())
        final_out_mask_sitk.SetDirection(imagesitk.GetDirection())
        return final_out_mask_sitk

    def inference_patch(self, imagesitk, newSpacing=(0.5, 0.5, 0.5)):
        # resize image and normalization,should rewrite
        _, resizeimagesitk = resize_image_itk(imagesitk, newSpacing, imagesitk.GetSpacing(), sitk.sitkLinear)
        imageresize = sitk.GetArrayFromImage(imagesitk)
        # transpose (D,H,W,C) order to (C,D,H,W) order
        D, H, W = np.shape(imageresize)[0], np.shape(imageresize)[1], np.shape(imageresize)[2]
        imageresize = np.reshape(imageresize, (D, H, W, 1))
        imageresize = np.transpose(imageresize, (3, 0, 1, 2))
        # predict patch
        stepx = self.image_width // 2
        stepy = self.image_height // 2
        stepz = self.image_depth // 2
        # check reize image size is or not smaller than net input size
        x_size = max(W, self.image_width)
        y_size = max(H, self.image_height)
        z_size = max(D, self.image_depth)
        newimageresize = np.zeros((1, z_size, y_size, x_size))
        newimageresize[:, 0:D, 0:H, 0:W] = imageresize[:, :, :, :]

        out_mask = np.zeros((z_size, y_size, x_size))
        out_mask_weight = np.zeros((z_size, y_size, x_size))
        for z in range(0, D, stepz):
            for y in range(0, H, stepy):
                for x in range(0, W, stepx):
                    x_min = x
                    x_max = x_min + self.image_width
                    if x_max > W:
                        x_max = W
                        x_min = W - self.image_width
                    y_min = y
                    y_max = y_min + self.image_height
                    if y_max > H:
                        y_max = H
                        y_min = H - self.image_height
                    z_min = z
                    z_max = z_min + self.image_depth
                    if z_max > D:
                        z_max = D
                        z_min = D - self.image_depth
                    patch_xs = imageresize[:, z_min:z_max, y_min:y_max, x_min:x_max]
                    mean = patch_xs.mean()
                    std = patch_xs.std()
                    patch_xs = (patch_xs - mean) / std
                    predictresult = self.predict(patch_xs)
                    predictresult = predictresult * std + mean
                    out_mask[z_min:z_max, y_min:y_max, x_min:x_max] = out_mask[
                                                                      z_min:z_max,
                                                                      y_min:y_max,
                                                                      x_min:x_max] + predictresult.copy()
                    out_mask_weight[z_min:z_max, y_min:y_max, x_min:x_max] = out_mask_weight[
                                                                             z_min:z_max,
                                                                             y_min:y_max,
                                                                             x_min:x_max] + 1.
        # resize mask to src image size,should rewrite
        out_mask = out_mask / out_mask_weight
        out_mask = np.around(out_mask)
        out_mask[out_mask < 0] = 0
        final_out_mask_sitk = sitk.GetImageFromArray(out_mask)
        final_out_mask_sitk.SetOrigin(imagesitk.GetOrigin())
        final_out_mask_sitk.SetSpacing(imagesitk.GetSpacing())
        final_out_mask_sitk.SetDirection(imagesitk.GetDirection())
        return final_out_mask_sitk

    def clear_GPU_cache(self):
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())
