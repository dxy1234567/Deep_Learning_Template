"""
This script is modified from the work of Abdelrahman Eldesokey.
Find more details from https://github.com/abdo-eldesokey/nconv
"""


from trainers.trainer import Trainer  # from CVLPyDL repo
import torch
import matplotlib.pyplot as plt
import os.path
from utils.AverageMeter import AverageMeter
from utils.saveTensorToImage import *
from utils.ErrorMetrics import *
from torchvision import transforms
import time
from modules.losses import *
import cv2
from utils.util import *

err_metrics = ['MAE()', 'RMSE()','iMAE()', 'iRMSE()']

class KittiDepthTrainer(Trainer):
    def __init__(self, net, params, optimizer, objective, lr_scheduler, dataloaders,
                 workspace_dir, sets=['train', 'val'], use_load_checkpoint=None, K= None):

        # Call the constructor of the parent class (trainer)
        super(KittiDepthTrainer, self).__init__(net, optimizer, lr_scheduler, objective, use_gpu=params['use_gpu'],
                                                workspace_dir=workspace_dir)

        self.lr_scheduler = lr_scheduler
        self.dataloaders = dataloaders
        self.use_load_checkpoint = use_load_checkpoint

        self.params = params
        self.save_chkpt_each = params['save_chkpt_each']
        self.sets = sets
        self.save_images = params['save_out_imgs']
        self.load_rgb = params['load_rgb'] if 'load_rgb' in params else False

        self.exp_name = params['exp_name']

        for s in self.sets: self.stats[s + '_loss'] = []

    ####### Training Function #######

    def train(self, max_epochs):
        print('#############################\n### Experiment Parameters ###\n#############################')
        for k, v in self.params.items(): print('{0:<22s} : {1:}'.format(k, v))

        # Load last save checkpoint
        if self.use_load_checkpoint != None:
            if isinstance(self.use_load_checkpoint, int):
                if self.use_load_checkpoint > 0:
                    print('=> Loading checkpoint {} ...'.format(self.use_load_checkpoint))
                    if self.load_checkpoint(self.use_load_checkpoint):
                        print('Checkpoint was loaded successfully!\n')
                    else:
                        print('Evaluating using initial parameters')
                elif self.use_load_checkpoint == -1:
                    print('=> Loading last checkpoint ...')
                    if self.load_checkpoint():
                        print('Checkpoint was loaded successfully!\n')
                    else:
                        print('Evaluating using initial parameters')
            elif isinstance(self.use_load_checkpoint, str):
                print('loading checkpoint from : ' + self.use_load_checkpoint)
                if self.load_checkpoint(self.use_load_checkpoint):
                    print('Checkpoint was loaded successfully!\n')
                else:
                    print('Evaluating using initial parameters')

        start_full_time = time.time()
        print('Start the %d th epoch at ' % self.epoch)
        print(time.strftime('%m.%d.%H:%M:%S', time.localtime(time.time())))

        # 记录损失值
        train_losses = []
        val_losses = []

        for epoch in range(self.epoch, max_epochs + 1):  # range function returns max_epochs-1
            start_epoch_time = time.time()

            self.epoch = epoch

            # Decay Learning Rate
           

            print('\nTraining Epoch {}: (lr={}) '.format(epoch, self.optimizer.param_groups[0]['lr']))  # , end=' '

            # Train the epoch

            for i in range (1):
                # 得到当前轮次的损失值
                loss_meter, loss_val = self.train_epoch()
                
                # 保存损失值
                train_losses.append(loss_meter['train'].avg)
                val_losses.append(loss_val.avg)

                # Add the average loss for this epoch to stats
                for s in self.sets: self.stats[s + '_loss'].append(loss_meter[s].avg)

            self.lr_scheduler.step()  # LR decay

            # Save checkpoint
            if self.use_save_checkpoint and (self.epoch) % self.save_chkpt_each == 0:
                self.save_checkpoint()
                print('\n => Checkpoint was saved successfully!\n')

            end_epoch_time = time.time()
            print('End the %d th epoch at ' % self.epoch)
            print(time.strftime('%m.%d.%H:%M:%S\n', time.localtime(time.time())))
            epoch_duration = end_epoch_time - start_epoch_time
            self.training_time += epoch_duration
            if self.params['print_time_each_epoch']:
                print('Have trained %.2f HRs, and %.2f HRs per epoch, [%s]\n' % (
                    self.training_time / 3600, epoch_duration / 3600, self.exp_name))

        # Save the final model
        torch.save(self.net, self.workspace_dir + '/final_model.pth')

        print("Training [%s] Finished using %.2f HRs." % (self.exp_name, self.training_time / 3600))
        
        # 保存损失值折线图
        plot_losses(train_losses, val_losses, 'output/losses/losses.png')
        write_log(train_losses, val_losses, 'output/log/log.txt')

        return self.net
    
    def calculate_eval(self):
        device = torch.device("cuda:" + str(self.params['gpu_id']) if torch.cuda.is_available() else "cpu")

        loss_val = AverageMeter()

        with torch.no_grad():
            for data in self.dataloaders["val"]:

                    torch.cuda.synchronize()

                    inputs_d, labels, item_idxs, inputs_gray, C = data
                    inputs_d = inputs_d.to(device)
                    C = C.to(device)
                    labels = labels.to(device)
                    inputs_gray = inputs_gray.to(device)

                    outputs = self.net(inputs_d, inputs_gray)


                    if len(outputs) > 1:
                        outputs = outputs[0]

                    torch.cuda.synchronize()

                    # Calculate loss for valid pixel in the ground truth
                    loss = self.objective(outputs, labels, self.epoch)

                    # statistics
                    loss_val.update(loss.item(), inputs_d.size(0))

                    torch.cuda.empty_cache()
        return loss_val

    def train_epoch(self):
        """
            训练一轮并返回该轮训练的损失值
        """
        device = torch.device("cuda:" + str(self.params['gpu_id']) if torch.cuda.is_available() else "cpu")

        loss_meter = {}
        for s in self.sets: loss_meter[s] = AverageMeter()
        for i in range(1):
            for s in self.sets:
                for data in self.dataloaders[s]:
                    start_iter_time = time.time()
                    inputs_d, labels, item_idxs, inputs_gray, C = data
                    C = C.to(device)
                    inputs_d = inputs_d.to(device)
                    labels = labels.to(device)
                    inputs_gray = inputs_gray.to(device)

                    outputs = self.net(inputs_d, inputs_gray)
                    # Calculate loss for valid pixel in the ground truth
                    loss11 = self.objective(outputs[0], labels)
                    loss12 = self.objective(outputs[1], labels)
                    loss14 = self.objective(outputs[2], labels)
                    loss18 = self.objective(outputs[3], labels)

                    if self.epoch < 5:
                        loss = loss18 + loss14 + loss12 + loss11
                    elif self.epoch < 10:
                        loss = 0.1 * loss18 + 0.1 * loss14 + 0.1 * loss12 + loss11
                    else:
                        loss = loss11

                    # backward + optimize only if in training phase
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()


                    # statistics
                    loss_meter[s].update(loss11.item(), inputs_d.size(0))

                    end_iter_time = time.time()
                    iter_duration = end_iter_time - start_iter_time
                    if self.params['print_time_each_iter']:
                        print('finish the iteration in %.2f s.\n' % (
                            iter_duration))
                        print('Loss within the curt iter: {:.8f}\n'.format(loss_meter[s].avg))

            print('[{}] Loss: {:.8f}'.format(s, loss_meter[s].avg))
            #
            loss_val = self.calculate_eval()
            print('[{}] Loss: {:.8f}'.format("val", loss_val.avg))
            torch.cuda.empty_cache()

        return loss_meter, loss_val

    ####### Evaluation Function #######

    def evaluate(self):
        print('< Evaluate mode ! >')

        # Load last save checkpoint

        if self.use_load_checkpoint != None:
            if isinstance(self.use_load_checkpoint, int):
                if self.use_load_checkpoint > 0:
                    print('=> Loading checkpoint {} ...'.format(self.use_load_checkpoint))
                    if self.load_checkpoint(self.use_load_checkpoint):
                        print('Checkpoint was loaded successfully!\n')
                    else:
                        print('Evaluating using initial parameters')
                elif self.use_load_checkpoint == -1:
                    print('=> Loading last checkpoint ...')
                    if self.load_checkpoint():
                        print('Checkpoint was loaded successfully!\n')
                    else:
                        print('Evaluating using initial parameters')
            elif isinstance(self.use_load_checkpoint, str):
                print('loading checkpoint from : ' + self.use_load_checkpoint)
                if self.load_checkpoint(self.use_load_checkpoint):
                    print('Checkpoint was loaded successfully!\n')
                else:
                    print('Evaluating using initial parameters')

        self.net.train(False)
        self.net.eval()

        # AverageMeters for Loss
        loss_meter = {}
        for s in self.sets: loss_meter[s] = AverageMeter()

        # AverageMeters for error metrics
        err = {}
        for m in err_metrics: err[m] = AverageMeter()

        # AverageMeters for time
        times = AverageMeter()

        device = torch.device("cuda:" + str(self.params['gpu_id']) if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            for s in self.sets:
                print('Evaluating on [{}] set, Epoch [{}] ! \n'.format(s, str(self.epoch - 1)))
                # Iterate over data.
                Start_time = time.time()
                for data in self.dataloaders[s]:

                    torch.cuda.synchronize()
                    start_time = time.time()

                    inputs_d, labels, item_idxs, inputs_gray, C = data
                    inputs_d = inputs_d.to(device)
                    C = C.to(device)
                    labels = labels.to(device)
                    inputs_gray = inputs_gray.to(device)

                    outputs = self.net(inputs_d, inputs_gray)


                    if len(outputs) > 1:
                        outputs = outputs[0]

                    torch.cuda.synchronize()
                    duration = time.time() - start_time
                    times.update(duration / inputs_d.size(0), inputs_d.size(0))

                    if s == 'selval' or s == 'val' or s == 'test':

                        # Calculate loss for valid pixel in the ground truth
                        loss = self.objective(outputs, labels, self.epoch)

                        # statistics
                        loss_meter[s].update(loss.item(), inputs_d.size(0))


                        # Convert data to depth in meters before error metrics
                        outputs[outputs == 0] = -1
                        if not self.load_rgb:
                            outputs[outputs == outputs[0, 0, 0, 0]] = -1
                        labels[labels == 0] = -1
                        if self.params['invert_depth']:
                            outputs = 1 / outputs
                            labels = 1 / labels
                        outputs[outputs == -1] = 0
                        labels[labels == -1] = 0
                        outputs *= self.params['data_normalize_factor'] / 1000
                        labels *= self.params['data_normalize_factor'] / 1000

                        # Calculate error metrics
                        for m in err_metrics:
                            if m.find('Delta') >= 0:
                                fn = globals()['Deltas']()
                                error = fn(outputs, labels)
                                err['Delta1'].update(error[0], inputs_d.size(0))
                                err['Delta2'].update(error[1], inputs_d.size(0))
                                err['Delta3'].update(error[2], inputs_d.size(0))
                                break
                            else:
                                fn = eval(m)  # globals()[m]()
                                error = fn(outputs, labels)
                                err[m].update(error.item(), inputs_d.size(0))

                    # Save output images (optional)

                    if s in ['test']:
                        outputs = outputs.data

                        outputs *= 1000

                        saveTensorToImage(outputs, item_idxs, os.path.join(self.workspace_dir,
                                                                           s + '_output_' + 'epoch_' + str(
                                                                               self.epoch)))



                average_time = (time.time() - Start_time) / len(self.dataloaders[s].dataset)

                print('Evaluation results on [{}]:\n============================='.format(s))
                print('[{}]: {:.8f}'.format('Loss', loss_meter[s].avg))
                for m in err_metrics: print('[{}]: {:.8f}'.format(m, err[m].avg))
                print('[{}]: {:.4f}'.format('Time', times.avg))
                print('[{}]: {:.4f}'.format('Time_av', average_time))

                # Save evaluation metric to text file
                fname = 'error_' + s + '_epoch_' + str(self.epoch - 1) + '.txt'
                with open(os.path.join(self.workspace_dir, fname), 'w') as text_file:
                    text_file.write(
                        'Evaluation results on [{}], Epoch [{}]:\n==========================================\n'.format(
                            s, str(self.epoch - 1)))
                    text_file.write('[{}]: {:.8f}\n'.format('Loss', loss_meter[s].avg))
                    for m in err_metrics: text_file.write('[{}]: {:.8f}\n'.format(m, err[m].avg))
                    text_file.write('[{}]: {:.4f}\n'.format('Time', times.avg))

                torch.cuda.empty_cache()


    def predict(self, depth, gray, gt=None):
        """
            Params:
                gt: ground truth depth map, used for evaluation

            
        """
        # 模型读取存档点
        if self.load_checkpoint(self.use_load_checkpoint):
            print('Checkpoint was loaded successfully!\n')

        device = torch.device("cuda:" + str(self.params['gpu_id']) if torch.cuda.is_available() else "cpu")
        depth = depth.to(device)
        gray = gray.to(device)

        with torch.no_grad():
            outputs = self.net(depth, gray)

        # 得到tensor型的输出
        output = outputs[0]
        if gt == None:
            return output
        else:
            loss_meter = AverageMeter()
            err = {}

            loss = self.objective(output, gt, self.epoch)
            loss_meter.update(loss.item(), depth.size(0))

            err_metrics = ['MAE', 'RMSE()','iMAE', 'iRMSE']
            for m in err_metrics:
                err[m] = AverageMeter()

            for m in err_metrics:  
                fn = eval(m)  # globals()[m]()
                error = fn(output, gt)
                err[m].update(error.item(), depth.size(0))

            print('Evaluation results:\n=============================')
            print('[{}]: {:.8f}'.format('Loss', loss_meter.avg))
            for m in err_metrics: print('[{}]: {:.8f}'.format(m, err[m].avg))
            
            return output, err

    def preprocess(self, depth, gray, norm_size):
        crop = transforms.CenterCrop((480, 832))

        # 输入原始数据，要进行数据处理。
        depth = np.array(depth, dtype=np.float16)
        gray = np.array(gray, dtype=np.float16) 

        # 数据大小调整
        depth = depth / norm_size  # [0,1]
        gray = gray / 255.0

        # Expand dims into Pytorch format
        depth = np.expand_dims(depth, 0)
        gray = np.expand_dims(gray, 0)
        depth = np.expand_dims(depth, 0)
        gray = np.expand_dims(gray, 0)

        # Convert to Pytorch Tensors
        depth = torch.tensor(depth, dtype=torch.float)
        gray = torch.tensor(gray, dtype=torch.float)

        depth = crop(depth)
        gray = crop(gray)

        return depth, gray




