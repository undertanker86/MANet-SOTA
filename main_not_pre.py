import seaborn as sns
from pathlib import Path
import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import datetime
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
from model.manet import manet
from sklearn.metrics import confusion_matrix, classification_report
now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")
data_path = '/home/zhaozengqun/datasets_static/RAFDB_Face/'
checkpoint_path = ''

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='/data2/cmdir/home/ioit_thql/QDHManet/MA-Net/data/Heatmaps_RafDb_MP_DLIB_sigma_15')
parser.add_argument('--checkpoint_path', type=str, default='/data2/cmdir/home/ioit_thql/QDHManet/MA-Net/checkpoint/' + 'model.pth')
parser.add_argument('--best_checkpoint_path', type=str, default='/data2/cmdir/home/ioit_thql/QDHManet/MA-Net/checkpoint/'+'model_best.pth')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=60, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default=checkpoint_path, type=str, metavar='PATH', help='path to checkpoint')
parser.add_argument('-e', '--evaluate', default=False, action='store_true', help='evaluate model on test set')
parser.add_argument('--beta', type=float, default=0.6)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
print('beta', args.beta)
EMOTION_MAPPING = {
    1: "Anger",
    2: "Disgust",
    3: "Fear",
    4: "Happiness",
    5: "Sadness",
    6: "Surprise",
    7: "Neutral"
}
EMOTION_MAPPING_0 = {k-1: v for k, v in EMOTION_MAPPING.items()}
class FERPlusDataset(Dataset):
    """
    Dataset class for FERPlus dataset with pre-divided train/val/test splits
    """
    def __init__(self, root_dir, split='train', image_size=(224, 224), transform_type=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.transform_type = transform_type

        self.global_mean = [0.485, 0.456, 0.406]
        self.global_std = [0.229, 0.224, 0.225]

        # Prepare dataset
        self.images, self.labels = self._load_dataset()

        # Print dataset statistics
        self._print_statistics()
                # Prepare transforms
        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])
        self.image_transform = transforms.Compose([
                self.base_transform,
                transforms.Normalize(mean=self.global_mean, std=self.global_std)
        ])

        # Additional train transforms
        if transform_type == 'train':
            self.train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.RandomErasing(scale=(0.02, 0.1))
            ])
        else:
            # Identity transform for validation/test
            self.train_transform = lambda x: x

    def _load_dataset(self):
        """Load dataset from preprocessed .npz file"""
        saved_data_path = os.path.join(self.root_dir, f'{self.split}.npz')

        try:
            data = np.load(saved_data_path)
            return data['images'], data['labels']

        except Exception as e:
            print(f"Error loading dataset: {e}")
            return [], []
    def _print_statistics(self):
        """Print dataset statistics"""
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        total_images = len(self.labels)

        print(f"\n{self.split} set statistics:")
        print(f"Total images: {total_images}")

        for label, count in zip(unique_labels, counts):
            emotion = EMOTION_MAPPING.get(label, f"Unknown({label})")
            percentage = (count / total_images) * 100
            print(f"{emotion}: {count} images ({percentage:.2f}%)")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image, label
        image = self.images[idx]
        label = self.labels[idx] - 1  # Adjust to 0-indexed classes

        # Transform image
        transformed_image = self.image_transform(image)
        transformed_image = self.train_transform(transformed_image)

        return transformed_image, label
def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """Plot training and validation metrics and save figures separately."""

    # Define save paths
    loss_plot_path = "outputs/loss_plot.png"
    accuracy_plot_path = "outputs/accuracy_plot.png"

    # Plot and save loss separately
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to prevent overlap

    # Plot and save accuracy separately
    plt.figure(figsize=(8, 5))
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to prevent overlap

    # Show both plots
    plt.show()
def count_parameters(model):
    """Count trainable and total parameters of the model"""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    best_acc = 0
    print('Training time: ' + now.strftime("%m-%d %H:%M"))

    # create model
    model = manet()
    #count_parameters(model)
    #count_parameters(model)
    model = torch.nn.DataParallel(model).cuda()
    #checkpoint = torch.load('/data2/cmdir/home/ioit_thql/QDHManet/MA-Net/checkpoint/Pretrained_on_MSCeleb.pth.tar', weights_only=False)
    #pre_trained_dict = checkpoint['state_dict']
    #model.load_state_dict(pre_trained_dict)
    model.module.fc_1 = torch.nn.Linear(512, 7).cuda()
    model.module.fc_2 = torch.nn.Linear(512, 7).cuda()
    count_parameters(model)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    recorder = RecorderMeter(args.epochs)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            recorder = checkpoint['recorder']
            best_acc = best_acc.to()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True

    # Data loading code
    #traindir = os.path.join(args.data, 'train_split')
    #valdir = os.path.join(args.data, 'val')
    #testdir = os.path.join(args.data, 'test')
    train_dataset = FERPlusDataset(args.data, split='train_split', transform_type='train')
    val_dataset = FERPlusDataset(args.data, split='val')
    test_dataset = FERPlusDataset(args.data, split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=2)
    #train_dataset = datasets.ImageFolder(traindir,
      #                                   transforms.Compose([transforms.RandomResizedCrop((224, 224)),
     #                                                        transforms.RandomHorizontalFlip(),
       #                                                      transforms.ToTensor()]))

    #test_dataset = datasets.ImageFolder(valdir,
      #                                  transforms.Compose([transforms.Resize((224, 224)),
     #                                                       transforms.ToTensor()]))

    #train_loader = torch.utils.data.DataLoader(train_dataset,
     #                                          batch_size=args.batch_size,
      #                                         shuffle=True,
       #                                        num_workers=args.workers,
        #                                       pin_memory=True)
    #val_loader = torch.utils.data.DataLoader(test_dataset,
     #                                        batch_size=args.batch_size,
      #                                       shuffle=False,
       #                                      num_workers=args.workers,
        #                                     pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []  
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        print('Current learning rate: ', current_learning_rate)
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write('Current learning rate: ' + str(current_learning_rate) + '\n')

        # train for one epoch
        train_acc, train_los = train(train_loader, model, criterion, optimizer, epoch, args)
        train_losses.append(float(train_los))
        train_accs.append(float(train_acc))
        # evaluate on validation set
        val_acc, val_los = validate(val_loader, model, criterion, args)
        val_losses.append(float(val_los))
        val_accs.append(float(val_acc))
        scheduler.step()

        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        curve_name = time_str + 'cnn.png'
        recorder.plot_curve(os.path.join('./log/', curve_name))

        # remember best acc and save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        print('Current best accuracy: ', best_acc.item())
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write('Current best accuracy: ' + str(best_acc.item()) + '\n')

        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'recorder': recorder}, is_best, args)
        end_time = time.time()
        epoch_time = end_time - start_time
        print("An Epoch Time: ", epoch_time)
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write(str(epoch_time) + '\n')
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
            # Load the best model
    checkpoint = torch.load(args.best_checkpoint_path, weights_only=False)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
    # Nếu checkpoint là state_dict trực tiếp
    else:
            model.load_state_dict(checkpoint)

    # Evaluate on test set
    #time_str = datetime.datetime.now().strftime("%m-%d-%H-%M")
    testing(test_loader, model, criterion, args ,EMOTION_MAPPING_0)
def train(train_loader, model, criterion, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    for i, (images, target) in enumerate(train_loader):

        images = images.cuda()
        target = target.cuda()

        # compute output
        output1, output2 = model(images)
        output = (args.beta * output1) + ((1-args.beta) * output2)
        loss = (args.beta * criterion(output1, target)) + ((1-args.beta) * criterion(output2, target))

        # measure accuracy and record loss
        acc1, _ = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i)

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            output1, output2 = model(images)
            output = (args.beta * output1) + ((1-args.beta) * output2)
            loss = (args.beta * criterion(output1, target)) + ((1 - args.beta) * criterion(output2, target))

            # measure accuracy and record loss
            acc, _ = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc[0], images.size(0))

            if i % args.print_freq == 0:
                progress.display(i)

        print(' **** Accuracy {top1.avg:.3f} *** '.format(top1=top1))
        with open('./log/' + time_str + 'log.txt', 'a') as f:
            f.write(' * Accuracy {top1.avg:.3f}'.format(top1=top1) + '\n')
    return top1.avg, losses.avg

def testing(test_loader, model, criterion, args, emotion_map):
    print("Testing..................")
    #losses = AverageMeter('Loss', ':.4f')
    #top1 = AverageMeter('Accuracy', ':6.3f')
    #progress = ProgressMeter(len(val_loader),
                             #[losses, top1],
                             #prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_labels = []
    test_loss = 0.0
    test_samples = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(test_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            output1, output2 = model(images)
            output = (args.beta * output1) + ((1-args.beta) * output2)
            loss = (args.beta * criterion(output1, target)) + ((1 - args.beta) * criterion(output2, target))

            batch_size = images.size(0)
            test_loss += loss.item() * batch_size
            test_samples += batch_size

            # measure accuracy and record loss
            acc, _ = accuracy(output, target, topk=(1, 5))
            #losses.update(loss.item(), images.size
            #top1.update(acc[0], images.size(0))
            
            preds = output.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(target.cpu().numpy())
            #if i % args.print_freq == 0:
                #progress.display(i)
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    avg_loss = test_loss / test_samples

        # in kết quả
    print(f"\nTest Loss: {avg_loss:.4f}")
    # Accuracy toàn bộ
    total_acc = (y_pred == y_true).mean() * 100                      
    
    print(f"Test Accuracy: {total_acc:.3f}%")                                                                                            
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(emotion_map))))
    
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, None] * 100
    plt.figure(figsize=(10, 8))                                      
    sns.heatmap(cm_pct, annot=True, fmt='.2f', cmap='Blues',                     
    xticklabels=[emotion_map[i] for i in range(len(emotion_map))],
    yticklabels=[emotion_map[i] for i in range(len(emotion_map))])
    plt.title('Confusion Matrix (%) on Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    fig_path = f"outputs/confusion_matrix.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved at: {fig_path}")

    # 4. Classification report
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(emotion_map))),
        target_names=[emotion_map[i] for i in range(len(emotion_map))],
        digits=4
    )
    print("\nClassification Report:")
    print(report)

        #print(' **** Accuracy {top1.avg:.3f} *** '.format(top1=top1))
        #with open('./log/' + time_str + 'log.txt', 'a') as f:
            #f.write(' * Accuracy {top1.avg:.3f}'.format(top1=top1) + '\n')
    #return top1.avg, losses.avg

def save_checkpoint(state, is_best, args):
    torch.save(state, args.checkpoint_path)
    if is_best:
        shutil.copyfile(args.checkpoint_path, args.best_checkpoint_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 30
        self.epoch_losses[idx, 1] = val_loss * 30
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1800, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print('Saved figure')
        plt.close(fig)


if __name__ == '__main__':
    main()
