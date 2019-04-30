from utils.dataset import SeismicDataset, ToTensor
from utils.models import AccelerationPredictor

from torchvision import transforms
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch import nn

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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


def train(model, loader, optimizer, criterion, seismo_mean, seismo_std, velocity_mean, velocity_std,
          writer=None, global_step=None, name=None, normalize=True,):
    model.train()
    train_losses = AverageMeter()

    for idx, batch in enumerate((loader)):
        x = torch.FloatTensor(batch['seismogram']).to(device)
        y = torch.FloatTensor(batch['velocity']).to(device)

        if normalize:
            x = (x - seismo_mean) / seismo_std
        else:
            x = torch.log(torch.abs(x))  # torch.log(x)
            x[1 - torch.isfinite(x)] = 0.0
        y = (y - velocity_mean) / velocity_std

        y_pred = model(x)

        loss = criterion(y, y_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.update(loss.item(), x.size(0))

        if writer is not None:
            writer.add_scalar(f"{name}/train_loss.avg", train_losses.avg, global_step=global_step + idx)

    return train_losses.avg


def validate(model, loader, criterion, seismo_mean, seismo_std, velocity_mean, velocity_std,
             writer=None, global_step=None, name=None, normalize=True):
    model.eval()
    validate_losses = AverageMeter()

    for idx, batch in enumerate((loader)):
        x = torch.FloatTensor(batch['seismogram']).to(device)
        y = torch.FloatTensor(batch['velocity']).to(device)

        if normalize:
            x = (x - seismo_mean) / seismo_std
        else:
            x = torch.log(torch.abs(x))  # torch.log(x)
            x[1 - torch.isfinite(x)] = 0.0
        y = (y - velocity_mean) / velocity_std

        y_pred = model(x)

        loss = criterion(y, y_pred)
        validate_losses.update(loss.item(), x.size(0))

        if writer is not None:
            writer.add_scalar(f"{name}/val_loss.avg", validate_losses.avg, global_step=global_step + idx)
    return validate_losses.avg


def calculate_mean_and_std(train_dataset):
    seismogram_stack = train_dataset[0]['seismogram'][None]
    velocity_stack = torch.FloatTensor(train_dataset[0]['velocity'][None])

    for i in range(1, len(train_dataset)):
        _el = train_dataset[i]
        seismogram_stack = torch.cat([seismogram_stack, (_el['seismogram'][None])], dim=0)
        velocity_stack = torch.cat([velocity_stack, torch.FloatTensor(_el['velocity'][None])], dim=0)

    seismo_mean = torch.mean(seismogram_stack).to(device)
    seismo_std = torch.std(seismogram_stack).to(device)

    velocity_mean = torch.mean(velocity_stack).to(device)
    velocity_std = torch.std(velocity_stack).to(device)

    return seismo_mean, seismo_std, velocity_mean, velocity_std


if __name__ == '__main__':
    batch_size = 32
    experiment_dir_name = 'training_logs/1'
    save_model_each = 10
    normalize = True
    num_epoch = 100

    train_dataset = SeismicDataset(seismo_dir='data/train/raw/',
                                   velocity_dir='data/train/outputs/',
                                   transform=transforms.Compose([ToTensor()]))
    val_dataset = SeismicDataset(seismo_dir='data/val/raw/',
                                 velocity_dir='data/val/outputs/',
                                 transform=transforms.Compose([ToTensor()]))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    print("Dataset is loaded.")

    seismo_mean, seismo_std, velocity_mean, velocity_std = calculate_mean_and_std(train_dataset)
    print("Mean and std are calculated.")

    model = AccelerationPredictor().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    criterion = nn.MSELoss().to(device)

    writer = SummaryWriter(log_dir=os.path.join(experiment_dir_name, 'logs'))
    os.makedirs(experiment_dir_name, exist_ok=True)

    print("Training is started:")
    for epoch in tqdm(range(num_epoch)):

        train_loss = train(model=model, loader=train_loader, optimizer=optimizer, criterion=criterion,
                           writer=writer, global_step=len(train_loader.dataset) * epoch,
                           name=f"{experiment_dir_name}_by_batch", normalize=normalize,
                           seismo_mean=seismo_mean, seismo_std=seismo_std,
                           velocity_mean=velocity_mean, velocity_std=velocity_std)

        val_loss = validate(model=model, loader=val_loader, criterion=criterion,
                            writer=writer, global_step=len(train_loader.dataset) * epoch,
                            name=f"{experiment_dir_name}_by_batch", normalize=normalize,
                            seismo_mean=seismo_mean, seismo_std=seismo_std,
                            velocity_mean=velocity_mean, velocity_std=velocity_std)

        model_name = f"emd_loss_epoch_{epoch}_train_{train_loss}_{val_loss}.pth"

        if epoch % save_model_each == 0:
            torch.save(model.state_dict(), os.path.join(experiment_dir_name, model_name))

        writer.add_scalar(f"{experiment_dir_name}_by_epoch/train_loss", train_loss, global_step=epoch)
        writer.add_scalar(f"{experiment_dir_name}_by_epoch/val_loss", val_loss, global_step=epoch)

        lr_scheduler.step()

        print("Epoch: {}, Train: {}, Val: {}".format(epoch, train_loss, val_loss))

    writer.export_scalars_to_json(os.path.join(experiment_dir_name, 'all_scalars.json'))
    writer.close()
