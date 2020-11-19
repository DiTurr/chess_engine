"""
Implementation of chess engine based of DNN.

@author: DiTurr

"""
import torch
import torch.nn as nn
import torch.nn.functional as F  # NOQA
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt

NUM_RESIDUAL_LAYERS = 20
NUM_CHANNELS_CONV2D = 256


class AlphaZeroModel:
    def __init__(self, max_epochs, training_generator, validation_generator, loss_function=None, optimizer=None):
        """

        """
        # input parameters as attributes
        self.model = AlphaZeroNet()
        if torch.cuda.is_available():
            self.model.cuda()
        self.max_epochs = max_epochs
        self.training_generator = training_generator
        self.validation_generator = validation_generator
        if loss_function is None:
            self.loss_function = AlphaLoss()
        else:
            self.loss_function = loss_function
        if optimizer is None:
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        else:
            self.optimizer = optimizer

        # calculated attributes
        self.history = None

    def train(self):
        """

        """
        # CUDA for PyTorch
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        # loop over epochs
        self.history = {"loss": [], "val_loss": []}
        for epoch in range(self.max_epochs):
            # create progress bar
            pbar = tqdm(total=len(self.training_generator) + len(self.validation_generator),
                        ascii=True, ncols=100, dynamic_ncols=True,
                        desc=str(epoch + 1).zfill(5) + "/" + str(self.max_epochs).zfill(5) + ": ")

            # training
            running_loss_batch_train = 0
            mean_loss_batch_train = None
            with torch.set_grad_enabled(True):
                for index, (x_batch, y_policy_batch, y_winner_batch) in enumerate(self.training_generator):
                    # transfer to GPU
                    x_batch = x_batch.to(device)
                    y_policy_batch = y_policy_batch.to(device)
                    y_winner_batch = y_winner_batch.to(device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # set model to training mode
                    self.model.train()

                    # forward + backward + optimize
                    (y_hat_winner_batch, y_hat_policy_batch) = self.model(x_batch)
                    loss_batch_train = self.loss_function(y_winner_batch, y_hat_winner_batch, y_policy_batch,
                                                          y_hat_policy_batch)
                    loss_batch_train.backward()
                    self.optimizer.step()

                    # printing information/statistics
                    running_loss_batch_train += loss_batch_train.item()
                    mean_loss_batch_train = running_loss_batch_train / (index + 1)
                    pbar.update(1)
                    pbar.set_postfix_str('Loss: {:.4f}'.format(mean_loss_batch_train))

            # validation
            running_loss_batch_val = 0
            mean_loss_batch_val = None
            with torch.set_grad_enabled(False):
                for index, (x_batch, y_policy_batch, y_winner_batch) in enumerate(self.validation_generator):
                    # transfer to GPU
                    x_batch = x_batch.to(device)
                    y_policy_batch = y_policy_batch.to(device)
                    y_winner_batch = y_winner_batch.to(device)

                    # set model to evaluate mode
                    self.model.eval()

                    # model computations: forward
                    (y_hat_winner_batch, y_hat_policy_batch) = self.model(x_batch)
                    loss_batch_val = self.loss_function(y_winner_batch, y_hat_winner_batch, y_policy_batch,
                                                        y_hat_policy_batch)

                    # printing information/statistics
                    running_loss_batch_val += loss_batch_val.item()
                    mean_loss_batch_val = running_loss_batch_val / (index + 1)
                    pbar.update(1)
                    pbar.set_postfix_str('Loss: {:.4f}; Validation Loss: {:.4f}'.
                                         format(mean_loss_batch_train, mean_loss_batch_val))

            # save information
            if mean_loss_batch_train is not None:
                self.history["loss"].append(mean_loss_batch_train)
            if mean_loss_batch_val is not None:
                self.history["val_loss"].append(mean_loss_batch_val)

            # close progress bar
            pbar.close()

        # return the training history
        return self.history

    def plot_history(self):
        """

        """
        fig, axs = plt.subplots()
        axs.plot(self.history["loss"])
        axs.plot(self.history["val_loss"])
        axs.set_title("AlphaZero Losses")
        plt.grid()
        plt.show()

    def save_model(self, path_model_save):
        """

        """
        torch.save(self.model, path_model_save)

    def save_checkpoint(self, path_save_checkpoint):
        """

        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path_save_checkpoint)


class ConvolutionalBlock(nn.Module):
    def __init__(self):
        super(ConvolutionalBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=19, out_channels=NUM_CHANNELS_CONV2D, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(NUM_CHANNELS_CONV2D)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = F.relu(x)
        return x


class BackboneNet(nn.Module):
    def __init__(self):
        super(BackboneNet, self).__init__()
        self.convolutional_block = ConvolutionalBlock()
        for block in range(NUM_RESIDUAL_LAYERS):
            setattr(self, "residual_layer_%i" % block, ResidualLayer(in_channels=NUM_CHANNELS_CONV2D,
                                                                     out_channels=NUM_CHANNELS_CONV2D))

    def forward(self, x):
        x = self.convolutional_block(x)
        for block in range(NUM_RESIDUAL_LAYERS):
            x = getattr(self, "residual_layer_%i" % block)(x)
        return x


class ValueHead(nn.Module):
    def __init__(self):
        super(ValueHead, self).__init__()
        self.conv = nn.Conv2d(in_channels=NUM_CHANNELS_CONV2D, out_channels=3, kernel_size=1)
        self.bn = nn.BatchNorm2d(3)
        self.fc1 = nn.Linear(in_features=3 * 8 * 8, out_features=32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.fc1(x.view(-1, 3 * 8 * 8))
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        return x


class PolicyHead(nn.Module):
    def __init__(self):
        super(PolicyHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=NUM_CHANNELS_CONV2D, out_channels=73, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(73)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = x.view(-1, 8 * 8 * 73)
        x = self.logsoftmax(x).exp()
        return x


class AlphaZeroNet(nn.Module):
    def __init__(self):
        super(AlphaZeroNet, self).__init__()
        self.backbone = BackboneNet()
        self.value_head = ValueHead()
        self.policy_head = PolicyHead()

    def forward(self, x):
        x = self.backbone(x)
        winner = self.value_head(x)
        policy = self.policy_head(x)
        return winner, policy


class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    @staticmethod
    def forward(y_winner, y_hat_winner, y_policy, y_hat_policy):
        """

        """
        winner_error = ((y_winner - y_hat_winner) ** 2).view(-1)
        policy_error = torch.sum(y_hat_policy * (1e-10 + y_policy).log(), 1)
        total_error = (winner_error - policy_error).mean()
        return total_error
