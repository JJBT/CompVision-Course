import torch
from torch import nn
import numpy as np
import time


class ResNet(nn.Module):
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def __init__(self, num_blocks, num_classes=10):
        super(ResNet, self).__init__()

        self.cur_channels = 64

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(out_channels=64, num_blocks=num_blocks[0], stride=1)
        self.layer2 = self._make_layer(out_channels=128, num_blocks=num_blocks[1])
        self.layer3 = self._make_layer(out_channels=256, num_blocks=num_blocks[2])
        self.layer4 = self._make_layer(out_channels=512, num_blocks=num_blocks[3])

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, num_classes)

        self.activation = nn.ReLU(inplace=True)

    def _make_layer(self, out_channels, num_blocks, stride=2):
        layers = list()
        layers.append(ResBlock(in_channels=self.cur_channels, out_channels=out_channels, stride=stride))

        for _ in range(num_blocks - 1):
            layers.append(ResBlock(in_channels=out_channels, out_channels=out_channels))

        self.cur_channels = out_channels
        return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.ReLU(inplace=True)

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            # Option B
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=2,
                    bias=False
                ),
                nn.BatchNorm2d(num_features=out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.activation(out)

        return out


def fit(net, X_train, y_train, X_test=None, y_test=None, n_epochs=30):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1.0e-3)

    batch_size = 100

    test_accuracy_history = []
    test_loss_history = []

    if (X_test is not None) and (y_test is not None):
        X_test = X_test.to(device)
        y_test = y_test.to(device)

    start_time = time.time()

    for epoch in range(n_epochs):
        order = np.random.permutation(X_train.shape[0])
        for start_index in range(0, X_train.shape[0], batch_size):
            optimizer.zero_grad()
            net.train()

            batch_indexes = order[start_index : start_index + batch_size]

            X_batch = X_train[batch_indexes].to(device)
            y_batch = y_train[batch_indexes].to(device)

            preds = net.forward(X_batch)

            loss_value = loss(preds, y_batch)
            loss_value.backward()

            optimizer.step()

        print('Epoch: {0}\t'
              'Time {time:.3f}\t'.format(epoch, time=time.time() - start_time))

        if (X_test is not None) and (y_test is not None):
            net.eval()
            test_preds = net.forward(X_test)
            test_loss = loss(test_preds, y_test).data.cpu()
            test_loss_history.append(test_loss)

            accuracy = (test_preds.argmax(dim=1) == y_test).float().mean().data.cpu()
            test_accuracy_history.append(accuracy)

            print('Loss {loss:.3f}\t'
                  'Accuracy {accuracy:.3f}'.format(loss=test_loss, accuracy=accuracy))

        return test_accuracy_history, test_loss_history
