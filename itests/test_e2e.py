from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from jackdaw.serializers.tensor import TorchSerializer
from tests.conftest import take_n

from jackdaw.artefact_decorator import artefacts

torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


@artefacts({TorchSerializer: ['conv1', 'conv2', 'fc1', 'fc2']})
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch, log_interval=10):
    model.train()
    for batch_idx, (data, target) in take_n(enumerate(train_loader), 10):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def eval(model, device, test_loader) -> List[float]:
    model.eval()
    test_loss = 0
    correct = 0
    predictions = []
    with torch.no_grad():
        for data, target in take_n(test_loader, 5):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            predictions += pred.tolist()
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return predictions


def assert_model_equivalence(model: torch.nn.Module, model_two: torch.nn.Module):
    assert (
            len(model_two.__artefact_slots__.keys()) > 0
            or len(model_two.__artefact_subclasses__) > 0
    )
    for submodel_name in model_two.__artefact_subclasses__:
        submodel = getattr(model, submodel_name)
        assert (
                len(submodel.__artefact_slots__.keys()) > 0
                or len(submodel.__artefact_subclasses__) > 0
        )
        sub_loaded_model = getattr(model_two, submodel_name)
        for artefact_name in submodel.__artefact_slots__.keys():
            if artefact_name in ["weight", "bias"]:
                assert torch.all(
                    getattr(submodel, artefact_name)
                    == getattr(sub_loaded_model, artefact_name)
                ), print(
                    f"Parameter didn't match in {artefact_name} for {submodel_name}"
                )


def test_save_load_pytorch_model():
    device = "cpu"
    lr = 1.0
    gamma = 0.7
    epochs = 1
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    dataset1 = datasets.MNIST(
        "../data", train=True, download=True, transform=transform
    )
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=100)
    test_loader = torch.utils.data.DataLoader(dataset2)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        _ = eval(model, device, test_loader)
        scheduler.step()

    test_predictions = eval(model, device, test_loader)
    model_id = model.dumps()

    loaded_model = Net()
    loaded_model.loads(model_id)
    loaded_model_predictions = eval(loaded_model, device, test_loader)

    assert_model_equivalence(model, loaded_model)
    assert all([x == y for (x, y) in zip(loaded_model_predictions, test_predictions)])


if __name__ == "__main__":
    test_save_load_pytorch_model()
