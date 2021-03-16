from os.path import join
from torchvision import transforms, datasets


def get_dataset(project_root, data, train_or_test=None):

    transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize(7),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    if data == "mnist" and train_or_test:
        return datasets.MNIST(
            join(project_root, "data/raw"),
            download=True,
            train=train_or_test == "train",
            transform=transform,
        )

    elif data == "cifar10" and train_or_test:
        return datasets.CIFAR10(
            join(project_root, "data/raw/CIFAR10"),
            download=True,
            train=train_or_test == "train",
            transform=transform,
        )

    elif data == "svhn" and train_or_test:
        return datasets.SVHN(
            join(project_root, "data/raw/SVHN"),
            download=True,
            split=train_or_test,
            transform=transform,
        )
