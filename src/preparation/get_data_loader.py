from torch.utils.data import DataLoader


def get_data_loader(dataset, batch_size, num_workers=2):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
