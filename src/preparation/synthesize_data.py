from os.path import join
import numpy as np
import torch
import pandas as pd


def synthesize_data(f, f_name, project_root, input_dim, N=10000):
    input = torch.rand(N, input_dim)
    output = f(input)
    print("output:\n", output.shape, "input:\n", input.shape)
    y = [np.random.choice([1, 0], p=[p.item(), 1 - p.item()]) for p in output]
    df = pd.DataFrame(
        torch.cat((input, output), dim=1).detach().numpy(),
        columns=["x1", "x2", "target_p"],
    )
    df["target_y"] = y
    # write a pandas dataframe to gzipped CSV file

    path = join(project_root, f"data/raw/{f_name}.csv.gz")
    print("saving to :", path)
    df.to_csv(path, index=False, compression="gzip")
    print("saved")
    return df
