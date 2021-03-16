# import sys
# import os
# import numpy as np
# import torch

# module_path = os.path.abspath("../..")
# if module_path not in sys.path:
#     sys.path.append(module_path)

# from model.hidden1 import LitHidden1

# # from codes.toy_models import ThreeNN
# # from codes.toy_constants import CONST_TOY_v0103 as CONST
# # from codes.data import make_path
# # from codes.functions import count_parameters
# # import numpy as np

# import pandas as pd
# import matplotlib.pyplot as plt
# import plotly.express as px
# import copy


# def visualizeModelOutput(
#     df, p="target_p", y="target_y", fig_show=False, save_to_directory_as=None
# ):
#     fig_p = px.scatter(df, x="x1", y="x2", color=p, range_color=[0, 1])
#     fig_y = px.scatter(df, x="x1", y="x2", color=y, opacity=0.7)
#     if fig_show:
#         fig_p.show()
#         fig_y.show()
#     if save_to_directory_as is not None:
#         fig_p.write_image(save_to_directory_as + "_p.png")
#         fig_y.write_image(save_to_directory_as + "_y.png")


# def print_parameters(model):
#     print("parameters:")
#     for p in model.params:
#         print(p.shape)
#     print("...Total number of parameters:", count_parameters(model))


# def getToyModel():
#     model = ThreeNN()
#     print_parameters(model)

#     with torch.no_grad():
#         model.classifier[0].weight = torch.nn.Parameter(
#             torch.tensor(
#                 [
#                     np.array([-20, 0.5, 10, 0.01, -50]),
#                     np.array([20, 0.1, -10, -30, -10]),
#                 ]
#             ).T.float()
#         )
#         model.classifier[0].bias = torch.nn.Parameter(
#             torch.zeros_like(model.classifier[0].bias).float()
#         )  # .linspace(1, -1, steps=5).float())
#         model.classifier[2].weight = torch.nn.Parameter(
#             torch.tensor([[-1, 0.001, 10, -10, -10]]).float()
#         )
#         model.classifier[2].bias = torch.nn.Parameter(
#             torch.zeros_like(model.classifier[2].bias).float()
#         )

#     model.setParameters()
#     print_parameters(model)

#     # model_dict = {
#     #     "state_dict": model.state_dict(),
#     # }

#     # model_name = f"toy3nn_v{VERSION}_target.pt"
#     # save_to_path = make_path(CONST["DIRECTORY_OF_MODELS"], model_name)
#     # if SAVE:
#     #     torch.save(model_dict, save_to_path)
#     #     print("SAVED TO '{}'".format(save_to_path))
#     # else:
#     #     print("DID NOT SAVE '{}'".format(save_to_path))

#     return model


# def addOutputToDf(model, df, key):
#     input = torch.tensor(df.loc[:, "x1":"x2"].to_numpy()).float()
#     output = model(input)
#     y = [np.random.choice([1, 0], p=[p.item(), 1 - p.item()]) for p in output]
#     df[key + "_p"] = output.detach().numpy()
#     df[key + "_y"] = y
#     return df


# def generateToyData(model, N=10000, train_or_test="train"):
#     input = torch.rand(N, 2)
#     output = model(input)
#     print(min(output).item(), max(output).item())
#     y = [np.random.choice([1, 0], p=[p.item(), 1 - p.item()]) for p in output]
#     df = pd.DataFrame(
#         torch.cat((input, output), dim=1).detach().numpy(),
#         columns=["x1", "x2", "target_p"],
#     )
#     df["target_y"] = y
#     # write a pandas dataframe to gzipped CSV file
#     save_to_path = make_path(
#         CONST["DIRECTORY_OF_DATASETS"], f"toy3nn{VERSION}_{train_or_test}.csv.gz"
#     )
#     print("saving to :", save_to_path)
#     df.to_csv(save_to_path, index=False, compression="gzip")
#     # df.to_hdf(save_to_path, key='stage', mode='w')
#     print("saved")
#     return df


# def loadToyData(version, batch_size=5, shuffle=False, train_or_test="train"):
#     read_from_path = make_path(
#         CONST["DIRECTORY_OF_DATASETS"], f"toy3nn{version}_{train_or_test}.csv.gz"
#     )
#     print("load data: ", read_from_path)
#     df = pd.read_csv(read_from_path, sep=",", dtype=np.float64)
#     x = torch.tensor(df.loc[:, "x1":"x2"].to_numpy()).reshape(-1, batch_size, 2).float()
#     y = (
#         torch.tensor(df.loc[:, "target_y"].to_numpy())
#         .reshape(-1, batch_size, 1)
#         .float()
#     )
#     return df, x, y


# # if __name__ == "__main__":
# #     VERSION = "0103"
# #     SAVE = True
# #     TRAIN_OR_TEST = "train"
# #     model = getToyModel()
# #     df = generateToyData(model, N=10000, train_or_test=TRAIN_OR_TEST)
# #     df = addOutputToDf(model, df, "check")
# #     visualizeModelOutput(df, fig_show=True, p="check_p", y="check_y")
# #     print("done")