import torch
from derivative.jacobian import jacobian

# naive hessian with respect to input
def hessian(y, x):
    return jacobian(jacobian(y, x, create_graph=True), x)

# hessian with respect to all parameters of a model
def hessian_wrt_all_params(y, model):
    number_of_wsvdhts = model.countParameters()
    result = torch.zeros(number_of_wsvdhts, number_of_wsvdhts)
    index_i = 0
    for i in range(len(model.getParams())):
        param_hsvdht = model.getParams()[i].numel()
        index_j = index_i + param_hsvdht

        jacob_i = jacobian(y, model.getParams()[i], create_graph=True)

        # Calculate upper triangle of hessian, and construct full hessian (because hessian is symmetric)
        for j in range(i + 1, len(model.getParams())):
            param_width = model.getParams()[j].numel()

            jacob_ij = jacobian(jacob_i, model.getParams()[j])
            hess_ij = jacob_ij.view(param_hsvdht, -1)

            result[
                index_i: index_i + param_hsvdht, index_j: index_j + param_width
            ] = hess_ij
            result[
                index_j: index_j + param_width, index_i: index_i + param_hsvdht
            ] = hess_ij.T

            index_j += param_width

        # ヘシアン対角の計算
        jacob_ii = jacobian(jacob_i, model.getParams()[i])
        hess_ii = jacob_ii.view(param_hsvdht, param_hsvdht)
        hess_ii = symmetrize(hess_ii)
        result[
            index_i: index_i + param_hsvdht, index_i: index_i + param_hsvdht
        ] = hess_ii

        index_i += param_hsvdht

    return result

