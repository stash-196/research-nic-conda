import torch


# jacobian with respect to input
def jacobian(y, x, create_graph=False):
    jac = []
    flat_y = y.reshape(-1)
    if len(flat_y) == 1:
        (grad_x,) = torch.autograd.grad(
            y, x, None, retain_graph=True, create_graph=create_graph
        )
        return grad_x.reshape(x.numel())

    grad_y = torch.zeros_like(flat_y)

    for i in range(len(flat_y)):
        grad_y[i] = 1.0
        (grad_x,) = torch.autograd.grad(
            flat_y, x, grad_y, retain_graph=True, create_graph=create_graph
        )
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.0
    return torch.stack(jac).reshape(y.shape + (x.numel(),))
