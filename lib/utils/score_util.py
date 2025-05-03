import torch


def compprob(p,
             children_map,
             group_sizes,
             n_samples,
             n_parents,
             device="cpu",
             **kwargs,
             ):

    group_sums = torch.zeros(n_samples, n_parents, device=device)
    group_sums.scatter_add_(1, children_map.expand(n_samples, -1), p)

    p_comp = 1.0 - group_sums

    return p, p_comp


def entcompprob(p,
                children_map,
                group_sizes,
                n_samples,
                n_parents,
                device="cpu",
                **kwargs,
                ):

    eps = 1e-12

    validate_tensor(p)

    group_sums = torch.zeros(n_samples, n_parents, device=device)
    group_sums.scatter_add_(1, children_map.expand(n_samples, -1), p + eps)
    p_norm = (p + eps) / (group_sums[:, children_map] + eps)

    validate_tensor(p_norm)

    x = torch.zeros(n_samples, n_parents, device=device)

    x.scatter_add_(1, children_map.expand(n_samples, -1), -1.0 * p_norm * torch.log(p_norm + eps))

    assert torch.isfinite(x).all()

    total_sums = group_sums + x
    result = p / total_sums[:, children_map]

    p_comp = x / total_sums

    # TODO: remove assertions when we are confident
    validate_tensor(p_comp)
    validate_tensor(result)

    result_sums = torch.zeros(n_samples, n_parents, device=device)
    result_sums.scatter_add_(1, children_map.expand(n_samples, -1), result)
    final_sums = result_sums + p_comp
    assert torch.allclose(final_sums, torch.ones_like(final_sums), atol=1e-4)

    return result, p_comp


def validate_tensor(tensor):
    assert not torch.isnan(tensor).any(), "Tensor contains NaN values!"
    assert not torch.isinf(tensor).any(), "Tensor contains Inf values!"
    assert (tensor >= 0.0).all(), "Tensor contains values less than 0.0!"
    assert (tensor <= 1.0).all(), "Tensor contains values greater than 1.0!"
