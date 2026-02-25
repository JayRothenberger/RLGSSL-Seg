import torch


def balance_samples(class_mask, factor=1.0, sorted=False):
    """
    return a sorted tensor of indices which when applied to the class mask will result
    in a tensor of labels which is balanced within a factor of the factor argument

    balanced tensor will have at most factor times more elements of the majority class than any other class.
    """
    assert factor >= 1, "factor must be greater than or equal to 1"

    unique_values = torch.unique(class_mask)

    value_inds = dict()
    min_len = float('inf')

    for value in unique_values:
        inds = torch.arange(class_mask.shape[0])[class_mask == value]
        value_inds[value] = inds
        min_len = min(min_len, len(inds))

    assert min_len > 0, "some class had no elements"

    if not sorted:
        return torch.cat([torch.arange(class_mask.shape[0])[v[:int(factor * min_len)]] for k, v in value_inds.items()], 0)[0]

    return torch.sort(torch.cat([torch.arange(class_mask.shape[0])[v[:int(factor * min_len)]] for k, v in value_inds.items()], 0))[0]