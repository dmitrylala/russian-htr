from torch import nn


def count_parameters(module: nn.Module) -> int:
	total = 0
	for p in module.parameters():
		if p.requires_grad:
			total += p.numel()
	return total
