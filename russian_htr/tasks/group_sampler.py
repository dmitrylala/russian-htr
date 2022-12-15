import numpy as np

from torch.utils.data import Dataset, BatchSampler


class GroupSampler(BatchSampler):
	def __init__(self, data_src: Dataset, batch_size: int, drop_last: bool = False):
		"""
		BatchSampler with following logic:
			Each batch should contain samples from the same group.
			Indexes of such samples are stored in mapping with group_ids as keys.
		"""
		super().__init__(sampler=None, batch_size=batch_size, drop_last=drop_last)
		if not hasattr(data_src, 'group2idxs'):
			raise ValueError(f"{data_src} should have group2idxs attribute")

		self.group2idxs = data_src.group2idxs.copy()

	def __iter__(self):
		group2idxs = self.group2idxs.copy()
		while group2idxs:
			group_id = np.random.choice(list(group2idxs.keys()))

			if len(group2idxs[group_id]) > self.batch_size:
				idxs = np.random.choice(group2idxs[group_id], self.batch_size, replace=False)
				group2idxs[group_id] = np.setdiff1d(group2idxs[group_id], idxs)
			else:
				idxs = group2idxs.pop(group_id)

				if self.drop_last:
					continue

			yield idxs
