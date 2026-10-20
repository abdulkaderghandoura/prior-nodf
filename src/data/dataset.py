from torch.utils.data import Dataset


class ObservationPoints(Dataset):

    def __init__(self, coords, signal=None, aux_coords=None):
        """
        coords: torch.tensor (N x 3) spatial coordinates
        signal: torch.tensor (N x M) M angular samples at each spatial location (optional)
        aux_coords: torch.tensor (N x S x 3) sub-voxel points of each location (optional)
        """
        super().__init__()
        self.coords = coords
        self.signal = signal
        self.aux_coords = aux_coords

    def __getitem__(self, idx):
        if self.aux_coords is not None:
            return {
                "coords": self.coords[idx, :],
                "signal": self.signal[idx, :],
                "aux_coords": self.aux_coords[idx, :]
            }

        if self.signal is not None:
            return {
                "coords": self.coords[idx, :],
                "signal": self.signal[idx, :]
            }

        return {"coords": self.coords[idx, :]}

    def __len__(self):
        return self.coords.shape[0]
