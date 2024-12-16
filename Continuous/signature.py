from __future__ import annotations

from dataclasses import dataclass
from os import listdir
from os.path import isfile, join

import numpy as np
import signatory
import torch
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


@dataclass
class Signatures:
    path: str = "./Aftershock/group0/"
    depth: int = 2
    device: str = "gpu" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        trajectories = [
            join(self.path, f)
            for f in listdir(self.path)
            if isfile(join(self.path, f)) and "stateData" in f
        ]

        trajectories = sorted(trajectories)[0:7]

        trajectories = {
            int(t.split("stateData")[-1].split(".")[0]):
            np.transpose(np.load(t, allow_pickle=True)["O_Optimal"], [1, 0])
            for t in trajectories
        }

        self.signatures = {
            y: self.get_all_signatures(trajectory)
            for y, trajectory in trajectories.items()
        }

    def get_all_signatures(self, path: list[list[float]]) -> list[list[float]]:
        path = torch.from_numpy(path)[None]
        signature_path = signatory.Path(path, self.depth)

        signatures = [(1,)]

        for i in range(2, path.shape[1] + 1):
            signature = signature_path.signature(0, i)[0]
            signature = tuple(signature.tolist())
            signatures.append(signature)

        assert len(signatures) == path.shape[1]
        return tuple(signatures)

    def get_signature(self, path: list[list[float]]) -> list[float]:
        if path.shape[0] == 1:
            return [1]

        path = torch.from_numpy(path)[None]
        return signatory.signature(path, depth=self.depth).tolist()


if __name__ == "__main__":
    x = Signatures(device="cpu")

    loaded_data1 = np.load('./Aftershock/group0/stateData01.npz', allow_pickle=True)
    loaded_data2 = np.load('./Aftershock/group0/stateData01.npz', allow_pickle=True)



    O_Optimal1 = loaded_data1['O_Optimal']
    O_Optimal2 = loaded_data2['O_Optimal']

    sig1 = x.get_all_signatures(O_Optimal1[0:2, :].T)
    sig2 = x.get_all_signatures(O_Optimal2[0:2, :].T)
    #sig = x.get_all_signatures(O_Optimal[0:2, :].T)[0]

    distance_dtw, path = fastdtw(sig1[1:10+1], sig1[1:], dist=euclidean)
    print(distance_dtw)