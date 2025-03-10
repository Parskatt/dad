import torch
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel
import gc
from pathlib import Path
import dad
from dad.types import Detector

class CheckPoint:
    def __init__(self, dir):
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: Detector,
        optimizer,
        lr_scheduler,
        n,
    ):
        assert model is not None
        if isinstance(model, (DataParallel, DistributedDataParallel)):
            model = model.module
        states = {
            "model": model.state_dict(),
            "n": n,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        }
        torch.save(states, self.dir / "model_latest.pth")
        dad.logger.info(f"Saved states {list(states.keys())}, at step {n}")

    def load(
        self,
        model: Detector,
        optimizer,
        lr_scheduler,
        n,
    ):
        if not (self.dir / "model_latest.pth").exists():
            return model, optimizer, lr_scheduler, n

        states = torch.load(self.dir / "model_latest.pth")
        if "model" in states:
            model.load_state_dict(states["model"])
        if "n" in states:
            n = states["n"] if states["n"] else n
        if "optimizer" in states:
            try:
                optimizer.load_state_dict(states["optimizer"])
            except Exception as e:
                dad.logger.warning(
                    f"Failed to load states for optimizer, with error {e}"
                )
        if "lr_scheduler" in states:
            lr_scheduler.load_state_dict(states["lr_scheduler"])
        dad.logger.info(f"Loaded states {list(states.keys())}, at step {n}")
        del states
        gc.collect()
        torch.cuda.empty_cache()
        return model, optimizer, lr_scheduler, n
