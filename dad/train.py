from typing import Iterable, Callable, Optional
import torch
from tqdm import tqdm
from dad.utils import to_best_device
import dad


def train_step(
    train_batch: dict[str, torch.Tensor],
    model: dad.Detector,
    objective: Callable[[dict, dict], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    grad_scaler: Optional[torch.amp.GradScaler] = None,
):
    optimizer.zero_grad()
    loss = objective(train_batch, model)
    if grad_scaler is not None:
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        loss.backward()
        optimizer.step()


def train_k_steps(
    n_0: int,
    k: int,
    dataloader: Iterable[dict[str, torch.Tensor]],
    model: dad.Detector,
    objective: Callable[[dict, dict], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    grad_scaler: Optional[torch.amp.GradScaler] = None,
    progress_bar: bool = True,
):
    for n in tqdm(range(n_0, n_0 + k), disable=not progress_bar, mininterval=10.0):
        batch = next(dataloader)
        model.train(True)
        batch = to_best_device(batch)
        train_step(
            train_batch=batch,
            model=model,
            objective=objective,
            optimizer=optimizer,
            grad_scaler=grad_scaler,
        )
        lr_scheduler.step()
        dad.GLOBAL_STEP += 1
