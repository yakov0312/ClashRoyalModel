import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from Model.policy import ClashRLModel


class ActionDataset(Dataset):
    def __init__(self, path):
        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        self.samples = data["samples"]

        if not self.samples:
            raise RuntimeError("Dataset contains no samples.")

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        return (
            sample["global_features"].float(),
            sample["board"].float(),
            sample["entities"].float(),
            sample["entity_mask"].bool(),
            torch.tensor(sample["action"], dtype=torch.long),
        )


def findLatestCheckpoint(checkpointDir):
    checkpoints = list(checkpointDir.glob("policy_update_*.pt"))

    if not checkpoints:
        return None

    checkpoints.sort(
        key=lambda path: int(path.stem.rsplit("_", 1)[1])
    )

    return checkpoints[-1]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)

    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/selfplay_run",
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Load a specific starting policy instead of the latest checkpoint.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of passes over the imitation dataset.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--update-offset",
        type=int,
        default=1,
        help="Policy update number added to the starting checkpoint.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Explicit output checkpoint path.",
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Training device. CPU is the default.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Stop if val_loss does not improve for this many epochs. 0 disables early stopping.",
    )

    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    if args.update_offset < 0:
        raise ValueError("--update-offset cannot be negative")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device cuda, but CUDA is unavailable.")

    device = torch.device(args.device)

    print(f"Device: {device}")

    checkpointDir = Path(args.checkpoint_dir)
    checkpointDir.mkdir(parents=True, exist_ok=True)

    dataset = ActionDataset(args.dataset)

    valSize = max(
        1,
        int(len(dataset) * args.val_split),
    )

    trainSize = len(dataset) - valSize

    if trainSize < 1:
        raise RuntimeError(
            "Dataset is too small for the selected validation split."
        )

    trainDataset, valDataset = random_split(
        dataset,
        [trainSize, valSize],
        generator=torch.Generator().manual_seed(42),
    )

    trainLoader = DataLoader(
        trainDataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    valLoader = DataLoader(
        valDataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    model = ClashRLModel().to(device)

    startCheckpoint = None

    if args.model:
        startCheckpoint = Path(args.model)

        if not startCheckpoint.exists():
            raise FileNotFoundError(
                f"model checkpoint not found: {startCheckpoint}"
            )
    else:
        startCheckpoint = findLatestCheckpoint(checkpointDir)

        if startCheckpoint is None:
            raise FileNotFoundError(
                f"No policy checkpoints found in {checkpointDir}"
            )

    print(f"Loading policy: {startCheckpoint}")

    checkpoint = torch.load(
        startCheckpoint,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise RuntimeError(f"BAD MODEL WEIGHT: {name}")

    sourceUpdate = int(checkpoint.get("update", 0))
    targetUpdate = sourceUpdate + args.update_offset

    if args.output:
        outputPath = Path(args.output)
    else:
        outputPath = checkpointDir / f"policy_update_{targetUpdate:04d}.pt"

    if outputPath == startCheckpoint:
        raise RuntimeError(
            "Output checkpoint would overwrite the starting checkpoint."
        )

    print(f"Source update: {sourceUpdate}")
    print(f"Target update: {targetUpdate}")
    print(f"Epochs: {args.epochs}")
    print(f"Output: {outputPath}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
    )

    bestValLoss = float("inf")
    epochsSinceImprovement = 0
    bestState = None

    for epoch in range(1, args.epochs + 1):
        model.train()

        trainLoss = 0.0
        trainCorrect = 0
        trainCount = 0

        for (
            globalFeatures,
            board,
            entities,
            entityMask,
            actions,
        ) in trainLoader:

            globalFeatures = globalFeatures.to(
                device,
                non_blocking=True,
            )
            board = board.to(
                device,
                non_blocking=True,
            )
            entities = entities.to(
                device,
                non_blocking=True,
            )
            entityMask = entityMask.to(
                device,
                non_blocking=True,
            )
            actions = actions.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(set_to_none=True)

            logits, _, _ = model(
                globalFeatures,
                board,
                entities,
                entityMask,
            )

            loss = F.cross_entropy(
                logits,
                actions,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss at epoch {epoch}: {loss.item()}"
                )

            loss.backward()

            badGradient = False

            for name, parameter in model.named_parameters():
                if (
                    parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                ):
                    print(f"BAD GRADIENT: {name}")
                    badGradient = True

            if badGradient:
                optimizer.zero_grad(set_to_none=True)
                continue

            gradNorm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            if not torch.isfinite(gradNorm):
                print(
                    f"WARNING: non-finite gradients at epoch {epoch}: "
                    f"{gradNorm}"
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.step()

            trainLoss += loss.item() * actions.size(0)
            trainCorrect += (
                logits.argmax(dim=-1) == actions
            ).sum().item()
            trainCount += actions.size(0)

        if trainCount == 0:
            raise RuntimeError(
                "No valid training updates were performed."
            )

        trainLoss /= trainCount
        trainAccuracy = trainCorrect / trainCount

        model.eval()

        valLoss = 0.0
        valCorrect = 0
        valCount = 0

        with torch.no_grad():
            for (
                globalFeatures,
                board,
                entities,
                entityMask,
                actions,
            ) in valLoader:

                globalFeatures = globalFeatures.to(
                    device,
                    non_blocking=True,
                )
                board = board.to(
                    device,
                    non_blocking=True,
                )
                entities = entities.to(
                    device,
                    non_blocking=True,
                )
                entityMask = entityMask.to(
                    device,
                    non_blocking=True,
                )
                actions = actions.to(
                    device,
                    non_blocking=True,
                )

                logits, _, _ = model(
                    globalFeatures,
                    board,
                    entities,
                    entityMask,
                )

                loss = F.cross_entropy(
                    logits,
                    actions,
                )

                if not torch.isfinite(loss):
                    print(
                        f"WARNING: non-finite validation loss "
                        f"at epoch {epoch}: {loss.item()}"
                    )
                    valLoss = float("nan")
                    break

                valLoss += loss.item() * actions.size(0)
                valCorrect += (
                    logits.argmax(dim=-1) == actions
                ).sum().item()
                valCount += actions.size(0)

        if valCount > 0:
            valLoss /= valCount
            valAccuracy = valCorrect / valCount
        else:
            valLoss = float("nan")
            valAccuracy = 0.0

        print(
            f"Epoch {epoch:3d}/{args.epochs} "
            f"train_loss={trainLoss:.4f} "
            f"train_acc={trainAccuracy * 100:.2f}% "
            f"val_loss={valLoss:.4f} "
            f"val_acc={valAccuracy * 100:.2f}%"
        )

        if torch.isfinite(torch.tensor(valLoss)) and valLoss < bestValLoss:
            bestValLoss = valLoss
            epochsSinceImprovement = 0

            bestState = {
                "model_state_dict": {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in model.state_dict().items()
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": valLoss,
                "val_accuracy": valAccuracy,
            }
        else:
            epochsSinceImprovement += 1

            if (
                args.patience > 0
                and epochsSinceImprovement >= args.patience
            ):
                print(
                    f"Early stopping: no val_loss improvement for "
                    f"{args.patience} epochs"
                )
                break

    if bestState is None:
        raise RuntimeError(
            "No valid checkpoint was produced."
        )

    outputPath.parent.mkdir(parents=True, exist_ok=True)

    bestState.update(
        {
            "update": targetUpdate,
            "source_update": sourceUpdate,
            "imitation": True,
            "source_checkpoint": str(startCheckpoint),
            "dataset": str(args.dataset),
            "num_actions": 2305,
        }
    )

    torch.save(
        bestState,
        outputPath,
    )

    print()
    print("Finished")
    print(f"Source policy update : {sourceUpdate}")
    print(f"New policy update    : {targetUpdate}")
    print(f"Best val loss        : {bestValLoss:.4f}")
    print(f"Saved to             : {outputPath}")


if __name__ == "__main__":
    main()
