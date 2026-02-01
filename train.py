import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import WikiText2Dataset
from model import LMConfig, Transformer


@dataclass
class ExperimentConfig:
    name: str
    num_layers: int
    d_model: int
    max_context_len: int
    batch_size: int
    num_heads: int
    dropout_rate: float
    num_epochs: int
    learning_rate: float


def load_data(max_context_len, batch_size):
    train_dataset = WikiText2Dataset(split="train", sequence_len=max_context_len)
    val_dataset = WikiText2Dataset(split="validation", sequence_len=max_context_len)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, train_dataset.vocab_size


def create_model(vocab_size, config, device):
    lm_config = LMConfig(
        vocab_size=vocab_size,
        max_context_len=config.max_context_len,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout_rate=config.dropout_rate,
    )
    model = Transformer(lm_config).to(device)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    num_batches = 0

    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)

        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validation", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)

            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def train_model(config, device=None):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"\n{'=' * 80}")
    print(f"Experiment: {config.name}")
    print(f"{'=' * 80}")

    train_loader, val_loader, vocab_size = load_data(
        config.max_context_len, config.batch_size
    )

    model = create_model(vocab_size, config, device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    criterion = nn.CrossEntropyLoss()

    num_params = count_parameters(model)
    print(f"Model: {config.num_layers} layers, {config.d_model} d_model")
    print(f"Sequence length: {config.max_context_len}, Batch size: {config.batch_size}")
    print(f"Parameters: {num_params / 1e6:.2f}M")
    print(f"Device: {device}")

    train_losses = []
    val_losses = []
    train_perplexities = []
    val_perplexities = []

    for epoch in range(1, config.num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        train_ppl = np.exp(train_loss)
        val_ppl = np.exp(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_perplexities.append(float(train_ppl))
        val_perplexities.append(float(val_ppl))

        print(
            f"Epoch {epoch:2d}/{config.num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train PPL: {train_ppl:.2f} | Val PPL: {val_ppl:.2f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        scheduler.step()

    return {
        "experiment_name": config.name,
        "config": {
            "num_layers": config.num_layers,
            "d_model": config.d_model,
            "max_context_len": config.max_context_len,
            "num_heads": config.num_heads,
            "dropout_rate": config.dropout_rate,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "learning_rate": config.learning_rate,
            "vocab_size": vocab_size,
        },
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_perplexities": train_perplexities,
        "val_perplexities": val_perplexities,
        "num_params": num_params,
        "best_val_ppl": min(val_perplexities),
    }


def save_result_to_json(result, output_dir, run_id=None):
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_name = result["experiment_name"]

    if run_id:
        filename = f"{experiment_name}_{run_id}.json"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{timestamp}.json"

    filepath = output_dir / filename

    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved results: {filename}")
    return filepath


def print_summary_table(results):
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"{'Model':<45} {'Layers':<8} {'Params (M)':<12} {'Val PPL':<10}")
    print("-" * 80)

    for result in results:
        config = result["config"]
        num_params = result["num_params"] / 1e6
        val_ppl = result["best_val_ppl"]

        model_name = f"TransformerLM (block size {config['max_context_len']})"

        print(
            f"{model_name:<45} {config['num_layers']:<8} {num_params:<12.2f} {val_ppl:<10.2f}"
        )

    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Transformer language model with custom configuration"
    )

    parser.add_argument("--name", type=str, required=True, help="Experiment name")
    parser.add_argument(
        "--num-layers", type=int, required=True, help="Number of transformer layers"
    )
    parser.add_argument("--d-model", type=int, required=True, help="Model dimension")
    parser.add_argument(
        "--max-context-len", type=int, required=True, help="Maximum context length"
    )
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size")

    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Number of attention heads (default: 4)",
    )
    parser.add_argument(
        "--dropout-rate", type=float, default=0.1, help="Dropout rate (default: 0.1)"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="training_results",
        help="Output directory for results (default: training_results)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run identifier for output filename",
    )

    return parser.parse_args()


def config_from_args(args):
    return ExperimentConfig(
        name=args.name,
        num_layers=args.num_layers,
        d_model=args.d_model,
        max_context_len=args.max_context_len,
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
    )


def main():
    args = parse_args()
    experiment = config_from_args(args)

    result = train_model(experiment)

    output_dir = Path(args.output_dir)
    save_path = save_result_to_json(result, output_dir, args.run_id)

    print_summary_table([result])

    print(f"\nResults saved to: {save_path.absolute()}")


if __name__ == "__main__":
    main()
