import os
import pickle
import random
import time
from typing import Dict, List, Tuple

import absl.app
import absl.flags
import numpy as np
import torch  # type: ignore
import yaml

import utils.utils as utils


# user flags
absl.flags.DEFINE_string("modality", None, "std, memory or encoder_memory")
absl.flags.DEFINE_bool("continue_train", False, "Whether to continue a stopped training run")
absl.flags.DEFINE_integer("log_interval", 100, "Log interval between prints during training process")
absl.flags.DEFINE_string(
    "memory_strategy",
    "default",
    "Memory strategy for memory-based models: default, balanced, or top_weight"
)
absl.flags.mark_flag_as_required("modality")
FLAGS = absl.flags.FLAGS


def _build_class_index_map(dataset) -> Dict[int, List[int]]:
    """
    Build a mapping {class_id: [dataset_indices]} by iterating directly over
    the dataset. This is robust to nested Subset objects.
    """
    class_to_indices: Dict[int, List[int]] = {}

    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label = int(label)
        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(idx)

    return class_to_indices


def _sample_balanced_memory_batch(
    dataset,
    class_to_indices: Dict[int, List[int]],
    batch_size: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a memory batch that is as balanced as possible across classes.
    """
    classes = list(class_to_indices.keys())
    random.shuffle(classes)

    num_classes = len(classes)
    if num_classes == 0:
        raise ValueError("No classes found in memory dataset for balanced sampling.")

    per_class = batch_size // num_classes
    remainder = batch_size % num_classes

    sampled_indices: List[int] = []

    for i, cls in enumerate(classes):
        num_to_sample = per_class + (1 if i < remainder else 0)
        cls_indices = class_to_indices[cls]

        if len(cls_indices) == 0:
            continue

        replace = len(cls_indices) < num_to_sample
        chosen = np.random.choice(cls_indices, size=num_to_sample, replace=replace)
        sampled_indices.extend(chosen.tolist())

    random.shuffle(sampled_indices)

    batch_imgs = []
    batch_labels = []
    for idx in sampled_indices:
        img, label = dataset[idx]
        batch_imgs.append(img)
        batch_labels.append(label)

    memory_input = torch.stack(batch_imgs).to(device)
    memory_labels = torch.tensor(batch_labels, device=device)

    return memory_input, memory_labels


def _build_top_weight_pool(
    dataset,
    fraction: float = 0.5
) -> List[int]:
    """
    Approximation of a top-weight recycling strategy:
    keep only a fixed subset of the memory dataset and reuse it more often.

    Since train.py does not expose internal memory weights directly, this
    implements the idea by repeatedly sampling from a reduced memory pool.
    """
    total = len(dataset)
    keep = max(1, int(total * fraction))
    all_indices = list(range(total))
    random.shuffle(all_indices)
    return all_indices[:keep]


def _sample_from_index_pool(
    dataset,
    index_pool: List[int],
    batch_size: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch from a specified pool of dataset indices.
    Sampling uses replacement if needed.
    """
    if len(index_pool) == 0:
        raise ValueError("Index pool is empty for memory sampling.")

    replace = len(index_pool) < batch_size
    sampled_indices = np.random.choice(index_pool, size=batch_size, replace=replace).tolist()
    random.shuffle(sampled_indices)

    batch_imgs = []
    batch_labels = []
    for idx in sampled_indices:
        img, label = dataset[idx]
        batch_imgs.append(img)
        batch_labels.append(label)

    memory_input = torch.stack(batch_imgs).to(device)
    memory_labels = torch.tensor(batch_labels, device=device)

    return memory_input, memory_labels


def _get_default_memory_batch(
    mem_loader: torch.utils.data.DataLoader,
    mem_iterator,
    device: torch.device
):
    """
    Get the next batch from mem_loader, restarting the iterator when needed.
    """
    try:
        memory_input, memory_labels = next(mem_iterator)
    except StopIteration:
        mem_iterator = iter(mem_loader)
        memory_input, memory_labels = next(mem_iterator)

    memory_input = memory_input.to(device)
    memory_labels = memory_labels.to(device)
    return memory_input, memory_labels, mem_iterator


def train_memory_model(
    model: torch.nn.Module,
    loaders: List[torch.utils.data.DataLoader],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss_criterion: torch.nn.modules.loss,
    num_epochs: int,
    device: torch.device
) -> torch.nn.Module:
    """
    Function to train a model with a Memory Wrap layer.

    loaders[0] = training loader
    loaders[1] = memory loader
    """
    train_loader, mem_loader = loaders

    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    strategy = FLAGS.memory_strategy.lower()
    use_balanced_memory = strategy == "balanced"
    use_top_weight_memory = strategy == "top_weight"

    class_to_indices = None
    top_weight_pool = None

    if use_balanced_memory:
        class_to_indices = _build_class_index_map(mem_loader.dataset)

    if use_top_weight_memory:
        # Recycle a smaller fixed portion of the memory pool.
        # 50% is a simple approximation of "top memory weight recycling".
        top_weight_pool = _build_top_weight_pool(mem_loader.dataset, fraction=0.5)

    mem_iterator = iter(mem_loader)

    for epoch in range(1, num_epochs + 1):
        for batch_idx, (data, y) in enumerate(train_loader):
            optimizer.zero_grad()

            data = data.to(device)
            y = y.to(device)

            batch_size = mem_loader.batch_size if mem_loader.batch_size is not None else len(y)

            if use_balanced_memory:
                memory_input, _ = _sample_balanced_memory_batch(
                    dataset=mem_loader.dataset,
                    class_to_indices=class_to_indices,
                    batch_size=batch_size,
                    device=device
                )
            elif use_top_weight_memory:
                memory_input, _ = _sample_from_index_pool(
                    dataset=mem_loader.dataset,
                    index_pool=top_weight_pool,
                    batch_size=batch_size,
                    device=device
                )
            else:
                memory_input, _, mem_iterator = _get_default_memory_batch(
                    mem_loader, mem_iterator, device
                )

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(data, memory_input)
                loss = loss_criterion(outputs, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if batch_idx % FLAGS.log_interval == 0:
                print(
                    "Train Epoch: {} [({:.0f}%({})]\t".format(
                        epoch,
                        100.0 * batch_idx / len(train_loader),
                        len(train_loader.dataset)
                    ),
                    end="\r"
                )

        scheduler.step()

    return model


def train_std_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss_criterion: torch.nn.modules.loss,
    num_epochs: int,
    device: torch.device = torch.device("cpu")
) -> torch.nn.Module:
    """Function to train standard models."""
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(1, num_epochs + 1):
        for batch_idx, (data, y) in enumerate(train_loader):
            optimizer.zero_grad()

            data = data.to(device)
            y = y.to(device)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(data)
                loss = loss_criterion(outputs, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if batch_idx % FLAGS.log_interval == 0:
                print(
                    "Train Epoch: {} [({:.0f}%({})]\t".format(
                        epoch,
                        100.0 * batch_idx / len(train_loader),
                        len(train_loader.dataset)
                    ),
                    end="\r"
                )

        scheduler.step()

    return model


def run_experiment(config: dict, modality: str):
    """
    Method to run an experiment. Each experiment is composed of n runs,
    defined in the config dictionary, where in each of them a new model is trained.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:{}".format(device))
    print("Memory strategy: {}".format(FLAGS.memory_strategy))

    dataset_name = config["dataset_name"]
    num_classes = config[dataset_name]["num_classes"]

    loss_criterion = torch.nn.CrossEntropyLoss()

    save = config["save"]

    strategy_name = (
        FLAGS.memory_strategy if modality in ["memory", "encoder_memory"] else "std"
    )
    path_saving_model = "models/{}/{}/{}/{}/{}/".format(
        dataset_name,
        FLAGS.modality,
        strategy_name,
        config["model"],
        config["train_examples"]
    )

    if save and not os.path.isdir(path_saving_model):
        os.makedirs(path_saving_model)

    learning_rate = float(config["optimizer"]["learning_rate"])
    weight_decay = float(config["optimizer"]["weight_decay"])
    nesterov = bool(config["optimizer"]["nesterov"])
    momentum = float(config["optimizer"]["momentum"])
    dict_optim = {
        "lr": learning_rate,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "nesterov": nesterov
    }
    opt_milestones = config[dataset_name]["opt_milestones"]

    run_acc = []
    initial_run = 0

    if FLAGS.continue_train:
        print("Restarting training process\n")
        info = pickle.load(open(path_saving_model + "conf.p", "rb"))
        initial_run = info["run_num"]
        run_acc = info["accuracies"]

    for run in range(initial_run, config["runs"]):
        run_time = time.time()
        utils.set_seed(run)

        model = utils.get_model(config["model"], num_classes, model_type=modality)
        model = model.to(device)

        optimizer = torch.optim.SGD(model.parameters(), **dict_optim)

        if dataset_name == "CINIC10":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config[dataset_name]["num_epochs"]
            )
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=opt_milestones
            )

        train_loader, _, test_loader, mem_loader = utils.get_loaders(config, run)

        if modality == "memory" or modality == "encoder_memory":
            model = train_memory_model(
                model,
                [train_loader, mem_loader],
                optimizer,
                scheduler,
                loss_criterion,
                config[dataset_name]["num_epochs"],
                device=device
            )
            train_time = time.time()

            cum_acc = []
            init_eval_time = time.time()
            for _ in range(5):
                best_acc, best_loss = utils.eval_memory(
                    model, test_loader, mem_loader, loss_criterion, device
                )
                cum_acc.append(best_acc)
            best_acc = np.mean(cum_acc)
            end_eval_time = time.time()

        else:
            model = train_std_model(
                model,
                train_loader,
                optimizer,
                scheduler,
                loss_criterion,
                config[dataset_name]["num_epochs"],
                device
            )
            train_time = time.time()
            init_eval_time = time.time()
            best_acc, best_loss = utils.eval_std(model, test_loader, loss_criterion, device)
            end_eval_time = time.time()

        run_acc.append(best_acc)

        if save and path_saving_model:
            saved_name = "{}.pt".format(run + 1)
            save_path = os.path.join(path_saving_model, saved_name)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "train_examples": config["train_examples"],
                    "mem_examples": config[config["dataset_name"]]["mem_examples"],
                    "model_name": config["model"],
                    "num_classes": num_classes,
                    "modality": modality,
                    "dataset_name": config["dataset_name"],
                    "memory_strategy": FLAGS.memory_strategy,
                },
                save_path
            )
            info = {"run_num": run + 1, "accuracies": run_acc}
            pickle.dump(info, open(path_saving_model + "conf.p", "wb"))

        print(
            "Run:{} | Best Loss:{:.4f} | Accuracy {:.2f} | Mean Accuracy:{:.2f} | Std Dev Accuracy:{:.2f}\tT:{:.2f}min\tE:{:.2f}".format(
                run + 1,
                best_loss,
                best_acc,
                np.mean(run_acc),
                np.std(run_acc),
                (train_time - run_time) / 60,
                (end_eval_time - init_eval_time) / 60,
            )
        )


def main(argv):
    config_file = open(r"config/train.yaml")
    config = yaml.safe_load(config_file)

    print("Model:{}\nSizeTrain:{}\n".format(config["model"], config["train_examples"]))
    run_experiment(config, FLAGS.modality)


if __name__ == "__main__":
    absl.app.run(main)