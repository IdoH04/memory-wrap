import os
import pickle
import random
import time
from typing import Dict, List, Optional, Tuple

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
    "Memory strategy for memory-based models: default or balanced"
)
absl.flags.mark_flag_as_required("modality")
FLAGS = absl.flags.FLAGS


def _extract_labels_from_dataset(dataset) -> Optional[np.ndarray]:
    """
    Try to recover labels from a dataset or nested Subset dataset.

    This is written defensively because torchvision datasets / subsets often store
    labels in slightly different places: labels, targets, or inside nested .dataset.
    """
    current = dataset

    # Unwrap nested torch.utils.data.Subset objects while remembering indices
    collected_indices = None
    while hasattr(current, "indices") and hasattr(current, "dataset"):
        subset_indices = np.array(current.indices)
        if collected_indices is None:
            collected_indices = subset_indices
        else:
            collected_indices = collected_indices[subset_indices]
        current = current.dataset

    labels = None
    for attr in ["labels", "targets"]:
        if hasattr(current, attr):
            labels = getattr(current, attr)
            break

    if labels is None:
        return None

    labels = np.array(labels)

    if collected_indices is not None:
        labels = labels[collected_indices]

    return labels


def _build_class_index_map(dataset) -> Optional[Dict[int, List[int]]]:
    """
    Build a mapping {class_id: [dataset_indices]} for the memory dataset.
    Returns None if labels cannot be recovered.
    """
    labels = _extract_labels_from_dataset(dataset)
    if labels is None:
        return None

    class_to_indices: Dict[int, List[int]] = {}
    unique_classes = np.unique(labels)

    for cls in unique_classes:
        class_to_indices[int(cls)] = np.where(labels == cls)[0].tolist()

    return class_to_indices


def _sample_balanced_memory_batch(
    dataset,
    class_to_indices: Dict[int, List[int]],
    batch_size: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a memory batch that is as balanced as possible across classes.

    If batch_size is not divisible by num_classes, the remainder is distributed
    by giving one extra sample to the first few classes in a shuffled class order.
    Sampling is done with replacement when a class has too few samples.
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

        # sample with replacement if needed
        replace = len(cls_indices) < num_to_sample
        chosen = np.random.choice(cls_indices, size=num_to_sample, replace=replace)
        sampled_indices.extend(chosen.tolist())

    # Shuffle the final sampled set so the batch is not class-blocked
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

    # Prepare memory strategy
    use_balanced_memory = FLAGS.memory_strategy.lower() == "balanced"

    class_to_indices = None
    if use_balanced_memory:
        class_to_indices = _build_class_index_map(mem_loader.dataset)
        if class_to_indices is None:
            print("[WARNING] Could not recover class labels from mem_loader.dataset.")
            print("[WARNING] Falling back to default memory sampling.")
            use_balanced_memory = False

    mem_iterator = iter(mem_loader)

    for epoch in range(1, num_epochs + 1):
        for batch_idx, (data, y) in enumerate(train_loader):
            optimizer.zero_grad()

            # input
            data = data.to(device)
            y = y.to(device)

            # memory input
            if use_balanced_memory:
                batch_size = mem_loader.batch_size if mem_loader.batch_size is not None else len(y)
                memory_input, _ = _sample_balanced_memory_batch(
                    dataset=mem_loader.dataset,
                    class_to_indices=class_to_indices,
                    batch_size=batch_size,
                    device=device
                )
            else:
                memory_input, _, mem_iterator = _get_default_memory_batch(
                    mem_loader, mem_iterator, device
                )

            # perform training step
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(data, memory_input)
                loss = loss_criterion(outputs, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # log stuff
            if batch_idx % FLAGS.log_interval == 0:
                print(
                    "Train Epoch: {} [({:.0f}%({})]\t".format(
                        epoch,
                        100.0 * batch_idx / len(train_loader),
                        len(train_loader.dataset)
                    ),
                    end="\r"
                )

        scheduler.step()  # increase scheduler step for each epoch

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

    # get dataset info
    dataset_name = config["dataset_name"]
    num_classes = config[dataset_name]["num_classes"]

    # training parameters
    loss_criterion = torch.nn.CrossEntropyLoss()

    # saving/loading stuff
    save = config["save"]
    path_saving_model = 'models/{}/{}/{}/{}/{}/'.format(
        dataset_name,
        FLAGS.modality,
        FLAGS.memory_strategy if FLAGS.modality in ['memory', 'encoder_memory'] else 'std',
        config['model'],
        config['train_examples']
    )
    if save and not os.path.isdir(path_saving_model):
        os.makedirs(path_saving_model)

    # optimizer parameters
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

        # get dataset
        train_loader, _, test_loader, mem_loader = utils.get_loaders(config, run)

        # training process
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

            # perform 5 times the validation to stabilize results
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

        # stats
        run_acc.append(best_acc)

        # save
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

        # log
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