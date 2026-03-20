import os
import sys
import csv
import random

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

sys.path.insert(0, PAPER_DIR)
sys.path.insert(0, REPO_ROOT)

import torch  # type: ignore
import torchvision  # type: ignore
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
import absl.flags
import absl.app
from torch.utils.data import DataLoader, Dataset, Subset  # type: ignore

import utils.datasets as datasets
import utils.utils as utils

# user flags
absl.flags.DEFINE_string("path_model", None, "Path to the trained model checkpoint")
absl.flags.DEFINE_integer("batch_size_test", 8, "Batch size for test samples")
absl.flags.DEFINE_string("dir_dataset", "../datasets/", "Directory where datasets are stored")
absl.flags.DEFINE_integer("num_images", 20, "Number of wrong-prediction examples to analyze")
absl.flags.DEFINE_integer("min_abs_index", 20, "Only use examples whose absolute dataset index is >= this value")
absl.flags.DEFINE_integer("memory_set_size", 100, "Number of memory samples in each custom memory set")
absl.flags.DEFINE_integer("recycle_top_k", 20, "Number of top-weight original memory samples to keep in recycling strategy")
absl.flags.DEFINE_integer("seed", 42, "Random seed")
absl.flags.mark_flag_as_required("path_model")

FLAGS = absl.flags.FLAGS


def resolve_absolute_index(dataset, idx):
    """Resolve the absolute index even if the dataset is wrapped in Subset."""
    if isinstance(dataset, Subset):
        parent_idx = dataset.indices[idx]
        return resolve_absolute_index(dataset.dataset, parent_idx)
    return idx


class IndexedDataset(Dataset):
    """Wrap a dataset so __getitem__ returns image, label, absolute_index."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        abs_idx = resolve_absolute_index(self.dataset, idx)
        return image, label, abs_idx


def get_class_names(dataset_name, num_classes):
    if dataset_name in ["CIFAR10", "CINIC10"]:
        return [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]
    return [str(i) for i in range(num_classes)]


def get_image(image, undo_normalization_fn, revert_norm=True):
    if revert_norm:
        im = undo_normalization_fn(image)
    else:
        im = image
    im = im.squeeze().cpu().detach().numpy()
    transformed_im = np.transpose(im, (1, 2, 0))
    return transformed_im


def collect_memory_by_class(memory_dataset):
    """
    Returns:
        all_memory_images: list of tensors
        all_memory_labels: list of ints
        class_to_indices: dict[class_label] -> list of indices
    """
    all_memory_images = []
    all_memory_labels = []
    class_to_indices = {}

    for idx in range(len(memory_dataset)):
        image, label = memory_dataset[idx]
        label_int = int(label)
        all_memory_images.append(image)
        all_memory_labels.append(label_int)

        if label_int not in class_to_indices:
            class_to_indices[label_int] = []
        class_to_indices[label_int].append(idx)

    return all_memory_images, all_memory_labels, class_to_indices


def build_memory_tensor_from_indices(all_memory_images, indices, device):
    selected = [all_memory_images[i] for i in indices]
    memory_tensor = torch.stack(selected, dim=0).to(device)
    return memory_tensor


def sample_predicted_class_memory(class_to_indices, predicted_label, memory_set_size, rng):
    indices = class_to_indices[int(predicted_label)]
    if len(indices) >= memory_set_size:
        chosen = rng.sample(indices, memory_set_size)
    else:
        chosen = [rng.choice(indices) for _ in range(memory_set_size)]
    return chosen


def sample_balanced_fallback_memory(class_to_indices, memory_set_size, rng):
    """
    Build a fallback memory set spread across all classes.
    """
    classes = sorted(class_to_indices.keys())
    num_classes = len(classes)
    per_class = max(1, memory_set_size // num_classes)

    chosen = []
    for c in classes:
        indices = class_to_indices[c]
        if len(indices) >= per_class:
            chosen.extend(rng.sample(indices, per_class))
        else:
            chosen.extend(rng.choice(indices) for _ in range(per_class))

    while len(chosen) < memory_set_size:
        c = rng.choice(classes)
        chosen.append(rng.choice(class_to_indices[c]))

    if len(chosen) > memory_set_size:
        chosen = chosen[:memory_set_size]

    return chosen


def build_top_weight_recycled_memory(
    original_memory_batch,
    sorted_idx_row,
    mem_val_row,
    class_to_indices,
    all_memory_images,
    device,
    memory_set_size,
    recycle_top_k,
    rng
):
    """
    Keep the top positively weighted samples from the original memory batch,
    then fill the rest with a balanced fallback memory set from the full memory dataset.
    """
    positive_sorted = sorted_idx_row[mem_val_row > 0].detach().cpu().tolist()
    keep_local = positive_sorted[:recycle_top_k]

    kept_tensors = [original_memory_batch[i].detach().cpu() for i in keep_local]

    num_needed = max(0, memory_set_size - len(kept_tensors))
    fallback_indices = sample_balanced_fallback_memory(class_to_indices, num_needed, rng) if num_needed > 0 else []
    fallback_tensors = [all_memory_images[i] for i in fallback_indices]

    combined = kept_tensors + fallback_tensors

    # If there were more kept samples than memory_set_size, trim.
    combined = combined[:memory_set_size]

    memory_tensor = torch.stack(combined, dim=0).to(device)
    return memory_tensor


def make_memory_grid(memory_tensor, weights_row, sorted_idx_row, undo_normalization_fn):
    """
    Build visualization of only positively contributing memory samples.
    """
    positive_indices = sorted_idx_row[weights_row > 0]

    if len(positive_indices) > 0:
        reduced_mem = undo_normalization_fn(memory_tensor[positive_indices])
        npimg = torchvision.utils.make_grid(reduced_mem, nrow=4).cpu().numpy()
        mem_img = (np.transpose(npimg, (1, 2, 0)) * 255).astype(np.uint8)
    else:
        mem_img = np.zeros((64, 64, 3), dtype=np.uint8)

    return mem_img


def save_result_figure(
    save_path,
    input_image,
    abs_idx,
    true_label,
    pred_label,
    class_names,
    mem_img,
    title_prefix,
    undo_normalization_fn
):
    fig = plt.figure(figsize=(6, 6), dpi=300)

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.imshow(
        (get_image(input_image, undo_normalization_fn) * 255).astype(np.uint8),
        interpolation="nearest",
        aspect="equal"
    )
    ax1.set_title(
        "{}\nAbsolute Index: {}\nTrue Class: {} | Predicted Class: {}".format(
            title_prefix,
            abs_idx,
            class_names[true_label],
            class_names[pred_label]
        ),
        fontsize=10
    )
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.imshow(mem_img, interpolation="nearest", aspect="equal")
    ax2.set_title("Used Memory Samples", fontsize=10)
    ax2.axis("off")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def evaluate_single_image(model, image_tensor, memory_tensor):
    """
    image_tensor: shape [1, C, H, W]
    memory_tensor: shape [M, C, H, W]
    """
    with torch.no_grad():
        outputs, rw = model(image_tensor, memory_tensor, return_weights=True)
        _, prediction = torch.max(outputs, 1)
        mem_val, memory_sorted_index = torch.sort(rw, descending=True)

    return int(prediction.item()), rw[0], mem_val[0], memory_sorted_index[0]


def run(path: str, dataset_dir: str):
    rng = random.Random(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    checkpoint = torch.load(path, map_location=device)
    modality = checkpoint["modality"]
    if modality not in ["memory", "encoder_memory"]:
        raise ValueError(
            "Model modality must be one of ['memory', 'encoder_memory'], not {}".format(modality)
        )

    dataset_name = checkpoint["dataset_name"]
    model = utils.get_model(
        checkpoint["model_name"],
        checkpoint["num_classes"],
        model_type=modality
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    class_names = get_class_names(dataset_name, checkpoint["num_classes"])

    train_examples = checkpoint["train_examples"]
    load_dataset = getattr(datasets, "get_" + dataset_name)
    undo_normalization = getattr(datasets, "undo_normalization_" + dataset_name)

    _, _, test_loader, mem_loader = load_dataset(
        dataset_dir,
        batch_size_train=50,
        batch_size_test=FLAGS.batch_size_test,
        batch_size_memory=100,
        size_train=train_examples
    )

    indexed_test_dataset = IndexedDataset(test_loader.dataset)
    test_loader = DataLoader(
        indexed_test_dataset,
        batch_size=FLAGS.batch_size_test,
        shuffle=False
    )

    memory_dataset = mem_loader.dataset
    all_memory_images, all_memory_labels, class_to_indices = collect_memory_by_class(memory_dataset)

    dir_save = os.path.join(
        "..", "images", "mem_images", dataset_name, modality,
        checkpoint["model_name"], "strategy_analysis"
    )
    os.makedirs(dir_save, exist_ok=True)

    summary_csv = os.path.join(dir_save, "summary.csv")
    rows = []

    memory_iter = iter(mem_loader)
    saved_count = 0

    with torch.no_grad():
        for batch_idx, (images, labels, abs_indices) in enumerate(test_loader):
            print(
                "Batch: {}/{} | Analyzed: {}/{}".format(
                    batch_idx + 1, len(test_loader), saved_count, FLAGS.num_images
                ),
                end="\r"
            )

            if saved_count >= FLAGS.num_images:
                break

            try:
                default_memory_batch, _ = next(memory_iter)
            except StopIteration:
                memory_iter = iter(mem_loader)
                default_memory_batch, _ = next(memory_iter)

            images = images.to(device)
            labels = labels.to(device)
            default_memory_batch = default_memory_batch.to(device)

            outputs, rw = model(images, default_memory_batch, return_weights=True)
            _, predictions = torch.max(outputs, 1)

            mem_val, memory_sorted_index = torch.sort(rw, descending=True)

            for ind in range(len(images)):
                if saved_count >= FLAGS.num_images:
                    break

                true_label = int(labels[ind].item())
                original_pred = int(predictions[ind].item())
                abs_idx = int(abs_indices[ind])

                if original_pred == true_label:
                    continue

                if abs_idx < FLAGS.min_abs_index:
                    continue

                sample_dir = os.path.join(
                    dir_save,
                    "wrong_{:02d}_absidx_{}".format(saved_count, abs_idx)
                )
                os.makedirs(sample_dir, exist_ok=True)

                input_selected = images[ind].unsqueeze(0)

                # -------------------------------------------------
                # 1) ORIGINAL MEMORY BATCH RESULT
                # -------------------------------------------------
                original_mem_img = make_memory_grid(
                    default_memory_batch,
                    mem_val[ind],
                    memory_sorted_index[ind],
                    undo_normalization
                )

                save_result_figure(
                    save_path=os.path.join(sample_dir, "original.png"),
                    input_image=input_selected,
                    abs_idx=abs_idx,
                    true_label=true_label,
                    pred_label=original_pred,
                    class_names=class_names,
                    mem_img=original_mem_img,
                    title_prefix="Original Memory Batch",
                    undo_normalization_fn=undo_normalization
                )

                # -------------------------------------------------
                # 2) PREDICTED-CLASS-ONLY MEMORY
                # -------------------------------------------------
                predicted_class_indices = sample_predicted_class_memory(
                    class_to_indices,
                    original_pred,
                    FLAGS.memory_set_size,
                    rng
                )
                predicted_class_memory = build_memory_tensor_from_indices(
                    all_memory_images,
                    predicted_class_indices,
                    device
                )

                pc_pred, pc_rw_row, pc_mem_val_row, pc_sorted_idx_row = evaluate_single_image(
                    model,
                    input_selected,
                    predicted_class_memory
                )

                predicted_class_mem_img = make_memory_grid(
                    predicted_class_memory,
                    pc_mem_val_row,
                    pc_sorted_idx_row,
                    undo_normalization
                )

                save_result_figure(
                    save_path=os.path.join(sample_dir, "predicted_class_only.png"),
                    input_image=input_selected,
                    abs_idx=abs_idx,
                    true_label=true_label,
                    pred_label=pc_pred,
                    class_names=class_names,
                    mem_img=predicted_class_mem_img,
                    title_prefix="Predicted-Class-Only Memory",
                    undo_normalization_fn=undo_normalization
                )

                # -------------------------------------------------
                # 3) TOP-WEIGHT MEMORY RECYCLING
                # -------------------------------------------------
                recycled_memory = build_top_weight_recycled_memory(
                    original_memory_batch=default_memory_batch,
                    sorted_idx_row=memory_sorted_index[ind],
                    mem_val_row=mem_val[ind],
                    class_to_indices=class_to_indices,
                    all_memory_images=all_memory_images,
                    device=device,
                    memory_set_size=FLAGS.memory_set_size,
                    recycle_top_k=FLAGS.recycle_top_k,
                    rng=rng
                )

                tw_pred, tw_rw_row, tw_mem_val_row, tw_sorted_idx_row = evaluate_single_image(
                    model,
                    input_selected,
                    recycled_memory
                )

                recycled_mem_img = make_memory_grid(
                    recycled_memory,
                    tw_mem_val_row,
                    tw_sorted_idx_row,
                    undo_normalization
                )

                save_result_figure(
                    save_path=os.path.join(sample_dir, "top_weight_recycling.png"),
                    input_image=input_selected,
                    abs_idx=abs_idx,
                    true_label=true_label,
                    pred_label=tw_pred,
                    class_names=class_names,
                    mem_img=recycled_mem_img,
                    title_prefix="Top-Weight Memory Recycling",
                    undo_normalization_fn=undo_normalization
                )

                rows.append({
                    "wrong_example_id": saved_count,
                    "abs_idx": abs_idx,
                    "true_label": true_label,
                    "true_class_name": class_names[true_label],
                    "original_pred": original_pred,
                    "original_pred_name": class_names[original_pred],

                    "predicted_class_only_pred": pc_pred,
                    "predicted_class_only_changed": int(pc_pred != original_pred),
                    "predicted_class_only_corrected": int(pc_pred == true_label),

                    "top_weight_recycling_pred": tw_pred,
                    "top_weight_recycling_changed": int(tw_pred != original_pred),
                    "top_weight_recycling_corrected": int(tw_pred == true_label),
                })

                saved_count += 1
                print("Analyzed {}/{} wrong predictions".format(saved_count, FLAGS.num_images), end="\r")

    print()

    fieldnames = [
        "wrong_example_id",
        "abs_idx",
        "true_label",
        "true_class_name",
        "original_pred",
        "original_pred_name",
        "predicted_class_only_pred",
        "predicted_class_only_changed",
        "predicted_class_only_corrected",
        "top_weight_recycling_pred",
        "top_weight_recycling_changed",
        "top_weight_recycling_corrected",
    ]

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("Done. Saved analysis to {}".format(dir_save))

    if rows:
        pc_changed = sum(int(r["predicted_class_only_changed"]) for r in rows)
        pc_corrected = sum(int(r["predicted_class_only_corrected"]) for r in rows)
        tw_changed = sum(int(r["top_weight_recycling_changed"]) for r in rows)
        tw_corrected = sum(int(r["top_weight_recycling_corrected"]) for r in rows)

        print("Predicted-class-only changed: {}/{}".format(pc_changed, len(rows)))
        print("Predicted-class-only corrected: {}/{}".format(pc_corrected, len(rows)))
        print("Top-weight recycling changed: {}/{}".format(tw_changed, len(rows)))
        print("Top-weight recycling corrected: {}/{}".format(tw_corrected, len(rows)))
    else:
        print("No qualifying wrong predictions found.")

    if saved_count < FLAGS.num_images:
        print(
            "Warning: only found {} wrong predictions with absolute index >= {}.".format(
                saved_count, FLAGS.min_abs_index
            )
        )


def main(argv):
    del argv
    run(FLAGS.path_model, FLAGS.dir_dataset)


if __name__ == "__main__":
    absl.app.run(main)