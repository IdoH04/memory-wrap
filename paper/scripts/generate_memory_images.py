import os
import sys

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
absl.flags.DEFINE_integer("num_images", 20, "Number of wrong-prediction explanation images to save")
absl.flags.DEFINE_integer("min_abs_index", 20, "Only save examples whose absolute dataset index is >= this value")
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


def run(path: str, dataset_dir: str):
    """
    Generate memory images only for wrong predictions.
    Each saved image includes:
    - absolute index
    - true class
    - predicted class
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    # load model
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

    # class names
    if dataset_name in ["CIFAR10", "CINIC10"]:
        name_classes = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]
    else:
        name_classes = [str(i) for i in range(checkpoint["num_classes"])]

    # load data
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

    memory_iter = iter(mem_loader)

    dir_save = os.path.join(
        "..", "images", "mem_images", dataset_name, modality,
        checkpoint["model_name"], "wrong_predictions"
    )
    os.makedirs(dir_save, exist_ok=True)

    def get_image(image, revert_norm=True):
        if revert_norm:
            im = undo_normalization(image)
        else:
            im = image
        im = im.squeeze().cpu().detach().numpy()
        transformed_im = np.transpose(im, (1, 2, 0))
        return transformed_im

    saved_count = 0

    with torch.no_grad():
        for batch_idx, (images, labels, abs_indices) in enumerate(test_loader):
            print(
                "Batch: {}/{} | Saved: {}/{}".format(
                    batch_idx + 1, len(test_loader), saved_count, FLAGS.num_images
                ),
                end="\r"
            )

            if saved_count >= FLAGS.num_images:
                break

            try:
                memory, _ = next(memory_iter)
            except StopIteration:
                memory_iter = iter(mem_loader)
                memory, _ = next(memory_iter)

            images = images.to(device)
            labels = labels.to(device)
            memory = memory.to(device)

            outputs, rw = model(images, memory, return_weights=True)
            _, predictions = torch.max(outputs, 1)

            mem_val, memory_sorted_index = torch.sort(rw, descending=True)

            for ind in range(len(images)):
                if saved_count >= FLAGS.num_images:
                    break

                true_label = labels[ind].item()
                pred_label = predictions[ind].item()
                abs_idx = int(abs_indices[ind])

                # wrong predictions only
                if pred_label == true_label:
                    continue

                # skip earliest dataset samples
                if abs_idx < FLAGS.min_abs_index:
                    continue

                input_selected = images[ind].unsqueeze(0)

                # memory samples with positive impact
                m_ec = memory_sorted_index[ind][mem_val[ind] > 0]

                if len(m_ec) > 0:
                    reduced_mem = undo_normalization(memory[m_ec])
                    npimg = torchvision.utils.make_grid(reduced_mem, nrow=4).cpu().numpy()
                    mem_img = (np.transpose(npimg, (1, 2, 0)) * 255).astype(np.uint8)
                else:
                    mem_img = np.zeros((64, 64, 3), dtype=np.uint8)

                fig = plt.figure(figsize=(6, 6), dpi=300)

                ax1 = fig.add_subplot(2, 1, 1)
                ax1.imshow(
                    (get_image(input_selected) * 255).astype(np.uint8),
                    interpolation="nearest",
                    aspect="equal"
                )
                ax1.set_title(
                    "Absolute Index: {}\nTrue Class: {} | Predicted Class: {}".format(
                        abs_idx,
                        name_classes[true_label],
                        name_classes[pred_label]
                    ),
                    fontsize=10
                )
                ax1.axis("off")

                ax2 = fig.add_subplot(2, 1, 2)
                ax2.imshow(mem_img, interpolation="nearest", aspect="equal")
                ax2.set_title("Used Memory Samples", fontsize=10)
                ax2.axis("off")

                fig.tight_layout()

                save_name = "wrong_{:02d}_absidx_{}.png".format(saved_count, abs_idx)
                fig.savefig(os.path.join(dir_save, save_name))
                plt.close(fig)

                saved_count += 1
                print("Saved {}/{} images".format(saved_count, FLAGS.num_images), end="\r")

    print()
    print("Done. Saved {} images to {}".format(saved_count, dir_save))

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