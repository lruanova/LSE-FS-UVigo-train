import itertools
import numpy as np
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import confusion_matrix
import difflib


def _plot_cm(cm: np.ndarray, labels: list[str], title: str, normalize: bool):
    fig, ax = plt.subplots(figsize=(12, 10))
    chosen_cmap = "viridis"
    im = ax.imshow(cm, interpolation="nearest", aspect="auto", cmap=chosen_cmap)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=title + (" (normalizada)" if normalize else ""),
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(
            j,
            i,
            format(cm[i, j], fmt),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=7,
        )
    fig.tight_layout()
    return fig


def log_confusion_matrix(
    seqs_true: list[list[int]],
    seqs_pred: list[list[int]],
    sklearn_labels: list[int],
    class_names: list[str],
    pad_id: int,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> None:
    """
    Computes and logs to WandB a confussion matrix per token, including blank (pad_id) token
    as an additional token.

    Args:
        seqs_true: list of true sequences
        seqs_pred: list of predicted sequences
        sklearn_labels: list of valid integer IDs
        class_names: list of corresponding alphabetic characters
        pad_id : id of the blank/padding token
        title: title of the plot
        normalize: use or not percentages.

    """
    # Adds blank token at beginning
    labels = [pad_id] + sklearn_labels
    names = ["<blank>"] + class_names

    all_true, all_pred = [], []
    for t_ids, p_ids in zip(seqs_true, seqs_pred):
        # for a pair of lists of tokens of different lengths with replacements,
        # omissions, insertions... SequenceMatcher returns this cases and classifies
        # in four opcodes (equal, replace, delete, insert) with corresponding
        # indexes (i1,i2),(j1,j2)

        matcher = difflib.SequenceMatcher(a=t_ids, b=p_ids)
        t_al, p_al = [], []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                # copy i1:i2 and j1:j2 to aligned output list
                t_al.extend(t_ids[i1:i2])
                p_al.extend(p_ids[j1:j2])
            elif tag == "replace":
                # As they have different lengths, two blocks
                # of same length are created before copying
                t_block = t_ids[i1:i2]
                p_block = p_ids[j1:j2]
                L = max(len(t_block), len(p_block))
                t_al.extend(t_block + [pad_id] * (L - len(t_block)))
                p_al.extend(p_block + [pad_id] * (L - len(p_block)))
            elif tag == "delete":
                # fill predicted with blanks to equal length
                t_al.extend(t_ids[i1:i2])
                p_al.extend([pad_id] * (i2 - i1))
            elif tag == "insert":
                # fill true with blanks
                t_al.extend([pad_id] * (j2 - j1))
                p_al.extend(p_ids[j1:j2])
        # If on same position both are blanks, skip them
        for tt, pp in zip(t_al, p_al):
            if tt == pad_id and pp == pad_id:
                continue

            all_true.append(tt)
            all_pred.append(pp)

    # >> Computes confussion matrix including blanks
    cm = confusion_matrix(all_true, all_pred, labels=labels)
    if normalize:
        cm = cm.astype(float)
        cm = (cm.T / (cm.sum(axis=1) + 1e-12)).T

    # >> Draws it and logs to wandb
    fig = _plot_cm(cm, labels=names, title=title, normalize=normalize)
    wandb.log({title.replace(" ", "_").lower(): wandb.Image(fig)})
    plt.close(fig)
