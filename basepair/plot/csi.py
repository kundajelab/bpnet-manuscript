from collections import defaultdict
import attr
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


@attr.s
class SeqNode:
    vocab = ['A', 'C', 'G', 'T']

    seq = attr.ib()
    value = attr.ib(0)
    fixed_positions = attr.ib([])

    @seq.validator
    def check(self, attribute, value):
        for s in value:
            if s not in self.vocab:
                raise ValueError(f"letter {s} not in vocab {self.vocab}")

    def mutated_sequence(self):
        """Generate all mutated sequences

        Args:
          fixed_positions: at those positions, don't mutate the sequence
        """
        for i in range(len(self.seq)):
            for alt in self.vocab:
                if i in self.fixed_positions or alt == self.seq[i]:
                    continue
                yield SeqNode(self.seq[:i] + alt + self.seq[i + 1:],
                              fixed_positions=self.fixed_positions + [i])


class SeqMutationTree(object):
    def __init__(self, root_node=None, children=[]):
        self.root_node = root_node
        self.children = children

    @classmethod
    def create(cls, root_node, max_hamming_distance=2):
        if max_hamming_distance < 0:
            raise ValueError("max_hamming_distance < 0")
        elif max_hamming_distance == 0:
            # we came to the leaf node
            return SeqMutationTree(root_node)
        else:
            return SeqMutationTree(root_node=root_node,
                                   children=[SeqMutationTree.create(n, max_hamming_distance - 1)
                                             for n in root_node.mutated_sequence()])

    def get_seq_by_depth(self, depth=0, values=None):
        if values is None:
            values = defaultdict(list)
        values[depth].append(self.root_node.seq)
        for child in self.children:
            child.get_seq_by_depth(depth + 1, values)
        return values

    def update_value(self, fn):
        """Update value using a sequence-based function

        Args:
          fn: function which takes the sequence as input and returns a value
        """
        # update root_node
        self.root_node.value = fn(self.root_node.seq)

        # update children
        for child in self.children:
            child.update_value(fn)

        return self


def plot_csi(vals, title=None, cmap_name='Blues', ax=None, figwidth=3,
             annotate_distance=False):
    """
    Args:
      vals: dictionary with key being the hamming distance and values being
        the list of different values to plot
    """

    max_val = max([max(v) for v in vals.values()])
    if ax is None:
        fig, ax = plt.subplots(figsize=(figwidth, figwidth))

    size = ax.get_xbound()[1] / (len(vals) - 0.5)

    cmap = plt.get_cmap(cmap_name)
    for hamming_dist in range(len(vals)):
        values = np.array(vals[hamming_dist]) / max_val
        if hamming_dist > 0:
            radius = hamming_dist * size + size / 2
            width = size
        else:
            radius = size / 2
            width = size / 2

        if hamming_dist > 1:
            edgecolor = None
        else:
            edgecolor = 'w'

        ax.pie(np.ones_like(values), radius=radius, colors=cmap(values),
               wedgeprops=dict(width=width,
                               edgecolor=edgecolor), startangle=90)
        ax.set(aspect="equal")
        if annotate_distance:
            ax.text(0, hamming_dist * size + size / 2, hamming_dist,
                    verticalalignment='center')
    if title is not None:
        ax.set_title(title)


def heatmap_csi(df, motif, task, score, cmap=None, ax=None, figsize=(15, 2), clim=(None, None)):
    """
    Args:
      dataframe with columns: hamming_distance, i, motif, score, seq, value
    """
    rows = df.hamming_distance.max() + 1
    dfs = df[(df.motif == motif) &
             (df.task == task) &
             (df.score == score)]
    columns = dfs.i.max() + 1
    m = np.zeros((rows, columns))

    for hd in range(rows):
        values = dfs[dfs.hamming_distance == hd]['value'].values

        if columns % len(values) != 0:
            raise ValueError("columns % len(values) != 0")

        rep = columns // len(values)
        m[hd, :] = values.repeat(rep)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    heatmap = sns.heatmap(m, ax=ax, cmap=cmap, vmin=clim[0], vmax=clim[1])
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.xaxis.set_ticks_position('none')
    ax.set_xlabel("Mutation position")
    ax.set_ylabel("Hamming distance")


def heatmap_csi_tasks(df, tasks, motif, score, cmap=None, subfigsize=(20, 2)):
    fig, axes = plt.subplots(len(tasks), 1, figsize=(subfigsize[0], subfigsize[1] * len(tasks)))
    vmin = df[(df.motif == motif) & (df.score == score)]['value'].min()
    vmax = df[(df.motif == motif) & (df.score == score)]['value'].max()
    for i, (task, ax) in enumerate(zip(tasks, axes)):
        heatmap_csi(df, motif, task, score, cmap=cmap, ax=ax, clim=[vmin, vmax])
        if i != len(task) - 1:
            ax.set_xlabel("")
        ax.set_title(task)
    return fig
