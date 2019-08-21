"""Module for running gSpan https://github.com/betterenvi/gSpan

```
# prepare dfi_subset
dfi_subset = dfi.query("match_weighted_cat!= 'low'").query("imp_weighted_cat == 'high'")
dfi_subset['pattern_cat'] = dfi_subset['pattern_name'].astype(str) + '/match=' + dfi_subset.match_weighted_cat.astype(str) + "/imp=" + dfi_subset.imp_weighted_cat.astype(str)
dfi_subset = dfi_subset[['pattern_cat', 'pattern_center']].dropna()

# Write gspan.data
dfi2graph_data(dfi_subset.groupby(dfi_subset.index), graph_file)

# Run gSpan
run_gspan(graph_file, output_file)

# Load the results file into List[nx.Graph]
graphs = load_gspan_results(output_file)
```
"""
import basepair
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


def write_graph(x, dist_min=10, dist_max=50, num_cat=False):
    def write_nodes(x):
        features = ['v', 'pattern_idx', 'pattern_cat']
        return x.assign(v="v")[features].to_csv(mode='a', sep=' ', header=False, index=False)

    def write_edges(x_pairs):
        features = ['e', 'pattern_idx_x', 'pattern_idx_y', 'edge_cat']
        return x_pairs.assign(e="e")[features].to_csv(mode='a', sep=' ', header=False, index=False)
    x['pattern_idx'] = np.arange(len(x))

    # create all pairs
    x_pairs = pd.merge(x[['pattern_cat', 'pattern_center', 'pattern_idx']],
                       x[['pattern_cat', 'pattern_center', 'pattern_idx']], right_index=True, left_index=True)
    x_pairs = x_pairs.query('pattern_idx_x != pattern_idx_y')
    x_pairs['pairwise_dist'] = np.abs(x_pairs['pattern_center_x'] - x_pairs['pattern_center_y'])

    # Filter edges
    x_pairs['edge_cat'] = pd.cut(x_pairs['pairwise_dist'], bins=[dist_min, dist_max],
                                 labels=[f'{dist_min}-{dist_max}'])
    x_pairs = x_pairs.dropna()
    if num_cat:
        x_pairs['edge_cat'] = x_pairs['edge_cat'].cat.codes
    else:
        x_pairs['edge_cat'] = x_pairs['edge_cat'].astype(str)

    return write_nodes(x) + write_edges(x_pairs)


def dfi2graph_data(dfi_grouped, output_file, verbose=True, n_jobs=10,
                   dist_min=10, dist_max=50, num_cat=False):
    """Write instances from genomic regions 

    Args:
      dfi_grouped: dfi table grouped by example_idx. Required columns in the grouped file: 'pattern_cat', 'pattern_center'
      output_file: output file path

    Example:
    ```
    # prepare dfi_subset
    dfi_subset = dfi.query("match_weighted_cat!= 'low'").query("imp_weighted_cat == 'high'")
    dfi_subset['pattern_cat'] = dfi_subset['pattern_name'].astype(str) + '/match=' + dfi_subset.match_weighted_cat.astype(str) + "/imp=" + dfi_subset.imp_weighted_cat.astype(str)
    dfi_subset = dfi_subset[['pattern_cat', 'pattern_center']].dropna()

    dfi2graph_data(dfi_subset.groupby(dfi_subset.index), graph_filt)
    ```
    """
    strings = Parallel(n_jobs=n_jobs)(delayed(write_graph)(group, 
                                                           dist_min=dist_min, dist_max=dist_max, num_cat=num_cat)
                                      for i, (name, group) in enumerate(tqdm(dfi_grouped, disable=not verbose)))
    with open(output_file, "w") as f:
        for i, s in enumerate(strings):
            f.write(f"t # {i}\n")
            f.write(s)
        f.write("t # -1")
        
    # Write also the group idx
    example_idx = dfi_grouped.size().index.values
    np.savetxt(str(output_file) + ".idx", example_idx)


def run_gspan(graph_file, output_file, min_support=10, min_num_vertices=1, where=True, **kwargs):
    """Run gSpan algorithm from https://github.com/betterenvi/gSpan

    Args:
      graph_file formatted as follows
        ```
        t # 0
        v 0 Oct4-Sox2/match=medium/imp=high
        v 1 Oct4-Sox2/match=high/imp=high
        v 2 Oct4-Sox2/match=high/imp=high
        v 3 Oct4-Sox2-deg/match=medium/imp=high
        e 0 1 10-50
        e 1 0 10-50
        t # 1
        v 0 Oct4-Sox2/match=medium/imp=high
        v 1 Nanog/match=medium/imp=high
        ...
        t # -1
        ```
      output_file: output file path
      min_support: minimal required support in order to display the output count
      min_num_vertices: minimal number of vertices in the graph
    """
    from gspan_mining import gSpan
    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        g = gSpan(graph_file, min_support=min_support, min_num_vertices=min_num_vertices, where=where, **kwargs)
        g.run()
    out = f.getvalue()
    with open(output_file, 'w') as f:
        f.write(out)


def load_gspan_results(file):
    """Parse the file produced by run_gspan
    """
    import networkx as nx
    out = []
    g = None
    with open(file) as f:
        for line in f:
            if line.startswith("t"):
                gid = line.strip().split(" ")[2]
                g = nx.Graph(name=gid)
            elif line.startswith("v"):
                v, vid, vname = line.strip().split(" ")
                g.add_node(vid, name=vname)
            elif line.startswith("e"):
                e, v1, v2, ename = line.strip().split(" ")
                g.add_edge(v1, v2, name=ename)
            elif line.startswith("Support: "):
                support = int(line.strip().replace("Support: ", ""))
                g.graph['support'] = support
            elif line.startswith("where: "):
                where = eval(line.strip().replace("where: ", ""))
                g.graph['where'] = where
            elif line.startswith("----------------"):
                # write the graph to the list
                out.append(g)
                g = None
    return out



def graph_colname(graph):
    edges = ",".join([v['name'] 
                     for e,v in graph.nodes.items()])
    nodes = ",".join([f"{i}-{j}" for i,j in graph.edges])
    return  edges + ";" + nodes


def load_gspan_as_features(gspan_output, gspan_idx_file):
    # load results
    graphs = load_gspan_results(gspan_output)
    example_idx = np.loadtxt(gspan_idx_file)
    
    graph_features = np.zeros((len(example_idx), len(graphs)), dtype=bool)

    for i, g in enumerate(graphs):
        assert g.graph['name'] == str(i)
        graph_features[g.graph['where'], i] = True

    assert np.all(graph_features.sum(0) == [g.graph['support'] for g in graphs])
    
    df = pd.DataFrame(graph_features,
                      columns=[graph_colname(g) for g in graphs],
                      index=example_idx.astype(int))
    df.index.name = 'example_idx'
    return df