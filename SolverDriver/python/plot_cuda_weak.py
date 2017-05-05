#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd
import ScalingFilenameParser as SFP

if __name__ == '__main__':
  # write the total dataset out, index=False, because we do not drop it above
  dataset = pd.read_csv('all_data.csv', low_memory=False)

  # set the index, verify it, and sort
  dataset.set_index(keys=['Experiment',
                          'problem_type', 'problem_nx', 'problem_ny', 'problem_nz',
                          'Timer Name',
                          'num_nodes', 'procs_per_node', 'cores_per_proc', 'threads_per_core',
                          'cuda_device_name'],
                    drop=False, inplace=True, verify_integrity=True)

  dataset.sort_values(inplace=True,
                      by=['Experiment',
                          'problem_type', 'problem_nx', 'problem_ny', 'problem_nz',
                          'Timer Name',
                          'procs_per_node', 'cores_per_proc', 'threads_per_core', 'num_nodes',
                          'cuda_device_name'])

  groups = dataset.groupby(['Experiment', 'problem_type', 'procs_per_node', 'cores_per_proc', 'threads_per_core',
                            'Timer Name'])
  for name, group in groups:
    if name[5] in ['0 - Total Time', '1 - Reseting Linear System', '4 - Constructing Solver', '5 - Solve']:
      continue

    my_tokens = SFP.getTokensFromDataFrameGroupBy(group)
    simple_fname = SFP.getScalingFilename(my_tokens, weak=True)
    simple_title = SFP.getScalingTitle(my_tokens, weak=True)

    # the number of HT combos we have
    num_cuda_arch = group['cuda_device_name'].nunique()

    fig_size = 5
    ax = []
    sec_ax = []
    fig = plt.figure()
    fig.set_size_inches(fig_size*num_cuda_arch, fig_size*1.30)

    my_nodes = np.array(list(map(int, group['num_nodes'].unique())))
    max_num_nodes = np.max(my_nodes)
    procs_per_node = int(group['procs_per_node'].max())

    print(my_nodes)

    dy = group['Max Aggregate Time'].max() * 0.05
    y_max = group['Max Aggregate Time'].max() + dy
    y_min = group['Max Aggregate Time'].min() - dy
    if y_min < 0:
      y_min = 0.0

    print("min: {}, max: {}, num_arches: {}".format(y_min, y_max,num_cuda_arch))

    # iterate over cuda device types
    idx = 0
    dev_groups = group.groupby('cuda_device_name')
    for dev_name, dev_group in dev_groups:
      print(dev_name)

      idx += 1
      ax_ = fig.add_subplot(1, 2, idx)
      ax_.scatter(dev_group['num_nodes'], dev_group['Max Aggregate Time'])
      ax_.set_ylabel('Runtime (s)')
      ax_.set_ylim([y_min, y_max])
      ax_.set_xlabel("Number of Nodes\n({} MPI Processes per Node)".format(procs_per_node))
      ax_.set_xticks(my_nodes)
      ax_.set_xlim([0.5, max_num_nodes*1.05])
      ax_.set_title('Raw Data\n({})'.format(dev_name))
      ax.append(ax_)

      print("Raw Data, idx={}".format(idx))

    fig.suptitle(simple_title, fontsize=18)
    plt.subplots_adjust(top=0.9, hspace=0.2)
    fig.tight_layout()
    plt.subplots_adjust(top=0.75)
    fig.savefig("{}.png".format(simple_fname), format='png', dpi=90)
    print("Wrote: {}.png".format(simple_fname))
    plt.close()

