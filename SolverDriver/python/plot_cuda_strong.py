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
                          'num_nodes', 'procs_per_node', 'cores_per_proc', 'threads_per_core',
                          'cuda_device_name'])

  groups = dataset.groupby(['Experiment', 'problem_type', 'Timer Name'])
  for name, group in groups:
    if name[2] in ['0 - Total Time', '1 - Reseting Linear System', '4 - Constructing Solver', '5 - Solve']:
      continue

    my_tokens = SFP.getTokensFromDataFrameGroupBy(group)
    simple_fname = SFP.getScalingFilename(my_tokens, strong=True)
    simple_title = SFP.getScalingTitle(my_tokens, strong=True)

    # the number of HT combos we have
    num_cuda_arch = group['cuda_device_name'].nunique()

    fig_size = 5
    ax = []
    sec_ax = []
    fig = plt.figure()
    fig.set_size_inches(fig_size*3, fig_size*num_cuda_arch*1.05)

    my_nodes = np.array(list(map(int, group['num_nodes'].unique())))
    max_num_nodes = np.max(my_nodes)
    procs_per_node = int(group['procs_per_node'].max())

    print(my_nodes)

    prob_size = int(group['problem_nx'].max() * group['problem_ny'].max() * group['problem_nz'].max())

    unknowns_per_proc = prob_size / (4 * my_nodes)
    unknowns_per_proc_label = ["%d" % member for member in unknowns_per_proc]

    dy = group['Max Aggregate Time'].max() * 0.05
    y_max = group['Max Aggregate Time'].max() + dy
    y_min = group['Max Aggregate Time'].min() - dy
    if y_min < 0:
      y_min = 0.0

    print("min: {}, max: {}, num_arches: {}".format(y_min, y_max,num_cuda_arch))

    # iterate over cuda device types
    dev_n = 0
    idx = 0
    dev_groups = group.groupby('cuda_device_name')
    for dev_name, dev_group in dev_groups:
      print(dev_name)
      # compute SpeedUp and Efficiency
      np1 = dev_group[dev_group['num_nodes'] == 1]['Max Aggregate Time'].values
      S = np1 / dev_group['Max Aggregate Time']
      E = 100.00 * S / dev_group['num_nodes']

      idx += 1
      ax_ = fig.add_subplot(num_cuda_arch, 3, idx)
      ax_.scatter(dev_group['num_nodes'], dev_group['Max Aggregate Time'])
      ax_.set_ylabel('Runtime (s)')
      ax_.set_ylim([y_min, y_max])
      if idx > 3:
        ax_.set_xlabel("Number of Nodes\n({} MPI Processes per Node)".format(procs_per_node))
      ax_.set_xticks(my_nodes)
      ax_.set_xlim([0.5, max_num_nodes*1.05])
      ax_.set_title('Raw Data\n({})'.format(dev_name))
      ax.append(ax_)

      print("Raw Data, idx={}".format(idx))

      idx += 1
      ax_ = fig.add_subplot(num_cuda_arch, 3, idx)
      # plt.loglog(group['omp_num_threads'], S)
      # plt.loglog(group['omp_num_threads'], group['omp_num_threads'])
      # plt.title('loglog S')
      ax_.scatter(dev_group['num_nodes'], S)
      ax_.plot(dev_group['num_nodes'], dev_group['num_nodes'])
      ax_.set_ylabel('Speed Up')
      ax_.set_ylim([0.5,max_num_nodes*1.05])
      ax_.set_yticks(my_nodes)
      if idx > 3:
        ax_.set_xlabel("Number of Nodes\n({} MPI Processes per Node)".format(procs_per_node))
      ax_.set_xticks(my_nodes)
      ax_.set_xlim([0.5, max_num_nodes*1.05])
      ax_.set_title('SpeedUp\n({})'.format(dev_name))
      # add unknowns per node
      # ax2 =ax_.twinx()
      # ax2.set_ylabel('Unknowns per Proc', color='r')
      # ax2.tick_params('y', colors='r')
      # ax2.set_ylim([0.5,max_num_nodes*1.05])
      # ax2.set_yticks(my_nodes)
      # ax2.set_yticklabels(unknowns_per_proc_label)
      # sec_ax.append(ax2)
      ax.append(ax_)
      #ax2.yaxis.set_major_formatter(FormatStrFormatter('%2.2e'))

      print("S, idx={}".format(idx))

      idx += 1
      ax_ = fig.add_subplot(num_cuda_arch, 3, idx)
      ax_.scatter(dev_group['num_nodes'], E)
      ax_.set_ylabel('Efficiency (%')
      ax_.set_ylim([0, 100*1.05])
      if idx > 3:
        ax_.set_xlabel("Number of Nodes\n({} MPI Processes per Node)".format(procs_per_node))
      ax_.set_xticks(my_nodes)
      ax_.set_xlim([0.5, max_num_nodes*1.05])
      ax_.set_title('Efficiency\n({})'.format(dev_name))
      ax.append(ax_)

      print("E, idx={}".format(idx))
      dev_n += 1

    fig.suptitle(simple_title, fontsize=18)
    plt.subplots_adjust(top=0.9, hspace=0.2)
    fig.tight_layout()
    plt.subplots_adjust(top=0.85)
    fig.savefig("{}.png".format(simple_fname), format='png', dpi=90)
    print("Wrote: {}.png".format(simple_fname))
    plt.close()

