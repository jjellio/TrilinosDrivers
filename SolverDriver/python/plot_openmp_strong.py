#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd
from pathlib import Path
import ScalingFilenameParser as SFP

FIG_SIZE = 5
MIN_NUM_NODES = 1
MAX_NUM_NODES = 64
FORCE_REPLOT = True
QUANTITY_OF_INTEREST='maxT'

if __name__ == '__main__':
  # write the total dataset out, index=False, because we do not drop it above
  dataset = pd.read_csv('all_data.csv', low_memory=False)

  # set the index, verify it, and sort
  dataset.set_index(keys=SFP.getIndexColumns(execspace_name='OpenMP'),
                    drop=False, inplace=True, verify_integrity=True)
  # sort
  dataset.sort_values(inplace=True,
                      by=SFP.getIndexColumns(execspace_name='OpenMP'))

  # remove the timers the driver adds
  dataset = dataset[~dataset['Timer Name'].isin(['0 - Total Time',
                                                  '1 - Reseting Linear System',
                                                  '2 - Adjusting Nullspace for BlockSize',
                                                  '3 - Constructing Preconditioner',
                                                  '4 - Constructing Solver',
                                                  '5 - Solve'])]
  # optionally restrict the data processed
  # Elasticity data is incomplete.
  restriction_query = '(problem_type != \"Elasticity3D\") & ' \
                      '(num_nodes >= {min_num_nodes}) & ' \
                      '(num_nodes <= {max_num_nodes}) & ' \
                      '(prec_attributes != \"-no-repartition\")' \
                      ''.format(min_num_nodes=MIN_NUM_NODES,
                                max_num_nodes=MAX_NUM_NODES)

  dataset = dataset.query(restriction_query)

  # enforce all plots use the same num_nodes. E.g., the axes will be consistent
  my_nodes = np.array(list(map(int, dataset['num_nodes'].unique())))
  my_num_nodes = dataset['num_nodes'].nunique()
  #
  my_ticks = np.arange(start=1, stop=my_num_nodes+1, step=1, dtype=int)
  max_num_nodes = np.max(my_nodes)

  expected_data_points = my_num_nodes
  print(my_nodes)

  # groups = dataset.groupby(['Experiment', 'problem_type', 'Timer Name', 'procs_per_node', 'cores_per_proc'])
  # timer_name_index = 2
  # there is a bug in Pandas. GroupBy cannot handle groupby keys that are none or nan.
  # For now, use Experiment, because it encapsulates the possible 'nan' keys

  omp_groupby_columns = SFP.getMasterGroupBy(execspace_name='OpenMP', scaling_type='strong')
  #print(omp_groupby_columns)
  timer_name_index = omp_groupby_columns.index('Timer Name')
  groups = dataset.groupby(omp_groupby_columns)

  for group_name, group in groups:
    # for a specific group of data, compute the scaling terms, which are things like min/max
    # this also flattens the timer creating a 'fat_timer_name'
    # essentially, this function computes data that is relevant to a group, but not the whole
    my_tokens = SFP.getTokensFromDataFrameGroupBy(group)
    simple_fname = SFP.getScalingFilename(my_tokens, strong=True)
    simple_title = SFP.getScalingTitle(my_tokens, strong=True)

    if not FORCE_REPLOT:
      my_file = Path("{}.png".format(simple_fname))
      if my_file.is_file():
        print("Skipping {}.png".format(simple_fname))
        continue

    print("My tokens:")
    print(my_tokens)
    print("Rebuilt filename: {}".format(SFP.rebuild_source_filename(my_tokens)))

    # the number of HT combos we have
    nhts = group['threads_per_core'].nunique()

    procs_per_node = int(group['procs_per_node'].unique())
    max_cores = int(group['cores_per_proc'].unique())

    prob_size = int(group['problem_nx'].max() * group['problem_ny'].max() * group['problem_nz'].max())

    dy = group[QUANTITY_OF_INTEREST].max() * 0.05
    y_max = group[QUANTITY_OF_INTEREST].max() + dy
    y_min = group[QUANTITY_OF_INTEREST].min() - dy
    if y_min < 0:
      y_min = 0.0

    np1 = group[(group['num_nodes'] == 1) & (group['threads_per_core'] == 1)][QUANTITY_OF_INTEREST].values

    if np1 == [] or len(np1) == 0:
      print('Obtained no values for the case of np==1 and threads_per_core==1. Skipping this data')
      continue
    elif len(np1) > 1:
      np1 = np.min(np1)

    spmv_index = list(group[(group['num_nodes'] == 1) & (group['threads_per_core'] == 1)].iloc[0].name)
    spmv_index[0] = 'Belos:CG: Operation Op*x'
    spmv_index = tuple(spmv_index)
    np1_spmv_time = dataset.iloc[dataset.index.get_loc(spmv_index)][QUANTITY_OF_INTEREST]
    print(np1_spmv_time)

    print("min: {}, max: {}".format(y_min, y_max))

    ax = []
    fig = plt.figure()
    fig.set_size_inches(FIG_SIZE * 3 * 1.05, FIG_SIZE * nhts * 1.1)

    # iterate over HTs
    ht_groups = group.groupby('threads_per_core')
    ht_n = 1
    max_ht_n = len(ht_groups)
    idx = 0
    for ht_name, ht_group in ht_groups:
      # this will fill in missing data
      my_agg_times = pd.DataFrame(columns=['num_nodes'], data=my_nodes)
      my_agg_times = pd.merge(my_agg_times, ht_group, on='num_nodes', how='left')
      # count the missing values
      num_missing_data_points = my_agg_times[QUANTITY_OF_INTEREST].isnull().values.ravel().sum()

      if num_missing_data_points != 0:
        print("Expected {expected_data_points} data points, Missing: {num_missing_data_points}, num_nodes column:".format(
          expected_data_points=expected_data_points,
          num_missing_data_points=num_missing_data_points))
        #print(my_agg_times[['num_nodes', QUANTITY_OF_INTEREST]])

      # compute SpeedUp and Efficiency
      S = np1 / my_agg_times[QUANTITY_OF_INTEREST]
      E = 100.00 * S / my_agg_times['num_nodes']
      S_spmv = np1_spmv_time / my_agg_times[QUANTITY_OF_INTEREST]

      c = ht_group[QUANTITY_OF_INTEREST].count()

      if my_ticks.size != my_agg_times['num_nodes'].count():
        print("Bad sizes, size x={}, size y={}, they should be equal".format(
          my_ticks.size, my_agg_times['num_nodes'].count()))

      my_agg_times['SpeedUp'] = S
      my_agg_times['Efficiency'] = E
      my_agg_times['SpeedUp SpMV'] = S_spmv

      # compute approximate nnz per proc
      my_agg_times['nnz'] = (prob_size / (my_agg_times['num_nodes']*procs_per_node))/1000.0

      # plot the raw data
      idx += 1
      ax_ = fig.add_subplot(nhts, 3, idx)
      ax_.scatter(my_ticks, my_agg_times[QUANTITY_OF_INTEREST])
      ax_.set_ylabel('Runtime (s)')
      ax_.set_ylim([y_min, y_max])
      # plot x label for the last HT group
      if ht_n == max_ht_n:
        ax_.set_xlabel("Number of Nodes\n({} MPI Processes per Node)".format(procs_per_node))
      ax_.set_xticks(my_ticks)
      ax_.set_xticklabels(my_nodes)
      ax_.set_xlim([0.5, my_num_nodes*1.05])

      # plot the titles for the first HT group
      if ht_n == 1:
        ax_.set_title('Raw Data\nAnnotation: Number of NP=1 SpMVs\n(HTs={:.0f})'.format(ht_name))
      else:
        # otherwise, print only the num hts
        ax_.set_title('(HTs={:.0f})'.format(ht_name))

      # add spmvs as a right side 2nd axes
      ax_twin = ax_.twinx()
      ax_twin.set_yticks(ax_.get_yticks() / np1_spmv_time)
      ax_twin.set_ylim([y_min/np1_spmv_time, y_max/np1_spmv_time])
      ax_twin.set_ylabel('Cost as Function of SpMVs')

      # label the runtimes as functions of SpMVs
      my_agg_times['num_np1_spmvs'] = my_agg_times[QUANTITY_OF_INTEREST].values / np1_spmv_time
      spmv_data_labels = ['{:.2f}'.format(i) for i in (my_agg_times['num_np1_spmvs'])]
      bad_value = False
      for label, x, y in zip(spmv_data_labels, my_ticks, my_agg_times['num_np1_spmvs'].values):
        ax_twin.annotate(
          label,
          xy=(x, y), xytext=(20, 20),
          textcoords='offset points', ha='right', va='bottom',
          bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
          arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

      ax.append(ax_)

      # plot speedup
      idx += 1
      ax_ = fig.add_subplot(nhts, 3, idx)
      SPEEDUP_YMIN = 0.75
      # pow2 scale, so make it slightly smaller than the next power
      SPEEDUP_YMAX = (max_num_nodes*2)*0.75
      # scatter the data points
      ax_.scatter(my_nodes, my_agg_times['SpeedUp'])
      # plot perfect scaling
      ax_.plot(my_nodes, my_nodes)
      ax_.set_ylabel('Speed Up')
      ax_.set_ylim([SPEEDUP_YMIN, SPEEDUP_YMAX])
      ax_.set_yscale('log', basey=2)
      ax_.set_xscale('log', basex=2)
      ax_.set_xticks(my_nodes)
      ax_.set_xticklabels(my_nodes)
      ax_.set_xlim([SPEEDUP_YMIN, SPEEDUP_YMAX])

      ax_.set_yticks(my_nodes)
      ax_.set_yticklabels(my_nodes)

      # annotate the speed up graph with data labels
      # mark values that fall below the minimum y-axis value
      data_labels = ['{:.1f}'.format(i) for i in my_agg_times['SpeedUp'].values]
      bad_value = False
      for label, x, y in zip(data_labels, my_nodes, my_agg_times['SpeedUp'].values):
        if y < 1.0:
          bad_value = True
          ax_.annotate(
            label,
            xy=(x, 1.0), xytext=(-20, 20),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='orange', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        else:
          ax_.annotate(
            label,
            xy=(x, y), xytext=(-20, 20),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

      if bad_value:
        ax_.annotate(
          'Orange values are below the Y-axis limit'.format(SPEEDUP_YMIN),
          xy=(np.log2(max_num_nodes), max_num_nodes), xytext=(0, -30),
          textcoords='offset points', ha='center', va='bottom',
          bbox=dict(boxstyle='round,pad=0.5', fc='orange', alpha=0.5))

      # plot x label for the last HT group
      if ht_n == max_ht_n:
        ax_.set_xlabel("Number of Nodes\n({} MPI Processes per Node)".format(procs_per_node))

      # plot the titles for the first HT group
      if ht_n == 1:
        ax_.set_title('SpeedUp\nAnnotation: Data Labels\n(HTs={:.0f})'.format(ht_name))
      else:
        # otherwise, print only the num hts
        ax_.set_title('(HTs={:.0f})'.format(ht_name))

      ax_twin = ax_.twiny()
      ax_twin.set_xscale('log', basex=2)
      ax_twin.set_xticks(my_nodes)
      proc_nnz_labels = ['{:d}'.format(int(round(i))) for i in my_agg_times['nnz'].values]
      ax_twin.set_xticklabels(proc_nnz_labels, y=0.92)
      ax_twin.set_xlim([SPEEDUP_YMIN, SPEEDUP_YMAX])
      ax_twin.tick_params(direction='in')
      ax_twin.xaxis.set_label_coords(0.5, 0.90)
      ax_twin.set_xlabel('nnz per process (thousands)')

      ax.append(ax_)

      # plot efficiency
      idx += 1
      ax_ = fig.add_subplot(nhts, 3, idx)
      ax_.scatter(my_ticks, my_agg_times['Efficiency'])
      ax_.set_ylabel('Efficiency (%')
      ax_.set_ylim([0, 100*1.05])
      # plot x label for the last HT group
      if ht_n == max_ht_n:
        ax_.set_xlabel("Number of Nodes\n({} MPI Processes per Node)".format(procs_per_node))
      ax_.set_xticks(my_ticks)
      ax_.set_xticklabels(my_nodes)
      ax_.set_xlim([0.5, my_num_nodes*1.05])
      # plot the titles for the first HT group
      if ht_n == 1:
        ax_.set_title('Efficiency\n\n(HTs={:.0f})'.format(ht_name))
      else:
        # otherwise, print only the num hts
        ax_.set_title('(HTs={:.0f})'.format(ht_name))
      ax.append(ax_)

      ht_n += 1

    fig.suptitle(simple_title, fontsize=18)
    plt.subplots_adjust(top=0.9, hspace=0.2)
    fig.tight_layout()
    plt.subplots_adjust(top=0.85)
    fig.savefig("{}.png".format(simple_fname), format='png', dpi=90)
    print("Wrote: {}.png".format(simple_fname))
    plt.close()
