#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd
from tableau import tableau20
from pathlib import Path
import copy
import ScalingFilenameParser as SFP
from operator import itemgetter

MIN_NUM_NODES = 1
MAX_NUM_NODES = 10000
FORCE_REPLOT = False
QUANTITY_OF_INTEREST='minT'
QUANTITY_OF_INTEREST_COUNT='minC'

QUANTITY_OF_INTEREST_MIN='minT'
QUANTITY_OF_INTEREST_MIN_COUNT='minC'
QUANTITY_OF_INTEREST_MAX='maxT'
QUANTITY_OF_INTEREST_MAX_COUNT='maxC'
QUANTITY_OF_INTEREST_THING='meanCT'
QUANTITY_OF_INTEREST_THING_COUNT='meanCC'

composite_count = 0

DECOMP_COLORS = {
  '64x1' : 'xkcd:greyish',
  '32x2' : 'xkcd:windows blue',
  '16x4' : 'xkcd:amber',
  '8x8'  : 'xkcd:faded green',
  '4x16' : 'xkcd:dusty purple'
}

def plot_composite(composite_group, my_nodes, my_ticks, average=False):
  decomp_groups = composite_group.groupby(['procs_per_node', 'cores_per_proc'])

  # for a specific group of data, compute the scaling terms, which are things like min/max
  # this also flattens the timer creating a 'fat_timer_name'
  # essentially, this function computes data that is relevant to a group, but not the whole
  my_tokens = SFP.getTokensFromDataFrameGroupBy(composite_group)
  simple_fname = SFP.getScalingFilename(my_tokens, weak=True, composite=True)
  simple_title = SFP.getScalingTitle(my_tokens, weak=True, composite=True)

  simple_fname = '{}-{}'.format(composite_count, simple_fname)
  global composite_count
  composite_count = composite_count + 1

  if not FORCE_REPLOT:
    my_file = Path("{}.png".format(simple_fname))
    if my_file.is_file():
      print("Skipping {}.png".format(simple_fname))
      return

  if average:
    new_quantity_of_interest = 'normalized_{}'.format(QUANTITY_OF_INTEREST)
    composite_group[new_quantity_of_interest] = composite_group[QUANTITY_OF_INTEREST] /\
                                                composite_group[QUANTITY_OF_INTEREST_COUNT]
    DATA_LABEL = new_quantity_of_interest
  else:
    DATA_LABEL = QUANTITY_OF_INTEREST

  dy = composite_group[DATA_LABEL].max() * 0.05
  y_max = composite_group[DATA_LABEL].max() + dy
  y_min = composite_group[DATA_LABEL].min() - dy
  if y_min < 0:
    y_min = 0.0

  # the number of HT combos we have
  nhts = composite_group['threads_per_core'].nunique()
  ndecomps = len(decomp_groups)

  my_num_nodes = my_nodes.size

  fig_size = 5
  fig = plt.figure()
  fig.set_size_inches(fig_size * nhts, fig_size * 1.35)

  ax = []

  for plot_idx in range(0, nhts):
    ax_ = fig.add_subplot(1, nhts, plot_idx + 1)
    ax.append(ax_)

  for decomp_group_name, decomp_group in decomp_groups:
    # label this decomp
    decomp_label = "{procs_per_node}x{cores_per_proc}".format(procs_per_node=decomp_group_name[0],
                                                              cores_per_proc=decomp_group_name[1])
    # iterate over HTs
    ht_groups = decomp_group.groupby('threads_per_core')
    ht_n = 1
    max_ht_n = len(ht_groups)
    plot_idx = 0
    for ht_name, ht_group in ht_groups:
      # this will fill in missing data
      timings = ht_group.groupby('num_nodes', as_index=False)[[QUANTITY_OF_INTEREST_MIN,
                                                               QUANTITY_OF_INTEREST_MAX,
                                                               QUANTITY_OF_INTEREST_THING]].sum()
      counts = ht_group.groupby('num_nodes', as_index=False)[[QUANTITY_OF_INTEREST_MIN_COUNT,
                                                              QUANTITY_OF_INTEREST_MAX_COUNT]].sum()

      timings[QUANTITY_OF_INTEREST_MIN] = timings[QUANTITY_OF_INTEREST_MIN] / counts[QUANTITY_OF_INTEREST_MIN_COUNT]
      timings[QUANTITY_OF_INTEREST_MAX] = timings[QUANTITY_OF_INTEREST_MAX] / counts[QUANTITY_OF_INTEREST_MAX_COUNT]
      barf_df = ht_group.groupby('num_nodes', as_index=False)[QUANTITY_OF_INTEREST_THING].mean()
      timings[QUANTITY_OF_INTEREST_THING] = barf_df[QUANTITY_OF_INTEREST_THING]

      my_agg_times = pd.DataFrame(columns=['num_nodes', 'ticks'], data=np.column_stack((my_nodes, my_ticks)))
      my_agg_times = pd.merge(my_agg_times, timings, on='num_nodes', how='left')
      # count the missing values
      num_missing_data_points = my_agg_times[DATA_LABEL].isnull().values.ravel().sum()

      if num_missing_data_points != 0:
        print(
          "Expected {expected_data_points} data points, Missing: {num_missing_data_points}".format(
            expected_data_points=my_num_nodes,
            num_missing_data_points=num_missing_data_points))

      print("x={}, y={}".format(my_agg_times['ticks'].count(),
                                my_agg_times['num_nodes'].count()))

      # ax[plot_idx].plot(my_agg_times['ticks'], my_agg_times[DATA_LABEL],
      #                   label=decomp_label, color=DECOMP_COLORS[decomp_label])

      ax[plot_idx].plot(my_agg_times['ticks'], my_agg_times[QUANTITY_OF_INTEREST_MIN],
                        label='min-{}'.format(decomp_label), color=DECOMP_COLORS[decomp_label])
      ax[plot_idx].plot(my_agg_times['ticks'], my_agg_times[QUANTITY_OF_INTEREST_MAX],
                        label='max-{}'.format(decomp_label), color=DECOMP_COLORS[decomp_label],
                        linestyle=':')
      # ax[plot_idx].plot(my_agg_times['ticks'], my_agg_times[QUANTITY_OF_INTEREST_THING],
      #                   label='meanCT-{}'.format(decomp_label), color=DECOMP_COLORS[decomp_label],
      #                   marker='o', fillstyle='none', linestyle='none')

      ax[plot_idx].set_ylabel('Runtime (s)')
      #ax[plot_idx].set_ylim([y_min, y_max])

      ax[plot_idx].set_xlabel("Number of Nodes")
      ax[plot_idx].set_xticks(my_ticks)
      ax[plot_idx].set_xticklabels(my_nodes, rotation=45)
      ax[plot_idx].set_xlim([0.5, my_num_nodes + 1])
      # plot the titles
      ax[plot_idx].set_title('Raw Data\n(HTs={:.0f})'.format(ht_name))
      plot_idx = plot_idx + 1

  best_ylims = [np.inf, -np.inf]
  for axis in ax:
    ylims = axis.get_ylim()
    best_ylims[0] = min(best_ylims[0], ylims[0])
    best_ylims[1] = max(best_ylims[1], ylims[1])

  for axis in ax:
    axis.set_ylim(best_ylims)

  handles, labels = ax[0].get_legend_handles_labels()
  fig.legend(handles, labels,
             title="Procs per Node x Cores per Proc",
             loc='lower center', ncol=ndecomps, bbox_to_anchor=(0.5, 0.0))

  fig.suptitle(simple_title, fontsize=18)
  # plt.subplots_adjust(top=0.9, hspace=0.2)
  fig.tight_layout()
  plt.subplots_adjust(top=0.78, bottom=0.20)
  try:
    fig.savefig("{}.png".format(simple_fname), format='png', dpi=180)
    print("Wrote: {}.png".format(simple_fname))
  except:
    print("FAILED writing {}.png".format(simple_fname))
    raise

  plt.close()


def load_dataset(dataset_name):
  print('Reading {}'.format(dataset_name))
  # write the total dataset out, index=False, because we do not drop it above
  dataset = pd.read_csv(dataset_name, low_memory=False)

  print('Read csv complete')

  # set the index, verify it, and sort
  dataset.set_index(keys=SFP.getIndexColumns(execspace_name='OpenMP'),
                    drop=False, inplace=True, verify_integrity=True)
  print('Verified index')

  # optionally restrict the data processed
  # Elasticity data is incomplete.
  restriction_query = '(problem_type != \"Elasticity3D\") & ' \
                      '(num_nodes >= {min_num_nodes}) & ' \
                      '(num_nodes <= {max_num_nodes}) & ' \
                      '(prec_attributes != \"-no-repartition\")' \
                      ''.format(min_num_nodes=MIN_NUM_NODES,
                                max_num_nodes=MAX_NUM_NODES)

  dataset = dataset.query(restriction_query)
  print('Restricted dataset')

  dataset.fillna(value='None', inplace=True)

  # sort
  dataset.sort_values(inplace=True,
                      by=SFP.getIndexColumns(execspace_name='OpenMP'))
  print('Sorted')

  # remove the timers the driver adds
  driver_dataset = dataset[dataset['Timer Name'].isin(['0 - Total Time',
                                                  '1 - Reseting Linear System',
                                                  '2 - Adjusting Nullspace for BlockSize',
                                                  '3 - Constructing Preconditioner',
                                                  '4 - Constructing Solver',
                                                  '5 - Solve'])]
  print('Gathered driver timers')

  # remove the timers the driver adds
  dataset = dataset[~dataset['Timer Name'].isin(['0 - Total Time',
                                                  '1 - Reseting Linear System',
                                                  '2 - Adjusting Nullspace for BlockSize',
                                                  '3 - Constructing Preconditioner',
                                                  '4 - Constructing Solver',
                                                  '5 - Solve'])]
  print('Removed driver timers')

  # reindex
  # set the index, verify it, and sort
  dataset.set_index(keys=SFP.getIndexColumns(execspace_name='OpenMP'),
                    drop=False, inplace=True, verify_integrity=True)
  driver_dataset.set_index(keys=SFP.getIndexColumns(execspace_name='OpenMP'),
                    drop=False, inplace=True, verify_integrity=True)
  print('Rebuilt truncated index')

  return dataset,driver_dataset

def get_ordered_timers(dataset):
  # find an order of timer names based on aggregate time
  ordered_timers = dataset.groupby(['Timer Name',
                                    'problem_type',
                                    'solver_name',
                                    'solver_attributes',
                                    'prec_name',
                                    'prec_attributes'], as_index=False).sum().sort_values(
    by=QUANTITY_OF_INTEREST, ascending=False)['Timer Name'].tolist()
  #
  # restriction_tokens = {'solver_name' : 'Constructor',
  #                       'solver_attributes' : '-Only',
  #                       'prec_name' : 'MueLu',
  #                       'prec_attributes' : '-repartition'}
  # get_timers(dataset, restriction_tokens=restriction_tokens)

  return ordered_timers


def get_timers(dataset, restriction_tokens={}):
  # The dataset contains many timers, that may be shared between experiments.
  # for example, solve vs setup timers may overlap, since solve implies a setup
  # restriction_tokens is a dict of column names and values that should be *enforced*
  # for timer selection, e.g., this is like a WHERE clause in SQL

  query_string = ' & '.join(['(\"{name}\" == \"{value}\")'.format(name=name, value=value)
                            for name,value in restriction_tokens.items() ])
  print(query_string)


def plot_dataset(dataset, driver_dataset, ordered_timers):
  # enforce all plots use the same num_nodes. E.g., the axes will be consistent
  my_nodes = np.array(list(map(int, dataset['num_nodes'].unique())))
  my_num_nodes = dataset['num_nodes'].nunique()
  #
  my_ticks = np.arange(start=1, stop=my_num_nodes+1, step=1, dtype=int)

  print(my_num_nodes)
  print(my_nodes)
  print(my_ticks)

  if len(my_nodes) != len(my_ticks):
    print('Length of ticks and nodes are different')
    exit(-1)

  # figure out the best SpMV time
  matvec_df = dataset[dataset['Timer Name'] == 'Belos:CG: Operation Op*x']
  matvec_df = matvec_df.groupby(['problem_type', 'num_nodes'])
  best_spmvs_idx = matvec_df[QUANTITY_OF_INTEREST].idxmin()

  best_spmvs_df = dataset.ix[best_spmvs_idx]
  best_spmvs_df.set_index(['problem_type', 'num_nodes'], drop=False, inplace=True, verify_integrity=True)
  pd.set_option('display.expand_frame_repr', False)
  print(best_spmvs_df[['problem_type',
                          'num_nodes',
                          'procs_per_node',
                          'cores_per_proc',
                          'threads_per_core',
                          QUANTITY_OF_INTEREST,
                          QUANTITY_OF_INTEREST_COUNT]])
  pd.set_option('display.expand_frame_repr', True)

  omp_groupby_columns = SFP.getMasterGroupBy(execspace_name='OpenMP', scaling_type='weak')
  omp_groupby_columns.remove('procs_per_node')
  omp_groupby_columns.remove('cores_per_proc')
  omp_groupby_columns.remove('solver_name')
  omp_groupby_columns.remove('solver_attributes')
  omp_groupby_columns.remove('prec_name')
  omp_groupby_columns.remove('prec_attributes')

  spmv_only_data = dataset[dataset['Timer Name'].str.match('^.* Operation Op\*x$')]
  spmv_only_data['Timer Name'] = 'Operation Op*x'

  spmv_agg_groups = spmv_only_data.groupby(omp_groupby_columns)

  for spmv_agg_name, spmv_agg_group in spmv_agg_groups:
    plot_composite(spmv_agg_group, my_nodes, my_ticks)

  #best_spmvs_df.rename(columns={QUANTITY_OF_INTEREST: 'best_spmv'}, inplace=True)
  #dataset = dataset.merge(best_spmvs_df, on=['problem_type', 'num_nodes'], suffixes=['', '_best_spmv'])

  # first, restrict the dataset to construction only muelu data
  restriction_tokens = {'solver_name' : 'Constructor',
                        'solver_attributes' : '-Only',
                        'prec_name' : 'MueLu',
                        'prec_attributes' : '-repartition'}

  stack_query_string = ' & '.join(['({name} == \"{value}\")'.format(
    name=name if ' ' not in name else '\"{}\"'.format(name),
    value=value)
                            for name,value in restriction_tokens.items() ])
  print(stack_query_string)
  dataset = dataset.query(stack_query_string)

  # group by timer and attributes, but keep all decomps together
  omp_groupby_columns = SFP.getMasterGroupBy(execspace_name='OpenMP', scaling_type='weak')
  omp_groupby_columns.remove('procs_per_node')
  omp_groupby_columns.remove('cores_per_proc')
  # print(omp_groupby_columns)
  timer_name_index = omp_groupby_columns.index('Timer Name')
  composite_groups = dataset.groupby(omp_groupby_columns)

  print(ordered_timers)
  foo = copy.deepcopy(composite_groups.groups).keys()
  # the timer name is the first element in the index. This
  # will take the index, and apply the sorting from ordered_timers to them.
  # This approach is not stable. That is, it does not preserve the order of
  # multiple occurences of the same name.
  sorted_composite_groups = sorted(foo, key=lambda x: ordered_timers.index(x[0]))
  print(sorted_composite_groups)

  for composite_group_name in sorted_composite_groups:
    composite_group = composite_groups.get_group(composite_group_name)
    plot_composite(composite_group, my_nodes, my_ticks)

  # construct a stacked plot.

  # first, restrict the dataset to construction only muelu data
  restriction_tokens = {'solver_name' : 'Constructor',
                        'solver_attributes' : '-Only',
                        'prec_name' : 'MueLu',
                        'prec_attributes' : '-repartition'}

  stack_query_string = ' & '.join(['({name} == \"{value}\")'.format(
    name=name if ' ' not in name else '\"{}\"'.format(name),
    value=value)
                            for name,value in restriction_tokens.items() ])
  print(stack_query_string)
  stack_dataset = dataset.query(stack_query_string)

  # group by attributes. The timers are included in the group this time
  stack_groupby_columns = SFP.getMasterGroupBy(execspace_name='OpenMP', scaling_type='weak')
  stack_groupby_columns.remove('Timer Name')
  stack_groupby_columns.remove('procs_per_node')
  stack_groupby_columns.remove('cores_per_proc')
  stack_groups = stack_dataset.groupby(stack_groupby_columns)

  stack_TOP = 20
  silly_top = 18
  color_map = tableau20()

  idx=0

  for stack_group_name, stack_group in stack_groups:
    # handle (total), (total, level=n), (level=n)

    stack_df = stack_group[stack_group['Timer Name'].str.match('^.*\(level=\d+\)$')]
    stack_df = stack_df[~stack_df['Timer Name'].str.contains('Solve')]

    print(stack_df['Timer Name'].unique())

    #timer_names = stack_df['Timer Name'].unique()
    timer_color_map = {}
    chosen_timers = list()

    timer_names = stack_df['Timer Name'].unique().tolist()
    timer_df = pd.DataFrame(index=timer_names, columns=[QUANTITY_OF_INTEREST])
    driver_total = 0.0

    # next group by the decomp
    decomp_groups = stack_df.groupby(['procs_per_node', 'cores_per_proc'])
    for decomp_group_name, decomp_group in decomp_groups:
      ht_groups = decomp_group.groupby('threads_per_core')
      for ht_name, ht_group in ht_groups:
        node_groups = ht_group.sort_values(by=QUANTITY_OF_INTEREST, ascending=False).groupby('num_nodes')

        y = np.ndarray([my_num_nodes, stack_TOP])
        y[:] = np.NAN

        fig_size = 10
        fig, ax = plt.subplots()
        fig.set_size_inches(fig_size, fig_size * 1.05)

        for node_name, node_group in node_groups:
          node_id, = np.where(my_nodes == int(node_name))
          print('node {} maps to index {}'.format(node_name, node_id))
          # if the number of timers is greater than the stack_TOP
          # then aggregate those lower in the list then the top count
          print(node_group['Timer Name'].count())
          print(node_group[QUANTITY_OF_INTEREST].count())

          timer_df = pd.concat([node_group, timer_df]).groupby(["Timer Name"], as_index=False)[QUANTITY_OF_INTEREST].sum()
          #print(timer_df)

          # using the timer_Names returned, construct a color lookup table
          # only do this once
          if timer_color_map == {}:
            ci = 0
            #timer_names = node_group['Timer Name'].tolist()
            for timer_name in node_group['Timer Name'].tolist():
              if ci < (stack_TOP-1):
                timer_color_map[timer_name] = ci
              else:
                timer_color_map[timer_name] = stack_TOP-1
              ci += 1
              #timer_stuff[timer_name] += node_group['Timer Name' == timer_name, QUANTITY_OF_INTEREST]

          # obtain an index, and query the driver's timer for the preconditioner setup
          row_index = list(node_group.head(1).iloc[0].name)
          row_index[0] = '3 - Constructing Preconditioner'
          row_index = tuple(row_index)
          # we now can query the driver for the total time spend creating the preconditioner
          #print(driver_dataset.loc[row_index, QUANTITY_OF_INTEREST])
          driver_total += driver_dataset.loc[row_index, QUANTITY_OF_INTEREST]

          num_values = node_group[QUANTITY_OF_INTEREST].count()

          # grab the y values
          if num_values > stack_TOP:
            #print(node_group.head(stack_TOP-1)[QUANTITY_OF_INTEREST].count())
            #print(node_group.head(stack_TOP - 1)[QUANTITY_OF_INTEREST].values)
            #print(node_group.tail(num_values - (stack_TOP - 1))[QUANTITY_OF_INTEREST].count())
            #print(node_group.tail(num_values - (stack_TOP - 1))[QUANTITY_OF_INTEREST].sum())
            y[node_id, :] = np.append(node_group.head(stack_TOP-1)[QUANTITY_OF_INTEREST].values,
                             node_group.tail(num_values - (stack_TOP - 1))[QUANTITY_OF_INTEREST].sum())
          else:
            y[node_id, 0:num_values] = node_group[QUANTITY_OF_INTEREST].values

          #print(y)
          #print(np.sum(y[node_id, :]))

        # ax.stackplot(my_nodes, y.T, colors=color_map, labels=timer_names)
        # plt.savefig('{}-{}x{}x{}.png'.format(idx,
        #                                      decomp_group_name[0],
        #                                      decomp_group_name[1], ht_name), bbox_inches='tight')
        # idx +=1

    timer_df.sort_values(by=QUANTITY_OF_INTEREST, ascending=False, inplace=True)
    timer_df = timer_df.reset_index(drop=True)
    timer_df['Percent'] = timer_df[QUANTITY_OF_INTEREST] / driver_total
    print(timer_df)
    print(timer_df['Percent'].sum())

    chosen_timers = timer_df.head(silly_top)['Timer Name'].tolist()

    decomp_groups = stack_df.groupby(['procs_per_node', 'cores_per_proc'])
    for decomp_group_name, decomp_group in decomp_groups:
      ht_groups = decomp_group.groupby('threads_per_core')
      for ht_name, ht_group in ht_groups:
        node_groups = ht_group.sort_values(by=QUANTITY_OF_INTEREST, ascending=False).groupby('num_nodes')

        for node_name, node_group in node_groups:
          node_id, = np.where(my_nodes == int(node_name))

          tmp_timer_names = node_group.head(silly_top)['Timer Name'].tolist()
          print('{}x{}x{}x{}'.format(node_name, decomp_group_name[0], decomp_group_name[1], ht_name))
          print(set(chosen_timers) - set(tmp_timer_names))
  exit(-1)
  #
  # stack_dataset.to_csv('stack.csv', index=False)
  # for stack_group_name, stack_group in stack_groups:
  #   # handle (total), (total, level=n), (level=n)
  #
  #   # first, reproduce analysis.py, which keeps only (level=n) timers
  #   stack_df = stack_group[stack_group['Timer Name'].str.match('^.*\(level=\d+\)$')]
  #   stack_df = stack_df[~stack_df['Timer Name'].str.contains('Solve')]
  #
  #   #timer_names = stack_df['Timer Name'].unique()
  #   timer_color_map = {}
  #   timer_names = list()
  #
  #   # next group by the decomp
  #   decomp_groups = stack_df.groupby(['procs_per_node', 'cores_per_proc'])
  #   for decomp_group_name, decomp_group in decomp_groups:
  #     ht_groups = decomp_group.groupby('threads_per_core')
  #     for ht_name, ht_group in ht_groups:
  #       node_groups = ht_group.sort_values(by=QUANTITY_OF_INTEREST, ascending=False).groupby('num_nodes')
  #
  #       y = np.ndarray([my_num_nodes, stack_TOP])
  #       y[:] = np.NAN
  #
  #       fig_size = 10
  #       fig, ax = plt.subplots()
  #       fig.set_size_inches(fig_size, fig_size * 1.05)
  #
  #       for node_name, node_group in node_groups:
  #         node_id, = np.where(my_nodes == int(node_name))
  #         print('node {} maps to index {}'.format(node_name, node_id))
  #         # if the number of timers is greater than the stack_TOP
  #         # then aggregate those lower in the list then the top count
  #         print(node_group['Timer Name'].count())
  #         print(node_group[QUANTITY_OF_INTEREST].count())
  #
  #         # using the timer_Names returned, construct a color lookup table
  #         # only do this once
  #         if timer_color_map == {}:
  #           ci = 0
  #           timer_names = node_group['Timer Name'].tolist()
  #           for timer_name in node_group['Timer Name'].tolist():
  #             if ci < (stack_TOP-1):
  #               timer_color_map[timer_name] = ci
  #             else:
  #               timer_color_map[timer_name] = stack_TOP-1
  #             ci += 1
  #
  #         # obtain an index, and query the driver's timer for the preconditioner setup
  #         row_index = list(node_group.head(1).iloc[0].name)
  #         row_index[0] = '3 - Constructing Preconditioner'
  #         row_index = tuple(row_index)
  #         # we now can query the driver for the total time spend creating the preconditioner
  #         print(driver_dataset.loc[row_index, QUANTITY_OF_INTEREST])
  #
  #         num_values = node_group[QUANTITY_OF_INTEREST].count()
  #
  #         # grab the y values
  #         if num_values > stack_TOP:
  #           print(node_group.head(stack_TOP-1)[QUANTITY_OF_INTEREST].count())
  #           print(node_group.head(stack_TOP - 1)[QUANTITY_OF_INTEREST].values)
  #           print(node_group.tail(num_values - (stack_TOP - 1))[QUANTITY_OF_INTEREST].count())
  #           print(node_group.tail(num_values - (stack_TOP - 1))[QUANTITY_OF_INTEREST].sum())
  #           y[node_id, :] = np.append(node_group.head(stack_TOP-1)[QUANTITY_OF_INTEREST].values,
  #                            node_group.tail(num_values - (stack_TOP - 1))[QUANTITY_OF_INTEREST].sum())
  #         else:
  #           y[node_id, 0:num_values] = node_group[QUANTITY_OF_INTEREST].values
  #
  #         print(y)
  #         print(np.sum(y[node_id, :]))
  #
  #       ax.stackplot(my_nodes, y.T, colors=color_map, labels=timer_names)
  #       plt.savefig('{}-{}x{}x{}.png'.format(idx,
  #                                            decomp_group_name[0],
  #                                            decomp_group_name[1], ht_name), bbox_inches='tight')
  #       idx +=1
  #
  #
  #
  # exit(-1)
  #
  # old_groups = dataset.groupby(['Experiment', 'problem_type', 'Timer Name', 'procs_per_node', 'cores_per_proc'])
  # # groups = dataset.groupby(['Experiment', 'problem_type', 'Timer Name', 'procs_per_node', 'cores_per_proc'])
  # # timer_name_index = 2
  # # there is a bug in Pandas. GroupBy cannot handle groupby keys that are none or nan.
  # # For now, use Experiment, because it encapsulates the possible 'nan' keys
  #
  # omp_groupby_columns = SFP.getMasterGroupBy(execspace_name='OpenMP', scaling_type='weak')
  # # print(omp_groupby_columns)
  # timer_name_index = omp_groupby_columns.index('Timer Name')
  # groups = dataset.groupby(omp_groupby_columns)
  #
  # print("Obtained: {} groups with full groupby, {} the old way".format(len(groups), len(old_groups)))
  # if len(groups) != len(old_groups):
  #   print("Error: lengths are not matching.")
  #   exit(-1)
  #
  # print(ordered_timers)
  # foo = copy.deepcopy(groups.groups).keys()
  #
  # print(foo)
  # sorted_groups = sorted(foo, key=lambda x: ordered_timers.index(x[0]))
  # print(sorted_groups)
  #
  # for group_name in sorted_groups:
  #   print('Group Name: ', group_name)
  #
  #   group = groups.get_group(group_name)
  #   # for a specific group of data, compute the scaling terms, which are things like min/max
  #   # this also flattens the timer creating a 'fat_timer_name'
  #   # essentially, this function computes data that is relevant to a group, but not the whole
  #   my_tokens = SFP.getTokensFromDataFrameGroupBy(group)
  #   simple_fname = SFP.getScalingFilename(my_tokens, weak=True)
  #   simple_title = SFP.getScalingTitle(my_tokens, weak=True)
  #
  #   #print("My tokens:")
  #   #print(my_tokens)
  #   print("Rebuilt filename: {}".format(SFP.rebuild_source_filename(my_tokens)))
  #
  #   if not FORCE_REPLOT:
  #     my_file = Path("{}.png".format(simple_fname))
  #     if my_file.is_file():
  #       print("Skipping {}.png".format(simple_fname))
  #       continue
  #
  #   # the number of HT combos we have
  #   nhts = group['threads_per_core'].nunique()
  #
  #   fig_size = 5
  #   ax = []
  #   fig = plt.figure()
  #   fig.set_size_inches(fig_size * nhts, fig_size * 1.05)
  #
  #   procs_per_node = int(group['procs_per_node'].max())
  #   max_cores = group['cores_per_proc'].max()
  #
  #   prob_size = int(group['problem_nx'].max() * group['problem_ny'].max() * group['problem_nz'].max())
  #
  #   dy = group[QUANTITY_OF_INTEREST].max() * 0.05
  #   y_max = group[QUANTITY_OF_INTEREST].max() + dy
  #   y_min = group[QUANTITY_OF_INTEREST].min() - dy
  #   if y_min < 0:
  #     y_min = 0.0
  #
  #   print("min: {}, max: {}".format(y_min, y_max))
  #
  #   # iterate over HTs
  #   ht_groups = group.groupby('threads_per_core')
  #   ht_n = 1
  #   max_ht_n = len(ht_groups)
  #   idx = 0
  #   for ht_name, ht_group in ht_groups:
  #     # this will fill in missing data
  #     my_agg_times = pd.DataFrame(columns=['num_nodes', 'ticks'], data=np.column_stack((my_nodes, my_ticks)))
  #     my_agg_times = pd.merge(my_agg_times, ht_group, on='num_nodes', how='left')
  #     # count the missing values
  #     num_missing_data_points = my_agg_times[QUANTITY_OF_INTEREST].isnull().values.ravel().sum()
  #
  #     my_agg_times = my_agg_times.merge(best_spmvs_df, on=['problem_type', 'num_nodes'], suffixes=['', '_best_spmv'])
  #
  #     if num_missing_data_points != 0:
  #       print(
  #       "Expected {expected_data_points} data points, Missing: {num_missing_data_points}".format(
  #         expected_data_points=expected_data_points,
  #         num_missing_data_points=num_missing_data_points))
  #
  #     print("x={}, y={}".format(my_agg_times['ticks'].count(),
  #                               my_agg_times['num_nodes'].count()))
  #
  #     idx += 1
  #     ax_ = fig.add_subplot(1, nhts, idx)
  #     ax_.scatter(my_agg_times['ticks'], my_agg_times[QUANTITY_OF_INTEREST] / my_agg_times[QUANTITY_OF_INTEREST_COUNT])
  #     ax_.set_ylabel('Runtime (s)')
  #     ax_.set_ylim([y_min, y_max])
  #     # plot x label for the last HT group
  #     ax_.set_xlabel("Number of Nodes\n({} MPI Processes per Node)".format(procs_per_node))
  #
  #     ax_.set_xticks(my_agg_times['ticks'])
  #     ax_.set_xticklabels(my_agg_times['num_nodes'], rotation=45)
  #
  #     # pow2 scale, so make it slightly smaller than the next power
  #     xlim_max = (max_num_nodes*2)*0.75
  #     ax_.set_xlim([0.5, my_num_nodes+1])
  #     # plot the titles
  #     ax_.set_title('Raw Data\n(HTs={:.0f})'.format(ht_name))
  #
  #     # label the runtimes as functions of SpMVs
  #     # normalize the time. num_spmvx = maxT / maxC * spmv_T / spmv_C
  #     my_agg_times['num_spmvs'] = (my_agg_times[QUANTITY_OF_INTEREST] / my_agg_times[QUANTITY_OF_INTEREST_COUNT]) * \
  #                                 (
  #                                   my_agg_times['{}_best_spmv'.format(QUANTITY_OF_INTEREST)] /
  #                                   my_agg_times['{}_best_spmv'.format(QUANTITY_OF_INTEREST_COUNT)]
  #                                 )
  #     try :
  #       spmv_data_labels = ['{:.2f} ({}x{}x{})'.format(num_spmv,
  #                                                      int(num_spmv_procs),
  #                                                      int(num_spmv_cores),
  #                                                      int(num_spmv_threads))
  #                           for num_spmv, num_spmv_procs, num_spmv_cores, num_spmv_threads
  #                             in zip(my_agg_times['num_spmvs'],
  #                                    my_agg_times['procs_per_node_best_spmv'],
  #                                    my_agg_times['cores_per_proc_best_spmv'],
  #                                    my_agg_times['threads_per_core_best_spmv'])]
  #     except:
  #       print(my_agg_times)
  #       my_agg_times.to_csv('bad_df.csv')
  #       exit(-1)
  #
  #     spmv_data_labels = ['{:.2f}'.format(num_spmv)
  #                         for num_spmv in my_agg_times['num_spmvs'] ]
  #
  #     for label, x, y in zip(spmv_data_labels, my_agg_times['ticks'], my_agg_times[QUANTITY_OF_INTEREST]):
  #       ax_.annotate(
  #         label,
  #         xy=(x, y), xytext=(-20, 20),
  #         textcoords='offset points', ha='right', va='bottom',
  #         bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
  #         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
  #
  #     ax.append(ax_)
  #
  #   fig.suptitle(simple_title, fontsize=18)
  #   plt.subplots_adjust(top=0.9, hspace=0.2)
  #   fig.tight_layout()
  #   plt.subplots_adjust(top=0.70)
  #   try:
  #     fig.savefig("{}.png".format(simple_fname), format='png', dpi=90)
  #     print("Wrote: {}.png".format(simple_fname))
  #   except:
  #     print("FAILED writing {}.png".format(simple_fname))
  #     continue
  #
  #   plt.close()

def main():
  dataset, driver_dataset = load_dataset('all_data.csv')
  # obtain a list of timer names ordered the aggregate time spent in each
  ordered_timers = get_ordered_timers(dataset)
  plot_dataset(dataset, driver_dataset, ordered_timers)


if __name__ == '__main__':
  main()