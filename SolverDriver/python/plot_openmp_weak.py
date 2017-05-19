#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
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

MIN_LINESTYLE='dotted'
MAX_LINESTYLE='solid'

plot_only_min = False
smooth_outliers = False
ht_consistent_yaxes = False
composite_count = 0

DECOMP_COLORS = {
  '64x1' : 'xkcd:greyish',
  '32x2' : 'xkcd:windows blue',
  '16x4' : 'xkcd:amber',
  '8x8'  : 'xkcd:faded green',
  '4x16' : 'xkcd:dusty purple'
}


def is_outlier(points, thresh=3.5):
  """
  Returns a boolean array with True if points are outliers and False
  otherwise.

  Parameters:
  -----------
      points : An numobservations by numdimensions array of observations
      thresh : The modified z-score to use as a threshold. Observations with
          a modified z-score (based on the median absolute deviation) greater
          than this value will be classified as outliers.

  Returns:
  --------
      mask : A numobservations-length boolean array.

  References:
  ----------
      Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
      Handle Outliers", The ASQC Basic References in Quality Control:
      Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
  """
  if len(points.shape) == 1:
    points = points[:, None]
  median = np.nanmedian(points, axis=0)
  diff = np.nansum((points - median) ** 2, axis=-1)
  diff = np.sqrt(diff)
  med_abs_deviation = np.median(diff)

  modified_z_score = 0.6745 * diff / med_abs_deviation

  return modified_z_score > thresh

# def timer_lookup_acrossDFs(df_source, df_dest, timer_name='3 - Constructing Preconditioner'):
#   print(df_source.groups())
#   # obtain an index, and query the driver's timer for the preconditioner setup
#   row_index = list(df_source.head(1).iloc[0].name)
#   row_index[0] = timer_name
#   row_index = tuple(row_index)
#   # we now can query the driver for the total time spend creating the preconditioner
#   adjacent_row = df_dest.loc[row_index]
#   print(adjacent_row)
#   return adjacent_row


def plot_composite(composite_group, my_nodes, my_ticks, driver_df, average=False,
                   annotate_with_driver=True, numbered_plots=True):

  # figure out the flat MPI time
  flat_mpi_df = composite_group.groupby(['procs_per_node', 'cores_per_proc', 'threads_per_core']).get_group((64,1,1))
  flat_mpi_df.rename(columns={QUANTITY_OF_INTEREST_MIN: 'flat_mpi_min',
                              QUANTITY_OF_INTEREST_MAX: 'flat_mpi_max'}, inplace=True)

  composite_group = pd.merge(composite_group, flat_mpi_df[['num_nodes',
                                                           'flat_mpi_min',
                                                           'flat_mpi_max']],
                             how='left',
                             on=['num_nodes'])

  decomp_groups = composite_group.groupby(['procs_per_node', 'cores_per_proc'])
  driver_decomp_groups = driver_df.groupby(['procs_per_node', 'cores_per_proc'])

  # for a specific group of data, compute the scaling terms, which are things like min/max
  # this also flattens the timer creating a 'fat_timer_name'
  # essentially, this function computes data that is relevant to a group, but not the whole
  my_tokens = SFP.getTokensFromDataFrameGroupBy(composite_group)
  simple_fname = SFP.getScalingFilename(my_tokens, weak=True, composite=True)
  simple_title = SFP.getScalingTitle(my_tokens, weak=True, composite=True)

  if numbered_plots:
    simple_fname = '{}-{}'.format(composite_count, simple_fname)
    global composite_count
    composite_count = composite_count + 1

  if not ht_consistent_yaxes:
    simple_fname = '{fname}-free_yaxis'.format(fname=simple_fname)

  if not FORCE_REPLOT:
    my_file = Path("{}.png".format(simple_fname))
    if my_file.is_file():
      print("Skipping {}.png".format(simple_fname))
      return

  if smooth_outliers:
    simple_fname = '{}-outliers-smoothed'.format(simple_fname)

  # the number of HT combos we have
  nhts = composite_group['threads_per_core'].nunique()
  ndecomps = len(decomp_groups)

  my_num_nodes = my_nodes.size

  fig_size = 5
  fig_size_height_inflation = 1.125
  fig_size_width_inflation  = 1.5
  fig = plt.figure()
  # width: there are two plots currently, currently I make these with a 'wide' aspect ratio
  # height: a factor of the number of HTs shown
  fig.set_size_inches(fig_size * 2.0 * fig_size_width_inflation,
                      fig_size * nhts * fig_size_height_inflation)

  ax = []
  factor_ax = []
  perc_ax = []
  indep_plot = []

  for plot_idx in range(0, nhts):
    ax_ = fig.add_subplot(nhts, 2, plot_idx*2 + 1 )
    perc_ax_ = fig.add_subplot(nhts, 2, plot_idx*2 + 2)
    perc_ax_.yaxis.set_major_formatter(FormatStrFormatter('%3.0f %%'))

    temp_fig = plt.figure()
    temp_fig.set_size_inches(fig_size * fig_size_width_inflation * 1.5,
                             fig_size * fig_size_height_inflation * 1.5)
    indep_plot.append(temp_fig)
    ax.append(ax_)
    factor_ax.append(ax_.twinx())
    perc_ax.append(perc_ax_)

  composite_group['flat_mpi_factor_min'] = composite_group[QUANTITY_OF_INTEREST_MIN] / composite_group['flat_mpi_min']
  composite_group['flat_mpi_factor_max'] = composite_group[QUANTITY_OF_INTEREST_MAX] / composite_group['flat_mpi_max']

  for decomp_group_name, decomp_group in decomp_groups:
    procs_per_node = int(decomp_group_name[0])
    cores_per_proc = int(decomp_group_name[1])
    # label this decomp
    decomp_label = "{procs_per_node}x{cores_per_proc}".format(procs_per_node=procs_per_node,
                                                              cores_per_proc=cores_per_proc)
    # iterate over HTs
    ht_groups = decomp_group.groupby('threads_per_core')
    driver_ht_groups = driver_decomp_groups.get_group(decomp_group_name).groupby('threads_per_core')
    ht_n = 1
    max_ht_n = len(ht_groups)
    plot_idx = 0
    for ht_name, ht_group in ht_groups:
      threads_per_core = int(ht_name)
      # Use aggregation. Since the dataset is noisy and we have multiple experiments worth of data
      # This is similar to ensemble type analysis
      # what you could do, is rather than sum, look at the variations between chunks of data and vote
      # outliers.
      timings = ht_group.groupby('num_nodes', as_index=False)[[QUANTITY_OF_INTEREST_MIN,
                                                               QUANTITY_OF_INTEREST_MAX,
                                                               QUANTITY_OF_INTEREST_THING,
                                                               'flat_mpi_min',
                                                               'flat_mpi_max']].sum()
      driver_timings = driver_ht_groups.get_group(ht_name).groupby('num_nodes',
                                                                   as_index=False)[
                                                                    [QUANTITY_OF_INTEREST_MIN,
                                                                     QUANTITY_OF_INTEREST_MAX,
                                                                     QUANTITY_OF_INTEREST_THING]].sum()

      # if we want average times, then sum the counts that correspond to these timers, and compute the mean
      if average:
        counts = ht_group.groupby('num_nodes', as_index=False)[[QUANTITY_OF_INTEREST_MIN_COUNT,
                                                                QUANTITY_OF_INTEREST_MAX_COUNT]].sum()

        driver_counts = driver_ht_groups.get_group(ht_name).groupby('num_nodes', as_index=False)[
                                                                      [QUANTITY_OF_INTEREST_MIN_COUNT,
                                                                       QUANTITY_OF_INTEREST_MAX_COUNT]].sum()


        timings[QUANTITY_OF_INTEREST_MIN] = timings[QUANTITY_OF_INTEREST_MIN] / counts[QUANTITY_OF_INTEREST_MIN_COUNT]
        timings[QUANTITY_OF_INTEREST_MAX] = timings[QUANTITY_OF_INTEREST_MAX] / counts[QUANTITY_OF_INTEREST_MAX_COUNT]

        driver_timings[QUANTITY_OF_INTEREST_MIN] = driver_timings[QUANTITY_OF_INTEREST_MIN] /\
                                                   driver_counts[QUANTITY_OF_INTEREST_MIN_COUNT]

        driver_timings[QUANTITY_OF_INTEREST_MAX] = driver_timings[QUANTITY_OF_INTEREST_MAX] /\
                                                   driver_counts[QUANTITY_OF_INTEREST_MAX_COUNT]

      driver_timings = driver_timings[['num_nodes', QUANTITY_OF_INTEREST_MIN,QUANTITY_OF_INTEREST_MAX]]
      driver_timings.rename(columns={QUANTITY_OF_INTEREST_MIN: 'Driver Min',
                                     QUANTITY_OF_INTEREST_MAX: 'Driver Max'}, inplace=True)

      barf_df = ht_group.groupby('num_nodes', as_index=False)[QUANTITY_OF_INTEREST_THING].mean()
      timings[QUANTITY_OF_INTEREST_THING] = barf_df[QUANTITY_OF_INTEREST_THING]

      my_agg_times = pd.DataFrame(columns=['num_nodes', 'ticks'], data=np.column_stack((my_nodes, my_ticks)))
      my_agg_times = pd.merge(my_agg_times, timings, on='num_nodes', how='left')
      my_agg_times = pd.merge(my_agg_times, driver_timings, on='num_nodes', how='left')

      # attempt to deal with the outliers
      if smooth_outliers:
        my_agg_times['max_is_outlier'] = is_outlier(my_agg_times[QUANTITY_OF_INTEREST_MAX].values, thresh=3.0)
        my_agg_times['min_is_outlier'] = is_outlier(my_agg_times[QUANTITY_OF_INTEREST_MIN].values, thresh=3.0)
        #print(my_agg_times)
        #df['flag'][df.name.str.contains('e$')] = 'Blue'
        my_agg_times[QUANTITY_OF_INTEREST_MAX][(my_agg_times['max_is_outlier'] == True)] = my_agg_times[QUANTITY_OF_INTEREST_MAX].median()
        my_agg_times[QUANTITY_OF_INTEREST_MIN][(my_agg_times['min_is_outlier'] == True)] = my_agg_times[QUANTITY_OF_INTEREST_MIN].median()
        #my_agg_times[(my_agg_times['max_is_outlier'] == True), QUANTITY_OF_INTEREST_MAX] = np.NAN
        #print(my_agg_times)

      # scale by 100 so these can be formatted as percentages
      my_agg_times['max_percent'] = my_agg_times[QUANTITY_OF_INTEREST_MAX] / my_agg_times['Driver Max'] * 100.00
      my_agg_times['min_percent'] = my_agg_times[QUANTITY_OF_INTEREST_MIN] / my_agg_times['Driver Max'] * 100.00

      # my_agg_times['flat_mpi_min_factor'] = my_agg_times[QUANTITY_OF_INTEREST_MIN] / my_agg_times['flat_mpi_min']
      # my_agg_times['flat_mpi_max_factor'] = my_agg_times[QUANTITY_OF_INTEREST_MAX] / my_agg_times['flat_mpi_max']

      # count the missing values, can use any quantity of interest for this
      num_missing_data_points = my_agg_times[QUANTITY_OF_INTEREST_MIN].isnull().values.ravel().sum()

      if num_missing_data_points != 0:
        print(
          "Expected {expected_data_points} data points, Missing: {num_missing_data_points}".format(
            expected_data_points=my_num_nodes,
            num_missing_data_points=num_missing_data_points))

      print("x={}, y={}, {}x{}x{}".format(my_agg_times['ticks'].count(),
                                my_agg_times['num_nodes'].count(),
                                procs_per_node,
                                cores_per_proc,
                                ht_name))

      if plot_only_min:
        # plot the data
        ax[plot_idx].plot(my_agg_times['ticks'], my_agg_times[QUANTITY_OF_INTEREST_MIN],
                          label='{}'.format(decomp_label), color=DECOMP_COLORS[decomp_label])
      else:
        # plot the data
        ax[plot_idx].plot(my_agg_times['ticks'], my_agg_times[QUANTITY_OF_INTEREST_MIN],
                          label='min-{}'.format(decomp_label), color=DECOMP_COLORS[decomp_label],
                          linestyle=MIN_LINESTYLE)
        ax[plot_idx].plot(my_agg_times['ticks'], my_agg_times[QUANTITY_OF_INTEREST_MAX],
                          label='max-{}'.format(decomp_label), color=DECOMP_COLORS[decomp_label],
                          linestyle=MAX_LINESTYLE)
      # ax[plot_idx].plot(my_agg_times['ticks'], my_agg_times[QUANTITY_OF_INTEREST_THING],
      #                   label='meanCT-{}'.format(decomp_label), color=DECOMP_COLORS[decomp_label],
      #                   marker='o', fillstyle='none', linestyle='none')
      ax[plot_idx].set_ylabel('Runtime (s)')

      # factor_ax[plot_idx].plot(my_agg_times['ticks'], my_agg_times['flat_mpi_min_factor'],
      #                          marker='o', fillstyle='none', linestyle='none', color=DECOMP_COLORS[decomp_label])
      # factor_ax[plot_idx].plot(my_agg_times['ticks'], my_agg_times['flat_mpi_max_factor'],
      #                          marker='x', fillstyle='none', linestyle='none', color=DECOMP_COLORS[decomp_label])
      # factor_ax[plot_idx].set_ylabel('Runtime as Factor of Flat MPI')

      # construct the independent figure
      if plot_only_min:
        # plot the data
        indep_plot[plot_idx].gca().plot(my_agg_times['ticks'], my_agg_times[QUANTITY_OF_INTEREST_MIN],
                          label='{}'.format(decomp_label), color=DECOMP_COLORS[decomp_label])
      else:
        indep_plot[plot_idx].gca().plot(my_agg_times['ticks'], my_agg_times[QUANTITY_OF_INTEREST_MIN],
                                        label='min-{}'.format(decomp_label), color=DECOMP_COLORS[decomp_label],
                                        linestyle=MIN_LINESTYLE)
        indep_plot[plot_idx].gca().plot(my_agg_times['ticks'], my_agg_times[QUANTITY_OF_INTEREST_MAX],
                                        label='max-{}'.format(decomp_label), color=DECOMP_COLORS[decomp_label],
                                        linestyle=MAX_LINESTYLE)
      indep_plot[plot_idx].gca().set_ylabel('Runtime (s)')
      indep_plot[plot_idx].gca().set_xlabel('Number of Nodes')
      indep_plot[plot_idx].gca().set_xticks(my_ticks)
      indep_plot[plot_idx].gca().set_xticklabels(my_nodes, rotation=45)
      indep_plot[plot_idx].gca().set_xlim([0.5, my_num_nodes + 1])
      indep_plot[plot_idx].gca().set_title('{}\n(HTs={:.0f})'.format(simple_title, ht_name))

      if plot_only_min:
        # plot the min/max percentage of total time
        perc_ax[plot_idx].plot(my_agg_times['ticks'], my_agg_times['min_percent'],
                          label='{}'.format(decomp_label), color=DECOMP_COLORS[decomp_label])
      else:
        # plot the min/max percentage of total time
        perc_ax[plot_idx].plot(my_agg_times['ticks'], my_agg_times['min_percent'],
                               label='min-{}'.format(decomp_label), color=DECOMP_COLORS[decomp_label])
        perc_ax[plot_idx].plot(my_agg_times['ticks'], my_agg_times['max_percent'],
                               label='max-{}'.format(decomp_label), color=DECOMP_COLORS[decomp_label],
                               linestyle=':')

      perc_ax[plot_idx].set_ylabel('Percentage of Total Time')

      if int(ht_name) != 1:
        ax[plot_idx].set_xlabel("Number of Nodes")
        ax[plot_idx].set_xticks(my_ticks)
        ax[plot_idx].set_xticklabels(my_nodes, rotation=45)
        ax[plot_idx].set_xlim([0.5, my_num_nodes + 1])

        perc_ax[plot_idx].set_xlabel("Number of Nodes")
        perc_ax[plot_idx].set_xticks(my_ticks)
        perc_ax[plot_idx].set_xticklabels(my_nodes, rotation=45)
        perc_ax[plot_idx].set_xlim([0.5, my_num_nodes + 1])

      # plot the titles
      if int(ht_name) == 1:
        ax[plot_idx].set_title('Raw Data\n(HTs={:.0f})'.format(ht_name))
        perc_ax[plot_idx].set_title('Percentage of Total Time\n(HTs={:.0f})'.format(ht_name))
        perc_ax[plot_idx].set_xticks([])
        ax[plot_idx].set_xticks([])
      else:
        ax[plot_idx].set_title('HTs={:.0f}'.format(ht_name))
        perc_ax[plot_idx].set_title('HTs={:.0f}'.format(ht_name))

      plot_idx = plot_idx + 1

  if ht_consistent_yaxes:
    best_ylims = [np.inf, -np.inf]
    for axis in ax:
      ylims = axis.get_ylim()
      best_ylims[0] = min(best_ylims[0], ylims[0])
      best_ylims[1] = max(best_ylims[1], ylims[1])

    # nothing below zero
    best_ylims[0] = max(best_ylims[0], 0.0)
    for axis in ax:
      axis.set_ylim(best_ylims)

    for figure in indep_plot:
      figure.gca().set_ylim(best_ylims)

    best_ylims = [np.inf, -np.inf]
    for axis in perc_ax:
      ylims = axis.get_ylim()
      best_ylims[0] = min(best_ylims[0], ylims[0])
      best_ylims[1] = max(best_ylims[1], ylims[1])

    for axis in perc_ax:
      axis.set_ylim(best_ylims)

  x=1
  for figure in indep_plot:
    handles, labels = figure.gca().get_legend_handles_labels()
    lgd = figure.legend(handles, labels,
               title="Procs per Node x Cores per Proc",
               loc='lower center', ncol=ndecomps, bbox_to_anchor=(0.5, 0.0))
    #figure.tight_layout()
    figure.subplots_adjust(bottom=0.20)
    try:
      figure.savefig("{}-{}.png".format(simple_fname, x), format='png', dpi=180)
      print("Wrote: {}-{}.png".format(simple_fname, x))
    except:
      print("FAILED writing {}-{}.png".format(simple_fname, x))
      raise
    x = x + 1
    plt.close(figure)

  handles, labels = ax[0].get_legend_handles_labels()
  fig.legend(handles, labels,
             title="Procs per Node x Cores per Proc",
             loc='lower center', ncol=ndecomps, bbox_to_anchor=(0.5, 0.0))

  fig.suptitle(simple_title, fontsize=18)
  # plt.subplots_adjust(top=0.9, hspace=0.2)
  fig.tight_layout()
  # this must be called after tight layout
  plt.subplots_adjust(top=0.85, bottom=0.15)
  try:
    fig.savefig("{}.png".format(simple_fname), format='png', dpi=180)
    print("Wrote: {}.png".format(simple_fname))
  except:
    print("FAILED writing {}.png".format(simple_fname))
    raise

  plt.close(fig)


def load_dataset(dataset_name):
  print('Reading {}'.format(dataset_name))
  # write the total dataset out, index=False, because we do not drop it above
  dataset = pd.read_csv(dataset_name, low_memory=False)

  print('Read csv complete')

  integral_columns = SFP.getIndexColumns(execspace_name='OpenMP')
  non_integral_names = ['Timer Name',
                           'problem_type',
                           'solver_name',
                           'solver_attributes',
                           'prec_name',
                           'prec_attributes',
                           'execspace_name',
                           'execspace_attributes']
  integral_columns = list(set(integral_columns).difference(set(non_integral_names)))
  dataset[integral_columns] = dataset[integral_columns].astype(np.int32)


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
  # dataset.sort_values(inplace=True,
  #                     by=SFP.getIndexColumns(execspace_name='OpenMP'))
  dataset.sort_index(inplace=True)
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


def plot_dataset(dataset, driver_dataset, ordered_timers,
                 total_time_key='',
                 restriction_tokens={}):

  # enforce all plots use the same num_nodes. i.e., the axes will be consistent
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
    plot_composite(spmv_agg_group, my_nodes, my_ticks, driver_dataset)

  #best_spmvs_df.rename(columns={QUANTITY_OF_INTEREST: 'best_spmv'}, inplace=True)
  #dataset = dataset.merge(best_spmvs_df, on=['problem_type', 'num_nodes'], suffixes=['', '_best_spmv'])

  # first, restrict the dataset to construction only muelu data
  if restriction_tokens:
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
  composite_groups = dataset.groupby(omp_groupby_columns)
  # group the driver timers the same way
  # it would be nice to pivot the driver timers so that they become columns in the actual data
  # e.g., each experiment has their corresponding driver timers as additional columns
  driver_composite_groups = driver_dataset.groupby(omp_groupby_columns)

  # if we sort timers, then construct a sorted set
  if ordered_timers:
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
      driver_constructor_name = list(composite_group_name)
      driver_constructor_name[0] = total_time_key
      plot_composite(composite_group,
                     my_nodes, my_ticks,
                     # this restricts the driver timers to those that match with the composite group
                     driver_composite_groups.get_group(tuple(driver_constructor_name)))
  else:
    for composite_group_name, composite_group in composite_groups:
      driver_constructor_name = list(composite_group_name)
      driver_constructor_name[0] = total_time_key
      plot_composite(composite_group,
                     my_nodes, my_ticks,
                     # this restricts the driver timers to those that match with the composite group
                     driver_composite_groups.get_group(tuple(driver_constructor_name)),
                     numbered_plots=False)

  exit(-1)
  # construct a stacked plot.

  # # first, restrict the dataset to construction only muelu data
  # restriction_tokens = {'solver_name' : 'Constructor',
  #                       'solver_attributes' : '-Only',
  #                       'prec_name' : 'MueLu',
  #                       'prec_attributes' : '-repartition'}
  #
  # stack_query_string = ' & '.join(['({name} == \"{value}\")'.format(
  #   name=name if ' ' not in name else '\"{}\"'.format(name),
  #   value=value)
  #                           for name,value in restriction_tokens.items() ])
  # print(stack_query_string)
  # stack_dataset = dataset.query(stack_query_string)
  stack_dataset = dataset

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
          row_index[0] = total_time_key
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

  study_type = "muelu_constructor"

  if study_type == 'muelu_constructor':
    plot_dataset(dataset, driver_dataset, ordered_timers,
                 total_time_key='3 - Constructing Preconditioner',
                 restriction_tokens={ 'solver_name' : 'Constructor',
                                      'solver_attributes' : '-Only',
                                      'prec_name' : 'MueLu',
                                      'prec_attributes' : '-repartition' })
  else:
    plot_dataset(dataset, driver_dataset, ordered_timers=[],
                 total_time_key='5 - Solve')



if __name__ == '__main__':
  main()