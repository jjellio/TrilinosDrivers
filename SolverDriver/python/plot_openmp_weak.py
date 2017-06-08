#!/usr/bin/env python3
"""plotter.py

Usage:
  plotter.py -d DATASET [-s STUDY]
  plotter.py (-h | --help)

Options:
  -h --help                      Show this screen.
  -d DATASET --dataset=DATASET   Input file [default: all_data.csv]
  -s STUDY --study=STUDY         Type of analysis, weak/strong [default: strong]
"""

from   docopt import docopt
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

PLOT_ONLY_MIN = False
SMOOTH_OUTLIERS = False
HT_CONSISTENT_YAXES = False

# define the colors used for each deomp type
DECOMP_COLORS = {
  '64x1'      : 'xkcd:greyish',
  '32x2'      : 'xkcd:windows blue',
  '16x4'      : 'xkcd:amber',
  '8x8'       : 'xkcd:faded green',
  '4x16'      : 'xkcd:dusty purple',
  'flat_mpi'  : 'xkcd:black'
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


def plot_composite(composite_group, my_nodes, my_ticks, driver_df, average=False,
                   annotate_with_driver=True, numbered_plots_idx=-1):
  """
  Plot all decompositions on a single figure.

  This plotter expects a dataframe with all data that should be plotted. It also takes as parameters
  a list of nodes and ticks, which allows the graphs to have the same x-axes across all figures.

  By default, the code will plot the total aggregate time. Using the average flag, the aggregate time will be
  averaged using the call count. This is undesired, because the call count as reported by Teuchos is not necessarily
  the real call count, because timers are often nested, and the call count will increase each time the timer is hit,
  even if the timer is actually running.

  Numbering the plots uses a global counter to add a number prepended to each plots's filename. E.g., call this function
  with groups in a specific order and you can number the plots using that same order.

  :param composite_group: dataframe (or groupbyDataframe) of stuff to plot
  :param my_nodes:  a list of nodes that will be shared by all plots for the xaxis.
  :param my_ticks: the actual x location for the tick marks, which will be labeled using the node list
  :param driver_df: dataframe containing timers from the driver itself (not Trilinos/kernel timers)
  :param average: flag that will enable averaging
  :param annotate_with_driver: Not used. This is always on, and enables the calculation of percent total time.
  :param numbered_plots: add a number to each plot
  :return: nothing
  """

  # figure out the flat MPI time
  # ideally, we would want to query the Serial execution space here... but that is kinda complicated, we likely
  # need to add an argument that is a serial execution space dataframe, as the groupby logic expects the execution
  # space to be the same
  flat_mpi_df = composite_group.groupby(['procs_per_node', 'cores_per_proc', 'threads_per_core']).get_group((64,1,1))
  flat_mpi_df.rename(columns={QUANTITY_OF_INTEREST_MIN: 'flat_mpi_min',
                              QUANTITY_OF_INTEREST_MAX: 'flat_mpi_max'}, inplace=True)

  # add the flat mpi times as columns to all decompositions
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

  # if numbered, then prepend the number to the filename
  # and increment the count.
  if numbered_plots_idx >= 0:
    simple_fname = '{}-{}'.format(numbered_plots_idx, simple_fname)

  # whether or not the yaxes should be the same for all HT combos plotted
  if not HT_CONSISTENT_YAXES:
    simple_fname = '{fname}-free_yaxis'.format(fname=simple_fname)

  # whether we should replot images that already exist.
  if not FORCE_REPLOT:
    my_file = Path("{}.png".format(simple_fname))
    if my_file.is_file():
      print("Skipping {}.png".format(simple_fname))
      return

  # if we apply any type of smoothing, then note this in the filename
  if SMOOTH_OUTLIERS:
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
      if SMOOTH_OUTLIERS:
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

      if PLOT_ONLY_MIN:
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
      if PLOT_ONLY_MIN:
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

      if PLOT_ONLY_MIN:
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

  if HT_CONSISTENT_YAXES:
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

  return dataset, driver_dataset

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
                 restriction_tokens={},
                 scaling_type='weak',
                 number_plots=True):

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

  # # figure out the best SpMV time
  # matvec_df = dataset[dataset['Timer Name'] == 'Belos:CG: Operation Op*x']
  # matvec_df = matvec_df.groupby(['problem_type', 'num_nodes'])
  # best_spmvs_idx = matvec_df[QUANTITY_OF_INTEREST].idxmin()
  #
  # best_spmvs_df = dataset.ix[best_spmvs_idx]
  # best_spmvs_df.set_index(['problem_type', 'num_nodes'], drop=False, inplace=True, verify_integrity=True)
  # pd.set_option('display.expand_frame_repr', False)
  # print(best_spmvs_df[['problem_type',
  #                         'num_nodes',
  #                         'procs_per_node',
  #                         'cores_per_proc',
  #                         'threads_per_core',
  #                         QUANTITY_OF_INTEREST,
  #                         QUANTITY_OF_INTEREST_COUNT]])
  # pd.set_option('display.expand_frame_repr', True)

  omp_groupby_columns = SFP.getMasterGroupBy(execspace_name='OpenMP', scaling_type=scaling_type)
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
  omp_groupby_columns = SFP.getMasterGroupBy(execspace_name='OpenMP', scaling_type=scaling_type)
  omp_groupby_columns.remove('procs_per_node')
  omp_groupby_columns.remove('cores_per_proc')
  # print(omp_groupby_columns)
  composite_groups = dataset.groupby(omp_groupby_columns)
  # group the driver timers the same way
  # it would be nice to pivot the driver timers so that they become columns in the actual data
  # e.g., each experiment has their corresponding driver timers as additional columns
  driver_composite_groups = driver_dataset.groupby(omp_groupby_columns)

  # optional numbered plots
  numbered_plots_idx = -1

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

    # loop over the sorted names, which are index tuples
    for composite_group_name in sorted_composite_groups:
      composite_group = composite_groups.get_group(composite_group_name)
      # construct an index into the driver_df by changing the timer label to match the driver's
      # global total label
      driver_constructor_name = list(composite_group_name)
      driver_constructor_name[0] = total_time_key

      plot_composite(composite_group=composite_group,
                     my_nodes=my_nodes,
                     my_ticks=my_ticks,
                     driver_df=driver_composite_groups.get_group(tuple(driver_constructor_name)),
                     numbered_plots_idx=numbered_plots_idx)

  else:
    # loop over the groups using the built in iterator (name,group)
    for composite_group_name, composite_group in composite_groups:
      # construct an index into the driver_df by changing the timer label to match the driver's
      # global total label
      driver_constructor_name = list(composite_group_name)
      driver_constructor_name[0] = total_time_key

      # increment this counter first, because it starts at the sentinel value of -1, which means no numbers
      if number_plots:
        numbered_plots_idx += 1

      plot_composite(composite_group=composite_group,
                     my_nodes=my_nodes,
                     my_ticks=my_ticks,
                     driver_df=driver_composite_groups.get_group(tuple(driver_constructor_name)),
                     numbered_plots_idx=numbered_plots_idx)


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
  # Process input
  options = docopt(__doc__)

  dataset_filename  = options['--dataset']
  study_type        = options['--study']

  main()
