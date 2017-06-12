#!/usr/bin/env python3
"""plotter.py

Usage:
  plotter.py [-d DATASET -s STUDY]
  plotter.py (-h | --help)

Options:
  -h --help                      Show this screen.
  -d DATASET --dataset=DATASET   Input file [default: all_data.csv]
  -s STUDY --study=STUDY         Type of analysis, weak/strong [default: strong]
"""

from docopt import docopt
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from pathlib import Path
import copy
import ScalingFilenameParser as SFP
# from operator import itemgetter

MIN_NUM_NODES = 1
MAX_NUM_NODES = 10000
FORCE_REPLOT  = False
QUANTITY_OF_INTEREST       = 'minT'
QUANTITY_OF_INTEREST_COUNT = 'minC'

QUANTITY_OF_INTEREST_MIN         = 'minT'
QUANTITY_OF_INTEREST_MIN_COUNT   = 'minC'
QUANTITY_OF_INTEREST_MAX         = 'maxT'
QUANTITY_OF_INTEREST_MAX_COUNT   = 'maxC'
QUANTITY_OF_INTEREST_THING       = 'meanCT'
QUANTITY_OF_INTEREST_THING_COUNT = 'meanCC'

MIN_LINESTYLE = 'dotted'
MAX_LINESTYLE = 'solid'

PLOT_ONLY_MIN       = False
SMOOTH_OUTLIERS     = False
HT_CONSISTENT_YAXES = True

# define the colors used for each deomp type
DECOMP_COLORS = {
  '64x1'      : 'xkcd:greyish',
  '32x2'      : 'xkcd:windows blue',
  '16x4'      : 'xkcd:amber',
  '8x8'       : 'xkcd:faded green',
  '4x16'      : 'xkcd:dusty purple',
  'flat_mpi'  : 'xkcd:black'
}


###############################################################################
def sanity_check():
  """
  Report the version number of the core packages we use

  :return: Nothing
  """
  import matplotlib
  print('matplotlib: {}'.format(matplotlib.__version__))
  print('numpy: {}'.format(np.__version__))
  print('pandas: {}'.format(pd.__version__))


###############################################################################
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


# def add_flat_mpi_column(flat_mpi_dataset, other_dataset, scaling_study_type):
#   """
#   Add the flat MPI data to the other dataset
#
#   :param [in] flat_mpi_dataset: Dataframe obtained by load_dataset
#   :param [in/out] other_dataset: Dataframe obtained by load_dataset
#   :return: nothing, modifies the other dataset
#   """
#   # figure out the flat MPI time
#   # ideally, we would want to query the Serial execution space here... but that is kinda complicated, we likely
#   # need to add an argument that is a serial execution space dataframe, as the groupby logic expects the execution
#   # space to be the same
#   SFP.getMasterGroupBy(execspace_name='Serial', scaling_type=scaling_study_type)
#   flat_mpi_df = flat_mpi_data_group.groupby(
#                                         ['procs_per_node', 'cores_per_proc', 'threads_per_core']).get_group((64,1,1))
#   flat_mpi_df.rename(columns={QUANTITY_OF_INTEREST_MIN: 'flat_mpi_min',
#                               QUANTITY_OF_INTEREST_MAX: 'flat_mpi_max',
#                               QUANTITY_OF_INTEREST_MIN_COUNT : 'flat_mpi_min_count',
#                               QUANTITY_OF_INTEREST_MIN_COUNT : 'flat_mpi_max_count'}, inplace=True)
#
#   # add the flat mpi times as columns to all decompositions
#   # the join only uses num_nodes, as the other decomp information is only relevant for hybrid runs
#   other_dataset_data_group.merge(flat_mpi_df[['num_nodes',
#                                               'flat_mpi_min',
#                                               'flat_mpi_max',
#                                               'flat_mpi_min_count',
#                                               'flat_mpi_max_count']],
#                       how='left',
#                       on=['num_nodes'],
#                       indicator=True)


# def add_flat_mpi_column(flat_mpi_data_group, other_dataset_data_group):
#   """
#   This data should be grouped!  Given two groups of data, add the flat MPI data to the other data group
#
#   :param [in] flat_mpi_data_group: Dataframe obtained by a groupby, but not grouped at the decomp level
#   :param [in/out] other_dataset_data_group: Dataframe obtained by a groupby, but not grouped at the decomp level
#   :return: nothing, modifies the other dataset
#   """
#   # figure out the flat MPI time
#   # ideally, we would want to query the Serial execution space here... but that is kinda complicated, we likely
#   # need to add an argument that is a serial execution space dataframe, as the groupby logic expects the execution
#   # space to be the same
#   flat_mpi_df = flat_mpi_data_group.groupby(['procs_per_node', 'cores_per_proc', 'threads_per_core']).get_group((64,1,1))
#   flat_mpi_df.rename(columns={QUANTITY_OF_INTEREST_MIN: 'flat_mpi_min',
#                               QUANTITY_OF_INTEREST_MAX: 'flat_mpi_max',
#                               QUANTITY_OF_INTEREST_MIN_COUNT : 'flat_mpi_min_count',
#                               QUANTITY_OF_INTEREST_MIN_COUNT : 'flat_mpi_max_count'}, inplace=True)
#
#   # add the flat mpi times as columns to all decompositions
#   # the join only uses num_nodes, as the other decomp information is only relevant for hybrid runs
#   other_dataset_data_group.merge(flat_mpi_df[['num_nodes',
#                                               'flat_mpi_min',
#                                               'flat_mpi_max',
#                                               'flat_mpi_min_count',
#                                               'flat_mpi_max_count']],
#                       how='left',
#                       on=['num_nodes'],
#                       indicator=True)

def get_plottable_dataframe(plottable_df, data_group, data_name, driver_groups, average=False):
  # Use aggregation. Since the dataset is noisy and we have multiple experiments worth of data
  # This is similar to ensemble type analysis
  # what you could do, is rather than sum, look at the variations between chunks of data and vote
  # on outliers/anomalous values.
  timings = data_group.groupby('num_nodes', as_index=False)[[QUANTITY_OF_INTEREST_MIN,
                                                           QUANTITY_OF_INTEREST_MAX,
                                                           QUANTITY_OF_INTEREST_THING]].sum()

  driver_timings = driver_groups.get_group(data_name).groupby('num_nodes',
                                                               as_index=False)[
    [QUANTITY_OF_INTEREST_MIN,
     QUANTITY_OF_INTEREST_MAX,
     QUANTITY_OF_INTEREST_THING]].sum()

  # if we want average times, then sum the counts that correspond to these timers, and compute the mean
  if average:
    counts = data_group.groupby('num_nodes', as_index=False)[[QUANTITY_OF_INTEREST_MIN_COUNT,
                                                            QUANTITY_OF_INTEREST_MAX_COUNT]].sum()

    driver_counts = driver_groups.get_group(data_name).groupby('num_nodes', as_index=False)[
      [QUANTITY_OF_INTEREST_MIN_COUNT,
       QUANTITY_OF_INTEREST_MAX_COUNT]].sum()

    timings[QUANTITY_OF_INTEREST_MIN] = timings[QUANTITY_OF_INTEREST_MIN] / counts[QUANTITY_OF_INTEREST_MIN_COUNT]
    timings[QUANTITY_OF_INTEREST_MAX] = timings[QUANTITY_OF_INTEREST_MAX] / counts[QUANTITY_OF_INTEREST_MAX_COUNT]

    driver_timings[QUANTITY_OF_INTEREST_MIN] = driver_timings[QUANTITY_OF_INTEREST_MIN] / \
                                               driver_counts[QUANTITY_OF_INTEREST_MIN_COUNT]

    driver_timings[QUANTITY_OF_INTEREST_MAX] = driver_timings[QUANTITY_OF_INTEREST_MAX] / \
                                               driver_counts[QUANTITY_OF_INTEREST_MAX_COUNT]

  driver_timings = driver_timings[['num_nodes', QUANTITY_OF_INTEREST_MIN, QUANTITY_OF_INTEREST_MAX]]
  driver_timings.rename(columns={QUANTITY_OF_INTEREST_MIN: 'Driver Min',
                                 QUANTITY_OF_INTEREST_MAX: 'Driver Max'}, inplace=True)

  barf_df = data_group.groupby('num_nodes', as_index=False)[QUANTITY_OF_INTEREST_THING].mean()
  timings[QUANTITY_OF_INTEREST_THING] = barf_df[QUANTITY_OF_INTEREST_THING]

  print(plottable_df)
  plottable_df = plottable_df.merge(timings, on='num_nodes', how='left')
  print(plottable_df)
  plottable_df = plottable_df.merge(driver_timings, on='num_nodes', how='left')
  print(plottable_df)

  # attempt to deal with the outliers
  if SMOOTH_OUTLIERS:
    plottable_df['max_is_outlier'] = is_outlier(plottable_df[QUANTITY_OF_INTEREST_MAX].values, thresh=3.0)
    plottable_df['min_is_outlier'] = is_outlier(plottable_df[QUANTITY_OF_INTEREST_MIN].values, thresh=3.0)
    # print(my_agg_times)
    # df['flag'][df.name.str.contains('e$')] = 'Blue'
    plottable_df[QUANTITY_OF_INTEREST_MAX][(plottable_df['max_is_outlier'] is True)] = plottable_df[
      QUANTITY_OF_INTEREST_MAX].median()
    plottable_df[QUANTITY_OF_INTEREST_MIN][(plottable_df['min_is_outlier'] is True)] = plottable_df[
      QUANTITY_OF_INTEREST_MIN].median()
    # my_agg_times[(my_agg_times['max_is_outlier'] == True), QUANTITY_OF_INTEREST_MAX] = np.NAN
    # print(my_agg_times)

  # scale by 100 so these can be formatted as percentages
  plottable_df['max_percent'] = plottable_df[QUANTITY_OF_INTEREST_MAX] / plottable_df['Driver Max'] * 100.00
  plottable_df['min_percent'] = plottable_df[QUANTITY_OF_INTEREST_MIN] / plottable_df['Driver Max'] * 100.00

  # my_agg_times['flat_mpi_min_factor'] = my_agg_times[QUANTITY_OF_INTEREST_MIN] / my_agg_times['flat_mpi_min']
  # my_agg_times['flat_mpi_max_factor'] = my_agg_times[QUANTITY_OF_INTEREST_MAX] / my_agg_times['flat_mpi_max']
  return plottable_df


def get_figures_and_axes(subplot_names,
                         fig_size=5.0,
                         fig_size_width_inflation=1.0,
                         fig_size_height_inflation=1.0,
                         num_plots_per_col=1):
  axes = dict()
  figures = dict()
  num_plots_per_row = len(subplot_names)

  figures['composite'] = []
  figures['independent'] = dict()

  for name in subplot_names:
    axes[name] = []
    figures['independent'][name] = []

  fig = plt.figure()
  # width: there are two plots currently, currently I make these with a 'wide' aspect ratio
  # height: a factor of the number of HTs shown
  fig.set_size_inches(fig_size * num_plots_per_row * fig_size_width_inflation,
                      fig_size * num_plots_per_col * fig_size_height_inflation)

  figures['composite'].append(fig)

  for row_idx in range(0, num_plots_per_col):
    col_idx = 1
    for name in subplot_names:
      ax_ = fig.add_subplot(num_plots_per_col, num_plots_per_row, row_idx * num_plots_per_col + col_idx)

      temp_fig = plt.figure()
      temp_fig.set_size_inches(fig_size * fig_size_width_inflation * 1.5,
                               fig_size * fig_size_height_inflation * 1.5)

      axes[name].append(ax_)
      figures['independent'][name].append(temp_fig)

      col_idx += 1

  return axes, figures


def plot_raw_data(ax, indep_ax, xticks, yvalues, linestyle, label, color):

  if PLOT_ONLY_MIN:
    # plot the data
    ax.plot(xticks,
            yvalues,
            label=label,
            color=color,
            linestyle=linestyle)

    if indep_ax:
      indep_ax.plot(xticks,
                    yvalues,
                    label=label,
                    color=color,
                    linestyle=linestyle)


def set_weak_scaling_axes(ax, x_ticks, nodes, title):
  ax.set_xlabel("Number of Nodes")
  ax.set_xticks(x_ticks)
  ax.set_xticklabels(nodes, rotation=45)
  ax.set_xlim([0.5, len(nodes) + 1])


###############################################################################
def plot_composite_weak(composite_group,
                        my_nodes,
                        my_ticks,
                        driver_df,
                        average=False,
                        numbered_plots_idx=-1,
                        generate_indpendents=True,
                        scaling_study_type='weak'):
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
  :param numbered_plots_idx: if non-negative, then add this number before the filename
  :param scaling_study_type: weak/strong (TODO add onnode)
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
  # the join only uses num_nodes, as the other decomp information is only relevant for hybrid runs
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
  max_hts = composite_group['threads_per_core'].max()
  min_hts = composite_group['threads_per_core'].min()
  ndecomps = len(decomp_groups)

  my_num_nodes = my_nodes.size

  fig_size = 5
  fig_size_height_inflation = 1.125
  fig_size_width_inflation  = 1.5

  subplot_names = ['raw_data', 'percent_total', 'scaling_factor']
  axes, figures = get_figures_and_axes(subplot_names=subplot_names,
                                       fig_size=fig_size,
                                       fig_size_width_inflation=fig_size_width_inflation,
                                       fig_size_height_inflation=fig_size_height_inflation,
                                       num_plots_per_col=nhts)

  for ax in axes['percent_total']:
    ax.yaxis.set_major_formatter(FormatStrFormatter('%3.0f %%'))

  for fig in figures['percent_total']:
    fig.gca().yaxis.set_major_formatter(FormatStrFormatter('%3.0f %%'))

  # composite_group['flat_mpi_factor_min'] = composite_group[QUANTITY_OF_INTEREST_MIN] / composite_group['flat_mpi_min']
  # composite_group['flat_mpi_factor_max'] = composite_group[QUANTITY_OF_INTEREST_MAX] / composite_group['flat_mpi_max']

  for decomp_group_name, decomp_group in decomp_groups:
    procs_per_node = int(decomp_group_name[0])
    cores_per_proc = int(decomp_group_name[1])
    # label this decomp
    decomp_label = "{procs_per_node}x{cores_per_proc}".format(procs_per_node=procs_per_node,
                                                              cores_per_proc=cores_per_proc)
    # iterate over HTs
    ht_groups = decomp_group.groupby('threads_per_core')
    driver_ht_groups = driver_decomp_groups.get_group(decomp_group_name).groupby('threads_per_core')

    row_idx = 0
    for ht_name, ht_group in ht_groups:
      my_agg_times = pd.DataFrame(columns=['num_nodes', 'ticks'], data=np.column_stack((my_nodes, my_ticks)))

      my_agg_times = get_plottable_dataframe(my_agg_times, ht_group, ht_name, driver_ht_groups)

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
        plot_raw_data(ax=axes['raw_data'][row_idx],
                      indep_ax=figures['independents']['raw_data'][row_idx].gca(),
                      xticks=my_agg_times['ticks'],
                      yvalues=my_agg_times[QUANTITY_OF_INTEREST_MIN],
                      linestyle='.',
                      label='{}'.format(decomp_label),
                      color=DECOMP_COLORS[decomp_label])
      else:
        # plot the data
        plot_raw_data(ax=axes['raw_data'][row_idx],
                      indep_ax=figures['independents']['raw_data'][row_idx].gca(),
                      xticks=my_agg_times['ticks'],
                      yvalues=my_agg_times[QUANTITY_OF_INTEREST_MIN],
                      linestyle=MIN_LINESTYLE,
                      label='min-{}'.format(decomp_label),
                      color=DECOMP_COLORS[decomp_label])

        plot_raw_data(ax=axes['raw_data'][row_idx],
                      indep_ax=figures['independents']['raw_data'][row_idx].gca(),
                      xticks=my_agg_times['ticks'],
                      yvalues=my_agg_times[QUANTITY_OF_INTEREST_MAX],
                      linestyle=MAX_LINESTYLE,
                      label='max-{}'.format(decomp_label),
                      color=DECOMP_COLORS[decomp_label])

      axes['raw_data'][row_idx].set_ylabel('Runtime (s)')

      # construct the independent figure
      figures['independents']['raw_data'][row_idx].gca().set_ylabel('Runtime (s)')
      figures['independents']['raw_data'][row_idx].gca().set_xlabel('Number of Nodes')
      figures['independents']['raw_data'][row_idx].gca().set_xticks(my_ticks)
      figures['independents']['raw_data'][row_idx].gca().set_xticklabels(my_nodes, rotation=45)
      figures['independents']['raw_data'][row_idx].gca().set_xlim([0.5, my_num_nodes + 1])
      figures['independents']['raw_data'][row_idx].gca().set_title('{}\n(HTs={:.0f})'.format(simple_title, ht_name))

      # factor_ax[plot_idx].plot(my_agg_times['ticks'], my_agg_times['flat_mpi_min_factor'],
      #                          marker='o', fillstyle='none', linestyle='none', color=DECOMP_COLORS[decomp_label])
      # factor_ax[plot_idx].plot(my_agg_times['ticks'], my_agg_times['flat_mpi_max_factor'],
      #                          marker='x', fillstyle='none', linestyle='none', color=DECOMP_COLORS[decomp_label])
      # factor_ax[plot_idx].set_ylabel('Runtime as Factor of Flat MPI')
      if PLOT_ONLY_MIN:
        # plot the data
        plot_raw_data(ax=axes['percent_total'][row_idx],
                      indep_ax=figures['independents']['percent_total'][row_idx].gca(),
                      xticks=my_agg_times['ticks'],
                      yvalues=my_agg_times['min_percent'],
                      linestyle='.',
                      label='{}'.format(decomp_label),
                      color=DECOMP_COLORS[decomp_label])
      else:
        # plot the data
        plot_raw_data(ax=axes['percent_total'][row_idx],
                      indep_ax=figures['independents']['percent_total'][row_idx].gca(),
                      xticks=my_agg_times['ticks'],
                      yvalues=my_agg_times['min_percent'],
                      linestyle='.',
                      label='min-{}'.format(decomp_label),
                      color=DECOMP_COLORS[decomp_label])

        plot_raw_data(ax=axes['percent_total'][row_idx],
                      indep_ax=figures['independents']['percent_total'][row_idx].gca(),
                      xticks=my_agg_times['ticks'],
                      yvalues=my_agg_times['max_percent'],
                      linestyle=':',
                      label='max-{}'.format(decomp_label),
                      color=DECOMP_COLORS[decomp_label])

      axes['percent_total'][row_idx].set_ylabel('Percentage of Total Time')

      # use min_hts and max_hts + nhts to label the axes
      if int(ht_name) != 1:
        set_weak_scaling_axes(ax=axes['raw_data'][row_idx],
                              x_ticks=my_ticks,
                              nodes=my_nodes,
                              title='HTs={:.0f}'.format(ht_name))

        set_weak_scaling_axes(ax=figures['independents']['raw_data'][row_idx].gca(),
                              x_ticks=my_ticks,
                              nodes=my_nodes,
                              title='HTs={:.0f}'.format(ht_name))

        set_weak_scaling_axes(ax=axes['percent_total'][row_idx],
                              x_ticks=my_ticks,
                              nodes=my_nodes,
                              title='HTs={:.0f}'.format(ht_name))

        set_weak_scaling_axes(ax=figures['independents']['percent_total'][row_idx].gca(),
                              x_ticks=my_ticks,
                              nodes=my_nodes,
                              title='HTs={:.0f}'.format(ht_name))
        # ax[plot_idx].set_xlabel("Number of Nodes")
        # ax[plot_idx].set_xticks(my_ticks)
        # ax[plot_idx].set_xticklabels(my_nodes, rotation=45)
        # ax[plot_idx].set_xlim([0.5, my_num_nodes + 1])

        # perc_ax[plot_idx].set_xlabel("Number of Nodes")
        # perc_ax[plot_idx].set_xticks(my_ticks)
        # perc_ax[plot_idx].set_xticklabels(my_nodes, rotation=45)
        # perc_ax[plot_idx].set_xlim([0.5, my_num_nodes + 1])

      # plot the titles
      if int(ht_name) == 1:
        ax[plot_idx].set_title('Raw Data\n(HTs={:.0f})'.format(ht_name))
        perc_ax[plot_idx].set_title('Percentage of Total Time\n(HTs={:.0f})'.format(ht_name))
        perc_ax[plot_idx].set_xticks([])
        ax[plot_idx].set_xticks([])
      else:
        ax[plot_idx].set_title('HTs={:.0f}'.format(ht_name))
        perc_ax[plot_idx].set_title('HTs={:.0f}'.format(ht_name))

      plot_idx += 1

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


###############################################################################
def plot_composite_strong(composite_group,
                          my_nodes,
                          my_ticks,
                          driver_df,
                          average=False,
                          numbered_plots_idx=-1,
                          scaling_study_type='strong'):
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
  :param numbered_plots_idx: if non-negative, then add this number before the filename
  :param scaling_study_type: weak/strong (TODO add onnode)
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
  # the join only uses num_nodes, as the other decomp information is only relevant for hybrid runs
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
  simple_fname = SFP.getScalingFilename(my_tokens, strong=True, composite=True)
  simple_title = SFP.getScalingTitle(my_tokens, strong=True, composite=True)

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
  fig_size_height_inflation = 1.2
  fig_size_width_inflation  = 1.2
  fig = plt.figure()
  # width: there are three plots currently for strong scaling plots, we really want the aspect ratio
  # as close to 1.0 as possible
  # height: a factor of the number of HTs shown
  fig.set_size_inches(fig_size * 3.0 * fig_size_width_inflation,
                      fig_size * nhts * fig_size_height_inflation)

  axes = dict()
  figures = dict()
  axes['raw_data']   = []
  axes['speedup']    = []
  axes['efficiency'] = []

  figures['composite'] = []
  figures['independent'] = []

  for plot_idx in range(0, nhts):
    ax_ = fig.add_subplot(nhts, 2, plot_idx*2 + 1)
    perc_ax_ = fig.add_subplot(nhts, 2, plot_idx*2 + 2)
    perc_ax_.yaxis.set_major_formatter(FormatStrFormatter('%3.0f %%'))

    temp_fig = plt.figure()
    temp_fig.set_size_inches(fig_size * fig_size_width_inflation * 1.5,
                             fig_size * fig_size_height_inflation * 1.5)
    indep_plot.append(temp_fig)
    ax.append(ax_)
    factor_ax.append(ax_.twinx())
    perc_ax.append(perc_ax_)

  # composite_group['flat_mpi_factor_min'] = composite_group[QUANTITY_OF_INTEREST_MIN] / composite_group['flat_mpi_min']
  # composite_group['flat_mpi_factor_max'] = composite_group[QUANTITY_OF_INTEREST_MAX] / composite_group['flat_mpi_max']

  for decomp_group_name, decomp_group in decomp_groups:
    procs_per_node = int(decomp_group_name[0])
    cores_per_proc = int(decomp_group_name[1])
    # label this decomp
    decomp_label = "{procs_per_node}x{cores_per_proc}".format(procs_per_node=procs_per_node,
                                                              cores_per_proc=cores_per_proc)
    # iterate over HTs
    ht_groups = decomp_group.groupby('threads_per_core')
    driver_ht_groups = driver_decomp_groups.get_group(decomp_group_name).groupby('threads_per_core')

    plot_idx = 0
    for ht_name, ht_group in ht_groups:
      threads_per_core = int(ht_name)

      my_agg_times = pd.DataFrame(columns=['num_nodes', 'ticks'], data=np.column_stack((my_nodes, my_ticks)))

      my_agg_times = get_plottable_dataframe(my_agg_times, ht_group, ht_name, driver_ht_groups)

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


###############################################################################
def load_dataset(dataset_filename):
  """
  Load a CSV datafile. This assumes the data was parsed from YAML using the parser in this directory

  This loader currently restricts the dataset loaded to a specific range of node counts.
  It also removes problematic data, which are runs that never finished or were know to be excessively buggy.
  E.g., The Elasticity3D matrix, or running MueLu without repartitioning

  The loaded data is verified using pandas index and sorted by the index

  :param dataset_filename: path/filename to load
  :return: two pandas dataframes: timer_dataset and driver_dataset
           timer_dataset contains data for the actual timers inside Trilinos functions
           driver_dataset contains data generated by the driver which includes total time,
           preconditioner setup time, and solve time.
  """
  print('Reading {}'.format(dataset_filename))
  # write the total dataset out, index=False, because we do not drop it above
  dataset = pd.read_csv(dataset_filename, low_memory=False)

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
  # integral_columns = list(set(integral_columns).difference(set(non_integral_names)))
  # dataset[integral_columns] = dataset[integral_columns].astype(np.int32)

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


###############################################################################
def get_ordered_timers(dataset, rank_by_column_name):
  """
  Given a dataset, construct and ordered list of Timer Names that ranks based on the largest aggregate (sum)
  quantity of interest (rank by column name).

  :param dataset: Dataframe created by load_dataset()
  :param rank_by_column_name: the column to rank by
  :return: list of timer names in sorted order (descending)
  """
  # find an order of timer names based on aggregate time
  ordered_timers = dataset.groupby(['Timer Name',
                                    'problem_type',
                                    'solver_name',
                                    'solver_attributes',
                                    'prec_name',
                                    'prec_attributes'], as_index=False).sum().sort_values(
    by=rank_by_column_name, ascending=False)['Timer Name'].tolist()

  return ordered_timers


###############################################################################
def dict_to_pandas_query_string(kv):
  """
  convert a dict to a string of ("key" == "value") & ("key" == "value")...

  :param kv: dict of key value pairs
  :return: string for querying
  """
  query_string = ' & '.join(
    [
      '({name} == \"{value}\")'.format(
        # this quotes the lhs string if it contains a space
        name=name if ' ' not in name else '\"{}\"'.format(name),
        value=value)
      # end of the format class
      for name, value in kv.items()
    ])
  return query_string


###############################################################################
def get_aggregate_groups(dataset, scaling_type,
                         timer_name_rename='Operation Op*x',
                         timer_name_re_str='^.* Operation Op\*x$'):
  """
  given a dataset, use the regex provided to create a new dataset with timer names matching the RE,
  and then rename those timers to timer_name_rename

  This function is designed to gather all SpMV timers that are comparable, and then
  label them all using the same timer name.

  :param dataset: Dataframe created by load_dataset
  :param scaling_type: the type of scaling study being performed. This impacts how we query the dataset
  :param timer_name_rename: rename the 'Timer Name' values in the result to this string
  :param timer_name_re_str: use this regular expression to match timer names
  :return:
  """
  spmv_groupby_columns = SFP.getMasterGroupBy(execspace_name='OpenMP', scaling_type=scaling_type)
  spmv_groupby_columns.remove('procs_per_node')
  spmv_groupby_columns.remove('cores_per_proc')
  spmv_groupby_columns.remove('solver_name')
  spmv_groupby_columns.remove('solver_attributes')
  spmv_groupby_columns.remove('prec_name')
  spmv_groupby_columns.remove('prec_attributes')

  spmv_only_data = dataset[dataset['Timer Name'].str.match(timer_name_re_str)]
  spmv_only_data['Timer Name'] = timer_name_rename

  spmv_agg_groups = spmv_only_data.groupby(spmv_groupby_columns)
  return spmv_agg_groups


###############################################################################
def plot_composite(composite_group,
                   my_nodes,
                   my_ticks,
                   scaling_study_type,
                   numbered_plots_idx,
                   driver_df):
  if scaling_study_type == 'weak':
    plot_composite_weak(composite_group=composite_group,
                        my_nodes=my_nodes,
                        my_ticks=my_ticks,
                        numbered_plots_idx=numbered_plots_idx,
                        driver_df=driver_df)

  elif scaling_study_type == 'strong':
    plot_composite_strong(composite_group=composite_group,
                        my_nodes=my_nodes,
                        my_ticks=my_ticks,
                        numbered_plots_idx=numbered_plots_idx,
                        driver_df=driver_df)


###############################################################################
def plot_dataset(dataset,
                 driver_dataset,
                 ordered_timers,
                 total_time_key='',
                 restriction_tokens={},
                 scaling_type='weak',
                 number_plots=True):
  """
  The main function for plotting, which will call plot_composite many times. This function
  is also the logical place to add additional plotting features such as stacked plots

  :param dataset: timer data from a Dataframe created by load_dataset
  :param driver_dataset: driver data from a Dataframe created by load_dataset
  :param ordered_timers: optional list of timer names that should be plotted. (order will be preserved)
  :param total_time_key: timer name in driver_df that should represent the total time for a single experiment
  :param restriction_tokens: optional constraints to impose on the data, e.g., only constructor data
                             This could be removed and enforced prior to calling this function
  :param scaling_type: weak/strong, determines what type of plot will be generated
                       Ideally, this could be inferred from the data
  :param number_plots: Whether plots should be numbered as they are created. If ordered_timers is provided,
                       This will create plots with a numeric value prepended to the filename that ranks the timers
  :return: nothing
  """
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

  # optional numbered plots
  numbered_plots_idx = -1

  spmv_agg_groups = get_aggregate_groups(dataset=dataset, scaling_type=scaling_type)

  # plot the aggregate spmv data, e.g., all data regardless of experiment so long as the problem size is the
  for spmv_agg_name, spmv_agg_group in spmv_agg_groups:
    # increment this counter first, because it starts at the sentinel value of -1, which means no numbers
    if number_plots:
      numbered_plots_idx += 1

    plot_composite(composite_group=spmv_agg_group,
                   my_nodes=my_nodes,
                   my_ticks=my_ticks,
                   scaling_study_type=scaling_type,
                   numbered_plots_idx=numbered_plots_idx,
                   driver_df=driver_dataset)

  # restrict the dataset if requested
  if restriction_tokens:
    restriction_query_string = dict_to_pandas_query_string(restriction_tokens)

    print(restriction_query_string)
    dataset = dataset.query(restriction_query_string)
    driver_dataset = driver_dataset.query(restriction_query_string)

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

      # increment this counter first, because it starts at the sentinel value of -1, which means no numbers
      if number_plots:
        numbered_plots_idx += 1

      plot_composite(composite_group=composite_group,
                     my_nodes=my_nodes,
                     my_ticks=my_ticks,
                     scaling_study_type=scaling_type,
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
                     scaling_study_type=scaling_type,
                     driver_df=driver_composite_groups.get_group(tuple(driver_constructor_name)),
                     numbered_plots_idx=numbered_plots_idx)


###############################################################################
def main(dataset_filename,
         scaling_study_type):
  print(dataset_filename)
  dataset, driver_dataset = load_dataset(dataset_filename=dataset_filename)

  # obtain a list of timer names ordered the aggregate time spent in each
  ordered_timers = get_ordered_timers(dataset=dataset,
                                      rank_by_column_name=QUANTITY_OF_INTEREST)

  study_type = "muelu_constructor"

  if study_type == 'muelu_constructor':
    plot_dataset(dataset=dataset,
                 driver_dataset=driver_dataset,
                 ordered_timers=ordered_timers,
                 total_time_key='3 - Constructing Preconditioner',
                 restriction_tokens={'solver_name' : 'Constructor',
                                     'solver_attributes' : '-Only',
                                     'prec_name' : 'MueLu',
                                     'prec_attributes' : '-repartition'},
                 scaling_type=scaling_study_type)
  else:
    plot_dataset(dataset=dataset,
                 driver_dataset=driver_dataset,
                 ordered_timers=[],
                 total_time_key='5 - Solve',
                 scaling_type=scaling_study_type)


###############################################################################
if __name__ == '__main__':
  sanity_check()

  # Process input
  _arg_options = docopt(__doc__)

  _arg_dataset_filename  = _arg_options['--dataset']
  _arg_study_type        = _arg_options['--study']

  main(dataset_filename=_arg_dataset_filename,
       scaling_study_type=_arg_study_type)
