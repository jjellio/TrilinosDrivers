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


def get_plottable_dataframe(plottable_df, data_group, data_name, driver_groups, compute_strong_terms=False):
  # Use aggregation. Since the dataset is noisy and we have multiple experiments worth of data
  # This is similar to ensemble type analysis
  # what you could do, is rather than sum, look at the variations between chunks of data and vote
  # on outliers/anomalous values.
  timings = data_group.groupby('num_nodes', as_index=False)[[QUANTITY_OF_INTEREST_MIN,
                                                             QUANTITY_OF_INTEREST_MIN_COUNT,
                                                             QUANTITY_OF_INTEREST_MAX,
                                                             QUANTITY_OF_INTEREST_MAX_COUNT,
                                                             QUANTITY_OF_INTEREST_THING,
                                                             QUANTITY_OF_INTEREST_THING_COUNT,
                                                             'numsteps',
                                                             'flat_mpi_min',
                                                             'flat_mpi_max',
                                                             'flat_mpi_min_count',
                                                             'flat_mpi_max_count',
                                                             'flat_mpi_numsteps']].sum()

  driver_timings = driver_groups.get_group(data_name).groupby('num_nodes',
                                                               as_index=False)[
    [QUANTITY_OF_INTEREST_MIN,
     QUANTITY_OF_INTEREST_MIN_COUNT,
     QUANTITY_OF_INTEREST_MAX,
     QUANTITY_OF_INTEREST_MAX_COUNT,
     QUANTITY_OF_INTEREST_THING,
     QUANTITY_OF_INTEREST_THING_COUNT,
     'numsteps']].sum()

  driver_timings = driver_timings[['num_nodes',
                                   QUANTITY_OF_INTEREST_MIN,
                                   QUANTITY_OF_INTEREST_MIN_COUNT,
                                   QUANTITY_OF_INTEREST_MAX,
                                   QUANTITY_OF_INTEREST_MAX_COUNT,
                                   QUANTITY_OF_INTEREST_THING,
                                   QUANTITY_OF_INTEREST_THING_COUNT,
                                   'numsteps']]

  driver_timings.rename(columns={QUANTITY_OF_INTEREST_MIN         : 'driver_min',
                                 QUANTITY_OF_INTEREST_MIN_COUNT   : 'driver_min_count',
                                 QUANTITY_OF_INTEREST_MAX         : 'driver_max',
                                 QUANTITY_OF_INTEREST_MAX_COUNT   : 'driver_max_count',
                                 QUANTITY_OF_INTEREST_THING       : 'driver_thing',
                                 QUANTITY_OF_INTEREST_THING_COUNT : 'driver_thing',
                                 'numsteps'                       : 'driver_numsteps'}, inplace=True)

  pd.set_option('display.expand_frame_repr', False)
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
  # make sure the numsteps parameter is the same, otherwise scale the data appropriately
  plottable_df['max_percent_t'] = plottable_df[QUANTITY_OF_INTEREST_MAX] / plottable_df['driver_max'] * 100.00
  plottable_df['min_percent_t'] = plottable_df[QUANTITY_OF_INTEREST_MIN] / plottable_df['driver_min'] * 100.00

  print(plottable_df)

  plottable_df.loc[~(plottable_df['numsteps'] == plottable_df['driver_numsteps']), 'max_percent_t'] = \
    (plottable_df[QUANTITY_OF_INTEREST_MAX] / plottable_df['numsteps']) \
                   / (plottable_df['driver_max'] / plottable_df['driver_numsteps']) * 100.00

  plottable_df.loc[~(plottable_df['numsteps'] == plottable_df['driver_numsteps']), 'min_percent_t'] = \
    (plottable_df[QUANTITY_OF_INTEREST_MIN] / plottable_df['numsteps']) \
                   / (plottable_df['driver_max'] / plottable_df['driver_numsteps']) * 100.00

  plottable_df['min_percent'] = plottable_df[['max_percent_t', 'min_percent_t']].min(axis=1)
  plottable_df['max_percent'] = plottable_df[['max_percent_t', 'min_percent_t']].max(axis=1)
  print(plottable_df)
  plottable_df = plottable_df.drop('min_percent_t', 1)
  plottable_df = plottable_df.drop('max_percent_t', 1)

  plottable_df['flat_mpi_factor_min_t'] = (plottable_df[QUANTITY_OF_INTEREST_MIN] / plottable_df['numsteps']) \
                                      / (plottable_df['flat_mpi_min'] / plottable_df['flat_mpi_numsteps'])

  plottable_df['flat_mpi_factor_max_t'] = (plottable_df[QUANTITY_OF_INTEREST_MAX] / plottable_df['numsteps']) \
                                      / (plottable_df['flat_mpi_max'] / plottable_df['flat_mpi_numsteps'])

  plottable_df['flat_mpi_factor_min'] = plottable_df[['flat_mpi_factor_max_t', 'flat_mpi_factor_min_t']].min(axis=1)
  plottable_df['flat_mpi_factor_max'] = plottable_df[['flat_mpi_factor_max_t', 'flat_mpi_factor_min_t']].max(axis=1)
  print(plottable_df)
  plottable_df = plottable_df.drop('flat_mpi_factor_min_t', 1)
  plottable_df = plottable_df.drop('flat_mpi_factor_max_t', 1)

  if ~compute_strong_terms:
    pd.set_option('display.expand_frame_repr', True)
    return plottable_df

  # compute Speedup and Efficiency. Use Flat MPI as the baseline
  np1_min = plottable_df.loc[(plottable_df['num_nodes'] == 1), 'flat_mpi_min', 'flat_mpi_numsteps'].min()
  np1_max = plottable_df.loc[(plottable_df['num_nodes'] == 1), 'flat_mpi_max', 'flat_mpi_numsteps'].max()

  print(np1_min)
  print(np1_max)
  exit(-1)

  pd.set_option('display.expand_frame_repr', True)
  return plottable_df


def enforce_consistent_ylims(figures, axes):
  # if we want consistent axes by column, then enforce that here.
  if HT_CONSISTENT_YAXES:
    for column_name, column_map in axes.items():
      best_ylims = [np.inf, -np.inf]

      for axes_name, ax in column_map.items():
        ylims = ax.get_ylim()
        best_ylims[0] = min(best_ylims[0], ylims[0])
        best_ylims[1] = max(best_ylims[1], ylims[1])

      # nothing below zero
      best_ylims[0] = max(best_ylims[0], 0.0)
      # apply these limits to each plot in this column
      for axes_name, ax in column_map.items():
        ax.set_ylim(best_ylims)

      for figure_name, fig in figures['independent'][column_name].items():
        fig.gca().set_ylim(best_ylims)


def save_figures(figures, filename, close_figure=False):
  try:
    figures['composite'].savefig('{}.png'.format(filename),
                                 format='png',
                                 dpi=180)
    print('Wrote: {}.png'.format(filename))
  except:
    print('FAILED writing {}.png'.format(filename))
    raise

  if close_figure:
    plt.close(figures['composite'])

  for column_name in figures['independent']:
    for ht_name in figures['independent'][column_name]:
      fig_filename = '{base}-{col}-{ht}.png'.format(base=filename, col=column_name, ht=ht_name)

      try:
        figures['independent'][column_name][ht_name].savefig(fig_filename,
                                                             format='png',
                                                             dpi=180)
        print('Wrote: {}.png'.format(fig_filename))
      except:
        print('FAILED writing {}.png'.format(fig_filename))
        raise

      if close_figure:
        plt.close(figures['independent'][column_name][ht_name])


def get_figures_and_axes(subplot_names,
                         subplot_row_names,
                         fig_size=5.0,
                         fig_size_width_inflation=1.0,
                         fig_size_height_inflation=1.0):
  axes = dict()
  figures = dict()
  num_plots_per_row = len(subplot_names)
  num_plots_per_col = len(subplot_row_names)

  figures['independent'] = dict()

  for name in subplot_names:
    axes[name] = dict()
    figures['independent'][name] = dict()

  figures['composite'] = plt.figure()
  # width: there are two plots currently, currently I make these with a 'wide' aspect ratio
  # height: a factor of the number of HTs shown
  figures['composite'].set_size_inches(fig_size * num_plots_per_row * fig_size_width_inflation,
                      fig_size * num_plots_per_col * fig_size_height_inflation)

  for row_idx in range(0, num_plots_per_col):
    subplot_row_name = subplot_row_names[row_idx]
    col_idx = 1
    for subplot_col_name in subplot_names:
      axes[subplot_col_name][subplot_row_name] = figures['composite'].add_subplot(
        num_plots_per_col,
        num_plots_per_row,
        row_idx * num_plots_per_row + col_idx)

      figures['independent'][subplot_col_name][subplot_row_name] = plt.figure()
      figures['independent'][subplot_col_name][subplot_row_name].set_size_inches(
        fig_size * fig_size_width_inflation * 1.5,
        fig_size * fig_size_height_inflation * 1.5)

      col_idx += 1

  return axes, figures


def plot_raw_data(ax, indep_ax, xticks, yvalues, linestyle, label, color):

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


###############################################################################
def need_to_replot(simple_fname, subplot_names, ht_names):
  need_to_replot_ = False
  my_file = Path("{}.png".format(simple_fname))
  try:
    temp = my_file.resolve()
  except FileNotFoundError or RuntimeError:
    print("File {}.png does not exist triggering replot".format(simple_fname))
    need_to_replot_ = True

  for column_name in subplot_names:
    for ht_name in ht_names:
      fig_filename = '{base}-{col}-{ht}.png'.format(base=simple_fname, col=column_name, ht=ht_name)
      my_file = Path(fig_filename)
      try:
        temp = my_file.resolve()
      except FileNotFoundError or RuntimeError:
        print("File {} does not exist triggering replot".format(fig_filename))
        need_to_replot_ = True

  return need_to_replot_


def add_flat_mpi_data(composite_group,
                      allow_baseline_override=False):
  # figure out the flat MPI time
  # ideally, we would want to query the Serial execution space here... but that is kinda complicated, we likely
  # need to add an argument that is a serial execution space dataframe, as the groupby logic expects the execution
  # space to be the same
  try :
    flat_mpi_df = composite_group.groupby(['procs_per_node', 'cores_per_proc', 'threads_per_core']).get_group((64,1,1))
  except KeyError as e:
    if allow_baseline_override:
      flat_mpi_df = composite_group.groupby(['procs_per_node', 'cores_per_proc', 'threads_per_core']).get_group((32, 2, 1))
    else:
      raise e

  flat_mpi_df.rename(columns={QUANTITY_OF_INTEREST_MIN: 'flat_mpi_min',
                              QUANTITY_OF_INTEREST_MAX: 'flat_mpi_max',
                              QUANTITY_OF_INTEREST_MIN_COUNT : 'flat_mpi_min_count',
                              QUANTITY_OF_INTEREST_MAX_COUNT : 'flat_mpi_max_count',
                              'numsteps'                     : 'flat_mpi_numsteps'}, inplace=True)

  # make sure this is one value per num_nodes.
  flat_mpi_df = flat_mpi_df.groupby('num_nodes')[[
                             'flat_mpi_min',
                             'flat_mpi_max',
                             'flat_mpi_min_count',
                             'flat_mpi_max_count',
                             'flat_mpi_numsteps']].sum()

  # add the flat mpi times as columns to all decompositions
  # join on the number of nodes, which will replicate the min/max to each decomposition that used
  # that number of nodes.
  composite_group = pd.merge(composite_group,
                             flat_mpi_df[[
                                         'flat_mpi_min',
                                         'flat_mpi_max',
                                         'flat_mpi_min_count',
                                         'flat_mpi_max_count',
                                         'flat_mpi_numsteps']],
                             how='left',
                             left_on=['num_nodes'],
                             right_index=True)
  return composite_group

###############################################################################
def plot_composite_weak(composite_group,
                        my_nodes,
                        my_ticks,
                        driver_df,
                        average=False,
                        numbered_plots_idx=-1,
                        scaling_study_type='weak',
                        kwargs={}):
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
  :param kwargs: Optional plotting arguments:
                 show_percent_total : True/False, plot a column that show the percentage of total time
                                      Total time is defined by the driver_df, and is expected to be in the columns
                                      Driver Min and Driver Max.
                 TODO: add show_factor, which would plot the thread scaling as a ratio of the serial execution space
                       runtime.
  :return: nothing
  """
  show_percent_total = True
  show_factor = True

  print(kwargs.keys())
  if kwargs is not None:
    if 'show_percent_total' in kwargs.keys():
      print('in the keys')
      show_percent_total = kwargs['show_percent_total']
    if 'show_factor' in kwargs.keys():
      show_factor = kwargs['show_factor']

  # determine the flat MPI time
  composite_group = add_flat_mpi_data(composite_group)

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

  # if we apply any type of smoothing, then note this in the filename
  if SMOOTH_OUTLIERS:
    simple_fname = '{}-outliers-smoothed'.format(simple_fname)

  # the number of HT combos we have
  ht_names = composite_group['threads_per_core'].sort_values(ascending=True).unique()
  ndecomps = len(decomp_groups)

  my_num_nodes = my_nodes.size

  fig_size = 5
  fig_size_height_inflation = 1.125
  fig_size_width_inflation  = 1.5

  subplot_names = ['raw_data', 'percent_total']

  if show_percent_total is False:
    subplot_names.remove('percent_total')
  if show_factor:
    subplot_names.append('flat_mpi_factor')

  # whether we should replot images that already exist.
  if FORCE_REPLOT is False:
    if not need_to_replot(simple_fname, subplot_names, ht_names):
      print("Skipping {}.png".format(simple_fname))
      return

  axes, figures = get_figures_and_axes(subplot_names=subplot_names,
                                       subplot_row_names=ht_names,
                                       fig_size=fig_size,
                                       fig_size_width_inflation=fig_size_width_inflation,
                                       fig_size_height_inflation=fig_size_height_inflation)

  for decomp_group_name, decomp_group in decomp_groups:
    procs_per_node = int(decomp_group_name[0])
    cores_per_proc = int(decomp_group_name[1])
    # label this decomp
    decomp_label = "{procs_per_node}x{cores_per_proc}".format(procs_per_node=procs_per_node,
                                                              cores_per_proc=cores_per_proc)
    # iterate over HTs
    ht_groups = decomp_group.groupby('threads_per_core')
    driver_ht_groups = driver_decomp_groups.get_group(decomp_group_name).groupby('threads_per_core')

    for ht_name in ht_names:
      ht_group = ht_groups.get_group(ht_name)
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
        plot_raw_data(ax=axes['raw_data'][ht_name],
                      indep_ax=figures['independent']['raw_data'][ht_name].gca(),
                      xticks=my_agg_times['ticks'],
                      yvalues=my_agg_times[QUANTITY_OF_INTEREST_MIN],
                      linestyle='-',
                      label='{}'.format(decomp_label),
                      color=DECOMP_COLORS[decomp_label])
      else:
        # plot the data
        plot_raw_data(ax=axes['raw_data'][ht_name],
                      indep_ax=figures['independent']['raw_data'][ht_name].gca(),
                      xticks=my_agg_times['ticks'],
                      yvalues=my_agg_times[QUANTITY_OF_INTEREST_MIN],
                      linestyle=MIN_LINESTYLE,
                      label='min-{}'.format(decomp_label),
                      color=DECOMP_COLORS[decomp_label])

        plot_raw_data(ax=axes['raw_data'][ht_name],
                      indep_ax=figures['independent']['raw_data'][ht_name].gca(),
                      xticks=my_agg_times['ticks'],
                      yvalues=my_agg_times[QUANTITY_OF_INTEREST_MAX],
                      linestyle=MAX_LINESTYLE,
                      label='max-{}'.format(decomp_label),
                      color=DECOMP_COLORS[decomp_label])

      # factor_ax[plot_idx].plot(my_agg_times['ticks'], my_agg_times['flat_mpi_min_factor'],
      #                          marker='o', fillstyle='none', linestyle='none', color=DECOMP_COLORS[decomp_label])
      # factor_ax[plot_idx].plot(my_agg_times['ticks'], my_agg_times['flat_mpi_max_factor'],
      #                          marker='x', fillstyle='none', linestyle='none', color=DECOMP_COLORS[decomp_label])
      # factor_ax[plot_idx].set_ylabel('Runtime as Factor of Flat MPI')
      if show_percent_total:
        if PLOT_ONLY_MIN:
          # plot the data
          plot_raw_data(ax=axes['percent_total'][ht_name],
                        indep_ax=figures['independent']['percent_total'][ht_name].gca(),
                        xticks=my_agg_times['ticks'],
                        yvalues=my_agg_times['min_percent'],
                        linestyle='-',
                        label='{}'.format(decomp_label),
                        color=DECOMP_COLORS[decomp_label])
        else:
          # plot the data
          plot_raw_data(ax=axes['percent_total'][ht_name],
                        indep_ax=figures['independent']['percent_total'][ht_name].gca(),
                        xticks=my_agg_times['ticks'],
                        yvalues=my_agg_times['min_percent'],
                        linestyle=MIN_LINESTYLE,
                        label='min-{}'.format(decomp_label),
                        color=DECOMP_COLORS[decomp_label])

          plot_raw_data(ax=axes['percent_total'][ht_name],
                        indep_ax=figures['independent']['percent_total'][ht_name].gca(),
                        xticks=my_agg_times['ticks'],
                        yvalues=my_agg_times['max_percent'],
                        linestyle=MAX_LINESTYLE,
                        label='max-{}'.format(decomp_label),
                        color=DECOMP_COLORS[decomp_label])
      if show_factor:
        if PLOT_ONLY_MIN:
          # plot the data
          plot_raw_data(ax=axes['flat_mpi_factor'][ht_name],
                        indep_ax=figures['independent']['flat_mpi_factor'][ht_name].gca(),
                        xticks=my_agg_times['ticks'],
                        yvalues=my_agg_times['flat_mpi_factor_min'],
                        linestyle='-',
                        label='{}'.format(decomp_label),
                        color=DECOMP_COLORS[decomp_label])
        else:
          # plot the data
          plot_raw_data(ax=axes['flat_mpi_factor'][ht_name],
                        indep_ax=figures['independent']['flat_mpi_factor'][ht_name].gca(),
                        xticks=my_agg_times['ticks'],
                        yvalues=my_agg_times['flat_mpi_factor_min'],
                        linestyle=MIN_LINESTYLE,
                        label='min-{}'.format(decomp_label),
                        color=DECOMP_COLORS[decomp_label])

          plot_raw_data(ax=axes['flat_mpi_factor'][ht_name],
                        indep_ax=figures['independent']['flat_mpi_factor'][ht_name].gca(),
                        xticks=my_agg_times['ticks'],
                        yvalues=my_agg_times['flat_mpi_factor_max'],
                        linestyle=MAX_LINESTYLE,
                        label='max-{}'.format(decomp_label),
                        color=DECOMP_COLORS[decomp_label])

  # configure the axes for the plotted data
  for row_idx in range(0, len(ht_names)):
    ht_name = ht_names[row_idx]
    # for independent plots (e.g., the subplot plotted separately) we show all axes labels
    ## raw data
    figures['independent']['raw_data'][ht_name].gca().set_ylabel('Runtime (s)')
    figures['independent']['raw_data'][ht_name].gca().set_xlabel('Number of Nodes')
    figures['independent']['raw_data'][ht_name].gca().set_xticks(my_ticks)
    figures['independent']['raw_data'][ht_name].gca().set_xticklabels(my_nodes, rotation=45)
    figures['independent']['raw_data'][ht_name].gca().set_xlim([0.5, my_num_nodes + 1])
    figures['independent']['raw_data'][ht_name].gca().set_title('{}\n(HTs={:.0f})'.format(simple_title, ht_name))
    ## percentages
    if show_percent_total:
      figures['independent']['percent_total'][ht_name].gca().set_ylabel('Percentage of Total Time')
      figures['independent']['percent_total'][ht_name].gca().set_xlabel('Number of Nodes')
      figures['independent']['percent_total'][ht_name].gca().set_xticks(my_ticks)
      figures['independent']['percent_total'][ht_name].gca().set_xticklabels(my_nodes, rotation=45)
      figures['independent']['percent_total'][ht_name].gca().set_xlim([0.5, my_num_nodes + 1])
      figures['independent']['percent_total'][ht_name].gca().set_title('{}\n(HTs={:.0f})'.format(simple_title, ht_name))
      figures['independent']['percent_total'][ht_name].gca().yaxis.set_major_formatter(FormatStrFormatter('%3.0f %%'))
    ## factors
    if show_factor:
      figures['independent']['flat_mpi_factor'][ht_name].gca().set_ylabel('Ratio (smaller is better)')
      figures['independent']['flat_mpi_factor'][ht_name].gca().set_xlabel('Number of Nodes')
      figures['independent']['flat_mpi_factor'][ht_name].gca().set_xticks(my_ticks)
      figures['independent']['flat_mpi_factor'][ht_name].gca().set_xticklabels(my_nodes, rotation=45)
      figures['independent']['flat_mpi_factor'][ht_name].gca().set_xlim([0.5, my_num_nodes + 1])
      figures['independent']['flat_mpi_factor'][ht_name].gca().set_title('{}\n(HTs={:.0f})'.format(simple_title, ht_name))
    axes['raw_data'][ht_name].set_ylabel('Runtime (s)')
    if show_percent_total:
      axes['percent_total'][ht_name].set_ylabel('Percentage of Total Time')
      axes['percent_total'][ht_name].yaxis.set_major_formatter(FormatStrFormatter('%3.0f %%'))
    if show_factor:
      axes['flat_mpi_factor'][ht_name].set_ylabel('Ratio of Runtime to Flat MPI Time')

    # if this is the last row, then display x axes labels
    if row_idx == (len(ht_names) - 1):
      axes['raw_data'][ht_name].set_xlabel("Number of Nodes")
      axes['raw_data'][ht_name].set_xticks(my_ticks)
      axes['raw_data'][ht_name].set_xticklabels(my_nodes, rotation=45)
      axes['raw_data'][ht_name].set_xlim([0.5, my_num_nodes + 1])
      axes['raw_data'][ht_name].set_title('HTs={:.0f}'.format(ht_name))
      if show_percent_total:
        axes['percent_total'][ht_name].set_xlabel("Number of Nodes")
        axes['percent_total'][ht_name].set_xticks(my_ticks)
        axes['percent_total'][ht_name].set_xticklabels(my_nodes, rotation=45)
        axes['percent_total'][ht_name].set_xlim([0.5, my_num_nodes + 1])
        axes['percent_total'][ht_name].set_title('HTs={:.0f}'.format(ht_name))
      if show_factor:
        axes['flat_mpi_factor'][ht_name].set_xlabel("Number of Nodes")
        axes['flat_mpi_factor'][ht_name].set_xticks(my_ticks)
        axes['flat_mpi_factor'][ht_name].set_xticklabels(my_nodes, rotation=45)
        axes['flat_mpi_factor'][ht_name].set_xlim([0.5, my_num_nodes + 1])
        axes['flat_mpi_factor'][ht_name].set_title('HTs={:.0f}'.format(ht_name))

    # if this is the first row, display the full title, e.g., 'Foo \n Ht = {}'
    elif row_idx == 0:
      axes['raw_data'][ht_name].set_title('Raw Data\n(HTs={:.0f})'.format(ht_name))
      if show_percent_total:
        axes['percent_total'][ht_name].set_title('Percentage of Total Time\n(HTs={:.0f})'.format(ht_name))
      if show_factor:
        axes['flat_mpi_factor'][ht_name].set_title('Ratio of Runtime to Flat MPI Time\n(HTs={:.0f})'.format(ht_name))
      # delete the xticks, because we do not want any x axis labels
      axes['raw_data'][ht_name].set_xticks([])
      if show_percent_total:
        axes['percent_total'][ht_name].set_xticks([])
    # otherwise, this is a middle plot, show a truncated title
    else:
      axes['raw_data'][ht_name].set_title('HTs={:.0f}'.format(ht_name))
      if show_percent_total:
        axes['percent_total'][ht_name].set_title('HTs={:.0f}'.format(ht_name))
      if show_factor:
        axes['flat_mpi_factor'][ht_name].set_title('HTs={:.0f}')
      # delete the xticks, because we do not want any x axis labels
      axes['raw_data'][ht_name].set_xticks([])
      if show_percent_total:
        axes['percent_total'][ht_name].set_xticks([])
      if show_factor:
        axes['flat_mpi_factor'][ht_name].set_xticks([])

  # add a suptitle and configure the legend for each figure
  figures['composite'].suptitle(simple_title, fontsize=18)
  # we plot in a deterministic fashion, and so the order is consistent among all
  # axes plotted. This allows a single legend that is compatible with all plots.
  handles, labels = axes['raw_data'][ht_names[0]].get_legend_handles_labels()
  figures['composite'].legend(handles, labels,
                              title="Procs per Node x Cores per Proc",
                              loc='lower center', ncol=ndecomps, bbox_to_anchor=(0.5, 0.0))
  figures['composite'].tight_layout()
  # this must be called after tight layout
  figures['composite'].subplots_adjust(top=0.85, bottom=0.15)

  for column_name in figures['independent']:
    for fig_name, fig in figures['independent'][column_name].items():
      fig.legend(handles, labels,
                 title="Procs per Node x Cores per Proc",
                 loc='lower center', ncol=ndecomps, bbox_to_anchor=(0.5, 0.0))
      # add space since the titles are typically large
      fig.subplots_adjust(bottom=0.20)

  # save the free axis version of the figures
  save_figures(figures,
               filename='{basename}-free-yaxis'.format(basename=simple_fname),
               close_figure=False)

  # if we want consistent axes by column, then enforce that here.
  if HT_CONSISTENT_YAXES:
    enforce_consistent_ylims(figures, axes)

  # save the figures with the axes shared
  save_figures(figures, filename=simple_fname, close_figure=True)


###############################################################################
def plot_composite_strong(composite_group,
                        my_nodes,
                        my_ticks,
                        driver_df,
                        numbered_plots_idx=-1,
                        scaling_study_type='strong',
                        kwargs={}):
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
  :param numbered_plots_idx: if non-negative, then add this number before the filename
  :param scaling_study_type: weak/strong (TODO add onnode)
  :param kwargs: Optional plotting arguments:
                 show_percent_total : True/False, plot a column that show the percentage of total time
                                      Total time is defined by the driver_df, and is expected to be in the columns
                                      Driver Min and Driver Max.
                 TODO: add show_factor, which would plot the thread scaling as a ratio of the serial execution space
                       runtime.
  :return: nothing
  """
  show_percent_total = False
  show_speed_up = True
  show_efficiency = True
  show_factor = False

  print(kwargs.keys())
  if kwargs is not None:
    if 'show_percent_total' in kwargs.keys():
      print('in the keys')
      show_percent_total = kwargs['show_percent_total']
    if 'show_factor' in kwargs.keys():
      show_factor = kwargs['show_factor']
    if 'show_speed_up' in kwargs.keys():
      show_speed_up = kwargs['show_speed_up']
    if 'show_efficiency' in kwargs.keys():
      show_efficiency = kwargs['show_efficiency']

  # determine the flat MPI time
  composite_group = add_flat_mpi_data(composite_group, allow_baseline_override=True)

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

  # if we apply any type of smoothing, then note this in the filename
  if SMOOTH_OUTLIERS:
    simple_fname = '{}-outliers-smoothed'.format(simple_fname)

  # the number of HT combos we have
  ht_names = composite_group['threads_per_core'].sort_values(ascending=True).unique()
  ndecomps = len(decomp_groups)

  my_num_nodes = my_nodes.size

  fig_size = 5
  fig_size_height_inflation = 1.0
  fig_size_width_inflation  = 1.0

  subplot_names = ['raw_data', 'speed_up', 'efficiency']

  if show_speed_up is False:
    subplot_names.remove('speed_up')
  if show_efficiency is False:
    subplot_names.remove('efficiency')
  if show_percent_total:
    subplot_names.append('percent_total')
  if show_factor:
    subplot_names.append('flat_mpi_factor')

  # whether we should replot images that already exist.
  if FORCE_REPLOT is False:
    if not need_to_replot(simple_fname, subplot_names, ht_names):
      print("Skipping {}.png".format(simple_fname))
      return

  axes, figures = get_figures_and_axes(subplot_names=subplot_names,
                                       subplot_row_names=ht_names,
                                       fig_size=fig_size,
                                       fig_size_width_inflation=fig_size_width_inflation,
                                       fig_size_height_inflation=fig_size_height_inflation)

  for decomp_group_name, decomp_group in decomp_groups:
    procs_per_node = int(decomp_group_name[0])
    cores_per_proc = int(decomp_group_name[1])
    # label this decomp
    decomp_label = "{procs_per_node}x{cores_per_proc}".format(procs_per_node=procs_per_node,
                                                              cores_per_proc=cores_per_proc)
    # iterate over HTs
    ht_groups = decomp_group.groupby('threads_per_core')
    driver_ht_groups = driver_decomp_groups.get_group(decomp_group_name).groupby('threads_per_core')

    for ht_name in ht_names:
      ht_group = ht_groups.get_group(ht_name)
      my_agg_times = pd.DataFrame(columns=['num_nodes', 'ticks'], data=np.column_stack((my_nodes, my_ticks)))

      my_agg_times = get_plottable_dataframe(my_agg_times, ht_group, ht_name, driver_ht_groups,
                                             compute_strong_terms=True)

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
        plot_raw_data(ax=axes['raw_data'][ht_name],
                      indep_ax=figures['independent']['raw_data'][ht_name].gca(),
                      xticks=my_agg_times['ticks'],
                      yvalues=my_agg_times[QUANTITY_OF_INTEREST_MIN],
                      linestyle='-',
                      label='{}'.format(decomp_label),
                      color=DECOMP_COLORS[decomp_label])
      else:
        # plot the data
        plot_raw_data(ax=axes['raw_data'][ht_name],
                      indep_ax=figures['independent']['raw_data'][ht_name].gca(),
                      xticks=my_agg_times['ticks'],
                      yvalues=my_agg_times[QUANTITY_OF_INTEREST_MIN],
                      linestyle=MIN_LINESTYLE,
                      label='min-{}'.format(decomp_label),
                      color=DECOMP_COLORS[decomp_label])

        plot_raw_data(ax=axes['raw_data'][ht_name],
                      indep_ax=figures['independent']['raw_data'][ht_name].gca(),
                      xticks=my_agg_times['ticks'],
                      yvalues=my_agg_times[QUANTITY_OF_INTEREST_MAX],
                      linestyle=MAX_LINESTYLE,
                      label='max-{}'.format(decomp_label),
                      color=DECOMP_COLORS[decomp_label])

      if show_percent_total:
        if PLOT_ONLY_MIN:
          # plot the data
          plot_raw_data(ax=axes['percent_total'][ht_name],
                        indep_ax=figures['independent']['percent_total'][ht_name].gca(),
                        xticks=my_agg_times['ticks'],
                        yvalues=my_agg_times['min_percent'],
                        linestyle='-',
                        label='{}'.format(decomp_label),
                        color=DECOMP_COLORS[decomp_label])
        else:
          # plot the data
          plot_raw_data(ax=axes['percent_total'][ht_name],
                        indep_ax=figures['independent']['percent_total'][ht_name].gca(),
                        xticks=my_agg_times['ticks'],
                        yvalues=my_agg_times['min_percent'],
                        linestyle=MIN_LINESTYLE,
                        label='min-{}'.format(decomp_label),
                        color=DECOMP_COLORS[decomp_label])

          plot_raw_data(ax=axes['percent_total'][ht_name],
                        indep_ax=figures['independent']['percent_total'][ht_name].gca(),
                        xticks=my_agg_times['ticks'],
                        yvalues=my_agg_times['max_percent'],
                        linestyle=MAX_LINESTYLE,
                        label='max-{}'.format(decomp_label),
                        color=DECOMP_COLORS[decomp_label])
      if show_factor:
        if PLOT_ONLY_MIN:
          # plot the data
          plot_raw_data(ax=axes['flat_mpi_factor'][ht_name],
                        indep_ax=figures['independent']['flat_mpi_factor'][ht_name].gca(),
                        xticks=my_agg_times['ticks'],
                        yvalues=my_agg_times['flat_mpi_factor_min'],
                        linestyle='-',
                        label='{}'.format(decomp_label),
                        color=DECOMP_COLORS[decomp_label])
        else:
          # plot the data
          plot_raw_data(ax=axes['flat_mpi_factor'][ht_name],
                        indep_ax=figures['independent']['flat_mpi_factor'][ht_name].gca(),
                        xticks=my_agg_times['ticks'],
                        yvalues=my_agg_times['flat_mpi_factor_min'],
                        linestyle=MIN_LINESTYLE,
                        label='min-{}'.format(decomp_label),
                        color=DECOMP_COLORS[decomp_label])

          plot_raw_data(ax=axes['flat_mpi_factor'][ht_name],
                        indep_ax=figures['independent']['flat_mpi_factor'][ht_name].gca(),
                        xticks=my_agg_times['ticks'],
                        yvalues=my_agg_times['flat_mpi_factor_max'],
                        linestyle=MAX_LINESTYLE,
                        label='max-{}'.format(decomp_label),
                        color=DECOMP_COLORS[decomp_label])

  # configure the axes for the plotted data
  for row_idx in range(0, len(ht_names)):
    ht_name = ht_names[row_idx]
    # for independent plots (e.g., the subplot plotted separately) we show all axes labels
    ## raw data
    figures['independent']['raw_data'][ht_name].gca().set_ylabel('Runtime (s)')
    figures['independent']['raw_data'][ht_name].gca().set_xlabel('Number of Nodes')
    figures['independent']['raw_data'][ht_name].gca().set_xticks(my_ticks)
    figures['independent']['raw_data'][ht_name].gca().set_xticklabels(my_nodes, rotation=45)
    figures['independent']['raw_data'][ht_name].gca().set_xlim([0.5, my_num_nodes + 1])
    figures['independent']['raw_data'][ht_name].gca().set_title('{}\n(HTs={:.0f})'.format(simple_title, ht_name))
    ## percentages
    if show_percent_total:
      figures['independent']['percent_total'][ht_name].gca().set_ylabel('Percentage of Total Time')
      figures['independent']['percent_total'][ht_name].gca().set_xlabel('Number of Nodes')
      figures['independent']['percent_total'][ht_name].gca().set_xticks(my_ticks)
      figures['independent']['percent_total'][ht_name].gca().set_xticklabels(my_nodes, rotation=45)
      figures['independent']['percent_total'][ht_name].gca().set_xlim([0.5, my_num_nodes + 1])
      figures['independent']['percent_total'][ht_name].gca().set_title('{}\n(HTs={:.0f})'.format(simple_title, ht_name))
      figures['independent']['percent_total'][ht_name].gca().yaxis.set_major_formatter(FormatStrFormatter('%3.0f %%'))
    ## factors
    if show_factor:
      figures['independent']['flat_mpi_factor'][ht_name].gca().set_ylabel('Ratio (smaller is better)')
      figures['independent']['flat_mpi_factor'][ht_name].gca().set_xlabel('Number of Nodes')
      figures['independent']['flat_mpi_factor'][ht_name].gca().set_xticks(my_ticks)
      figures['independent']['flat_mpi_factor'][ht_name].gca().set_xticklabels(my_nodes, rotation=45)
      figures['independent']['flat_mpi_factor'][ht_name].gca().set_xlim([0.5, my_num_nodes + 1])
      figures['independent']['flat_mpi_factor'][ht_name].gca().set_title('{}\n(HTs={:.0f})'.format(simple_title, ht_name))
    axes['raw_data'][ht_name].set_ylabel('Runtime (s)')
    if show_percent_total:
      axes['percent_total'][ht_name].set_ylabel('Percentage of Total Time')
      axes['percent_total'][ht_name].yaxis.set_major_formatter(FormatStrFormatter('%3.0f %%'))
    if show_factor:
      axes['flat_mpi_factor'][ht_name].set_ylabel('Ratio of Runtime to Flat MPI Time')

    # if this is the last row, then display x axes labels
    if row_idx == (len(ht_names) - 1):
      axes['raw_data'][ht_name].set_xlabel("Number of Nodes")
      axes['raw_data'][ht_name].set_xticks(my_ticks)
      axes['raw_data'][ht_name].set_xticklabels(my_nodes, rotation=45)
      axes['raw_data'][ht_name].set_xlim([0.5, my_num_nodes + 1])
      axes['raw_data'][ht_name].set_title('HTs={:.0f}'.format(ht_name))
      if show_percent_total:
        axes['percent_total'][ht_name].set_xlabel("Number of Nodes")
        axes['percent_total'][ht_name].set_xticks(my_ticks)
        axes['percent_total'][ht_name].set_xticklabels(my_nodes, rotation=45)
        axes['percent_total'][ht_name].set_xlim([0.5, my_num_nodes + 1])
        axes['percent_total'][ht_name].set_title('HTs={:.0f}'.format(ht_name))
      if show_factor:
        axes['flat_mpi_factor'][ht_name].set_xlabel("Number of Nodes")
        axes['flat_mpi_factor'][ht_name].set_xticks(my_ticks)
        axes['flat_mpi_factor'][ht_name].set_xticklabels(my_nodes, rotation=45)
        axes['flat_mpi_factor'][ht_name].set_xlim([0.5, my_num_nodes + 1])
        axes['flat_mpi_factor'][ht_name].set_title('HTs={:.0f}'.format(ht_name))

    # if this is the first row, display the full title, e.g., 'Foo \n Ht = {}'
    elif row_idx == 0:
      axes['raw_data'][ht_name].set_title('Raw Data\n(HTs={:.0f})'.format(ht_name))
      if show_percent_total:
        axes['percent_total'][ht_name].set_title('Percentage of Total Time\n(HTs={:.0f})'.format(ht_name))
      if show_factor:
        axes['flat_mpi_factor'][ht_name].set_title('Ratio of Runtime to Flat MPI Time\n(HTs={:.0f})'.format(ht_name))
      # delete the xticks, because we do not want any x axis labels
      axes['raw_data'][ht_name].set_xticks([])
      if show_percent_total:
        axes['percent_total'][ht_name].set_xticks([])
    # otherwise, this is a middle plot, show a truncated title
    else:
      axes['raw_data'][ht_name].set_title('HTs={:.0f}'.format(ht_name))
      if show_percent_total:
        axes['percent_total'][ht_name].set_title('HTs={:.0f}'.format(ht_name))
      if show_factor:
        axes['flat_mpi_factor'][ht_name].set_title('HTs={:.0f}')
      # delete the xticks, because we do not want any x axis labels
      axes['raw_data'][ht_name].set_xticks([])
      if show_percent_total:
        axes['percent_total'][ht_name].set_xticks([])
      if show_factor:
        axes['flat_mpi_factor'][ht_name].set_xticks([])

  # add a suptitle and configure the legend for each figure
  figures['composite'].suptitle(simple_title, fontsize=18)
  # we plot in a deterministic fashion, and so the order is consistent among all
  # axes plotted. This allows a single legend that is compatible with all plots.
  handles, labels = axes['raw_data'][ht_names[0]].get_legend_handles_labels()
  figures['composite'].legend(handles, labels,
                              title="Procs per Node x Cores per Proc",
                              loc='lower center', ncol=ndecomps, bbox_to_anchor=(0.5, 0.0))
  figures['composite'].tight_layout()
  # this must be called after tight layout
  figures['composite'].subplots_adjust(top=0.85, bottom=0.15)

  for column_name in figures['independent']:
    for fig_name, fig in figures['independent'][column_name].items():
      fig.legend(handles, labels,
                 title="Procs per Node x Cores per Proc",
                 loc='lower center', ncol=ndecomps, bbox_to_anchor=(0.5, 0.0))
      # add space since the titles are typically large
      fig.subplots_adjust(bottom=0.20)

  # save the free axis version of the figures
  save_figures(figures,
               filename='{basename}-free-yaxis'.format(basename=simple_fname),
               close_figure=False)

  # if we want consistent axes by column, then enforce that here.
  if HT_CONSISTENT_YAXES:
    enforce_consistent_ylims(figures, axes)

  # save the figures with the axes shared
  save_figures(figures, filename=simple_fname, close_figure=True)


###############################################################################
def load_dataset(dataset_filename,
                 min_num_nodes=MIN_NUM_NODES,
                 max_num_nodes=MAX_NUM_NODES):
  """
  Load a CSV datafile. This assumes the data was parsed from YAML using the parser in this directory

  This loader currently restricts the dataset loaded to a specific range of node counts.
  It also removes problematic data, which are runs that never finished or were know to be excessively buggy.
  E.g., The Elasticity3D matrix, or running MueLu without repartitioning

  The loaded data is verified using pandas index and sorted by the index

  :param dataset_filename: path/filename to load
  :param min_num_nodes:
  :param max_num_nodes:
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
                      ''.format(min_num_nodes=min_num_nodes,
                                max_num_nodes=max_num_nodes)

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
                   driver_df,
                   **kwargs):
  if scaling_study_type == 'weak':
    plot_composite_weak(composite_group=composite_group,
                        my_nodes=my_nodes,
                        my_ticks=my_ticks,
                        numbered_plots_idx=numbered_plots_idx,
                        driver_df=driver_df,
                        kwargs=kwargs)

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
                   driver_df=driver_dataset,
                   show_percent_total=False)

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

  if scaling_study_type == 'weak':
    dataset, driver_dataset = load_dataset(dataset_filename=dataset_filename)
  else:
    dataset, driver_dataset = load_dataset(dataset_filename=dataset_filename,
                                           min_num_nodes=0,
                                           max_num_nodes=64)

  # obtain a list of timer names ordered the aggregate time spent in each
  ordered_timers = get_ordered_timers(dataset=dataset,
                                      rank_by_column_name=QUANTITY_OF_INTEREST)

  #study_type = "muelu_constructor"
  study_type = ''

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
