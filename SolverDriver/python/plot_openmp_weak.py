#!/usr/bin/env python3
"""plotter.py

Usage:
  plotter.py [--dataset=DATASET --study=STUDY_TYPE --scaling=SCALING_TYPE --force_replot --max_nodes=NUM --min_nodes=NUM [--min_only | --max_only]]
  plotter.py (-h | --help)

Options:
  -h --help                Show this screen.
  --dataset=DATASET        Input file [default: all_data.csv]
  --study=STUDY_TYPE       muelu_constructor, muelu_prec, solvers
  --scaling=SCALING_TYPE   Type of analysis, weak/strong [default: strong]
  --force_replot           Force replotting of existing data [default: False]
  --max_nodes=NUM          Fix the number of nodes [default: 100000]
  --min_nodes=NUM          Fix the number of nodes [default: 1]
  --min_only               Plot only the minimum values [default: False]
  --max_only               Plot only the maximum values [default: False]
"""
import matplotlib as mpl
#mpl.use('TkAgg')

from docopt import docopt
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from pathlib import Path
import copy
import ScalingFilenameParser as SFP
# from operator import itemgetter

COMPOSITE_PATH   = 'composites'
INDEPENDENT_PATH = 'composites/indep'

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

PLOT_MIN            = True
PLOT_MAX            = True
SMOOTH_OUTLIERS     = False
HT_CONSISTENT_YAXES = True
ANNOTATE_BEST       = False

HYPER_THREAD_LABEL = 'HT'

# define the colors used for each deomp type
DECOMP_COLORS = {
  '64x1'      : 'xkcd:greyish',
  '32x2'      : 'xkcd:windows blue',
  '16x4'      : 'xkcd:amber',
  '8x8'       : 'xkcd:faded green',
  '4x16'      : 'xkcd:dusty purple',
  'flat_mpi'  : 'xkcd:salmon',
  'best'      : 'xkcd:salmon',
  'worst'     : 'xkcd:greyish'
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


def get_plottable_dataframe(plottable_df, data_group, data_name, driver_groups, compute_strong_terms=False, magic=''):
  """
  Given a dataframe that is designed to be plotted. That is, it should have the appropriate x ticks as a column
  as well as a column to join on.  This is implemented as xticks and num_nodes. The purpose is that we want to handle
  missing data. By joining to this dataframe, we will obtain empty entries if the data is missing. This also guarantees
  that the plotted data all have the same shape.

  :param plottable_df: dataframe with num_nodes as a column
  :param data_group: a datagroup that is suitable to aggregate and plot. For timing/count data, this runtime will sum()
                     all entries grouped by the number of of nodes.
  :param data_name: the group lookup tuple used to obtain this datagroup. We will use this looked tuple to find the
                    related data in the driver_groups data group.
  :param driver_groups: Data grouped identically to the data_group, but this is all groups.
                        TODO: possibly pass in the driver_group directly and avoid the need for the data_name
  :param compute_strong_terms: Compute speedup and efficiency
  :return: plottable_df, populated with quantities of interest aggregated and various measures computed from those
           aggregates.
  """
  DEBUG_plottable_dataframe = False
  DIVIDE_BY_CALLCOUNTS = False
  DIVIDE_BY_NUMSTEPS = True
  TAKE_COLUMNWISE_MINMAX = False

  if DEBUG_plottable_dataframe:
    pd.set_option('display.expand_frame_repr', False)
    print(data_name)
    data_group.to_csv('datagroup-{}.csv'.format(data_name))

  # Use aggregation. This typically will do nothing, but in some cases there are many data points
  # per node count, and we need some way to flatten that data.  This assumes that experiments performed were the same
  # which is how the groupby() logic works  at the highest level. For strong scaling problem_type, problem size (global)
  # , solver and prec options are forced to be the same.
  # The real challenge is how to handle multiple datapoints. For the most part, we do this by dividing by the number
  # of samples taken (numsteps). This assumes that the basic experiment (a step) was the same in all cases.
  timings = data_group.groupby('num_nodes', as_index=False)[[QUANTITY_OF_INTEREST_MIN,
                                                             QUANTITY_OF_INTEREST_MIN_COUNT,
                                                             QUANTITY_OF_INTEREST_MAX,
                                                             QUANTITY_OF_INTEREST_MAX_COUNT,
                                                             QUANTITY_OF_INTEREST_THING,
                                                             QUANTITY_OF_INTEREST_THING_COUNT,
                                                             'numsteps']].sum()

  flat_mpi_timings = data_group.groupby('num_nodes', as_index=False)[['flat_mpi_min',
                                                                      'flat_mpi_max',
                                                                      'flat_mpi_min_count',
                                                                      'flat_mpi_max_count',
                                                                      'flat_mpi_numsteps']].first()
  timings = pd.merge(timings, flat_mpi_timings,
                     how='left',
                     on='num_nodes')

  driver_timings = driver_groups.get_group(data_name).groupby('num_nodes',
                                                               as_index=False)[ [QUANTITY_OF_INTEREST_MIN,
                                                                                 QUANTITY_OF_INTEREST_MIN_COUNT,
                                                                                 QUANTITY_OF_INTEREST_MAX,
                                                                                 QUANTITY_OF_INTEREST_MAX_COUNT,
                                                                                 QUANTITY_OF_INTEREST_THING,
                                                                                 QUANTITY_OF_INTEREST_THING_COUNT,
                                                                                 'numsteps']].sum()

  if DEBUG_plottable_dataframe:
    driver_groups.get_group(data_name).to_csv('driver_timings-{timer}-{d}.csv'.format(
      timer=driver_groups.get_group(data_name)['Timer Name'].unique(),
      d=magic))

  driver_timings = driver_timings[['num_nodes',
                                   QUANTITY_OF_INTEREST_MIN,
                                   QUANTITY_OF_INTEREST_MIN_COUNT,
                                   QUANTITY_OF_INTEREST_MAX,
                                   QUANTITY_OF_INTEREST_MAX_COUNT,
                                   QUANTITY_OF_INTEREST_THING,
                                   QUANTITY_OF_INTEREST_THING_COUNT,
                                   'numsteps']]

  # rename the driver timings to indicate they were from the driver
  driver_timings.rename(columns={QUANTITY_OF_INTEREST_MIN         : 'driver_min',
                                 QUANTITY_OF_INTEREST_MIN_COUNT   : 'driver_min_count',
                                 QUANTITY_OF_INTEREST_MAX         : 'driver_max',
                                 QUANTITY_OF_INTEREST_MAX_COUNT   : 'driver_max_count',
                                 QUANTITY_OF_INTEREST_THING       : 'driver_thing',
                                 QUANTITY_OF_INTEREST_THING_COUNT : 'driver_thing',
                                 'numsteps'                       : 'driver_numsteps'}, inplace=True)

  if DEBUG_plottable_dataframe: print(plottable_df)

  print('Decomp: {magic}:\n\t\tnumsteps:: data: {numsteps} driver: {numsteps_d}\n\t\tcallCounts:: data: {cc} driver: {cc_d}'.format(magic=magic,
                                                                                                                                    numsteps=timings['numsteps'].unique(),
                                                                                                                                    numsteps_d=driver_timings['driver_numsteps'].unique(),
                                                                                                                                    cc=timings[QUANTITY_OF_INTEREST_MIN_COUNT].unique(),
                                                                                                                                    cc_d=driver_timings['driver_min_count'].unique()))

  # merge the kernels timings
  plottable_df = plottable_df.merge(timings, on='num_nodes', how='left')

  if DEBUG_plottable_dataframe: print(plottable_df)

  # merge the driver timings
  plottable_df = plottable_df.merge(driver_timings, on='num_nodes', how='left')

  if DEBUG_plottable_dataframe: print(plottable_df)

  # attempt to deal with the outliers
  # this is risky, because there is no promise there are the same number of data points for every node count
  # Perhaps, normalize by the aggregate call count, which should be consistent for a specific decomp at a specific node
  # count
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

  # divide by the call count.
  # The hope is that the call count will be consistent for a specific decomposition
  # This gets messy for MueLu stuff. The root of the problem is that we are trying to compare different decomposition
  # runs. We also have more data for some points than others. Plotting raw data can also be influenced by this.
  # a total mess.
  if DIVIDE_BY_CALLCOUNTS:
    plottable_df['min_percent_t'] = \
                              (plottable_df[QUANTITY_OF_INTEREST_MIN] / plottable_df[QUANTITY_OF_INTEREST_MIN_COUNT]) \
                            / (plottable_df['driver_min'] / plottable_df['driver_min_count']) * 100.00

    plottable_df['max_percent_t'] =\
                              (plottable_df[QUANTITY_OF_INTEREST_MAX] / plottable_df[QUANTITY_OF_INTEREST_MAX_COUNT]) \
                            / (plottable_df['driver_max'] / plottable_df['driver_max_count']) * 100.00
  elif DIVIDE_BY_NUMSTEPS:
    plottable_df['min_percent_t'] = \
                              (plottable_df[QUANTITY_OF_INTEREST_MIN] / plottable_df['numsteps']) \
                            / (plottable_df['driver_min'] / plottable_df['driver_numsteps']) * 100.00

    plottable_df['max_percent_t'] =\
                              (plottable_df[QUANTITY_OF_INTEREST_MAX] / plottable_df['numsteps']) \
                            / (plottable_df['driver_max'] / plottable_df['driver_numsteps']) * 100.00
  else:
    plottable_df['min_percent_t'] = \
                              (plottable_df[QUANTITY_OF_INTEREST_MIN]) \
                            / (plottable_df['driver_min']) * 100.00

    plottable_df['max_percent_t'] =\
                              (plottable_df[QUANTITY_OF_INTEREST_MAX]) \
                            / (plottable_df['driver_max']) * 100.00

  if TAKE_COLUMNWISE_MINMAX:
    # this is a bit iffy. It depends on what you choose to divide by above. Should the 'min' in the data
    # be hardcoded to be the minOverProcs values, or does 'min' mean the conceptual minimum of the data.
    # I take the latter as the definition since this data is a summary of experiments.
    plottable_df['min_percent'] = plottable_df[['max_percent_t', 'min_percent_t']].min(axis=1)
    plottable_df['max_percent'] = plottable_df[['max_percent_t', 'min_percent_t']].max(axis=1)

    # drop the temporary columns
    plottable_df = plottable_df.drop('min_percent_t', 1)
    plottable_df = plottable_df.drop('max_percent_t', 1)
  else:
    plottable_df.rename(columns={'min_percent_t': 'min_percent',
                                 'max_percent_t': 'max_percent'}, inplace=True)

  plottable_df['min_percent'] = plottable_df['min_percent'].round(1)
  plottable_df['max_percent'] = plottable_df['max_percent'].round(1)

  # similar approach as above.
  if DIVIDE_BY_CALLCOUNTS:
    plottable_df['flat_mpi_factor_min_t'] = \
                              (plottable_df[QUANTITY_OF_INTEREST_MIN] / plottable_df[QUANTITY_OF_INTEREST_MIN_COUNT]) \
                            / (plottable_df['flat_mpi_min'] / plottable_df['flat_mpi_min_count'])

    plottable_df['flat_mpi_factor_max_t'] = \
                              (plottable_df[QUANTITY_OF_INTEREST_MAX] / plottable_df[QUANTITY_OF_INTEREST_MAX_COUNT]) \
                            / (plottable_df['flat_mpi_max'] / plottable_df['flat_mpi_max_count'])
  elif DIVIDE_BY_NUMSTEPS:
    plottable_df['flat_mpi_factor_min_t'] = \
                              (plottable_df[QUANTITY_OF_INTEREST_MIN] / plottable_df['numsteps']) \
                            / (plottable_df['flat_mpi_min'] / plottable_df['flat_mpi_numsteps'])

    plottable_df['flat_mpi_factor_max_t'] = \
                              (plottable_df[QUANTITY_OF_INTEREST_MAX] / plottable_df['numsteps']) \
                            / (plottable_df['flat_mpi_max'] / plottable_df['flat_mpi_numsteps'])
  else:
    plottable_df['flat_mpi_factor_min_t'] = \
                              (plottable_df[QUANTITY_OF_INTEREST_MIN]) \
                            / (plottable_df['flat_mpi_min'])

    plottable_df['flat_mpi_factor_max_t'] = \
                              (plottable_df[QUANTITY_OF_INTEREST_MAX]) \
                            / (plottable_df['flat_mpi_max'])

  if TAKE_COLUMNWISE_MINMAX:
    plottable_df['flat_mpi_factor_min'] = plottable_df[['flat_mpi_factor_max_t', 'flat_mpi_factor_min_t']].min(axis=1)
    plottable_df['flat_mpi_factor_max'] = plottable_df[['flat_mpi_factor_max_t', 'flat_mpi_factor_min_t']].max(axis=1)

    plottable_df = plottable_df.drop('flat_mpi_factor_min_t', 1)
    plottable_df = plottable_df.drop('flat_mpi_factor_max_t', 1)

  else:
    plottable_df.rename(columns={'flat_mpi_factor_min_t': 'flat_mpi_factor_min',
                                 'flat_mpi_factor_max_t': 'flat_mpi_factor_max'}, inplace=True)
  if DEBUG_plottable_dataframe:
    plottable_df.to_csv('plottable-{timer}-{d}.csv'.format(timer=data_group['Timer Name'].unique(),
                                                           d=magic))

  if compute_strong_terms is False:
    if DEBUG_plottable_dataframe: pd.set_option('display.expand_frame_repr', True)
    return plottable_df

  # compute Speedup and Efficiency. Use Flat MPI as the baseline
  np1_min_row = plottable_df.loc[plottable_df.loc[(plottable_df['num_nodes'] == 1), ['flat_mpi_min']].idxmin()]
  np1_max_row = plottable_df.loc[plottable_df.loc[(plottable_df['num_nodes'] == 1), ['flat_mpi_max']].idxmax()]

  # normalize by call count
  if DIVIDE_BY_CALLCOUNTS:
    np1_min = (np1_min_row['flat_mpi_min'] / np1_min_row['flat_mpi_min_count']).values
    np1_max = (np1_max_row['flat_mpi_min'] / np1_min_row['flat_mpi_min_count']).values
  elif DIVIDE_BY_NUMSTEPS:
    np1_min = (np1_min_row['flat_mpi_min'] / np1_min_row['flat_mpi_numsteps']).values
    np1_max = (np1_max_row['flat_mpi_min'] / np1_min_row['flat_mpi_numsteps']).values
  else:
    np1_min = (np1_min_row['flat_mpi_min']).values
    np1_max = (np1_max_row['flat_mpi_min']).values

  if DEBUG_plottable_dataframe: print(np1_min, np1_max)

  if DIVIDE_BY_CALLCOUNTS:
    plottable_df['speedup_min_t'] = np1_min /\
                              (plottable_df[QUANTITY_OF_INTEREST_MIN] / plottable_df[QUANTITY_OF_INTEREST_MIN_COUNT])

    plottable_df['speedup_max_t'] = np1_max /\
                              (plottable_df[QUANTITY_OF_INTEREST_MAX] / plottable_df[QUANTITY_OF_INTEREST_MAX_COUNT])
  elif DIVIDE_BY_NUMSTEPS:
    plottable_df['speedup_min_t'] = np1_min /\
                              (plottable_df[QUANTITY_OF_INTEREST_MIN] / plottable_df['numsteps'])

    plottable_df['speedup_max_t'] = np1_max /\
                              (plottable_df[QUANTITY_OF_INTEREST_MAX] / plottable_df['numsteps'])
  else:
    plottable_df['speedup_min_t'] = np1_min /\
                              (plottable_df[QUANTITY_OF_INTEREST_MIN])

    plottable_df['speedup_max_t'] = np1_max /\
                              (plottable_df[QUANTITY_OF_INTEREST_MAX])

  if DEBUG_plottable_dataframe: print(plottable_df)
  # this will ensure that the 'max' as plotted is the max value, not necessarily the maxOverProcs value
  # In most cases, for speedup, the max is the minOverProcs, and the min is the maxOverProcs.
  # but because we had to choose which value to consider base line (np1) value, this can cause a mess if
  # trying to stick to the TeuchosTimer notion of min and max.  Here, we report the minimum values and the maximum
  # values we observe.
  plottable_df['speedup_min'] = plottable_df[['speedup_max_t', 'speedup_min_t']].min(axis=1)
  plottable_df['speedup_max'] = plottable_df[['speedup_max_t', 'speedup_min_t']].max(axis=1)

  # drop the temporary columns
  plottable_df = plottable_df.drop('speedup_min_t', 1)
  plottable_df = plottable_df.drop('speedup_max_t', 1)

  # efficiency
  plottable_df['efficiency_min_t'] = plottable_df['speedup_min'] * 100.0 / plottable_df['num_nodes']
  plottable_df['efficiency_max_t'] = plottable_df['speedup_max'] * 100.0 / plottable_df['num_nodes']

  plottable_df['efficiency_min'] = plottable_df[['efficiency_max_t', 'efficiency_min_t']].min(axis=1)
  plottable_df['efficiency_max'] = plottable_df[['efficiency_max_t', 'efficiency_min_t']].max(axis=1)

  plottable_df = plottable_df.drop('efficiency_min_t', 1)
  plottable_df = plottable_df.drop('efficiency_max_t', 1)

  if DEBUG_plottable_dataframe:
    plottable_df.to_csv('plottable-{}.csv'.format(data_name))
    pd.set_option('display.expand_frame_repr', True)

  return plottable_df


def enforce_consistent_ylims(figures, axes):
  """
  Given the matplotlib figure and axes handles, determine global y limits

  :param figures: dict of dict of dict of figures
                  figures should be the independent figures, constructed as
                  independent:column_name:row_name
  :param axes: Handles for all axes that are part of the composite plot. e.g., axes[column_name][row_name]
  :return: Nothing
  """
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


def save_figures(figures,
                 filename,
                 close_figure=False,
                 composite=True,
                 independent=True,
                 independent_names=None):
  """
  Helper to save figures. Plots the composite figure, as well as the independent figures.

  :param figures: dict of figures. composite => figure,
                                   independent => column_name => row_name => figure
  :param filename: the base filename to use
  :param close_figure: boolean, whether to close the figures after saving
  :param composite: boolean, save the composite figure
  :param independent: boolean, save the independent figures
  :param independent_names: list or string, save specific row names only.
                            This is mainly used when we destructively modify the figures to annotate and collapse
                            them into a 'best' figure.
  :return:
  """

  if composite:
    try:
      fullpath = '{path}/{fname}'.format(path=COMPOSITE_PATH, fname=filename)

      figures['composite'].savefig('{}.png'.format(fullpath),
                                   format='png',
                                   dpi=180)
      print('Wrote: {}.png'.format(filename))
    except:
      print('FAILED writing {}.png'.format(filename))
      raise

    if close_figure:
      plt.close(figures['composite'])

  if independent:
    if independent_names is None:
      fig_names = None
    elif isinstance(independent_names, list):
      fig_names = independent_names
    else:
      fig_names = [independent_names]

    for column_name in figures['independent']:

      if fig_names is None:
        fig_names = figures['independent'][column_name].keys()

      for ht_name in fig_names:
        fig_filename = '{base}-{col}-{ht}.png'.format(base=filename, col=column_name, ht=ht_name)

        try:
          fullpath = '{path}/{fname}'.format(path=INDEPENDENT_PATH, fname=fig_filename)

          figures['independent'][column_name][ht_name].savefig(fullpath,
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


def plot_raw_data(ax, indep_ax, xvalues, yvalues, linestyle, label, color,
                  **kwargs):

  # plot the data
  ax.plot(xvalues,
          yvalues,
          label=label,
          color=color,
          linestyle=linestyle,
          **kwargs)

  if indep_ax:
    indep_ax.plot(xvalues,
                  yvalues,
                  label=label,
                  color=color,
                  linestyle=linestyle,
                  **kwargs)


###############################################################################
def need_to_replot(simple_fname, subplot_names, ht_names):
  """
  determine if a set of files exist.

  This turned out to be error prone. The issues seem to arrise when you work in a directory that is actually a symbolic
  Link from another filesystem, and then use relative paths. The trick appears to be to resolve the path. Simply
  testing if the file is a file can fail in odd ways.

  :param simple_fname: Base name to test
  :param subplot_names: The names of subplots (e.g., column names)
  :param ht_names:  The row labels for the plots (e.g., the ht names)
  :return: Nothing
  """
  need_to_replot_ = False
  filepath = '{path}/{fname}.png'.format(path=COMPOSITE_PATH,
                                         fname=simple_fname)
  my_file = Path(filepath)
  try:
    temp = my_file.resolve()
  except FileNotFoundError or RuntimeError:
    print("File {}.png does not exist triggering replot".format(filepath))
    need_to_replot_ = True

  for column_name in subplot_names:
    for ht_name in ht_names:
      fig_filename = '{base}-{col}-{ht}.png'.format(base=simple_fname, col=column_name, ht=ht_name)
      filepath = '{path}/{fname}'.format(path=INDEPENDENT_PATH,
                                         fname=fig_filename)
      my_file = Path(filepath)
      try:
        temp = my_file.resolve()
      except FileNotFoundError or RuntimeError:
        print("File {} does not exist triggering replot".format(filepath))
        need_to_replot_ = True

  return need_to_replot_


###############################################################################
def add_flat_mpi_data(composite_group,
                      allow_baseline_override=False):
  """
  Given a group of data that hopefully contains *both* Serial and OpenMP datapoints, create a 'flat_mpi' set of columns.

  This means we must aggregate this flat MPI information. We sum() all values.
  If no serial data is available, then we look for single threaded OpenMP data.

  :param composite_group: Grouped data that makes sense to compare. It should make sense to aggregate data in this group
                          by decomposition type when grouped by the number of nodes used. E.g., sum all flat_mpi data
  :param allow_baseline_override: Allow the use of OpenMP threaded data as a baseline. E.g., use 32x2.
  :return: composite_group with flat mpi columns added
  """
  # TODO: convey information to the plotter that Serial or OpenMP data was used.

  # figure out the flat MPI time
  # ideally, we would want to query the Serial execution space here... but that is kinda complicated, we likely
  # need to add an argument that is a serial execution space dataframe, as the groupby logic expects the execution
  # space to be the same
  serial_data = composite_group[composite_group['execspace_name'] == 'Serial']
  if serial_data.empty:
    # try to use OpenMP instead
    groupby_cols = ['procs_per_node', 'cores_per_proc', 'threads_per_core']
    try:
      # first try for the 64x1x1, this would need to be adjusted for other architectures
      flat_mpi_df = composite_group.groupby(groupby_cols).get_group((64,1,1))
    except KeyError as e:
      if allow_baseline_override:
        # if that fails, and we allow it, then downgrade to 32x2x1
        flat_mpi_df = composite_group.groupby(groupby_cols).get_group((32, 2, 1))
      else:
        raise e
  else:
    flat_mpi_df = serial_data
    print('Using Serial run data for flatMPI')

  flat_mpi_df.rename(columns={QUANTITY_OF_INTEREST_MIN: 'flat_mpi_min',
                              QUANTITY_OF_INTEREST_MAX: 'flat_mpi_max',
                              QUANTITY_OF_INTEREST_MIN_COUNT : 'flat_mpi_min_count',
                              QUANTITY_OF_INTEREST_MAX_COUNT : 'flat_mpi_max_count',
                              'numsteps'                     : 'flat_mpi_numsteps'}, inplace=True)

  # make sure this is one value per num_nodes.
  # this is a real pest.
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


def update_decomp_dataframe(decomp_dataframe,
                            my_agg_times,
                            ht_group,
                            ht_name,
                            procs_per_node,
                            cores_per_proc):
  keys = SFP.getIndexColumns(execspace_name='OpenMP')
  keys.remove('nodes')
  keys.remove('numsteps')
  keys.remove('Timer Name')
  keys.remove('omp_num_threads')
  keys.remove('execspace_attributes')
  keys.remove('timestamp')
  keys.remove('problem_type')
  keys.remove('problem_bs')
  keys.remove('problem_nx')
  keys.remove('problem_ny')
  keys.remove('problem_nz')
  keys.remove('prec_name')
  keys.remove('prec_attributes')
  keys.remove('solver_name')
  keys.remove('solver_attributes')

  tmp_df = my_agg_times.merge(ht_group[keys], on='num_nodes', how='left')
  del tmp_df['ticks']
  del tmp_df['flat_mpi_numsteps']
  del tmp_df['flat_mpi_min']
  del tmp_df['flat_mpi_min_count']
  del tmp_df['flat_mpi_max']
  del tmp_df['flat_mpi_max_count']
  del tmp_df['driver_thing']
  del tmp_df['driver_numsteps']
  del tmp_df['driver_min']
  del tmp_df['driver_min_count']
  del tmp_df['driver_max']
  del tmp_df['driver_max_count']
  del tmp_df['numsteps']
  # del tmp_df['num_nodes']
  # keys.remove('num_nodes')

  del tmp_df[QUANTITY_OF_INTEREST_MIN_COUNT]
  del tmp_df[QUANTITY_OF_INTEREST_MAX_COUNT]
  del tmp_df[QUANTITY_OF_INTEREST_THING_COUNT]
  del tmp_df[QUANTITY_OF_INTEREST_THING]

  tmp_df = tmp_df.rename(columns={'threads_per_core': HYPER_THREAD_LABEL})
  keys[keys.index('threads_per_core')] = HYPER_THREAD_LABEL

  tmp_df = tmp_df.rename(columns={'num_nodes': 'nodes'})
  keys[keys.index('num_nodes')] = 'nodes'

  tmp_df = tmp_df.rename(columns={'num_mpi_procs': 'MPI Procs'})
  keys[keys.index('num_mpi_procs')] = 'MPI Procs'

  tmp_df['procs_per_node'] = procs_per_node
  tmp_df['cores_per_proc'] = cores_per_proc
  tmp_df[HYPER_THREAD_LABEL] = ht_name

  tmp_df['MPI Procs'] = tmp_df['nodes'] * procs_per_node

  tmp_df[[QUANTITY_OF_INTEREST_MIN,
          QUANTITY_OF_INTEREST_MAX,
          'flat_mpi_factor_min',
          'flat_mpi_factor_max',
          'min_percent',
          'max_percent']] = \
    tmp_df[[QUANTITY_OF_INTEREST_MIN,
            QUANTITY_OF_INTEREST_MAX,
            'flat_mpi_factor_min',
            'flat_mpi_factor_max',
            'min_percent',
            'max_percent']].apply(lambda x: pd.Series.round(x, 2))

  tmp_df = tmp_df.drop_duplicates()
  tmp_df[['MPI Procs',
          'nodes',

          'procs_per_node',
          'cores_per_proc',
          HYPER_THREAD_LABEL]] = tmp_df[['MPI Procs',
                            'nodes',
                            'procs_per_node',
                            'cores_per_proc',
                            HYPER_THREAD_LABEL]].astype(np.int32)

  tmp_df = tmp_df.set_index(keys=keys,
                            drop=True,
                            verify_integrity=True)

  # tmp_df.to_csv('tmp.csv', index=True)
  decomp_dataframe = pd.concat([decomp_dataframe, tmp_df])
  
  decomp_dataframe = decomp_dataframe.set_index(keys=keys,
                                                drop=True,
                                                verify_integrity=True)
  return decomp_dataframe

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

  total_df = pd.DataFrame()

  print(kwargs.keys())
  if kwargs is not None:
    if 'show_percent_total' in kwargs.keys():
      print('in the keys')
      show_percent_total = kwargs['show_percent_total']
    if 'show_factor' in kwargs.keys():
      show_factor = kwargs['show_factor']

  # determine the flat MPI time
  composite_group = add_flat_mpi_data(composite_group,allow_baseline_override=True)

  decomp_groups = composite_group.groupby(['procs_per_node', 'cores_per_proc', 'execspace_name'])
  driver_decomp_groups = driver_df.groupby(['procs_per_node', 'cores_per_proc', 'execspace_name'])

  # determine the components that should be in the filename
  show_solver_name = (composite_group['solver_name'].nunique() == 1)
  show_solver_attributes = (composite_group['solver_attributes'].nunique() == 1)
  show_prec_name = (composite_group['prec_name'].nunique() == 1)
  show_prec_attributes = (composite_group['prec_attributes'].nunique() == 1)

  # for a specific group of data, compute the scaling terms, which are things like min/max
  # this also flattens the timer creating a 'fat_timer_name'
  # essentially, this function computes data that is relevant to a group, but not the whole
  my_tokens = SFP.getTokensFromDataFrameGroupBy(composite_group)
  simple_fname = SFP.getScalingFilename(my_tokens,
                                        weak=True,
                                        composite=True,
                                        solver_name=show_solver_name,
                                        solver_attributes=show_solver_attributes,
                                        prec_name=show_prec_name,
                                        prec_attributes=show_prec_attributes)
  simple_title = SFP.getScalingTitle(my_tokens, weak=True, composite=True)

  # if numbered, then prepend the number to the filename
  # and increment the count.
  if numbered_plots_idx >= 0:
    simple_fname = '{}-{}'.format(numbered_plots_idx, simple_fname)

  # if we apply any type of smoothing, then note this in the filename
  if SMOOTH_OUTLIERS:
    simple_fname = '{}-outliers-smoothed'.format(simple_fname)

  if (PLOT_MAX is True) and (PLOT_MIN is False):
    simple_fname = '{}-max-only'.format(simple_fname)
  elif (PLOT_MAX is False) and (PLOT_MIN is True):
    simple_fname = '{}-min-only'.format(simple_fname)

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
    execution_space = decomp_group_name[2]

    # label this decomp
    if execution_space == 'OpenMP':
      decomp_label = "{procs_per_node}x{cores_per_proc}".format(procs_per_node=procs_per_node,
                                                                cores_per_proc=cores_per_proc)
    elif execution_space == 'Serial':
      decomp_label = 'flat_mpi'

    # iterate over HTs
    ht_groups = decomp_group.groupby('threads_per_core')
    driver_ht_groups = driver_decomp_groups.get_group(decomp_group_name).groupby('threads_per_core')

    for ht_name in ht_names:

      plot_row = ht_name

      if execution_space == 'Serial':
        ht_name = ht_names[0]

      ht_group = ht_groups.get_group(ht_name)
      magic_str = '-{decomp_label}x{threads_per_core}'.format(decomp_label=decomp_label,
                                                              threads_per_core=ht_name)

      my_agg_times = pd.DataFrame(columns=['num_nodes', 'ticks'], data=np.column_stack((my_nodes, my_ticks)))

      my_agg_times = get_plottable_dataframe(my_agg_times, ht_group, ht_name, driver_ht_groups, magic=magic_str)

      # this assumes that idx=0 is an SpMV aggregate figure, which appears to be worthless.
      if numbered_plots_idx > 0 and plot_row == ht_name:
        total_df = update_decomp_dataframe(total_df,
                                           my_agg_times,
                                           ht_group,
                                           ht_name,
                                           procs_per_node,
                                           cores_per_proc)

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

      # plot the data
      if PLOT_MAX:
        # plot the max if requested
        plot_raw_data(ax=axes['raw_data'][plot_row],
                      indep_ax=figures['independent']['raw_data'][plot_row].gca(),
                      xvalues=my_agg_times['ticks'],
                      yvalues=my_agg_times[QUANTITY_OF_INTEREST_MAX],
                      linestyle=MAX_LINESTYLE,
                      label='max-{}'.format(decomp_label),
                      color=DECOMP_COLORS[decomp_label])

      if PLOT_MIN:
        # plot the max if requested
        plot_raw_data(ax=axes['raw_data'][plot_row],
                      indep_ax=figures['independent']['raw_data'][plot_row].gca(),
                      xvalues=my_agg_times['ticks'],
                      yvalues=my_agg_times[QUANTITY_OF_INTEREST_MIN],
                      linestyle=MIN_LINESTYLE,
                      label='min-{}'.format(decomp_label),
                      color=DECOMP_COLORS[decomp_label])

      if show_percent_total:
        # plot the data
        if PLOT_MAX:
          plot_raw_data(ax=axes['percent_total'][plot_row],
                        indep_ax=figures['independent']['percent_total'][plot_row].gca(),
                        xvalues=my_agg_times['ticks'],
                        yvalues=my_agg_times['max_percent'],
                        linestyle=MAX_LINESTYLE,
                        label='max-{}'.format(decomp_label),
                        color=DECOMP_COLORS[decomp_label])

        if PLOT_MIN:
          plot_raw_data(ax=axes['percent_total'][plot_row],
                        indep_ax=figures['independent']['percent_total'][plot_row].gca(),
                        xvalues=my_agg_times['ticks'],
                        yvalues=my_agg_times['min_percent'],
                        linestyle=MIN_LINESTYLE,
                        label='min-{}'.format(decomp_label),
                        color=DECOMP_COLORS[decomp_label])
      if show_factor:
        # plot the data
        if PLOT_MAX:
          plot_raw_data(ax=axes['flat_mpi_factor'][plot_row],
                        indep_ax=figures['independent']['flat_mpi_factor'][plot_row].gca(),
                        xvalues=my_agg_times['ticks'],
                        yvalues=my_agg_times['flat_mpi_factor_max'],
                        linestyle=MAX_LINESTYLE,
                        label='max-{}'.format(decomp_label),
                        color=DECOMP_COLORS[decomp_label])

        if PLOT_MIN:
          plot_raw_data(ax=axes['flat_mpi_factor'][plot_row],
                        indep_ax=figures['independent']['flat_mpi_factor'][plot_row].gca(),
                        xvalues=my_agg_times['ticks'],
                        yvalues=my_agg_times['flat_mpi_factor_min'],
                        linestyle=MIN_LINESTYLE,
                        label='min-{}'.format(decomp_label),
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
    figures['independent']['raw_data'][ht_name].gca().set_xlim([0.5, my_num_nodes + 0.5])
    figures['independent']['raw_data'][ht_name].gca().set_title('{}\n({HT_LABEL}={HT_NUM:.0f})'.format(simple_title,
                                                                                                       HT_LABEL=HYPER_THREAD_LABEL,
                                                                                                       HT_NUM=ht_name))
    ## percentages
    if show_percent_total:
      figures['independent']['percent_total'][ht_name].gca().set_ylabel('Percentage of Total Time')
      figures['independent']['percent_total'][ht_name].gca().set_xlabel('Number of Nodes')
      figures['independent']['percent_total'][ht_name].gca().set_xticks(my_ticks)
      figures['independent']['percent_total'][ht_name].gca().set_xticklabels(my_nodes, rotation=45)
      figures['independent']['percent_total'][ht_name].gca().set_xlim([0.5, my_num_nodes + 0.5])
      figures['independent']['percent_total'][ht_name].gca().set_title('{}\n({HT_LABEL}={HT_NUM:.0f})'.format(simple_title,
                                                                                                              HT_LABEL=HYPER_THREAD_LABEL,
                                                                                                              HT_NUM=ht_name))
      figures['independent']['percent_total'][ht_name].gca().yaxis.set_major_formatter(FormatStrFormatter('%3.0f %%'))

    ## factors
    if show_factor:
      figures['independent']['flat_mpi_factor'][ht_name].gca().set_ylabel('Ratio (smaller is better)')
      figures['independent']['flat_mpi_factor'][ht_name].gca().set_xlabel('Number of Nodes')
      figures['independent']['flat_mpi_factor'][ht_name].gca().set_xticks(my_ticks)
      figures['independent']['flat_mpi_factor'][ht_name].gca().set_xticklabels(my_nodes, rotation=45)
      figures['independent']['flat_mpi_factor'][ht_name].gca().set_xlim([0.5, my_num_nodes + 0.5])
      figures['independent']['flat_mpi_factor'][ht_name].gca().set_title('{}\n({HT_LABEL}={HT_NUM:.0f})'.format(simple_title,
                                                                                                                HT_LABEL=HYPER_THREAD_LABEL,
                                                                                                                HT_NUM=ht_name))

    axes['raw_data'][ht_name].set_ylabel('Runtime (s)')
    axes['raw_data'][ht_name].set_xlim([0.5, my_num_nodes + 0.5])

    if show_percent_total:
      axes['percent_total'][ht_name].set_ylabel('Percentage of Total Time')
      axes['percent_total'][ht_name].yaxis.set_major_formatter(FormatStrFormatter('%3.0f %%'))
      axes['percent_total'][ht_name].set_xlim([0.5, my_num_nodes + 0.5])

    if show_factor:
      axes['flat_mpi_factor'][ht_name].set_ylabel('Ratio of Runtime to Flat MPI Time')
      axes['flat_mpi_factor'][ht_name].set_xlim([0.5, my_num_nodes + 0.5])

    # if this is the last row, then display x axes labels
    if row_idx == (len(ht_names) - 1):
      axes['raw_data'][ht_name].set_xlabel("Number of Nodes")
      axes['raw_data'][ht_name].set_xticks(my_ticks)
      axes['raw_data'][ht_name].set_xticklabels(my_nodes, rotation=45)
      axes['raw_data'][ht_name].set_title('{HT_LABEL}={HT_NUM:.0f}'.format(HT_LABEL=HYPER_THREAD_LABEL,
                                                                           HT_NUM=ht_name))

      if show_percent_total:
        axes['percent_total'][ht_name].set_xlabel("Number of Nodes")
        axes['percent_total'][ht_name].set_xticks(my_ticks)
        axes['percent_total'][ht_name].set_xticklabels(my_nodes, rotation=45)
        axes['percent_total'][ht_name].set_title('{HT_LABEL}={HT_NUM:.0f}'.format(HT_LABEL=HYPER_THREAD_LABEL,
                                                                                  HT_NUM=ht_name))

      if show_factor:
        axes['flat_mpi_factor'][ht_name].set_xlabel("Number of Nodes")
        axes['flat_mpi_factor'][ht_name].set_xticks(my_ticks)
        axes['flat_mpi_factor'][ht_name].set_xticklabels(my_nodes, rotation=45)
        axes['flat_mpi_factor'][ht_name].set_title('{HT_LABEL}={HT_NUM:.0f}'.format(HT_LABEL=HYPER_THREAD_LABEL,
                                                                                    HT_NUM=ht_name))

    # if this is the first row, display the full title, e.g., 'Foo \n Ht = {}'
    elif row_idx == 0:
      axes['raw_data'][ht_name].set_title('Raw Data\n({HT_LABEL}={HT_NUM:.0f})'.format(HT_LABEL=HYPER_THREAD_LABEL,
                                                                                       HT_NUM=ht_name))
      axes['raw_data'][ht_name].set_xticks([])

      if show_percent_total:
        axes['percent_total'][ht_name].set_title('Percentage of Total Time\n({HT_LABEL}={HT_NUM:.0f})'.format(HT_LABEL=HYPER_THREAD_LABEL,
                                                                                                              HT_NUM=ht_name))
        axes['percent_total'][ht_name].set_xticks([])

      if show_factor:
        axes['flat_mpi_factor'][ht_name].set_title('Ratio of Runtime to Flat MPI Time\n({HT_LABEL}={HT_NUM:.0f})'.format(HT_LABEL=HYPER_THREAD_LABEL,
                                                                                                                         HT_NUM=ht_name))
        axes['flat_mpi_factor'][ht_name].set_xticks([])

    # otherwise, this is a middle plot, show a truncated title
    else:
      axes['raw_data'][ht_name].set_title('{HT_LABEL}={HT_NUM:.0f}'.format(HT_LABEL=HYPER_THREAD_LABEL,
                                                                           HT_NUM=ht_name))
      axes['raw_data'][ht_name].set_xticks([])

      if show_percent_total:
        axes['percent_total'][ht_name].set_title('{HT_LABEL}={HT_NUM:.0f}'.format(HT_LABEL=HYPER_THREAD_LABEL,
                                                                                  HT_NUM=ht_name))
        axes['percent_total'][ht_name].set_xticks([])

      if show_factor:
        axes['flat_mpi_factor'][ht_name].set_title('{HT_LABEL}={HT_NUM:.0f}'.format(HT_LABEL=HYPER_THREAD_LABEL,
                                                                                    HT_NUM=ht_name))
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

  # add legends
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

  total_df.to_csv('{fname}.csv'.format(fname=simple_fname))
  total_df.to_latex('{fname}.tex'.format(fname=simple_fname),
                    longtable=True)

  # add labels for the best
  if ANNOTATE_BEST:
    for column_name in figures['independent']:
      for fig_name, fig in figures['independent'][column_name].items():
        annotate_best(ax=fig.gca(),
                      ax_id=fig_name,
                      objective='min')

    # save the free axis version of the figures
    save_figures(figures,
                 filename='{basename}-free-yaxis-best'.format(basename=simple_fname),
                 close_figure=False)

  # if we want consistent axes by column, then enforce that here.
  if HT_CONSISTENT_YAXES:
    enforce_consistent_ylims(figures, axes)

  # save the figures with the axes shared
  save_figures(figures, filename=simple_fname, close_figure=True)

  if ANNOTATE_BEST:
    for column_name in figures['independent']:
      if column_name == 'speedup' or column_name == 'efficiency':
        annotate_best_column(figures=figures['independent'][column_name],
                             axes_name_to_destroy=ht_names[0],
                             objective='min')

    # save the figures with the axes shared
    save_figures(figures,
                 filename='{fname}-overall'.format(fname=simple_fname),
                 composite=False,
                 independent=True,
                 independent_names=ht_names[0])


def axes_to_df(ax, ax_id):
  import re

  label_pattern = re.compile('(min|max)-(\d+x\d+|flat_mpi)')

  # most axes have min and max data points
  # track each separately
  all_data = pd.DataFrame(columns=['type', 'ax_id', 'x', 'label', 'y'])

  for line in ax.lines:
    label = line.get_label()

    m = label_pattern.match(label)
    if m:
      if m.group(1) == 'min':
        data_type = 'min'
        label = label.replace('min-', '', 1)
      elif m.group(1) == 'max':
        data_type = 'max'
        label = label.replace('max-', '', 1)
    else:
      continue

    df = pd.DataFrame(columns=['type', 'ax_id', 'x', 'label', 'y'])
    df['x'] = line.get_xdata()
    df['y'] = line.get_ydata()
    df['label'] = label
    df['type']  = data_type
    df['ax_id'] = ax_id
    all_data = pd.concat([all_data, df])

  all_data = all_data.set_index(['type', 'ax_id', 'x', 'label'], drop=False)
  print(all_data)
  return all_data


def annotate_best_column(figures, axes_name_to_destroy, objective='min'):
  import re

  df = pd.DataFrame()
  for fig_name, fig in figures.items():
    tmp_df = axes_to_df(fig.gca(), fig_name)
    df = pd.concat([df, tmp_df])

  df['ax_id'] = df['ax_id'].astype(np.int32)

  if objective == 'min':
    try:
      best_overall = df.loc[df.groupby(['type','x'])['y'].idxmin()].groupby('type')
      worst_overall = df.loc[df.groupby(['type','x'])['y'].idxmax()].groupby('type')
    except:
      print('Had a problem annotating the best. Skipping this axes')
      return
  elif objective == 'max':
    try:
      best_overall = df.loc[df.groupby(['type','x'])['y'].idxmax()].groupby('type')
      worst_overall = df.loc[df.groupby(['type', 'x'])['y'].idxmin()].groupby('type')
    except:
      print('Had a problem annotating the best. Skipping this axes')
      return
  else:
    raise ValueError('objective must be min or max.')

  ax = figures[axes_name_to_destroy].gca()
  # wipe out the lines
  ax.lines = []
  ax.texts = []
  # destroy the legend
  for l in figures[axes_name_to_destroy].legends:
    l.remove()

  current_title = ax.get_title()
  current_title = re.sub(r"\(?HTs=\d\)?", "", current_title)
  ax.set_title(current_title)

  linestyles = dict()
  linestyles['min'] = MIN_LINESTYLE
  linestyles['max'] = MAX_LINESTYLE

  # plot the new data
  for plot_type, df in best_overall:
    # if plot_type != objective:
    #   continue

    if plot_type == objective:
      ax.plot(df['x'], df['y'], linestyle='solid', label='best-{}'.format(plot_type), color=DECOMP_COLORS['best'])
    else:
      ax.plot(df['x'], df['y'], linestyle='dotted', label='best-{}'.format(plot_type), color=DECOMP_COLORS['best'])
      # do not add data labels
      continue

    # label the times
    data_labels = ['{decomp}x{ht}'.format(decomp=row['label'], ht=row['ax_id']) if row['label'] != 'flat_mpi' else '{decomp}'.format(decomp=row['label']) for index, row in df.iterrows()]

    x_offset = 40
    y_offset = 20
    for label, x, y in zip(data_labels, df['x'], df['y']):
      ax.annotate(
        label,
        xy=(x, y), xytext=(x_offset, y_offset),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

  # plot the new data
  for plot_type, df in worst_overall:
    # if plot_type == objective:
    #   continue

    if plot_type != objective:
      ax.plot(df['x'], df['y'], linestyle='solid', label='worst-{}'.format(plot_type), color=DECOMP_COLORS['worst'])
    else:
      ax.plot(df['x'], df['y'], linestyle='dotted', label='worst-{}'.format(plot_type), color=DECOMP_COLORS['worst'])
      # do not add data labels
      continue

    #ax.plot(df['x'], df['y'], linestyle=linestyles[plot_type], label='worst-{}'.format(plot_type))

    # label the times
    data_labels = ['{decomp}x{ht}'.format(decomp=row['label'], ht=row['ax_id']) if row['label'] != 'flat_mpi' else '{decomp}'.format(decomp=row['label']) for index, row in df.iterrows()]

    x_offset = 40
    y_offset = 20
    for label, x, y in zip(data_labels, df['x'], df['y']):
      ax.annotate(
        label,
        xy=(x, y), xytext=(x_offset, y_offset),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

  handles, labels = ax.get_legend_handles_labels()
  ax.legend(handles, labels)
  ax.relim()
  ax.autoscale()
  figures[axes_name_to_destroy].tight_layout()


def annotate_best(ax, ax_id, objective='min'):
  df = axes_to_df(ax, ax_id)
  PLOT_BEST_MIN_ONLY=True
  df['ax_id'] = df['ax_id'].astype(np.int32)

  if objective == 'min':
    try:
      best_overall = df.loc[df.groupby(['type','x'])['y'].idxmin()].groupby('type')
      best_by_id   = df.loc[df.groupby(['type','ax_id', 'x'])['y'].idxmin()].groupby(['type','ax_id'])
    except:
      print('Had a problem annotating the best. Skipping this axes')
      return
  elif objective == 'max':
    try:
      best_overall = df.loc[df.groupby(['type','x'])['y'].idxmax()].groupby('type')
      best_by_id   = df.loc[df.groupby(['type','ax_id', 'x'])['y'].idxmax()].groupby(['type','ax_id'])
    except:
      print('Had a problem annotating the best. Skipping this axes')
      return
  else:
    raise ValueError('objective must be min or max.')

  linestyles = dict()
  linestyles['min'] = 'o'
  linestyles['max'] = '+'

  # plot the new data
  for plot_type, df in best_overall:
    if plot_type != objective:
      continue

    # label the times
    data_labels = ['{decomp}x{ht}'.format(decomp=row['label'], ht=row['ax_id']) if row['label'] != 'flat_mpi' else '{decomp}'.format(decomp=row['label']) for index, row in df.iterrows()]

    x_offset = 40
    y_offset = 20
    for label, x, y in zip(data_labels, df['x'], df['y']):
      ax.annotate(
        label,
        xy=(x, y), xytext=(x_offset, y_offset),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))


def flatten_axes_to_best(ax, all_data, objective='min'):
  PLOT_BEST_MIN_ONLY=True
  all_data['ax_id'] = all_data['ax_id'].astype(np.int32)
  print(all_data.dtypes)

  if objective == 'min':
    best_overall = all_data.loc[all_data.groupby(['type','x'])['y'].idxmin()].groupby('type')
    best_by_id   = all_data.loc[all_data.groupby(['type','ax_id', 'x'])['y'].idxmin()].groupby(['type','ax_id'])
  elif objective == 'max':
    best_overall = all_data.loc[all_data.groupby(['type','x'])['y'].idxmax()].groupby('type')
    best_by_id   = all_data.loc[all_data.groupby(['type','ax_id', 'x'])['y'].idxmax()].groupby(['type','ax_id'])
  else:
    raise ValueError('objective must be min or max.')

  # delete the lines on the ax
  ax.lines = []

  linestyles = dict()
  linestyles['min'] = 'o'
  linestyles['max'] = '+'

  # plot the new data
  for plot_type, df in best_overall:
    if PLOT_BEST_MIN_ONLY and plot_type != 'min':
      continue

    print('plot_type', plot_type)
    label = 'best-{}'.format(plot_type)
    kwargs = dict(marker=linestyles[plot_type],
                  markersize=8,
                  fillstyle='none')

    for index, row in df.iterrows():

      plot_raw_data(ax=ax,
                    indep_ax=None,
                    xvalues=row['x'],
                    yvalues=row['y'],
                    linestyle=None,
                    label=label,
                    color=DECOMP_COLORS[row['label']],
                    **kwargs)
      label = None

    # label the times
    data_labels = ['{decomp}x{ht}'.format(decomp=row['label'], ht=row['ax_id']) for index, row in df.iterrows()]
    bad_value = False
    x_offset = 40 if plot_type == 'min' else -20
    y_offset = 20 if plot_type == 'min' else 20
    for label, x, y in zip(data_labels, df['x'], df['y']):
      ax.annotate(
        label,
        xy=(x, y), xytext=(x_offset, y_offset),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))


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
  show_factor = True

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

  decomp_groups = composite_group.groupby(['procs_per_node', 'cores_per_proc', 'execspace_name'])
  driver_decomp_groups = driver_df.groupby(['procs_per_node', 'cores_per_proc', 'execspace_name'])

  # determine the components that should be in the filename
  show_solver_name = (composite_group['solver_name'].nunique() == 1)
  show_solver_attributes = (composite_group['solver_attributes'].nunique() == 1)
  show_prec_name = (composite_group['prec_name'].nunique() == 1)
  show_prec_attributes = (composite_group['prec_attributes'].nunique() == 1)

  # for a specific group of data, compute the scaling terms, which are things like min/max
  # this also flattens the timer creating a 'fat_timer_name'
  # essentially, this function computes data that is relevant to a group, but not the whole
  my_tokens = SFP.getTokensFromDataFrameGroupBy(composite_group)
  simple_fname = SFP.getScalingFilename(my_tokens, strong=True, composite=True,
                                        solver_name=show_solver_name,
                                        solver_attributes=show_solver_attributes,
                                        prec_name=show_prec_name,
                                        prec_attributes=show_prec_attributes)
  simple_title = SFP.getScalingTitle(my_tokens, strong=True, composite=True)

  # if numbered, then prepend the number to the filename
  # and increment the count.
  if numbered_plots_idx >= 0:
    simple_fname = '{}-{}'.format(numbered_plots_idx, simple_fname)

  # if we apply any type of smoothing, then note this in the filename
  if SMOOTH_OUTLIERS:
    simple_fname = '{}-outliers-smoothed'.format(simple_fname)

  if (PLOT_MAX is True) and (PLOT_MIN is False):
    simple_fname = '{}-max-only'.format(simple_fname)
  elif (PLOT_MAX is False) and (PLOT_MIN is True):
    simple_fname = '{}-min-only'.format(simple_fname)

  # the number of HT combos we have
  ht_names = composite_group['threads_per_core'].sort_values(ascending=True).unique()
  ndecomps = len(decomp_groups)

  my_num_nodes = my_nodes.size

  SPEEDUP_YMAX = (my_nodes[-1] * 2) * 0.75
  SPEEDUP_YMIN = 0.75

  fig_size = 5.5
  fig_size_height_inflation = 1.0
  fig_size_width_inflation  = 1.0

  subplot_names = ['raw_data', 'speedup', 'efficiency']

  if show_speed_up is False:
    subplot_names.remove('speedup')
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
    execution_space = decomp_group_name[2]

    # label this decomp
    if execution_space == 'OpenMP':
      decomp_label = "{procs_per_node}x{cores_per_proc}".format(procs_per_node=procs_per_node,
                                                                cores_per_proc=cores_per_proc)
    elif execution_space == 'Serial':
      decomp_label = 'flat_mpi'

    # iterate over HTs
    ht_groups = decomp_group.groupby('threads_per_core')
    driver_ht_groups = driver_decomp_groups.get_group(decomp_group_name).groupby('threads_per_core')

    for ht_name in ht_names:

      plot_row = ht_name
      
      if execution_space == 'Serial':
        ht_name = ht_names[0]

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

      if PLOT_MAX:
        # plot the data
        plot_raw_data(ax=axes['raw_data'][plot_row],
                      indep_ax=figures['independent']['raw_data'][plot_row].gca(),
                      xvalues=my_agg_times['ticks'],
                      yvalues=my_agg_times[QUANTITY_OF_INTEREST_MIN],
                      linestyle='-',
                      label='{}'.format(decomp_label),
                      color=DECOMP_COLORS[decomp_label])
      if PLOT_MIN:
        plot_raw_data(ax=axes['raw_data'][plot_row],
                      indep_ax=figures['independent']['raw_data'][plot_row].gca(),
                      xvalues=my_agg_times['ticks'],
                      yvalues=my_agg_times[QUANTITY_OF_INTEREST_MIN],
                      linestyle=MIN_LINESTYLE,
                      label='min-{}'.format(decomp_label),
                      color=DECOMP_COLORS[decomp_label])

      if show_speed_up:
        # plot a straight line
        plot_raw_data(ax=axes['speedup'][plot_row],
                      indep_ax=figures['independent']['speedup'][plot_row].gca(),
                      xvalues=my_agg_times['num_nodes'],
                      yvalues=my_agg_times['num_nodes'],
                      linestyle='-',
                      label=None,
                      color='black')

        if PLOT_MAX:
          # plot the data
          plot_raw_data(ax=axes['speedup'][plot_row],
                        indep_ax=figures['independent']['speedup'][plot_row].gca(),
                        xvalues=my_agg_times['num_nodes'],
                        yvalues=my_agg_times['speedup_max'],
                        linestyle=MAX_LINESTYLE,
                        label='max-{}'.format(decomp_label),
                        color=DECOMP_COLORS[decomp_label])
        if PLOT_MIN:
          plot_raw_data(ax=axes['speedup'][plot_row],
                        indep_ax=figures['independent']['speedup'][plot_row].gca(),
                        xvalues=my_agg_times['num_nodes'],
                        yvalues=my_agg_times['speedup_min'],
                        linestyle=MIN_LINESTYLE,
                        label='min-{}'.format(decomp_label),
                        color=DECOMP_COLORS[decomp_label])

      if show_efficiency:
        if PLOT_MAX:
          # plot the data
          plot_raw_data(ax=axes['efficiency'][plot_row],
                        indep_ax=figures['independent']['efficiency'][plot_row].gca(),
                        xvalues=my_agg_times['ticks'],
                        yvalues=my_agg_times['efficiency_max'],
                        linestyle=MAX_LINESTYLE,
                        label='max-{}'.format(decomp_label),
                        color=DECOMP_COLORS[decomp_label])
        if PLOT_MIN:
          plot_raw_data(ax=axes['efficiency'][plot_row],
                        indep_ax=figures['independent']['efficiency'][plot_row].gca(),
                        xvalues=my_agg_times['ticks'],
                        yvalues=my_agg_times['efficiency_min'],
                        linestyle=MIN_LINESTYLE,
                        label='min-{}'.format(decomp_label),
                        color=DECOMP_COLORS[decomp_label])

      if show_percent_total:
        if PLOT_MAX:
          # plot the data
          plot_raw_data(ax=axes['percent_total'][plot_row],
                        indep_ax=figures['independent']['percent_total'][plot_row].gca(),
                        xvalues=my_agg_times['ticks'],
                        yvalues=my_agg_times['max_percent'],
                        linestyle=MAX_LINESTYLE,
                        label='max-{}'.format(decomp_label),
                        color=DECOMP_COLORS[decomp_label])
        if PLOT_MIN:
          plot_raw_data(ax=axes['percent_total'][plot_row],
                        indep_ax=figures['independent']['percent_total'][plot_row].gca(),
                        xvalues=my_agg_times['ticks'],
                        yvalues=my_agg_times['min_percent'],
                        linestyle=MIN_LINESTYLE,
                        label='min-{}'.format(decomp_label),
                        color=DECOMP_COLORS[decomp_label])

      if show_factor:
        if PLOT_MAX:
          # plot the data
          plot_raw_data(ax=axes['flat_mpi_factor'][plot_row],
                        indep_ax=figures['independent']['flat_mpi_factor'][plot_row].gca(),
                        xvalues=my_agg_times['ticks'],
                        yvalues=my_agg_times['flat_mpi_factor_max'],
                        linestyle=MAX_LINESTYLE,
                        label='max-{}'.format(decomp_label),
                        color=DECOMP_COLORS[decomp_label])
        if PLOT_MIN:
          plot_raw_data(ax=axes['flat_mpi_factor'][plot_row],
                        indep_ax=figures['independent']['flat_mpi_factor'][plot_row].gca(),
                        xvalues=my_agg_times['ticks'],
                        yvalues=my_agg_times['flat_mpi_factor_min'],
                        linestyle=MIN_LINESTYLE,
                        label='min-{}'.format(decomp_label),
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
    figures['independent']['raw_data'][ht_name].gca().set_xlim([0.5, my_num_nodes + 0.5])
    figures['independent']['raw_data'][ht_name].gca().set_title('{}\n({HT_LABEL}={HT_NUM:.0f})'.format(simple_title,
                                                                                                       HT_LABEL=HYPER_THREAD_LABEL,
                                                                                                       HT_NUM=ht_name))
    ## speedup
    if show_speed_up:
      figures['independent']['speedup'][ht_name].gca().set_ylabel('Speedup Relative to Flat MPI')
      figures['independent']['speedup'][ht_name].gca().set_xlabel('Number of Nodes')
      figures['independent']['speedup'][ht_name].gca().set_title('{}\n({HT_LABEL}={HT_NUM:.0f})'.format(simple_title,
                                                                                                        HT_LABEL=HYPER_THREAD_LABEL,
                                                                                                        HT_NUM=ht_name))
      figures['independent']['speedup'][ht_name].gca().set_yscale('log', basey=2)
      figures['independent']['speedup'][ht_name].gca().set_xscale('log', basex=2)

      figures['independent']['speedup'][ht_name].gca().set_yticks(my_nodes)
      figures['independent']['speedup'][ht_name].gca().set_yticklabels(my_nodes)

      figures['independent']['speedup'][ht_name].gca().set_xticks(my_nodes)
      figures['independent']['speedup'][ht_name].gca().set_xticklabels(my_nodes, rotation=45)

      figures['independent']['speedup'][ht_name].gca().set_xlim([SPEEDUP_YMIN, SPEEDUP_YMAX])
      figures['independent']['speedup'][ht_name].gca().set_ylim([SPEEDUP_YMIN, SPEEDUP_YMAX])

      figures['independent']['speedup'][ht_name].gca().grid(True)
    ## efficiency
    if show_efficiency:
      figures['independent']['efficiency'][ht_name].gca().set_ylabel('Efficiency')
      figures['independent']['efficiency'][ht_name].gca().set_xlabel('Number of Nodes')
      figures['independent']['efficiency'][ht_name].gca().set_xticks(my_ticks)
      figures['independent']['efficiency'][ht_name].gca().set_xticklabels(my_nodes, rotation=45)
      figures['independent']['efficiency'][ht_name].gca().set_xlim([0.5, my_num_nodes + 0.5])
      figures['independent']['efficiency'][ht_name].gca().set_title('{}\n({HT_LABEL}={HT_NUM:.0f})'.format(simple_title,
                                                                                                           HT_LABEL=HYPER_THREAD_LABEL,
                                                                                                           HT_NUM=ht_name))
      figures['independent']['efficiency'][ht_name].gca().yaxis.set_major_formatter(FormatStrFormatter('%3.0f %%'))
    ## percentages
    if show_percent_total:
      figures['independent']['percent_total'][ht_name].gca().set_ylabel('Percentage of Total Time')
      figures['independent']['percent_total'][ht_name].gca().set_xlabel('Number of Nodes')
      figures['independent']['percent_total'][ht_name].gca().set_xticks(my_ticks)
      figures['independent']['percent_total'][ht_name].gca().set_xticklabels(my_nodes, rotation=45)
      figures['independent']['percent_total'][ht_name].gca().set_xlim([0.5, my_num_nodes + 0.5])
      figures['independent']['percent_total'][ht_name].gca().set_title('{}\n({HT_LABEL}={HT_NUM:.0f})'.format(simple_title,
                                                                                                              HT_LABEL=HYPER_THREAD_LABEL,
                                                                                                              HT_NUM=ht_name))
      figures['independent']['percent_total'][ht_name].gca().yaxis.set_major_formatter(FormatStrFormatter('%3.0f %%'))
    ## factors
    if show_factor:
      figures['independent']['flat_mpi_factor'][ht_name].gca().set_ylabel('Ratio (smaller is better)')
      figures['independent']['flat_mpi_factor'][ht_name].gca().set_xlabel('Number of Nodes')
      figures['independent']['flat_mpi_factor'][ht_name].gca().set_xticks(my_ticks)
      figures['independent']['flat_mpi_factor'][ht_name].gca().set_xticklabels(my_nodes, rotation=45)
      figures['independent']['flat_mpi_factor'][ht_name].gca().set_xlim([0.5, my_num_nodes + 0.5])
      figures['independent']['flat_mpi_factor'][ht_name].gca().set_title('{}\n({HT_LABEL}={HT_NUM:.0f})'.format(simple_title,
                                                                                                                HT_LABEL=HYPER_THREAD_LABEL,
                                                                                                                HT_NUM=ht_name))

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
      axes['raw_data'][ht_name].set_xlim([0.5, my_num_nodes + 0.5])
      axes['raw_data'][ht_name].set_title('{HT_LABEL}={HT_NUM:.0f}'.format(HT_LABEL=HYPER_THREAD_LABEL,
                                                                           HT_NUM=ht_name))

      if show_speed_up:
        axes['speedup'][ht_name].set_xlabel("Number of Nodes")
        axes['speedup'][ht_name].set_title('{HT_LABEL}={HT_NUM:.0f}'.format(HT_LABEL=HYPER_THREAD_LABEL,
                                                                            HT_NUM=ht_name))

        axes['speedup'][ht_name].set_yscale('log', basey=2)
        axes['speedup'][ht_name].set_xscale('log', basex=2)

        axes['speedup'][ht_name].set_yticks(my_nodes)
        axes['speedup'][ht_name].set_yticklabels(my_nodes)

        axes['speedup'][ht_name].set_xticks(my_nodes)
        axes['speedup'][ht_name].set_xticklabels(my_nodes, rotation=45)

        axes['speedup'][ht_name].set_xlim([SPEEDUP_YMIN, SPEEDUP_YMAX])
        axes['speedup'][ht_name].set_ylim([SPEEDUP_YMIN, SPEEDUP_YMAX])

        axes['speedup'][ht_name].grid(True)

      if show_efficiency:
        axes['efficiency'][ht_name].set_xlabel("Number of Nodes")
        axes['efficiency'][ht_name].set_xticks(my_ticks)
        axes['efficiency'][ht_name].set_xticklabels(my_nodes, rotation=45)
        axes['efficiency'][ht_name].set_xlim([0.5, my_num_nodes + 0.5])
        axes['efficiency'][ht_name].set_title('{HT_LABEL}={HT_NUM:.0f}'.format(HT_LABEL=HYPER_THREAD_LABEL,
                                                                               HT_NUM=ht_name))

      if show_percent_total:
        axes['percent_total'][ht_name].set_xlabel("Number of Nodes")
        axes['percent_total'][ht_name].set_xticks(my_ticks)
        axes['percent_total'][ht_name].set_xticklabels(my_nodes, rotation=45)
        axes['percent_total'][ht_name].set_xlim([0.5, my_num_nodes + 0.5])
        axes['percent_total'][ht_name].set_title('{HT_LABEL}={HT_NUM:.0f}'.format(HT_LABEL=HYPER_THREAD_LABEL,
                                                                                  HT_NUM=ht_name))
      if show_factor:
        axes['flat_mpi_factor'][ht_name].set_xlabel("Number of Nodes")
        axes['flat_mpi_factor'][ht_name].set_xticks(my_ticks)
        axes['flat_mpi_factor'][ht_name].set_xticklabels(my_nodes, rotation=45)
        axes['flat_mpi_factor'][ht_name].set_xlim([0.5, my_num_nodes + 0.5])
        axes['flat_mpi_factor'][ht_name].set_title('{HT_LABEL}={HT_NUM:.0f}'.format(HT_LABEL=HYPER_THREAD_LABEL,
                                                                                    HT_NUM=ht_name))

    # if this is the first row, display the full title, e.g., 'Foo \n Ht = {}'
    elif row_idx == 0:
      axes['raw_data'][ht_name].set_title('Raw Data\n({HT_LABEL}={HT_NUM:.0f})'.format(HT_LABEL=HYPER_THREAD_LABEL,
                                                                                       HT_NUM=ht_name))
      # delete the xticks, because we do not want any x axis labels
      axes['raw_data'][ht_name].set_xticks([])
      if show_speed_up:
        axes['speedup'][ht_name].set_title('Speed Up\n({HT_LABEL}={HT_NUM:.0f})'.format(HT_LABEL=HYPER_THREAD_LABEL,
                                                                                        HT_NUM=ht_name))

        axes['speedup'][ht_name].set_yscale('log', basey=2)
        axes['speedup'][ht_name].set_xscale('log', basex=2)

        axes['speedup'][ht_name].set_yticks(my_nodes)
        axes['speedup'][ht_name].set_yticklabels(my_nodes)

        axes['speedup'][ht_name].set_xticks(my_nodes)
        axes['speedup'][ht_name].set_xticklabels(my_nodes, rotation=45)

        axes['speedup'][ht_name].grid(True)

        # force a redraw
        axes['speedup'][ht_name].set_xticklabels([])
        for tic in axes['speedup'][ht_name].xaxis.get_major_ticks():
          tic.tick1On = tic.tick2On = False

        axes['speedup'][ht_name].set_xlim([SPEEDUP_YMIN, SPEEDUP_YMAX])
        axes['speedup'][ht_name].set_ylim([SPEEDUP_YMIN, SPEEDUP_YMAX])

      if show_efficiency:
        axes['efficiency'][ht_name].set_title('Efficiency\n({HT_LABEL}={HT_NUM:.0f})'.format(HT_LABEL=HYPER_THREAD_LABEL,
                                                                                             HT_NUM=ht_name))
        axes['efficiency'][ht_name].set_xticks([])
      if show_percent_total:
        axes['percent_total'][ht_name].set_title('Percentage of Total Time\n({HT_LABEL}={HT_NUM:.0f})'.format(HT_LABEL=HYPER_THREAD_LABEL,
                                                                                                              HT_NUM=ht_name))
        axes['percent_total'][ht_name].set_xticks([])
      if show_factor:
        axes['flat_mpi_factor'][ht_name].set_title('Ratio of Runtime to Flat MPI Time\n({HT_LABEL}={HT_NUM:.0f})'.format(HT_LABEL=HYPER_THREAD_LABEL,
                                                                                                                         HT_NUM=ht_name))
        axes['flat_mpi_factor'][ht_name].set_xticks([])

    else:
      # otherwise, this is a middle plot, show a truncated title
      axes['raw_data'][ht_name].set_title('{HT_LABEL}={HT_NUM:.0f}'.format(HT_LABEL=HYPER_THREAD_LABEL,
                                                                           HT_NUM=ht_name))
      # delete the xticks, because we do not want any x axis labels
      axes['raw_data'][ht_name].set_xticks([])

      if show_speed_up:
        axes['speedup'][ht_name].set_title('{HT_LABEL}={HT_NUM:.0f}'.format(HT_LABEL=HYPER_THREAD_LABEL,
                                                                            HT_NUM=ht_name))

        axes['speedup'][ht_name].set_yscale('log', basey=2)
        axes['speedup'][ht_name].set_xscale('log', basex=2)

        axes['speedup'][ht_name].set_yticks(my_nodes)
        axes['speedup'][ht_name].set_yticklabels(my_nodes)

        axes['speedup'][ht_name].set_xticks(my_nodes)
        axes['speedup'][ht_name].set_xticklabels(my_nodes, rotation=45)

        axes['speedup'][ht_name].set_xlim([SPEEDUP_YMIN, SPEEDUP_YMAX])
        axes['speedup'][ht_name].set_ylim([SPEEDUP_YMIN, SPEEDUP_YMAX])

        axes['speedup'][ht_name].grid(True)

        # force a redraw
        axes['speedup'][ht_name].set_xticks([])
      if show_efficiency:
        axes['efficiency'][ht_name].set_title('{HT_LABEL}={HT_NUM:.0f}'.format(HT_LABEL=HYPER_THREAD_LABEL,
                                                                               HT_NUM=ht_name))
        axes['efficiency'][ht_name].set_xticks([])
      if show_percent_total:
        axes['percent_total'][ht_name].set_title('{HT_LABEL}={HT_NUM:.0f}'.format(HT_LABEL=HYPER_THREAD_LABEL,
                                                                                  HT_NUM=ht_name))
        axes['percent_total'][ht_name].set_xticks([])
      if show_factor:
        axes['flat_mpi_factor'][ht_name].set_title('{HT_LABEL}={HT_NUM:.0f}'.format(HT_LABEL=HYPER_THREAD_LABEL,
                                                                                    HT_NUM=ht_name))
        axes['flat_mpi_factor'][ht_name].set_xticks([])

  # add a suptitle and configure the legend for each figure
  figures['composite'].suptitle(simple_title, fontsize=18)
  # we plot in a deterministic fashion, and so the order is consistent among all
  # axes plotted. This allows a single legend that is compatible with all plots.
  handles, labels = axes['raw_data'][ht_names[0]].get_legend_handles_labels()
  figures['composite'].legend(handles, labels,
                              title="Procs per Node x Cores per Proc",
                              loc='lower center',
                              ncol=ndecomps,
                              bbox_to_anchor=(0.5, 0.0))
  figures['composite'].tight_layout()
  # this must be called after tight layout
  figures['composite'].subplots_adjust(top=0.85, bottom=0.15)

  # add legends
  for column_name in figures['independent']:
    for fig_name, fig in figures['independent'][column_name].items():
      #handles, labels = fig.gca().get_legend_handles_labels()
      fig.legend(handles, labels,
                 title="Procs per Node x Cores per Proc",
                 loc='lower center', ncol=ndecomps, bbox_to_anchor=(0.5, 0.0))
      # add space since the titles are typically large
      fig.subplots_adjust(bottom=0.20)

  # save the free axis version of the figures
  save_figures(figures,
               filename='{basename}-free-yaxis'.format(basename=simple_fname),
               close_figure=False)

  for column_name in figures['independent']:
    for fig_name, fig in figures['independent'][column_name].items():
      if column_name == 'speedup' or column_name == 'efficiency':
        annotate_best(ax=fig.gca(),
                      ax_id=fig_name,
                      objective='max')
      else:
        annotate_best(ax=fig.gca(),
                      ax_id=fig_name,
                      objective='min')

  # save the free axis version of the figures
  save_figures(figures,
               filename='{basename}-free-yaxis-best'.format(basename=simple_fname),
               close_figure=False)

  # if we want consistent axes by column, then enforce that here.
  if HT_CONSISTENT_YAXES:
    enforce_consistent_ylims(figures, axes)

  # save the figures with the axes shared
  save_figures(figures, filename=simple_fname)

  for column_name in figures['independent']:
    if column_name == 'speedup' or column_name == 'efficiency':
      annotate_best_column(figures=figures['independent'][column_name],
                           axes_name_to_destroy=ht_names[0],
                           objective='max')
    else:
      annotate_best_column(figures=figures['independent'][column_name],
                           axes_name_to_destroy=ht_names[0],
                           objective='min')

  # save the figures with the axes shared
  save_figures(figures,
               filename='{fname}-overall'.format(fname=simple_fname),
               composite=False,
               independent=True,
               independent_names=ht_names[0])


###############################################################################
def load_dataset(dataset_filename,
                 min_num_nodes=1,
                 max_num_nodes=1000000):
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

  # integral_columns = SFP.getIndexColumns(execspace_name='OpenMP')
  # non_integral_names = ['Timer Name',
  #                          'problem_type',
  #                          'solver_name',
  #                          'solver_attributes',
  #                          'prec_name',
  #                          'prec_attributes',
  #                          'execspace_name',
  #                          'execspace_attributes']
  # integral_columns = list(set(integral_columns).difference(set(non_integral_names)))
  # dataset[integral_columns] = dataset[integral_columns].astype(np.int32)

  # set the index, verify it, and sort
  dataset = dataset.set_index(keys=SFP.getIndexColumns(execspace_name='OpenMP'),
                              drop=False, verify_integrity=True)
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

  dataset = dataset.fillna(value='None')

  # sort
  # dataset.sort_values(inplace=True,
  #                     by=SFP.getIndexColumns(execspace_name='OpenMP'))
  dataset = dataset.sort_index()
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
  dataset = dataset.set_index(keys=SFP.getIndexColumns(execspace_name='OpenMP'),
                              drop=False,
                              verify_integrity=True)

  driver_dataset = driver_dataset.set_index(keys=SFP.getIndexColumns(execspace_name='OpenMP'),
                                            drop=False,
                                            verify_integrity=True)
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
  # be very careful, there will be duplicate decomp groups
  spmv_groupby_columns.remove('execspace_name')

  spmv_only_data = dataset[dataset['Timer Name'].str.match(timer_name_re_str)]
  spmv_only_data.loc[:, 'Timer Name'] = timer_name_rename

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
  import os as os
  if not os.path.exists(COMPOSITE_PATH):
    os.makedirs(COMPOSITE_PATH)
  if not os.path.exists(INDEPENDENT_PATH):
    os.makedirs(INDEPENDENT_PATH)

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
  # be very careful, there will be duplicate decomp groups
  omp_groupby_columns.remove('execspace_name')

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
def main():

  sanity_check()

  # Process input
  _arg_options = docopt(__doc__)

  dataset_filename  = _arg_options['--dataset']
  study_type        = _arg_options['--study']
  max_num_nodes          = _arg_options['--max_nodes']
  min_num_nodes          = _arg_options['--min_nodes']
  scaling_study_type      = _arg_options['--scaling']

  global FORCE_REPLOT
  FORCE_REPLOT           = _arg_options['--force_replot']

  if _arg_options['--min_only']:
    global PLOT_MAX
    PLOT_MAX = False

  if _arg_options['--max_only']:
    global PLOT_MIN
    PLOT_MIN = False

  print('study: {study}\nscaling_type: {scaling}\ndataset: {data}'.format(study=study_type,
                                                                          scaling=scaling_study_type,
                                                                          data=dataset_filename))
  print('Max Nodes: {max}\tMin Nodes: {min}'.format(max=max_num_nodes, min=min_num_nodes))

  if scaling_study_type == 'weak':
    dataset, driver_dataset = load_dataset(dataset_filename=dataset_filename,
                                           min_num_nodes=min_num_nodes,
                                           max_num_nodes=max_num_nodes)
  else:
    dataset, driver_dataset = load_dataset(dataset_filename=dataset_filename,
                                           min_num_nodes=0,
                                           max_num_nodes=64)

  ordered_timers = []

  if study_type == 'muelu_constructor':
    total_time_key = '3 - Constructing Preconditioner'
    restriction_tokens = {'solver_name' : 'Constructor',
                          'solver_attributes' : '-Only',
                          'prec_name' : 'MueLu',
                          'prec_attributes' : '-repartition'}

    # obtain a list of timer names ordered the aggregate time spent in each
    ordered_timers = get_ordered_timers(dataset=dataset,
                                        rank_by_column_name=QUANTITY_OF_INTEREST)

  elif study_type == 'muelu_prec':
    total_time_key = '5 - Solve'
    restriction_tokens = {'solver_name' : 'CG',
                          'solver_attributes' : '',
                          'prec_name' : 'MueLu',
                          'prec_attributes' : '-repartition'}

    # obtain a list of timer names ordered the aggregate time spent in each
    ordered_timers = get_ordered_timers(dataset=dataset,
                                        rank_by_column_name=QUANTITY_OF_INTEREST)
  elif study_type == 'solvers':
    total_time_key = '5 - Solve'
    restriction_tokens = {'prec_name' : 'None'}
  else:
    raise ValueError('unknown study_type ({})'.format(study_type))

  plot_dataset(dataset=dataset,
               driver_dataset=driver_dataset,
               ordered_timers=ordered_timers,
               total_time_key=total_time_key,
               scaling_type=scaling_study_type,
               restriction_tokens=restriction_tokens)


###############################################################################
if __name__ == '__main__':
  main()
