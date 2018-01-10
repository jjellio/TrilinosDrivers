#!/usr/bin/env python3
"""plotter.py

Usage:
  plotter.py --study=STUDY_TYPE
            [--scaling=SCALING_TYPE]
            [--dataset=DATASET]
            [--baseline=DATASET]
            [--no_baseline_comparison_shading]
            [--baseline_linestyle=STYLE]
            [--legend]
            [--force_replot]
            [--max_nodes=NUM]
            [--min_nodes=NUM]
            [--min_procs_per_node=NUM]
            [--max_procs_per_node=NUM]
            [--min_only | --max_only]
            [--ymin=NUM]
            [--ymax=NUM]
            [--normalize_y]
            [--annotate_filenames]
            [--plot=FIGURES]
            [--plot_titles=TITLES]
            [--expected_baseline_speedup=NUM]
            [--restrict_timer_labels=LABEL_RESTRICTION]
            [--sort_timer_labels=TIMING]
            [--number_plots=BOOL]
            [--verbose=N]
            [--average=<averaging>]
            [--quiet]
            [--img_format=FORMAT]
            [--img_dpi=NUM]
            [--timer_name_remapping=TIMER_MAPPING_FILE]


  plotter.py (-h | --help)

Options:
  -h --help                Show this screen.
  --dataset=DATASET        Input file [default: all_data.csv]
  --baseline=DATASET       Input file that should be used to form baseline comparisons
  --baseline_linestyle=STYLE  Override the default line style [default: default]
  --no_baseline_comparison_shading  Shade the dataset that is compared to the baseline [default: False]
  --study=STUDY_TYPE       muelu_constructor, muelu_prec, solvers
  --scaling=SCALING_TYPE   Type of analysis, weak/strong [default: strong]
  --force_replot           Force replotting of existing data [default: False]
  --legend                 Plot a legend [default: False]
  --max_nodes=NUM          Fix the number of nodes [default: 100000]
  --min_nodes=NUM          Fix the number of nodes [default: 1]
  --min_procs_per_node=NUM  Restrict the number processes per node [default: 4]
  --max_procs_per_node=NUM  Restrict the number processes per node [default: 64]
  --min_only               Plot only the minimum values [default: False]
  --max_only               Plot only the maximum values [default: False]
  --ymin=NUM|FIGURE=NUM,...  Restrict the number processes per node [default: -1.0]
  --ymax=NUM|FIGURE=NUM,...  Restrict the number processes per node [default: -1.0]
  --normalize_y            Normalize the y axis between [0,1] [default: False]
  --annotate_filenames     Add data filenames to figures [default: False]
  --plot=FIGURE[,FIGURE]   Plot specific figures [default: raw_data]
  --plot_titles=TITLES     Plot titles [default: none]
  --expected_baseline_speedup=NUM   draw a line on bl_speedup at this y intercept [default : 0.0]
  --restrict_timer_labels=LABEL_RESTRICTION   Plot timer labels matching REGEX [default : '']
  --sort_timer_labels=TIMING  sort the timer labels by a specific timing [default: None]
  --number_plots=BOOL  Number the plots [default: True]
  --verbose=N  Print details as execution goes [default: 1]
  --average=<averaging>   Average the times using callcounts, numsteps, or none [default: none]
  --quiet   Be quiet [default: False]
  --img_format=FORMAT  Format of the output images [default: pdf]
  --img_dpi=NUM     DPI of images [default: 150]
  --timer_name_remapping=TIMER_MAPPING_FILE  List of timer names that should be considered the same

Arguments:

      STYLE: default - line style is the same, no change
             other - Any valid matplotlib style dotted, solid, dashed

      FIGURES: raw_data : scatter plot of raw data
               percent_total : attempt to plot a figure of this timer as a percentage of at total time
               flat_mpi_factor : the ratio of the raw_data to the flat MPI data
               bl_speedup : plot the speedup of the raw_data over the baseline
               bl_perc_diff    : relative difference of baseline and data (bl-data)/bl *100
               composite       : generate a single figure with all requested figures

      LABEL_RESTRICTION:
        spmv : aggregate SpMVs that make sense
        muelu_levels : analyze only muelu top level timers
        regex : a valid regular expression

      TIMING:
        maxT : MaxOverProcs
        minT : MinOverProcs
        meanT : MeanOverProcs
        meanCT : MeanOverCallCounts

      TITLES:
        none : print no title
        ht   : only label hyperthreads
        full : complete descriptive title

      averaging: cc - call counts
                 ns - num_steps
                 none - do not average

      FORMAT: png, pdf, ...

      TIMER_MAPPING_FILE: file with each line being a ';' (semicolon) delimited list of similar timer names


"""
import matplotlib as mpl
mpl.use('TkAgg')

from docopt import docopt
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from pathlib import Path
import os
import copy
import re
import sys
import ScalingFilenameParser as SFP

# from operator import itemgetter

VERBOSITY = 0
BE_QUIET = False

SPMV_FIG = True

PLOT_DIRS = {
  'composite': 'composites',
  'independent': 'standalone',
  'node': 'nodes',
  'latex': 'latex',
  'csv': 'csv'
}

IMG_FORMAT = "png"
IMG_DPI = 150

FORCE_REPLOT = True
QUANTITY_OF_INTEREST = 'minT'
QUANTITY_OF_INTEREST_COUNT = 'minC'

QUANTITY_OF_INTEREST_MIN = 'minT'
QUANTITY_OF_INTEREST_MIN_COUNT = 'minC'
QUANTITY_OF_INTEREST_MAX = 'maxT'
QUANTITY_OF_INTEREST_MAX_COUNT = 'maxC'
QUANTITY_OF_INTEREST_THING = 'meanCT'
QUANTITY_OF_INTEREST_THING_COUNT = 'meanCC'

AVERAGE_BY = 'none'

MIN_LINESTYLE = 'dotted'
MAX_LINESTYLE = 'solid'

BASELINE_LINESTYLE = 'default'
SHADE_BASELINE_COMPARISON = True

MIN_MARKER = 's'
MAX_MARKER = 'o'
STANDALONE_FONT_SIZE = 20

MIN_STYLE = {
  'linewidth': 5,
  'marker': MIN_MARKER,
  'markersize': 10,
  'fillstyle': 'none'
}

MAX_STYLE = {
  'linewidth': 5,
  'marker': MAX_MARKER,
  'markersize': 10,
  'fillstyle': 'none'
}

PLOT_MIN = True
PLOT_MAX = True
SMOOTH_OUTLIERS = False
HT_CONSISTENT_YAXES = True
ANNOTATE_BEST = False
PLOT_LEGEND = False
PLOT_TITLES = 'none'
PLOT_TITLES_FULL = 'full'
PLOT_TITLES_HT = 'ht'
PLOT_TITLES_NONE = 'none'

DO_YMIN_OVERRIDE = False
DO_YMAX_OVERRIDE = False
YMIN_OVERRIDE = {}
YMAX_OVERRIDE = {}
DO_NORMALIZE_Y = False
HAVE_BASELINE = False
BASELINE_DATASET_DF = None
BASELINE_DRIVER_DF = None

EXPECTED_BASELINE_SPEEDUP = 0.0

BASELINE_DATASET_FILE = ''
DATASET_FILE = ''

PLOTS_TO_GENERATE = {
  'raw_data': True,
  'percent_total': False,
  'flat_mpi_factor': False,
  'bl_speedup': False,
  'bl_perc_diff': False,
  'composite': False
}

ANNOTATE_DATASET_FILENAMES = False

BASELINE_DECOMP = dict(procs_per_node=int(64),
                       cores_per_proc=int(1),
                       threads_per_core=int(1),
                       execspace_name='Serial')
BASELINE_DECOMP_TUPLE = (BASELINE_DECOMP['procs_per_node'],
                         BASELINE_DECOMP['cores_per_proc'],
                         BASELINE_DECOMP['execspace_name'])

HYPER_THREAD_LABEL = 'HT'

# define the colors used for each deomp type
DECOMP_COLORS = {
  '64x1': 'xkcd:greyish',
  '32x2': 'xkcd:windows blue',
  '16x4': 'xkcd:amber',
  '8x8': 'xkcd:faded green',
  '4x16': 'xkcd:dusty purple',
  '2x32': 'xkcd:peach',
  '1x64': 'xkcd:cyan',
  'flat_mpi': 'xkcd:red',
  # 'flat_mpi'  : 'xkcd:salmon',
  'best': 'xkcd:salmon',
  'worst': 'xkcd:greyish'
}


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='*'):
  """
  Call in a loop to create terminal progress bar
  @params:
      iteration   - Required  : current iteration (Int)
      total       - Required  : total iterations (Int)
      prefix      - Optional  : prefix string (Str)
      suffix      - Optional  : suffix string (Str)
      decimals    - Optional  : positive number of decimals in percent complete (Int)
      length      - Optional  : character length of bar (Int)
      fill        - Optional  : bar fill character (Str)
  """
  percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
  filledLength = int(length * iteration // total)
  bar = fill * filledLength + '-' * (length - filledLength)
  print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
  # Print New Line on Complete
  if iteration == total:
    print()
  sys.stdout.flush()


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
  TAKE_COLUMNWISE_MINMAX = False

  DIVIDE_BY_CALLCOUNTS = False
  DIVIDE_BY_NUMSTEPS = False

  have_driver_timings = (driver_groups is not None)  # and (driver_groups.empty is False)

  if AVERAGE_BY == 'ns':
    DIVIDE_BY_NUMSTEPS = True
  elif AVERAGE_BY == 'cc':
    DIVIDE_BY_CALLCOUNTS = True

  # global FOO
  # print('FOO', FOO)

  if DEBUG_plottable_dataframe:
    pd.set_option('display.expand_frame_repr', False)
    print('=========== DATA GROUP ===============')
    print(data_name)
    data_group.to_csv('datagroup-{}-{}.csv'.format(data_name, FOO))

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

  # merge the kernels timings
  plottable_df = plottable_df.merge(timings, on='num_nodes', how='left')

  if DEBUG_plottable_dataframe: print(plottable_df)

  # if the driver dataframe is not empty, then gather info using it
  if have_driver_timings:
    driver_timings = driver_groups.get_group(data_name).groupby('num_nodes',
                                                                as_index=False)[[QUANTITY_OF_INTEREST_MIN,
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
    driver_timings.rename(columns={QUANTITY_OF_INTEREST_MIN: 'driver_min',
                                   QUANTITY_OF_INTEREST_MIN_COUNT: 'driver_min_count',
                                   QUANTITY_OF_INTEREST_MAX: 'driver_max',
                                   QUANTITY_OF_INTEREST_MAX_COUNT: 'driver_max_count',
                                   QUANTITY_OF_INTEREST_THING: 'driver_thing',
                                   QUANTITY_OF_INTEREST_THING_COUNT: 'driver_thing',
                                   'numsteps': 'driver_numsteps'}, inplace=True)

    if DEBUG_plottable_dataframe: print(plottable_df)

    if VERBOSITY & 1024:
      print(
        'Decomp: {magic}:\n\t\tnumsteps:: data: {numsteps} driver: {numsteps_d}\n\t\tcallCounts:: data: {cc} driver: {cc_d}'.format(
          magic=magic,
          numsteps=timings['numsteps'].unique(),
          numsteps_d=driver_timings['driver_numsteps'].unique(),
          cc=timings[QUANTITY_OF_INTEREST_MIN_COUNT].unique(),
          cc_d=driver_timings['driver_min_count'].unique()))

    # merge the driver timings
    plottable_df = plottable_df.merge(driver_timings, on='num_nodes', how='left')

    if DEBUG_plottable_dataframe: print(plottable_df)

    if DEBUG_plottable_dataframe:
      if timings['numsteps'].unique() != driver_timings['driver_numsteps'].unique():
        plottable_df.to_csv('plottable.csv', index=True)
        raise IndexError

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
    plottable_df[QUANTITY_OF_INTEREST_MIN] = plottable_df[QUANTITY_OF_INTEREST_MIN] / plottable_df[
      QUANTITY_OF_INTEREST_MIN_COUNT]
    plottable_df[QUANTITY_OF_INTEREST_MAX] = plottable_df[QUANTITY_OF_INTEREST_MAX] / plottable_df[
      QUANTITY_OF_INTEREST_MAX_COUNT]

    plottable_df['flat_mpi_min'] = plottable_df['flat_mpi_min'] / plottable_df['flat_mpi_min_count']
    plottable_df['flat_mpi_max'] = plottable_df['flat_mpi_max'] / plottable_df['flat_mpi_max_count']

    if have_driver_timings:
      plottable_df['driver_min'] = plottable_df['driver_min'] / plottable_df['driver_min_count']
      plottable_df['driver_min'] = plottable_df['driver_max'] / plottable_df['driver_max_count']

  elif DIVIDE_BY_NUMSTEPS:
    plottable_df[QUANTITY_OF_INTEREST_MIN] = plottable_df[QUANTITY_OF_INTEREST_MIN] / plottable_df['numsteps']
    plottable_df[QUANTITY_OF_INTEREST_MAX] = plottable_df[QUANTITY_OF_INTEREST_MAX] / plottable_df['numsteps']

    plottable_df['flat_mpi_min'] = plottable_df['flat_mpi_min'] / plottable_df['flat_mpi_numsteps']
    plottable_df['flat_mpi_max'] = plottable_df['flat_mpi_max'] / plottable_df['flat_mpi_numsteps']

    if have_driver_timings:
      plottable_df['driver_min'] = plottable_df['driver_min'] / plottable_df['driver_numsteps']
      plottable_df['driver_min'] = plottable_df['driver_max'] / plottable_df['driver_numsteps']

  if have_driver_timings:
    # compute the percent of total
    plottable_df['min_percent_t'] = (plottable_df[QUANTITY_OF_INTEREST_MIN]) / (plottable_df['driver_min']) * 100.00
    plottable_df['max_percent_t'] = (plottable_df[QUANTITY_OF_INTEREST_MAX]) / (plottable_df['driver_max']) * 100.00

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
  plottable_df['flat_mpi_factor_min_t'] = (plottable_df[QUANTITY_OF_INTEREST_MIN]) / (plottable_df['flat_mpi_min'])
  plottable_df['flat_mpi_factor_max_t'] = (plottable_df[QUANTITY_OF_INTEREST_MAX]) / (plottable_df['flat_mpi_max'])

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

  plottable_df['speedup_min_t'] = np1_min / (plottable_df[QUANTITY_OF_INTEREST_MIN])
  plottable_df['speedup_max_t'] = np1_max / (plottable_df[QUANTITY_OF_INTEREST_MAX])

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


###########################################################################################
def compute_scaling_metrics(plottable_df,
                            dataset,
                            total_time_dataset,
                            decomp_label,
                            compute_strong_terms=False):
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
    print(decomp_label)
    dataset.to_csv('datagroup-{}.csv'.format(decomp_label))

  # Use aggregation. This typically will do nothing, but in some cases there are many data points
  # per node count, and we need some way to flatten that data.  This assumes that experiments performed were the same
  # which is how the groupby() logic works  at the highest level. For strong scaling problem_type, problem size (global)
  # , solver and prec options are forced to be the same.
  # The real challenge is how to handle multiple datapoints. For the most part, we do this by dividing by the number
  # of samples taken (numsteps). This assumes that the basic experiment (a step) was the same in all cases.
  datum_count_by_node_count = dataset.groupby('num_nodes').size()
  if DEBUG_plottable_dataframe:
    if datum_count_by_node_count.nunique() > 1:
      print('Data points used for aggregate timing:')
      print(datum_count_by_node_count)
    else:
      print('Using {} datapoints per node count'.format(datum_count_by_node_count.first))

  timings = dataset.groupby('num_nodes', as_index=False)[[QUANTITY_OF_INTEREST_MIN,
                                                          QUANTITY_OF_INTEREST_MIN_COUNT,
                                                          QUANTITY_OF_INTEREST_MAX,
                                                          QUANTITY_OF_INTEREST_MAX_COUNT,
                                                          QUANTITY_OF_INTEREST_THING,
                                                          QUANTITY_OF_INTEREST_THING_COUNT,
                                                          'numsteps']].sum()

  flat_mpi_timings = dataset.groupby('num_nodes', as_index=False)[['flat_mpi_min',
                                                                   'flat_mpi_max',
                                                                   'flat_mpi_min_count',
                                                                   'flat_mpi_max_count',
                                                                   'flat_mpi_numsteps']].first()
  timings = pd.merge(timings, flat_mpi_timings,
                     how='left',
                     on='num_nodes')

  total_time_timings = total_time_dataset.groupby('num_nodes',
                                                  as_index=False)[[QUANTITY_OF_INTEREST_MIN,
                                                                   QUANTITY_OF_INTEREST_MIN_COUNT,
                                                                   QUANTITY_OF_INTEREST_MAX,
                                                                   QUANTITY_OF_INTEREST_MAX_COUNT,
                                                                   QUANTITY_OF_INTEREST_THING,
                                                                   QUANTITY_OF_INTEREST_THING_COUNT,
                                                                   'numsteps']].sum()

  if DEBUG_plottable_dataframe:
    total_time_dataset.to_csv('driver_timings-{timer}-{d}.csv'.format(
      timer=total_time_dataset['Timer Name'].unique(),
      d=decomp_label))

  total_time_timings = total_time_timings[['num_nodes',
                                           QUANTITY_OF_INTEREST_MIN,
                                           QUANTITY_OF_INTEREST_MIN_COUNT,
                                           QUANTITY_OF_INTEREST_MAX,
                                           QUANTITY_OF_INTEREST_MAX_COUNT,
                                           QUANTITY_OF_INTEREST_THING,
                                           QUANTITY_OF_INTEREST_THING_COUNT,
                                           'numsteps']]

  # rename the driver timings to indicate they were from the driver
  total_time_timings.rename(columns={QUANTITY_OF_INTEREST_MIN: 'driver_min',
                                     QUANTITY_OF_INTEREST_MIN_COUNT: 'driver_min_count',
                                     QUANTITY_OF_INTEREST_MAX: 'driver_max',
                                     QUANTITY_OF_INTEREST_MAX_COUNT: 'driver_max_count',
                                     QUANTITY_OF_INTEREST_THING: 'driver_thing',
                                     QUANTITY_OF_INTEREST_THING_COUNT: 'driver_thing',
                                     'numsteps': 'driver_numsteps'}, inplace=True)

  if DEBUG_plottable_dataframe: print(plottable_df)

  print(
    'Decomp: {magic}:\n\t\tnumsteps:: data: {numsteps} driver: {numsteps_d}\n\t\tcallCounts:: data: {cc} driver: {cc_d}'.format(
      magic=decomp_label,
      numsteps=timings['numsteps'].unique(),
      numsteps_d=total_time_timings['driver_numsteps'].unique(),
      cc=timings[QUANTITY_OF_INTEREST_MIN_COUNT].unique(),
      cc_d=total_time_timings['driver_min_count'].unique()))

  # merge the kernels timings
  plottable_df = plottable_df.merge(timings, on='num_nodes', how='left')

  if DEBUG_plottable_dataframe: print(plottable_df)

  # merge the driver timings
  plottable_df = plottable_df.merge(total_time_timings, on='num_nodes', how='left')

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

    plottable_df['max_percent_t'] = \
      (plottable_df[QUANTITY_OF_INTEREST_MAX] / plottable_df[QUANTITY_OF_INTEREST_MAX_COUNT]) \
      / (plottable_df['driver_max'] / plottable_df['driver_max_count']) * 100.00
  elif DIVIDE_BY_NUMSTEPS:
    plottable_df['min_percent_t'] = \
      (plottable_df[QUANTITY_OF_INTEREST_MIN] / plottable_df['numsteps']) \
      / (plottable_df['driver_min'] / plottable_df['driver_numsteps']) * 100.00

    plottable_df['max_percent_t'] = \
      (plottable_df[QUANTITY_OF_INTEREST_MAX] / plottable_df['numsteps']) \
      / (plottable_df['driver_max'] / plottable_df['driver_numsteps']) * 100.00
  else:
    plottable_df['min_percent_t'] = \
      (plottable_df[QUANTITY_OF_INTEREST_MIN]) \
      / (plottable_df['driver_min']) * 100.00

    plottable_df['max_percent_t'] = \
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
    plottable_df.to_csv('plottable-{timer}-{d}.csv'.format(timer=dataset['Timer Name'].unique(),
                                                           d=decomp_label))

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
    plottable_df['speedup_min_t'] = np1_min / \
                                    (plottable_df[QUANTITY_OF_INTEREST_MIN] / plottable_df[
                                      QUANTITY_OF_INTEREST_MIN_COUNT])

    plottable_df['speedup_max_t'] = np1_max / \
                                    (plottable_df[QUANTITY_OF_INTEREST_MAX] / plottable_df[
                                      QUANTITY_OF_INTEREST_MAX_COUNT])
  elif DIVIDE_BY_NUMSTEPS:
    plottable_df['speedup_min_t'] = np1_min / \
                                    (plottable_df[QUANTITY_OF_INTEREST_MIN] / plottable_df['numsteps'])

    plottable_df['speedup_max_t'] = np1_max / \
                                    (plottable_df[QUANTITY_OF_INTEREST_MAX] / plottable_df['numsteps'])
  else:
    plottable_df['speedup_min_t'] = np1_min / \
                                    (plottable_df[QUANTITY_OF_INTEREST_MIN])

    plottable_df['speedup_max_t'] = np1_max / \
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
    plottable_df.to_csv('plottable-{}.csv'.format(decomp_label))
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


def enforce_override_ylims(figures, axes):
  """
  Given the matplotlib figure and axes handles, Override the current ylims

  :param figures: dict of dict of dict of figures
                  figures should be the independent figures, constructed as
                  independent:column_name:row_name
  :param axes: Handles for all axes that are part of the composite plot. e.g., axes[column_name][row_name]
  :return: Nothing
  """
  # if we want consistent axes by column, then enforce that here.
  if DO_YMAX_OVERRIDE or DO_YMIN_OVERRIDE:
    for column_name, column_map in axes.items():
      # apply these limits to each plot in this column
      for axes_name, ax in column_map.items():
        current_ylims = list(ax.get_ylim())
        if column_name in YMIN_OVERRIDE:
          current_ylims[0] = float(YMIN_OVERRIDE[column_name])
        else:
          if VERBOSITY & 1024: print('plot {} does not have override'.format(column_name))
        if column_name in YMAX_OVERRIDE:
          current_ylims[1] = float(YMAX_OVERRIDE[column_name])
        else:
          if VERBOSITY & 1024: print('plot {} does not have override'.format(column_name))
        ax.set_ylim(current_ylims)

      for figure_name, fig in figures['independent'][column_name].items():
        current_ylims = list(fig.gca().get_ylim())
        if column_name in YMIN_OVERRIDE:
          current_ylims[0] = float(YMIN_OVERRIDE[column_name])
        else:
          if VERBOSITY & 1024: print('plot {} does not have override'.format(column_name))
        if column_name in YMAX_OVERRIDE:
          current_ylims[1] = float(YMAX_OVERRIDE[column_name])
        else:
          if VERBOSITY & 1024: print('plot {} does not have override'.format(column_name))
        ax.set_ylim(current_ylims)

        fig.gca().set_ylim(current_ylims)


###########################################################################################
def save_figures(figures,
                 filename,
                 sub_dir=None,
                 close_figure=False,
                 independent_names=None,
                 composite=True,
                 independent=True):
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
  fullpath = ''

  # write the composite image if it was requested
  if PLOTS_TO_GENERATE['composite']:
    try:
      fullpath = get_img_filepath(base_name=filename,
                                  ext=IMG_FORMAT,
                                  path=PLOT_DIRS['composite'],
                                  sub_dir=sub_dir,
                                  composite=True)

      print(fullpath)
      figures['composite'].savefig(fullpath,
                                   format=IMG_FORMAT,
                                   dpi=IMG_DPI)
      print('Wrote: {}'.format(os.path.basename(fullpath)))
    except:
      print('FAILED writing {}'.format(os.path.basename(fullpath)))
      raise

  # create a second figure for the legend
  # we assume the composite figure has a legend attached always
  handles, labels = figures['composite'].gca().get_legend_handles_labels()
  ncols = len(labels)
  legend_path = '{path}/legend-{ncols}.{format}'.format(path=PLOT_DIRS['composite'], ncols=ncols, format=IMG_FORMAT)
  legend_file = Path(legend_path)
  try:
    temp = legend_file.resolve()
  except FileNotFoundError or RuntimeError:
    import pylab
    try:
      figLegend = pylab.figure(figsize=(12, 1))
      figLegend.legend(handles, labels,
                       title="Procs per Node x Cores per Proc",
                       ncol=ncols,
                       loc='lower center')
      figLegend.savefig(legend_path,
                        format=IMG_FORMAT,
                        dpi=IMG_DPI)
      plt.close(figLegend)
      if VERBOSITY & 1024:
        print('Wrote: {}'.format(os.path.basename(legend_path)))
    except:
      print('FAILED writing {}'.format(os.path.basename(legend_path)))
      raise

  if close_figure:
    plt.close(figures['composite'])

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

      try:
        fullpath = get_img_filepath(base_name=filename,
                                    plot_name=column_name,
                                    ht_name=ht_name,
                                    ext=IMG_FORMAT,
                                    path=PLOT_DIRS['independent'],
                                    sub_dir=sub_dir,
                                    independent=True)

        figures['independent'][column_name][ht_name].savefig(fullpath,
                                                             format=IMG_FORMAT,
                                                             dpi=IMG_DPI)
        if VERBOSITY & 1024:
          print('Wrote: {}'.format(os.path.basename(fullpath)))
      except:
        print('FAILED writing {}'.format(os.path.basename(fullpath)))
        raise

      if close_figure:
        plt.close(figures['independent'][column_name][ht_name])


###########################################################################################
def close_figures(figures):
  """
  Helper to save figures. Plots the composite figure, as well as the independent figures.

  :param figures: dict of figures. composite => figure,
                                   independent => column_name => row_name => figure
  :return:
  """

  plt.close(figures['composite'])

  for column_name in figures['independent']:
    fig_names = figures['independent'][column_name].keys()

    for ht_name in fig_names:
      plt.close(figures['independent'][column_name][ht_name])


###########################################################################################
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
  if ax:
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
def need_to_replot(simple_fname,
                   subplot_names,
                   ht_names,
                   sub_dir,
                   composite=False,
                   independent=True):
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

  if composite:
    filepath = get_img_filepath(base_name=simple_fname,
                                ext=IMG_FORMAT,
                                path=PLOT_DIRS['composite'],
                                composite=True)
    my_file = Path(filepath)
    try:
      # this is very annoying. PathLib seems to behave differently across versions and operating systems
      # what we do, is attempt to resolve the path. On Linux, this will attempt to locate the file
      # on Mac OS, this resolves the directory (which will always exist as it is created before plotting)
      # to handle this, we resolve then test if the resolve thing is a file.
      # We could return instantly on failure, but waiting until the end allows us to print out all of the
      # missing files.
      temp = my_file.resolve()
      if temp.is_file() is False:
        need_to_replot_ = True
    except FileNotFoundError or RuntimeError:
      if VERBOSITY & 1024:
        print("File {} does not exist triggering replot".format(filepath))
      need_to_replot_ = True

  if independent is False:
    return need_to_replot_

  for column_name in subplot_names:
    for ht_name in ht_names:
      fullpath = get_img_filepath(base_name=simple_fname,
                                  plot_name=column_name,
                                  ht_name=ht_name,
                                  ext=IMG_FORMAT,
                                  path=PLOT_DIRS['independent'],
                                  independent=True)
      my_file = Path(fullpath)
      try:
        temp = my_file.resolve()
        if temp.is_file() is False:
          need_to_replot_ = True
      except FileNotFoundError or RuntimeError:
        if VERBOSITY & 1024:
          print("File {} does not exist triggering replot".format(fullpath))
        need_to_replot_ = True

  return need_to_replot_


def get_img_filepath(base_name,
                     plot_name=None,
                     ht_name=None,
                     sub_dir=None,
                     ext=None,
                     path=None,
                     composite=False,
                     independent=False):
  if sub_dir is None:
    sub_dir = ''
  else:
    sub_dir = '/' + sub_dir

  fullpath = ''
  if composite:
    fullpath = '{path}{sub_dir}/{base_name}.{ext}'.format(path=path,
                                                          sub_dir=sub_dir,
                                                          base_name=base_name,
                                                          ext=ext)
  elif independent:
    fig_filename = '{base}-{ht}.{ext}'.format(base=base_name,
                                              ht=ht_name,
                                              ext=ext)

    fullpath = '{path}/{plot_name}{sub_dir}/{fname}'.format(path=path,
                                                            sub_dir=sub_dir,
                                                            plot_name=plot_name,
                                                            fname=fig_filename)
  else:
    print('get_img_filepath called without composite or independent set')
    exit(-1)

  return fullpath


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
      flat_mpi_df = composite_group.groupby(groupby_cols).get_group((64, 1, 1))
    except KeyError as e:
      if allow_baseline_override:
        # if that fails, and we allow it, then downgrade to 32x2x1
        flat_mpi_df = composite_group.groupby(groupby_cols).get_group((32, 2, 1))
      else:
        raise e
  else:
    flat_mpi_df = serial_data
    if VERBOSITY & 1024:
      print('Using Serial run data for flatMPI')

  flat_mpi_df.rename(columns={QUANTITY_OF_INTEREST_MIN: 'flat_mpi_min',
                              QUANTITY_OF_INTEREST_MAX: 'flat_mpi_max',
                              QUANTITY_OF_INTEREST_MIN_COUNT: 'flat_mpi_min_count',
                              QUANTITY_OF_INTEREST_MAX_COUNT: 'flat_mpi_max_count',
                              'numsteps': 'flat_mpi_numsteps'}, inplace=True)

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
  # del tmp_df['flat_mpi_min']
  del tmp_df['flat_mpi_min_count']
  del tmp_df['flat_mpi_max']
  del tmp_df['flat_mpi_max_count']
  try:
    del tmp_df['driver_thing']
    del tmp_df['driver_numsteps']
    del tmp_df['driver_min']
    del tmp_df['driver_min_count']
    del tmp_df['driver_max']
    del tmp_df['driver_max_count']
  except:
    pass
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

  tmp_df['decomp_label'] = str(procs_per_node) + 'x' + str(cores_per_proc)
  tmp_df.loc[tmp_df['execspace_name'] == 'Serial', 'decomp_label'] = 'flat_mpi'

  tmp_df['MPI Procs'] = tmp_df['nodes'] * procs_per_node

  # tmp_df[[QUANTITY_OF_INTEREST_MIN,
  #         QUANTITY_OF_INTEREST_MAX,
  #         'flat_mpi_factor_min',
  #         'flat_mpi_factor_max',
  #         'min_percent',
  #         'max_percent']] = \
  #   tmp_df[[QUANTITY_OF_INTEREST_MIN,
  #           QUANTITY_OF_INTEREST_MAX,
  #           'flat_mpi_factor_min',
  #           'flat_mpi_factor_max',
  #           'min_percent',
  #           'max_percent']].apply(lambda x: pd.Series.round(x, 2))

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
  try:
    decomp_dataframe = decomp_dataframe.set_index(keys=keys,
                                                  drop=True,
                                                  verify_integrity=True)
  except:
    pass

  return decomp_dataframe


###############################################################################
def plot_composite_weak(composite_group,
                        my_nodes,
                        my_ticks,
                        driver_df,
                        expected_sub_dirs,
                        average=False,
                        numbered_plots_idx=-1,
                        write_latex_and_csv=True,
                        baseline_group=None,
                        baseline_dr_df=None,
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
  total_df = pd.DataFrame()

  if composite_group is None:
    print("Composite Group is none?")
    raise LookupError

  have_driver_timings = (driver_df is not None) and (driver_df.empty is False) and \
                        (baseline_dr_df is not None) and (baseline_dr_df.empty is False)

  # determine the flat MPI time
  try:
    composite_group = add_flat_mpi_data(composite_group, allow_baseline_override=True)
  except:
    print('WTF?')
    composite_group.to_csv('failed.csv')
    raise

  if HAVE_BASELINE:
    if baseline_group is None:
      if VERBOSITY & 1024:
        print("Skipping, have a baseline but don't have this timer in the baseline")
      return

    bl_composite_group = add_flat_mpi_data(baseline_group, allow_baseline_override=True)

  decomp_groups = composite_group.groupby(['procs_per_node', 'cores_per_proc', 'execspace_name'])
  if have_driver_timings:
    driver_decomp_groups = driver_df.groupby(['procs_per_node', 'cores_per_proc', 'execspace_name'])

  if HAVE_BASELINE:
    if (VERBOSITY & 1024): print('have a baseline!')
    bl_decomp_groups = bl_composite_group.groupby(['procs_per_node', 'cores_per_proc', 'execspace_name'])
    if have_driver_timings:
      bl_dr_decomp_groups = baseline_dr_df.groupby(['procs_per_node', 'cores_per_proc', 'execspace_name'])

  # determine the components that should be in the filename
  show_solver_name = (composite_group['solver_name'].nunique() == 1)
  show_solver_attributes = (composite_group['solver_attributes'].nunique() == 1)
  show_prec_name = (composite_group['prec_name'].nunique() == 1)
  show_prec_attributes = (composite_group['prec_attributes'].nunique() == 1)

  # for a specific group of data, compute the scaling terms, which are things like min/max
  # this also flattens the timer creating a 'fat_timer_name'
  # essentially, this function computes data that is relevant to a group, but not the whole
  try:
    my_tokens = SFP.getTokensFromDataFrameGroupBy(composite_group)
  except KeyError as k:
    print(composite_group)
    raise k

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

  # the number of HT combos we have
  ht_names = composite_group['threads_per_core'].sort_values(ascending=True).unique()
  ndecomps = len(decomp_groups)

  my_num_nodes = my_nodes.size

  fig_size = 5
  fig_size_height_inflation = 1.125
  fig_size_width_inflation = 1.5

  SUBPLOT_NAMES = [k for k in PLOTS_TO_GENERATE.keys() if PLOTS_TO_GENERATE[k]]  # ['raw_data', 'percent_total']

  if 'bl_speedup' in SUBPLOT_NAMES:
    bl_speedup_expected = (EXPECTED_BASELINE_SPEEDUP > 0.0)

  # whether we should replot images that already exist.
  if FORCE_REPLOT is False:
    missing_files = False
    for sub_dir in expected_sub_dirs:
      missing_file = need_to_replot(simple_fname, SUBPLOT_NAMES, ht_names, sub_dir=sub_dir)
      if (VERBOSITY & 1024) and missing_file: print("Missing {}/{}.png".format(sub_dir,
                                                                               simple_fname))
      missing_files = missing_files or missing_file
    if missing_files is False:
      if VERBOSITY & 1024: print("Skipping".format(simple_fname))

  axes, figures = get_figures_and_axes(subplot_names=SUBPLOT_NAMES,
                                       subplot_row_names=ht_names,
                                       fig_size=fig_size,
                                       fig_size_width_inflation=fig_size_width_inflation,
                                       fig_size_height_inflation=fig_size_height_inflation)
  # whether we have data to make a baseline comparison
  have_global_baseline = False

  if HAVE_BASELINE:
    # gather data for a global baseline plot e.g., flat MPI from another study
    try:
      bl_global_decomp_group = bl_decomp_groups.get_group(BASELINE_DECOMP_TUPLE)
      bl_global_decomp_ht_groups = bl_global_decomp_group.groupby('threads_per_core')
      bl_global_decomp_ht_group = bl_global_decomp_ht_groups.get_group((BASELINE_DECOMP['threads_per_core']))

      if not bl_global_decomp_ht_group.empty:
        have_global_baseline = True
        # for plot_row in axes['raw_data']:
        #   axes['bl_perc_diff'][plot_row] = False

      if have_driver_timings:
        bl_dr_global_decomp_group = bl_dr_decomp_groups.get_group(BASELINE_DECOMP_TUPLE)
        bl_dr_global_decomp_ht_groups = bl_dr_global_decomp_group.groupby('threads_per_core')
        bl_dr_global_decomp_ht_group = bl_dr_global_decomp_ht_groups.get_group((BASELINE_DECOMP['threads_per_core']))
    except:
      pass

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

    if have_driver_timings:
      driver_ht_groups = driver_decomp_groups.get_group(decomp_group_name).groupby('threads_per_core')
    else:
      driver_ht_groups = None

    for ht_name in ht_names:

      plot_row = ht_name

      if execution_space == 'Serial':
        ht_name = ht_names[0]

      ht_group = ht_groups.get_group(ht_name)

      ####################################################################################################################
      # attempt to gather data for baseline tracking

      # whether we have data to make a baseline comparison
      have_decomp_baseline = False

      if HAVE_BASELINE:
        try:
          # this series of groupings is for clarity.
          # this is the same order of ops as we do for the regular data
          # what we effectively do, is group by procs, cores, execspace, threads
          bl_decomp_group = bl_decomp_groups.get_group(decomp_group_name)
          bl_decomp_ht_groups = bl_decomp_group.groupby('threads_per_core')
          bl_decomp_ht_group = bl_decomp_ht_groups.get_group(ht_name)

          if not bl_decomp_ht_group.empty:
            # if we got something, then we can form a comparison
            have_decomp_baseline = True

          if have_driver_timings:
            bl_dr_decomp_group = bl_dr_decomp_groups.get_group(decomp_group_name)
            bl_dr_decomp_ht_groups = bl_dr_decomp_group.groupby('threads_per_core')
            bl_dr_decomp_ht_group = bl_dr_decomp_ht_groups.get_group(ht_name)
          else:
            bl_dr_decomp_ht_groups = None
        except:
          pass

      magic_str = '-{decomp_label}x{threads_per_core}'.format(decomp_label=decomp_label,
                                                              threads_per_core=ht_name)

      my_agg_times = pd.DataFrame(columns=['num_nodes', 'ticks'], data=np.column_stack((my_nodes, my_ticks)))
      my_agg_times = get_plottable_dataframe(my_agg_times, ht_group, ht_name, driver_ht_groups, magic=magic_str)

      bl_agg_times = pd.DataFrame(columns=['num_nodes', 'ticks'], data=np.column_stack((my_nodes, my_ticks)))
      if have_decomp_baseline:
        bl_agg_times = get_plottable_dataframe(bl_agg_times, bl_decomp_ht_group, ht_name, bl_dr_decomp_ht_groups,
                                               magic='bl' + magic_str)
        bl_agg_times.rename(columns={QUANTITY_OF_INTEREST_MIN: 'bl_min',
                                     QUANTITY_OF_INTEREST_MAX: 'bl_max'},
                            inplace=True)
        my_agg_times = pd.merge(my_agg_times, bl_agg_times[['bl_min', 'bl_max', 'num_nodes']],
                                how='left',
                                on=['num_nodes'])

        if PLOTS_TO_GENERATE['bl_perc_diff']:
          my_agg_times['bl_max_diff'] = ((my_agg_times['bl_max'] - my_agg_times[QUANTITY_OF_INTEREST_MAX])
                                         / my_agg_times['bl_max']) * 100
          my_agg_times['bl_min_diff'] = ((my_agg_times['bl_min'] - my_agg_times[QUANTITY_OF_INTEREST_MIN])
                                         / my_agg_times['bl_min']) * 100

        if PLOTS_TO_GENERATE['bl_speedup']:
          my_agg_times['bl_speedup_max'] = my_agg_times['bl_max'] / my_agg_times[QUANTITY_OF_INTEREST_MAX]
          my_agg_times['bl_speedup_min'] = my_agg_times['bl_min'] / my_agg_times[QUANTITY_OF_INTEREST_MIN]

      # this assumes that idx=0 is an SpMV aggregate figure, which appears to be worthless.
      if write_latex_and_csv:  # and plot_row == ht_name:
        total_df = update_decomp_dataframe(total_df,
                                           my_agg_times,
                                           ht_group,
                                           plot_row,  # ht_name,
                                           procs_per_node,
                                           cores_per_proc)

      # count the missing values, can use any quantity of interest for this
      num_missing_data_points = my_agg_times[QUANTITY_OF_INTEREST_MIN].isnull().values.ravel().sum()

      if num_missing_data_points != 0 and VERBOSITY & 1024:
        print(
          "Expected {expected_data_points} data points, Missing: {num_missing_data_points}".format(
            expected_data_points=my_num_nodes,
            num_missing_data_points=num_missing_data_points))

      if VERBOSITY & 1024:
        print("x={}, y={}, {}x{}x{}".format(my_agg_times['ticks'].count(),
                                            my_agg_times['num_nodes'].count(),
                                            procs_per_node,
                                            cores_per_proc,
                                            ht_name))

      # plot the data
      if PLOT_MAX:
        # plot the max if requested

        # if we are doing baseline comparisons, then make sure this data is shaded
        if HAVE_BASELINE and SHADE_BASELINE_COMPARISON:
          import matplotlib.patheffects as pe
          MAX_STYLE['path_effects'] = [pe.Stroke(linewidth=18, foreground=DECOMP_COLORS[decomp_label], alpha=0.25),
                                       pe.Normal()]

        plot_raw_data(ax=axes['raw_data'][plot_row],
                      indep_ax=figures['independent']['raw_data'][plot_row].gca(),
                      xvalues=my_agg_times['ticks'],
                      yvalues=my_agg_times[QUANTITY_OF_INTEREST_MAX],
                      linestyle=MAX_LINESTYLE,
                      label='max-{}'.format(decomp_label),
                      color=DECOMP_COLORS[decomp_label],
                      **MAX_STYLE)

        if HAVE_BASELINE:
          MAX_STYLE['path_effects'] = None

      if PLOT_MIN:

        # if we are doing baseline comparisons, then make sure this data is shaded
        if HAVE_BASELINE and SHADE_BASELINE_COMPARISON:
          import matplotlib.patheffects as pe
          MIN_STYLE['path_effects'] = [pe.Stroke(linewidth=18, foreground=DECOMP_COLORS[decomp_label], alpha=0.25),
                                       pe.Normal()]

        # plot the max if requested
        plot_raw_data(ax=axes['raw_data'][plot_row],
                      indep_ax=figures['independent']['raw_data'][plot_row].gca(),
                      xvalues=my_agg_times['ticks'],
                      yvalues=my_agg_times[QUANTITY_OF_INTEREST_MIN],
                      linestyle=MIN_LINESTYLE,
                      label='min-{}'.format(decomp_label),
                      color=DECOMP_COLORS[decomp_label],
                      **MIN_STYLE)

        if HAVE_BASELINE:
          MIN_STYLE['path_effects'] = None

      if have_decomp_baseline:
        import copy
        import matplotlib.patheffects as pe
        if PLOT_MAX:
          if BASELINE_LINESTYLE == 'default':
            baseline_linestyle = MAX_LINESTYLE
          else:
            baseline_linestyle = BASELINE_LINESTYLE

          m = copy.deepcopy(MAX_STYLE)
          m['marker'] = None
          # m['path_effects'] = [pe.Stroke(linewidth=15, foreground=DECOMP_COLORS[decomp_label], alpha=0.25), pe.Normal()]
          # m['path_effects'] = [pe.Stroke(linewidth=5, foreground=DECOMP_COLORS[decomp_label], alpha=0.25), pe.Normal()]
          # m['path_effects'] = [pe.SimpleLineShadow(shadow_color=DECOMP_COLORS[decomp_label]), pe.Normal()]
          # plot the max if requested
          plot_raw_data(ax=axes['raw_data'][plot_row],
                        indep_ax=figures['independent']['raw_data'][plot_row].gca(),
                        xvalues=my_agg_times['ticks'],
                        yvalues=my_agg_times['bl_max'],
                        linestyle=baseline_linestyle,
                        label=None,
                        color='black',
                        **m)
          m['path_effects'] = None
          m['marker'] = '*'
          plot_raw_data(ax=axes['raw_data'][plot_row],
                        indep_ax=figures['independent']['raw_data'][plot_row].gca(),
                        xvalues=my_agg_times['ticks'],
                        yvalues=my_agg_times['bl_max'],
                        linestyle=baseline_linestyle,
                        label='Prior-max-{}'.format(decomp_label),
                        color=DECOMP_COLORS[decomp_label],
                        **m)

          if PLOTS_TO_GENERATE['bl_perc_diff']:
            # plot percent diff
            plot_raw_data(ax=axes['bl_perc_diff'][plot_row],
                          indep_ax=figures['independent']['bl_perc_diff'][plot_row].gca(),
                          xvalues=my_agg_times['ticks'],
                          yvalues=my_agg_times['bl_max_diff'],
                          linestyle=baseline_linestyle,
                          label='Perc. Diff-max-{}'.format(decomp_label),
                          color=DECOMP_COLORS[decomp_label],
                          **MAX_STYLE)

          if PLOTS_TO_GENERATE['bl_speedup']:
            # if we draw the expected line do it first
            if bl_speedup_expected:
              axes['bl_speedup'][plot_row].axhline(
                y=EXPECTED_BASELINE_SPEEDUP, color='black', linestyle='-')
              figures['independent']['bl_speedup'][plot_row].gca().axhline(
                y=EXPECTED_BASELINE_SPEEDUP, color='black', linestyle='-')

            # plot speedup of maxT over Baseline_maxT
            plot_raw_data(ax=axes['bl_speedup'][plot_row],
                          indep_ax=figures['independent']['bl_speedup'][plot_row].gca(),
                          xvalues=my_agg_times['ticks'],
                          yvalues=my_agg_times['bl_speedup_max'],
                          linestyle=baseline_linestyle,
                          label='Speedup-max-{}'.format(decomp_label),
                          color=DECOMP_COLORS[decomp_label],
                          **MAX_STYLE)

        if PLOT_MIN:
          if BASELINE_LINESTYLE == 'default':
            baseline_linestyle = MIN_LINESTYLE
          else:
            baseline_linestyle = BASELINE_LINESTYLE

          m = copy.deepcopy(MIN_STYLE)
          m['marker'] = '*'
          # plot the max if requested
          plot_raw_data(ax=axes['raw_data'][plot_row],
                        indep_ax=figures['independent']['raw_data'][plot_row].gca(),
                        xvalues=my_agg_times['ticks'],
                        yvalues=my_agg_times['bl_min'],
                        linestyle=baseline_linestyle,
                        label='Prior-{}'.format(decomp_label),
                        color=DECOMP_COLORS[decomp_label],
                        **m)

          # plot percent diff
          if PLOTS_TO_GENERATE['bl_perc_diff']:
            plot_raw_data(ax=axes['bl_perc_diff'][plot_row],
                          indep_ax=figures['independent']['bl_perc_diff'][plot_row].gca(),
                          xvalues=my_agg_times['ticks'],
                          yvalues=my_agg_times['bl_min_diff'],
                          linestyle=baseline_linestyle,
                          label='Perc. Diff-min-{}'.format(decomp_label),
                          color=DECOMP_COLORS[decomp_label],
                          **MIN_STYLE)

          # plot speedup of maxT over Baseline_maxT
          if PLOTS_TO_GENERATE['bl_speedup']:
            # if we draw the expected line do it first
            if bl_speedup_expected:
              axes['bl_speedup'][plot_row].axhline(
                y=EXPECTED_BASELINE_SPEEDUP, color='black', linestyle='-')
              figures['independent']['bl_speedup'][plot_row].gca().axhline(
                y=EXPECTED_BASELINE_SPEEDUP, color='black', linestyle='-')

            plot_raw_data(ax=axes['bl_speedup'][plot_row],
                          indep_ax=figures['independent']['bl_speedup'][plot_row].gca(),
                          xvalues=my_agg_times['ticks'],
                          yvalues=my_agg_times['bl_speedup_min'],
                          linestyle=baseline_linestyle,
                          label='Speedup-max-{}'.format(decomp_label),
                          color=DECOMP_COLORS[decomp_label],
                          **MIN_STYLE)

      if PLOTS_TO_GENERATE['percent_total']:
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

      if PLOTS_TO_GENERATE['flat_mpi_factor']:
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
    # construct a standard title, which may not be used
    standard_title = '{}\n{bolding_pre}({HT_LABEL}={HT_NUM:.0f}){bolding_post}'.format(simple_title,
                                                                                       HT_LABEL=HYPER_THREAD_LABEL,
                                                                                       HT_NUM=ht_name,
                                                                                       bolding_pre=r'$\bf{',
                                                                                       bolding_post=r'}$')

    ht_title = '{bolding_pre}{HT_LABEL}={HT_NUM:.0f}{bolding_post}'.format(HT_LABEL=HYPER_THREAD_LABEL,
                                                                           HT_NUM=ht_name,
                                                                           bolding_pre=r'$\bf{',
                                                                           bolding_post=r'}$')

    # for independent plots (e.g., the subplot plotted separately) we show all axes labels
    ## raw data
    figures['independent']['raw_data'][ht_name].gca().set_ylabel('Runtime (s)')
    figures['independent']['raw_data'][ht_name].gca().set_xlabel('Number of Nodes')
    figures['independent']['raw_data'][ht_name].gca().set_xticks(my_ticks)
    figures['independent']['raw_data'][ht_name].gca().set_xticklabels(my_nodes, rotation=45)
    figures['independent']['raw_data'][ht_name].gca().set_xlim([0.5, my_num_nodes + 0.5])
    if PLOT_TITLES == PLOT_TITLES_FULL:
      figures['independent']['raw_data'][ht_name].gca().set_title(standard_title)
    elif PLOT_TITLES == PLOT_TITLES_HT:
      figures['independent']['raw_data'][ht_name].gca().set_title(ht_title)

    ## percent diff
    if have_decomp_baseline and PLOTS_TO_GENERATE['bl_perc_diff']:
      figures['independent']['bl_perc_diff'][ht_name].gca().set_ylabel('Percentage Improvement over Prior')
      figures['independent']['bl_perc_diff'][ht_name].gca().set_xlabel('Number of Nodes')
      figures['independent']['bl_perc_diff'][ht_name].gca().set_xticks(my_ticks)
      figures['independent']['bl_perc_diff'][ht_name].gca().set_xticklabels(my_nodes, rotation=45)
      figures['independent']['bl_perc_diff'][ht_name].gca().set_xlim([0.5, my_num_nodes + 0.5])
      if PLOT_TITLES == PLOT_TITLES_FULL:
        figures['independent']['bl_perc_diff'][ht_name].gca().set_title(standard_title)
      elif PLOT_TITLES == PLOT_TITLES_HT:
        figures['independent']['bl_perc_diff'][ht_name].gca().set_title(ht_title)

    if have_decomp_baseline and PLOTS_TO_GENERATE['bl_speedup']:
      figures['independent']['bl_speedup'][ht_name].gca().set_ylabel('Speedup')
      figures['independent']['bl_speedup'][ht_name].gca().set_xlabel('Number of Nodes')
      figures['independent']['bl_speedup'][ht_name].gca().set_xticks(my_ticks)
      figures['independent']['bl_speedup'][ht_name].gca().set_xticklabels(my_nodes, rotation=45)
      figures['independent']['bl_speedup'][ht_name].gca().set_xlim([0.5, my_num_nodes + 0.5])
      if PLOT_TITLES == PLOT_TITLES_FULL:
        figures['independent']['bl_speedup'][ht_name].gca().set_title(standard_title)
      elif PLOT_TITLES == PLOT_TITLES_HT:
        figures['independent']['bl_speedup'][ht_name].gca().set_title(ht_title)

    ## percentages
    if PLOTS_TO_GENERATE['percent_total']:
      figures['independent']['percent_total'][ht_name].gca().set_ylabel('Percentage of Total Time')
      figures['independent']['percent_total'][ht_name].gca().set_xlabel('Number of Nodes')
      figures['independent']['percent_total'][ht_name].gca().set_xticks(my_ticks)
      figures['independent']['percent_total'][ht_name].gca().set_xticklabels(my_nodes, rotation=45)
      figures['independent']['percent_total'][ht_name].gca().set_xlim([0.5, my_num_nodes + 0.5])
      if PLOT_TITLES == PLOT_TITLES_FULL:
        figures['independent']['percent_total'][ht_name].gca().set_title(standard_title)
      elif PLOT_TITLES == PLOT_TITLES_HT:
        figures['independent']['percent_total'][ht_name].gca().set_title(ht_title)

      figures['independent']['percent_total'][ht_name].gca().yaxis.set_major_formatter(FormatStrFormatter('%3.0f %%'))

    ## factors
    if PLOTS_TO_GENERATE['flat_mpi_factor']:
      figures['independent']['flat_mpi_factor'][ht_name].gca().set_ylabel('Ratio (smaller is better)')
      figures['independent']['flat_mpi_factor'][ht_name].gca().set_xlabel('Number of Nodes')
      figures['independent']['flat_mpi_factor'][ht_name].gca().set_xticks(my_ticks)
      figures['independent']['flat_mpi_factor'][ht_name].gca().set_xticklabels(my_nodes, rotation=45)
      figures['independent']['flat_mpi_factor'][ht_name].gca().set_xlim([0.5, my_num_nodes + 0.5])
      if PLOT_TITLES == PLOT_TITLES_FULL:
        figures['independent']['flat_mpi_factor'][ht_name].gca().set_title(standard_title)
      elif PLOT_TITLES == PLOT_TITLES_HT:
        figures['independent']['flat_mpi_factor'][ht_name].gca().set_title(ht_title)

    # adjust the font size for all plots
    for plot_name in figures['independent'].keys():
      tmp_ax = figures['independent'][plot_name][ht_name].gca()
      for item in ([tmp_ax.xaxis.label, tmp_ax.yaxis.label] +
                     tmp_ax.get_xticklabels() + tmp_ax.get_yticklabels()):
        item.set_fontsize(STANDALONE_FONT_SIZE)
      tmp_ax.title.set_fontsize(STANDALONE_FONT_SIZE)

    axes['raw_data'][ht_name].set_ylabel('Runtime (s)')
    axes['raw_data'][ht_name].set_xlim([0.5, my_num_nodes + 0.5])

    if PLOTS_TO_GENERATE['percent_total']:
      axes['percent_total'][ht_name].set_ylabel('Percentage of Total Time')
      axes['percent_total'][ht_name].yaxis.set_major_formatter(FormatStrFormatter('%3.0f %%'))
      axes['percent_total'][ht_name].set_xlim([0.5, my_num_nodes + 0.5])

    if PLOTS_TO_GENERATE['flat_mpi_factor']:
      axes['flat_mpi_factor'][ht_name].set_ylabel('Ratio of Runtime to Flat MPI Time')
      axes['flat_mpi_factor'][ht_name].set_xlim([0.5, my_num_nodes + 0.5])

    # if this is the last row, then display x axes labels
    if row_idx == (len(ht_names) - 1):
      axes['raw_data'][ht_name].set_xlabel("Number of Nodes")
      axes['raw_data'][ht_name].set_xticks(my_ticks)
      axes['raw_data'][ht_name].set_xticklabels(my_nodes, rotation=45)
      axes['raw_data'][ht_name].set_title('{HT_LABEL}={HT_NUM:.0f}'.format(HT_LABEL=HYPER_THREAD_LABEL,
                                                                           HT_NUM=ht_name))

      if PLOTS_TO_GENERATE['percent_total']:
        axes['percent_total'][ht_name].set_xlabel("Number of Nodes")
        axes['percent_total'][ht_name].set_xticks(my_ticks)
        axes['percent_total'][ht_name].set_xticklabels(my_nodes, rotation=45)
        axes['percent_total'][ht_name].set_title('{HT_LABEL}={HT_NUM:.0f}'.format(HT_LABEL=HYPER_THREAD_LABEL,
                                                                                  HT_NUM=ht_name))

      if PLOTS_TO_GENERATE['flat_mpi_factor']:
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

      if PLOTS_TO_GENERATE['percent_total']:
        title = 'Percentage of Total Time\n({HT_LABEL}={HT_NUM:.0f})'.format(HT_LABEL=HYPER_THREAD_LABEL,
                                                                             HT_NUM=ht_name)
        axes['percent_total'][ht_name].set_title(title)
        axes['percent_total'][ht_name].set_xticks([])

      if PLOTS_TO_GENERATE['flat_mpi_factor']:
        title = 'Ratio of Runtime to Flat MPI Time\n({HT_LABEL}={HT_NUM:.0f})'.format(HT_LABEL=HYPER_THREAD_LABEL,
                                                                                      HT_NUM=ht_name)
        axes['flat_mpi_factor'][ht_name].set_title(title)
        axes['flat_mpi_factor'][ht_name].set_xticks([])

    # otherwise, this is a middle plot, show a truncated title
    else:
      title = '{HT_LABEL}={HT_NUM:.0f}'.format(HT_LABEL=HYPER_THREAD_LABEL,
                                               HT_NUM=ht_name)
      axes['raw_data'][ht_name].set_title(title)
      axes['raw_data'][ht_name].set_xticks([])

      if PLOTS_TO_GENERATE['percent_total']:
        title = '{HT_LABEL}={HT_NUM:.0f}'.format(HT_LABEL=HYPER_THREAD_LABEL,
                                                 HT_NUM=ht_name)
        axes['percent_total'][ht_name].set_title(title)
        axes['percent_total'][ht_name].set_xticks([])

      if PLOTS_TO_GENERATE['flat_mpi_factor']:
        title = '{HT_LABEL}={HT_NUM:.0f}'.format(HT_LABEL=HYPER_THREAD_LABEL,
                                                 HT_NUM=ht_name)
        axes['flat_mpi_factor'][ht_name].set_title(title)
        axes['flat_mpi_factor'][ht_name].set_xticks([])

  annotate_file_names(figures['independent'])

  # add a suptitle and configure the legend for each figure
  figures['composite'].suptitle(simple_title, fontsize=18)
  # we plot in a deterministic fashion, and so the order is consistent among all
  # axes plotted. This allows a single legend that is compatible with all plots.
  handles, labels = axes['raw_data'][ht_names[0]].get_legend_handles_labels()
  if PLOT_LEGEND:
    figures['composite'].legend(handles, labels,
                                title="Procs per Node x Cores per Proc",
                                loc='lower center', ncol=ndecomps, bbox_to_anchor=(0.5, 0.0))
  figures['composite'].tight_layout()
  # this must be called after tight layout
  figures['composite'].subplots_adjust(top=0.85, bottom=0.15)

  # add legends
  for column_name in figures['independent']:
    # # Why?
    # if column_name not in ['raw_data', 'bl_perc_diff']:
    #   continue
    for fig_name, fig in figures['independent'][column_name].items():
      if PLOT_LEGEND:
        fig.legend(handles, labels,
                   title="Procs per Node x Cores per Proc",
                   loc='lower center', ncol=ndecomps, bbox_to_anchor=(0.5, 0.0))
        # add space since the titles are typically large
        fig.subplots_adjust(bottom=0.20)
      # use a tight layout, this can trim padding, it can also be a pain
      fig.tight_layout()

  # save the free axis version of the figures
  save_figures(figures,
               filename='{basename}'.format(basename=simple_fname),
               sub_dir='free-yaxis',
               close_figure=False)

  if write_latex_and_csv:
    total_df.to_csv('{path}/{fname}.csv'.format(path=PLOT_DIRS['csv'],
                                                fname=simple_fname))
    total_df.to_latex('{path}/{fname}.tex'.format(path=PLOT_DIRS['latex'],
                                                  fname=simple_fname),
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
                 filename='{basename}'.format(basename=simple_fname),
                 sub_dir='free-yaxis-best',
                 close_figure=False)

  # if we want consistent axes by column, then enforce that here.
  if HT_CONSISTENT_YAXES:
    enforce_consistent_ylims(figures, axes)

  # save the figures with the axes shared
  save_figures(figures,
               filename=simple_fname,
               sub_dir='shared_ht',
               close_figure=False)

  if ANNOTATE_BEST:
    for column_name in figures['independent']:
      if column_name == 'speedup' or column_name == 'efficiency':
        annotate_best_column(figures=figures['independent'][column_name],
                             axes_name_to_destroy=ht_names[0],
                             objective='min')

    # save the figures with the axes shared
    save_figures(figures,
                 filename='{fname}'.format(fname=simple_fname),
                 sub_dir='overall',
                 composite=False,
                 independent=True,
                 independent_names=ht_names[0])

  # y axes overrides
  if DO_YMAX_OVERRIDE or DO_YMIN_OVERRIDE:
    enforce_override_ylims(figures, axes)

    # save the override axis version of the figures
    save_figures(figures,
                 filename='{basename}'.format(basename=simple_fname),
                 sub_dir='or',
                 close_figure=False)

  close_figures(figures)

  decomp_labels = total_df['decomp_label'].unique()
  ind = np.arange(len(decomp_labels))
  agg_groups = total_df.groupby('nodes')
  for num_nodes, node_group in agg_groups:
    node_group.loc[node_group['decomp_label'] == 'flat_mpi', 'maxT'] = 0.0
    for ht_name, ht_group in node_group.groupby('HT'):
      try:
        bl_data = ht_group['bl_max']
        data = ht_group['maxT']
        S_data = ht_group['bl_speedup_max'].tolist()

        print(S_data)
        fig, ax = plt.subplots()
        ax.bar(ind, bl_data, 0.35)
        data_rects = ax.bar(ind + 0.35, data, 0.35)
        """
        Attach a text label above each bar displaying speedup
        """
        for i in range(len(decomp_labels) - 1):
          if ~np.isfinite(S_data[i]):
            continue
          print(S_data[i])
          rect = data_rects[i]
          height = rect.get_height()
          ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                  '{0:.1f}'.format(S_data[i]),
                  ha='center', va='bottom')

        ax.set_ylabel('Time (s) (average)')
        ax.set_title('{} nodes'.format(num_nodes))
        ax.set_xticks(ind + 0.35 / 2)
        ax.set_xticklabels(decomp_labels)
        ax.legend(['Prior', 'LTG'])
        fullpath = 'max_only/nodes/{}-{}-{}.{}'.format(simple_fname,
                                                       num_nodes,
                                                       ht_name,
                                                       IMG_FORMAT)
        fig.savefig(fullpath,
                    format=IMG_FORMAT,
                    dpi=IMG_DPI)
        plt.close(fig)
        print('wrote: ', fullpath)
      except:
        print('failed bar chart for nodes=', num_nodes, ' and HT=', ht_name)
        pass

  total_df['Timer name'] = my_tokens['Timer Name']
  return total_df


def annotate_file_names(figures):
  if ANNOTATE_DATASET_FILENAMES:
    for column_name in figures:
      for fig_name, fig in figures[column_name].items():
        fig.gca().annotate('Data: ' + DATASET_FILE,
                           xy=(0, 1), xycoords='axes fraction', fontsize=14,
                           xytext=(0, 0), textcoords='offset points',
                           ha='left', va='top')
        if HAVE_BASELINE:
          fig.gca().annotate('Baseline: ' + BASELINE_DATASET_FILE,
                             xy=(1, 1), xycoords='axes fraction', fontsize=14,
                             xytext=(0, 0), textcoords='offset points',
                             ha='right', va='top')


###########################################################################################
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
    df['type'] = data_type
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
      best_overall = df.loc[df.groupby(['type', 'x'])['y'].idxmin()].groupby('type')
      worst_overall = df.loc[df.groupby(['type', 'x'])['y'].idxmax()].groupby('type')
    except:
      print('Had a problem annotating the best. Skipping this axes')
      return
  elif objective == 'max':
    try:
      best_overall = df.loc[df.groupby(['type', 'x'])['y'].idxmax()].groupby('type')
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
    data_labels = [
      '{decomp}x{ht}'.format(decomp=row['label'], ht=row['ax_id']) if row['label'] != 'flat_mpi' else '{decomp}'.format(
        decomp=row['label']) for index, row in df.iterrows()]

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

    # ax.plot(df['x'], df['y'], linestyle=linestyles[plot_type], label='worst-{}'.format(plot_type))

    # label the times
    data_labels = [
      '{decomp}x{ht}'.format(decomp=row['label'], ht=row['ax_id']) if row['label'] != 'flat_mpi' else '{decomp}'.format(
        decomp=row['label']) for index, row in df.iterrows()]

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
  PLOT_BEST_MIN_ONLY = True
  df['ax_id'] = df['ax_id'].astype(np.int32)

  if objective == 'min':
    try:
      best_overall = df.loc[df.groupby(['type', 'x'])['y'].idxmin()].groupby('type')
      best_by_id = df.loc[df.groupby(['type', 'ax_id', 'x'])['y'].idxmin()].groupby(['type', 'ax_id'])
    except:
      print('Had a problem annotating the best. Skipping this axes')
      return
  elif objective == 'max':
    try:
      best_overall = df.loc[df.groupby(['type', 'x'])['y'].idxmax()].groupby('type')
      best_by_id = df.loc[df.groupby(['type', 'ax_id', 'x'])['y'].idxmax()].groupby(['type', 'ax_id'])
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
    data_labels = [
      '{decomp}x{ht}'.format(decomp=row['label'], ht=row['ax_id']) if row['label'] != 'flat_mpi' else '{decomp}'.format(
        decomp=row['label']) for index, row in df.iterrows()]

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
  PLOT_BEST_MIN_ONLY = True
  all_data['ax_id'] = all_data['ax_id'].astype(np.int32)
  print(all_data.dtypes)

  if objective == 'min':
    best_overall = all_data.loc[all_data.groupby(['type', 'x'])['y'].idxmin()].groupby('type')
    best_by_id = all_data.loc[all_data.groupby(['type', 'ax_id', 'x'])['y'].idxmin()].groupby(['type', 'ax_id'])
  elif objective == 'max':
    best_overall = all_data.loc[all_data.groupby(['type', 'x'])['y'].idxmax()].groupby('type')
    best_by_id = all_data.loc[all_data.groupby(['type', 'ax_id', 'x'])['y'].idxmax()].groupby(['type', 'ax_id'])
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
                          write_latex_and_csv=True,
                          baseline_group=None,
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

  :param write_latex_and_csv: TODO for strong.
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
  fig_size_width_inflation = 1.0

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
                      yvalues=my_agg_times[QUANTITY_OF_INTEREST_MAX],
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
      figures['independent']['percent_total'][ht_name].gca().set_title(
        '{}\n({HT_LABEL}={HT_NUM:.0f})'.format(simple_title,
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
      figures['independent']['flat_mpi_factor'][ht_name].gca().set_title(
        '{}\n({HT_LABEL}={HT_NUM:.0f})'.format(simple_title,
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
        axes['efficiency'][ht_name].set_title(
          'Efficiency\n({HT_LABEL}={HT_NUM:.0f})'.format(HT_LABEL=HYPER_THREAD_LABEL,
                                                         HT_NUM=ht_name))
        axes['efficiency'][ht_name].set_xticks([])
      if show_percent_total:
        axes['percent_total'][ht_name].set_title(
          'Percentage of Total Time\n({HT_LABEL}={HT_NUM:.0f})'.format(HT_LABEL=HYPER_THREAD_LABEL,
                                                                       HT_NUM=ht_name))
        axes['percent_total'][ht_name].set_xticks([])
      if show_factor:
        axes['flat_mpi_factor'][ht_name].set_title(
          'Ratio of Runtime to Flat MPI Time\n({HT_LABEL}={HT_NUM:.0f})'.format(HT_LABEL=HYPER_THREAD_LABEL,
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
      # handles, labels = fig.gca().get_legend_handles_labels()
      fig.legend(handles, labels,
                 title="Procs per Node x Cores per Proc",
                 loc='lower center', ncol=ndecomps, bbox_to_anchor=(0.5, 0.0))
      # add space since the titles are typically large
      fig.subplots_adjust(bottom=0.20)

  # save the free axis version of the figures
  save_figures(figures,
               filename='{basename}-free-yaxis'.format(basename=simple_fname),
               close_figure=False)

  # produce a legend for the objects in the other figure
  pylab.figlegend(*ax.get_legend_handles_labels(), loc='upper left')

  if ANNOTATE_BEST:
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

  if ANNOTATE_BEST:
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
                 max_num_nodes=1000000,
                 min_procs_per_node=1,
                 max_procs_per_node=1000000,
                 timer_name_mappings={},
                 demangle_muelu=True):
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
  if VERBOSITY & 1:
    print('Reading {}'.format(dataset_filename))
  # write the total dataset out, index=False, because we do not drop it above
  dataset = pd.read_csv(dataset_filename, low_memory=False)

  if demangle_muelu:
    import TeuchosTimerUtils as TTU
    dataset = TTU.demange_muelu_timer_names_df(dataset)

  if VERBOSITY & 1:
    print('Read csv complete')

  if timer_name_mappings:
    if VERBOSITY & 1:
      print('Applying timer label remapping')
    dataset = adjust_timer_names(dataset, timer_name_mappings)

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
  index_columns = SFP.getIndexColumns(execspace_name='OpenMP')

  new_columns = list(dataset)
  index_columns = list(set(index_columns).intersection(set(new_columns)))

  dataset = dataset.set_index(keys=index_columns,
                              drop=False,
                              verify_integrity=True)
  if VERBOSITY & 1:
    print('Verified index')

  # optionally restrict the data processed
  # Elasticity data is incomplete.
  restriction_query = '(problem_type != \"Elasticity3D\") & ' \
                      '(num_nodes >= {min_num_nodes}) & ' \
                      '(num_nodes <= {max_num_nodes}) & ' \
                      '(procs_per_node >= {min_procs_per_node}) & ' \
                      '(procs_per_node <= {max_procs_per_node}) & ' \
                      '(prec_attributes != \"-no-repartition\")' \
                      ''.format(min_num_nodes=min_num_nodes,
                                max_num_nodes=max_num_nodes,
                                min_procs_per_node=min_procs_per_node,
                                max_procs_per_node=max_procs_per_node)

  dataset = dataset.query(restriction_query)
  # dataset = dataset[dataset['procs_per_node'].isin([64, 4])]
  # dataset = dataset[~dataset['num_nodes'].isin([4, 32, 768])]
  if VERBOSITY & 1:
    print('Restricted dataset')

  dataset = dataset.fillna(value='None')

  # sort
  # dataset.sort_values(inplace=True,
  #                     by=SFP.getIndexColumns(execspace_name='OpenMP'))
  dataset = dataset.sort_index()
  if VERBOSITY & 1:
    print('Sorted')

  # remove the timers the driver adds
  driver_dataset = dataset[dataset['Timer Name'].isin(['0 - Total Time',
                                                       '1 - Reseting Linear System',
                                                       '2 - Adjusting Nullspace for BlockSize',
                                                       '3 - Constructing Preconditioner',
                                                       '4 - Constructing Solver',
                                                       '5 - Solve'])]
  if VERBOSITY & 1:
    print('Gathered driver timers')

  # remove the timers the driver adds
  dataset = dataset[~dataset['Timer Name'].isin(['0 - Total Time',
                                                 '1 - Reseting Linear System',
                                                 '2 - Adjusting Nullspace for BlockSize',
                                                 '3 - Constructing Preconditioner',
                                                 '4 - Constructing Solver',
                                                 '5 - Solve'])]
  if VERBOSITY & 1:
    print('Removed driver timers')

  # reindex
  # set the index, verify it, and sort
  dataset = dataset.set_index(keys=index_columns,
                              drop=False,
                              verify_integrity=True)

  driver_dataset = driver_dataset.set_index(keys=index_columns,
                                            drop=False,
                                            verify_integrity=True)
  if VERBOSITY & 1:
    print('Rebuilt truncated index')

  dataset = dataset.sort_index(by=index_columns)
  driver_dataset = driver_dataset.sort_index(by=index_columns)

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
  # life would be easier if we cleaned up the indexing.
  # use the index to verify, then reindex so we always use columns
  tmp = dataset.groupby(['Timer Name',
                         'problem_type',
                         'solver_name',
                         'solver_attributes',
                         'prec_name',
                         'prec_attributes'])[rank_by_column_name].sum()

  ordered_timers = tmp.reset_index().sort_values(by=rank_by_column_name,

                                                 ascending=False)['Timer Name'].tolist()

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
  spmv_groupby_columns.remove('numsteps')

  spmv_only_data = dataset[dataset['Timer Name'].str.match(timer_name_re_str)]
  if spmv_only_data.empty:
    spmv_only_data = dataset[dataset['Timer Name'].str.match('OPT::Apply')]
    if VERBOSITY & 1024:
      print('Using OPT::Apply Data')

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
                   expected_sub_dirs,
                   baseline_group=None,
                   baseline_dr_df=None,
                   **kwargs):
  if scaling_study_type == 'weak':
    df = plot_composite_weak(composite_group=composite_group,
                             baseline_group=baseline_group,
                             baseline_dr_df=baseline_dr_df,
                             my_nodes=my_nodes,
                             my_ticks=my_ticks,
                             numbered_plots_idx=numbered_plots_idx,
                             driver_df=driver_df,
                             expected_sub_dirs=expected_sub_dirs,
                             kwargs=kwargs)

  elif scaling_study_type == 'strong':
    df = plot_composite_strong(composite_group=composite_group,
                               baseline_group=baseline_group,
                               baseline_dr_df=baseline_dr_df,
                               my_nodes=my_nodes,
                               my_ticks=my_ticks,
                               numbered_plots_idx=numbered_plots_idx,
                               driver_df=driver_df)
  else:
    print('Unknown scaling study type: ', scaling_study_type)
    raise RuntimeError

  return df


###############################################################################
def get_plottable_dataset(full_dataset,
                          total_time_dataset,
                          comparable_timers,
                          scaling_type,
                          expected_nodes):
  if scaling_type == 'strong':
    compute_strong_scaling_terms = True
  else:
    compute_strong_scaling_terms = False

  # we use the expected nodes and ticks to ensure all plots have the same axes
  my_ticks = np.arange(start=1, stop=len(expected_nodes) + 1, step=1, dtype=int)

  # group by timer and attributes, but keep all decomps together
  omp_groupby_columns = SFP.getMasterGroupBy(execspace_name='OpenMP', scaling_type=scaling_type)
  omp_groupby_columns.remove('procs_per_node')
  omp_groupby_columns.remove('cores_per_proc')
  # be very careful, there will be duplicate decomp groups
  omp_groupby_columns.remove('execspace_name')

  # find the index of the Timer Name field in the groupBy clause
  timer_name_index = omp_groupby_columns.index('Timer Name')

  result_df = pd.DataFrame()

  try:
    if isinstance(comparable_timers, str):
      dataset = full_dataset[full_dataset['Timer Name'] == comparable_timers]
    else:
      dataset = full_dataset[full_dataset['Timer Name'].isin(comparable_timers)]
  except:
    import sys
    print('Failed to gather comparable_timers', sys.exc_info()[0])
    return result_df

  dataset = dataset.reset_index(drop=True)
  total_time_dataset = total_time_dataset.reset_index(drop=True)

  # group data into comparable chunks
  composite_groups = dataset.groupby(omp_groupby_columns)
  # be very careful, there will be duplicate decomp groups
  omp_groupby_columns.remove('Timer Name')
  total_time_dataset_composite_groups = total_time_dataset.groupby(omp_groupby_columns)

  total_time_timer_names = total_time_dataset['Timer Name'].unique()

  if len(total_time_timer_names) > 1:
    raise (ValueError(
      'Total time dataset contains too many timers. We expect only one. Timers present:{}'.format(
        ','.join(total_time_timer_names))))

  # process each comparable chunk:
  for composite_group_name, composite_group in composite_groups:
    # determine the flat MPI time
    tmp_df = add_flat_mpi_data(composite_group, allow_baseline_override=True)

    # construct an index into the driver_df by changing the timer label to match the driver's
    # global total label
    total_time_composite_group_name = list(composite_group_name)
    # delete the timer name, it is indexed by the order of the groupby clause
    del total_time_composite_group_name[timer_name_index]
    # query the matching data
    total_time_dataset_composite_group = total_time_dataset_composite_groups.get_group(
      tuple(total_time_composite_group_name))

    decomp_groups = tmp_df.groupby(['procs_per_node', 'cores_per_proc', 'execspace_name'])
    total_time_dataset_decomp_groups = total_time_dataset_composite_group.groupby(
      ['procs_per_node', 'cores_per_proc', 'execspace_name'])

    # the number of HT combos we have
    ht_names = tmp_df['threads_per_core'].sort_values(ascending=True).unique()

    for decomp_group_name, decomp_group in decomp_groups:
      procs_per_node = int(decomp_group_name[0])
      cores_per_proc = int(decomp_group_name[1])
      execution_space = decomp_group_name[2]

      # iterate over HTs
      ht_groups = decomp_group.groupby('threads_per_core')
      # look up this decomp in the total_time dataset
      total_time_dataset_ht_groups = total_time_dataset_decomp_groups.get_group(decomp_group_name).groupby(
        'threads_per_core')

      for ht_name in ht_names:

        if execution_space == 'Serial':
          if ht_name == ht_names[0]:
            # no HT for serial
            total_time_dataset_ht_group = total_time_dataset_ht_groups.get_group(ht_name)
          else:
            continue
        else:
          total_time_dataset_ht_group = total_time_dataset_ht_groups.get_group(ht_name)

        # label this decomp
        if execution_space == 'OpenMP':
          decomp_label = "{procs_per_node}x{cores_per_proc}x{HT}".format(procs_per_node=procs_per_node,
                                                                         cores_per_proc=cores_per_proc,
                                                                         HT=ht_name)
        elif execution_space == 'Serial':
          decomp_label = 'flat_mpi'

        ht_group = ht_groups.get_group(ht_name)

        my_agg_times = pd.DataFrame(columns=['num_nodes', 'ticks'], data=np.column_stack((expected_nodes, my_ticks)))

        my_agg_times = compute_scaling_metrics(plottable_df=my_agg_times,
                                               dataset=ht_group,
                                               total_time_dataset=total_time_dataset_ht_group,
                                               decomp_label=decomp_label,
                                               compute_strong_terms=compute_strong_scaling_terms)
        my_agg_times['procs_per_node'] = procs_per_node
        my_agg_times['cores_per_proc'] = cores_per_proc
        my_agg_times['threads_per_core'] = ht_name
        my_agg_times['execspace_name'] = execution_space

        result_df = pd.concat([result_df, my_agg_times])

  # check the integrity of this aggregate data
  result_df = result_df.set_index(
    ['num_nodes', 'procs_per_node', 'cores_per_proc', 'threads_per_core', 'execspace_name'],
    drop=True)
  result_df = result_df.reset_index(drop=False)
  return result_df


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
  my_ticks = np.arange(start=1, stop=my_num_nodes + 1, step=1, dtype=int)

  if VERBOSITY & 512:
    print(my_num_nodes)
    print(my_nodes)
    print(my_ticks)

  if len(my_nodes) != len(my_ticks):
    print('Length of ticks and nodes are different')
    exit(-1)

  # restrict the timers
  '''
    A*P-1: All I&X
    A*P-1: All Multiply
    A*P-1: All Setup
    A*P-1: I&X Alloc
    A*P-1: I&X Collective-0
    A*P-1: I&X Import-2
    A*P-1: I&X Import-3
    A*P-1: I&X Import-4
    A*P-1: I&X RemoteMap
    A*P-1: M5 Cmap
    A*P-1: Core
    A*P-1: ESFC
    A*P-1: Sort
    
    # apply this renaming

    A*P-1: All I&X;TpetraExt MueLu::A*P-1: MMM All I&X
    A*P-1: All Multiply;TpetraExt MueLu::A*P-1: MMM All Multiply
    A*P-1: All Setup;TpetraExt MueLu::A*P-1: MMM All Setup
    A*P-1: I&X Alloc;TpetraExt MueLu::A*P-1: MMM I&X Alloc
    A*P-1: I&X Collective-0;TpetraExt MueLu::A*P-1: MMM I&X Collective-0
    A*P-1: I&X Import-2;TpetraExt MueLu::A*P-1: MMM I&X Import-2
    A*P-1: I&X Import-3;TpetraExt MueLu::A*P-1: MMM I&X Import-3
    A*P-1: I&X Import-4;TpetraExt MueLu::A*P-1: MMM I&X Import-4
    A*P-1: I&X RemoteMap;petraExt MueLu::A*P-1: MMM I&X RemoteMap
    A*P-1: M5 Cmap;petraExt MueLu::A*P-1: MMM M5 Cmap
    A*P-1: Core;TpetraExt MueLu::A*P-1: MMM Newmatrix OpenMPCore;TpetraExt MueLu::A*P-1: MMM Newmatrix SerialCore
    A*P-1: ESFC;TpetraExt MueLu::A*P-1: MMM Newmatrix OpenMPESFC;TpetraExt MueLu::A*P-1: MMM Newmatrix ESFC
    A*P-1: Sort;TpetraExt MueLu::A*P-1: MMM Newmatrix OpenMPSort;TpetraExt MueLu::A*P-1: MMM Newmatrix Final Sort
    
    # high level
    MueLu: RAPFactory: Computing Ac (level=1)
    MueLu: SaPFactory: Prolongator smoothing (level=1)
    MueLu: TentativePFactory: Build (level=1)
    MueLu: UncoupledAggregationFactory: Build (level=0)
    MueLu: CoordinatesTransferFactory: Build (level=1)
    MueLu: CoalesceDropFactory: Build (level=1)
    MueLu: CoarseMapFactory: Build (level=1)
    MueLu: AmalgamationFactory: Build (level=1)
    MueLu: NullspaceFactory: Nullspace factory (level=1)
    
    # ESFC
    'Tpetra MueLu::A*P-1: ESFC-G-Maps'
    'Tpetra MueLu::A*P-1: ESFC-G-Setup'
    'Tpetra MueLu::A*P-1: ESFC-G-cGC (noconst)'
    'Tpetra MueLu::A*P-1: ESFC-G-cIS'
    'Tpetra MueLu::A*P-1: ESFC-G-fLG'
    'Tpetra MueLu::A*P-1: ESFC-G-mIXcheckE'
    'Tpetra MueLu::A*P-1: ESFC-G-mIXcheckI'
    'Tpetra MueLu::A*P-1: ESFC-G-mIXmake'
    'Tpetra MueLu::A*P-1: ESFC-M-Graph'
    'Tpetra MueLu::A*P-1: ESFC-M-cGC'
    'Tpetra MueLu::A*P-1: ESFC-M-cIS'
    'Tpetra MueLu::A*P-1: ESFC-M-fLGAM'
  '''
  # ESFC
  restricted_timers = [
    'Tpetra MueLu::A*P-1: ESFC-M-Graph',
    'Tpetra MueLu::A*P-1: ESFC-G-Setup',
    'Tpetra MueLu::A*P-1: ESFC-G-Maps',
    'Tpetra MueLu::A*P-1: ESFC-G-mIXcheckE',
    'Tpetra MueLu::A*P-1: ESFC-G-mIXcheckI',
    'Tpetra MueLu::A*P-1: ESFC-G-mIXmake',
    'Tpetra MueLu::A*P-1: ESFC-G-cGC (noconst)',
    'Tpetra MueLu::A*P-1: ESFC-G-cGC (const)',
    'Tpetra MueLu::A*P-1: ESFC-G-fLG',
    'Tpetra MueLu::A*P-1: ESFC-G-cIS',
    'Tpetra MueLu::A*P-1: ESFC-M-cGC',
    'Tpetra MueLu::A*P-1: ESFC-M-fLGAM',
    'Tpetra MueLu::A*P-1: ESFC-M-cIS',
  ]
  find_replace = {
    'Tpetra MueLu::A*P-1: ': '',
  }
  import collections
  timer_groups = collections.OrderedDict()
  timer_groups['Tpetra MueLu::A*P-1: ESFC-M-Graph'] =\
    [
      'Tpetra MueLu::A*P-1: ESFC-G-Setup',
      'Tpetra MueLu::A*P-1: ESFC-G-Maps',
      'Tpetra MueLu::A*P-1: ESFC-G-mIXcheckE',
      'Tpetra MueLu::A*P-1: ESFC-G-mIXcheckI',
      'Tpetra MueLu::A*P-1: ESFC-G-mIXmake',
      'Tpetra MueLu::A*P-1: ESFC-G-cGC (noconst)',
      'Tpetra MueLu::A*P-1: ESFC-G-cGC (const)',
      'Tpetra MueLu::A*P-1: ESFC-G-fLG',
      'Tpetra MueLu::A*P-1: ESFC-G-cIS']
  timer_groups['Tpetra MueLu::A*P-1: ESFC-M-cGC']   = []
  timer_groups['Tpetra MueLu::A*P-1: ESFC-M-fLGAM'] = []
  timer_groups['Tpetra MueLu::A*P-1: ESFC-M-cIS']   = []

  '''
  # high level
  restricted_timers = [
    'MueLu: RAPFactory: Computing Ac (level=1)',
    'MueLu: SaPFactory: Prolongator smoothing (level=1)',
    'MueLu: TentativePFactory: Build (level=1)',
    'MueLu: UncoupledAggregationFactory: Build (level=0)',
    'MueLu: CoordinatesTransferFactory: Build (level=1)',
    'MueLu: CoalesceDropFactory: Build (level=1)',
    'MueLu: CoarseMapFactory: Build (level=1)',
    'MueLu: AmalgamationFactory: Build (level=1)',
    'MueLu: NullspaceFactory: Nullspace factory (level=1)'
  ]
  find_replace = {
    'MueLu: ': '',
    'Factory: ': '',
    'Nullspace: ': '',
    'level=': ''
  }
  '''

  '''
  restricted_timers = [
    'A*P-1: All I&X',
    'A*P-1: All Multiply',
    'A*P-1: All Setup',
    'A*P-1: I&X Alloc',
    'A*P-1: I&X Collective-0',
    'A*P-1: I&X Import-2',
    'A*P-1: I&X Import-3',
    'A*P-1: I&X Import-4',
    'A*P-1: I&X RemoteMap',
    'A*P-1: M5 Cmap',
    'A*P-1: Core',
    'A*P-1: ESFC',
    'A*P-1: Sort']
    
  find_replace = {
    'A*P-1: ' : ''
  }
  '''
  # add nice labels for the various thread combos
  dataset['decomp_label'] = dataset['procs_per_node'].map(str)  + 'x' + dataset['cores_per_proc'].map(str)
  dataset.loc[ dataset['execspace_name'] == 'Serial', 'decomp_label'] = 'flat_mpi'
  decomp_labels = ['flat_mpi', '64x1', '32x2' , '16x4', '8x8', '4x16']

  ndecomps = len(decomp_labels)
  print(decomp_labels)

  # grab the total time
  total_timings = dataset[ dataset['Timer Name'] == 'MueLu: Hierarchy: Setup(total)' ].copy()

  # construct the ordered timers
  ordered_timers = []
  for parent, children in timer_groups.items():
    ordered_timers.append(parent)
    if children:
      for child in children:
        ordered_timers.append(child)

  # restrict the data to the timers we care about
  dataset = dataset[ dataset['Timer Name'].isin(restricted_timers) ]

  ht_nums = dataset['threads_per_core'].unique()

  # set the index to what we need to do a join for flat MPI data
  dataset = dataset.set_index(['num_nodes', 'Timer Name', 'threads_per_core'])
  # restrict to the flat MPI data
  flat_mpi_data = dataset[dataset['decomp_label'] == 'flat_mpi']

  # replicate the data to other HT counts
  for threads_per_core in ht_nums:
    if threads_per_core == 1:
      continue
    print(threads_per_core)
    dummy_df = flat_mpi_data.copy()
    dummy_df = dummy_df.reset_index()
    dummy_df['threads_per_core'] = threads_per_core
    dummy_df = dummy_df.set_index(['num_nodes', 'Timer Name', 'threads_per_core'])
    dataset = pd.concat([dataset, dummy_df])

  # need to do this, because we modified the underlying dataframe above
  flat_mpi_data = dataset[dataset['decomp_label'] == 'flat_mpi']
  # now join against the data, and add flat MPI info
  dataset = dataset.join(flat_mpi_data[['minT', 'maxT', 'meanT', 'numsteps']],
                         rsuffix='_flat_mpi')

  dataset = dataset.reset_index()
  dataset['normalized_minT'] = dataset['minT'] / dataset['minT_flat_mpi']

  dataset.to_csv('blah.csv', index=True)

  # foreach node count, for each HT count
  data_by_node_count_groups = dataset.groupby('num_nodes')
  for node_count, node_group in data_by_node_count_groups:

    y_max = np.power(2, np.ceil(np.log2(node_group['normalized_minT'].max())))
    y_min = np.power(2.0, -4)
    ht_groups = node_group.groupby('threads_per_core')
    for threads_per_core, ht_group in ht_groups:
      my_decomps = ht_group['decomp_label'].unique().tolist()
      if 'flat_mpi' not in my_decomps:
        print('No Flat MPI data... skipping: ', node_count, threads_per_core)
        continue

      frozen_decomps = frozenset(my_decomps)
      decomp_labels = [x for x in decomp_labels if x in frozen_decomps]
      # do this every time to get the colors right if some are missing
      decomp_colors = [DECOMP_COLORS[x] for x in decomp_labels]

      df = ht_group.pivot(index='Timer Name', columns='decomp_label', values='normalized_minT')
      df = df[decomp_labels]

      df = df.reindex(ordered_timers)
      xcoords = []
      prior_coord = 0.0
      bar_width = 1.8
      bar_space = 2.0
      group_gap = 0.5
      vline_coords = []
      for parent, children in timer_groups.items():
        if prior_coord != 0.0:
          vline_coords.append(prior_coord)
          prior_coord += group_gap
        xcoords.append(prior_coord + bar_space)
        prior_coord = xcoords[-1]
        if children:
          for child in children:
            xcoords.append(prior_coord + bar_space)
            prior_coord = xcoords[-1]

      fname = '{nodes}_{ht}.{fmt}'.format(nodes=node_count, ht=threads_per_core, fmt=IMG_FORMAT)

      h = plt.figure(figsize=(21.5, 12))
      ax = h.add_subplot(111)

      df.plot(kind='bar', rot=45, legend=False, width=0.8, color=decomp_colors, ax=ax, fontsize=14)
      for patch in ax.patches:
        print(patch)

      ax.set_ylabel("SLOW DOWN! Over Flat MPI", fontsize=16)
      ax.set_xlabel("")
      print(ax.get_xticks())
      labels = ax.get_xticklabels()
      labels = [ticklabel.get_text() for ticklabel in labels]
      print(labels)
      for k,v in find_replace.items():
        labels = [ticklabel.replace(k, v) for ticklabel in labels]

      ax.set_xticklabels(labels, fontsize=16)
      ax.set_title('MMM, A*P-1, (Nodes={nodes}, HT={ht})'.format(nodes=node_count,
                                                                 ht=threads_per_core))

      ax.set_ylim([y_min, y_max])
      ax.set_yscale('log', basey=2)
      print(y_max)
      h.legend(title="Procs per Node x Cores per Proc",
               loc='upper right', ncol=ndecomps, fontsize=14)
      h.subplots_adjust(bottom=0.25)
      h.savefig(fname, format=IMG_FORMAT, dpi=IMG_DPI)
      # plt.show(block=True)
      plt.close(h)

      print('Wrote: ', fname)


###############################################################################
def main():
  global VERBOSITY
  global BE_QUIET
  global SPMV_FIG
  global PLOT_DIRS
  global IMG_FORMAT
  global IMG_DPI
  global FORCE_REPLOT
  global QUANTITY_OF_INTEREST
  global QUANTITY_OF_INTEREST_COUNT
  global QUANTITY_OF_INTEREST_MIN
  global QUANTITY_OF_INTEREST_MIN_COUNT
  global QUANTITY_OF_INTEREST_MAX
  global QUANTITY_OF_INTEREST_MAX_COUNT
  global QUANTITY_OF_INTEREST_THING
  global QUANTITY_OF_INTEREST_THING_COUNT

  global MIN_LINESTYLE
  global MAX_LINESTYLE

  global BASELINE_LINESTYLE
  global SHADE_BASELINE_COMPARISON

  global MIN_MARKER
  global MAX_MARKER
  global STANDALONE_FONT_SIZE

  global MIN_STYLE
  global MAX_STYLE

  global PLOT_MIN
  global PLOT_MAX
  global SMOOTH_OUTLIERS
  global HT_CONSISTENT_YAXES
  global ANNOTATE_BEST
  global PLOT_LEGEND
  global PLOT_TITLES
  global PLOT_TITLES_FULL
  global PLOT_TITLES_HT
  global PLOT_TITLES_NONE

  global DO_YMIN_OVERRIDE
  global DO_YMAX_OVERRIDE
  global YMIN_OVERRIDE
  global YMAX_OVERRIDE
  global DO_NORMALIZE_Y
  global HAVE_BASELINE
  global BASELINE_DATASET_DF
  global BASELINE_DRIVER_DF

  global EXPECTED_BASELINE_SPEEDUP
  global BASELINE_DATASET_FILE
  global DATASET_FILE
  global PLOTS_TO_GENERATE
  global ANNOTATE_DATASET_FILENAMES

  global BASELINE_DECOMP
  global BASELINE_DECOMP_TUPLE

  global HYPER_THREAD_LABEL
  global DECOMP_COLORS

  global AVERAGE_BY

  sanity_check()

  # Process input
  _arg_options = docopt(__doc__)
  VERBOSITY = int(_arg_options['--verbose'])
  BE_QUIET = _arg_options['--quiet']
  IMG_FORMAT = _arg_options['--img_format']
  IMG_DPI = int(_arg_options['--img_dpi'])

  dataset_filename = _arg_options['--dataset']
  study_type = _arg_options['--study']
  max_num_nodes = _arg_options['--max_nodes']
  min_num_nodes = _arg_options['--min_nodes']
  max_procs_per_node = _arg_options['--max_procs_per_node']
  min_procs_per_node = _arg_options['--min_procs_per_node']
  scaling_study_type = _arg_options['--scaling']

  averaging = str(_arg_options['--average']).lower()
  if averaging not in ['none', 'ns', 'cc']:
    print('Invalid averaging: ', averaging)
    averaging = 'none'
  AVERAGE_BY = averaging

  baseline_df_file = None
  plots_to_generate = _arg_options['--plot']
  FORCE_REPLOT = _arg_options['--force_replot']

  print(_arg_options)

  SHADE_BASELINE_COMPARISON = not _arg_options['--no_baseline_comparison_shading']
  print('SHADING:', SHADE_BASELINE_COMPARISON)

  if ',' in plots_to_generate:
    plots_to_generate = set(plots_to_generate.split(','))
  else:
    plots_to_generate = set([plots_to_generate])

  valid_plot_names = set(PLOTS_TO_GENERATE.keys()).intersection(plots_to_generate)
  for plot_name in valid_plot_names:
    PLOTS_TO_GENERATE[plot_name] = True

  if _arg_options['--expected_baseline_speedup']:
    EXPECTED_BASELINE_SPEEDUP = float(_arg_options['--expected_baseline_speedup'])

  if _arg_options['--plot_titles']:
    PLOT_TITLES = str(_arg_options['--plot_titles']).lower()

  if _arg_options['--min_only']:
    PLOT_MAX = False
    MIN_LINESTYLE = 'solid'
    for dir_name in PLOT_DIRS.keys():
      PLOT_DIRS[dir_name] = 'min_only/{}'.format(PLOT_DIRS[dir_name])

  if _arg_options['--max_only']:
    PLOT_MIN = False
    for dir_name in PLOT_DIRS.keys():
      PLOT_DIRS[dir_name] = 'max_only/{}'.format(PLOT_DIRS[dir_name])

  if _arg_options['--ymin'] != -1.0:
    global DO_YMIN_OVERRIDE
    DO_YMIN_OVERRIDE = True
    try:
      YMIN_OVERRIDE['raw_data'] = float(_arg_options['--ymin'])
    except ValueError:
      YMIN_OVERRIDE = dict(item.split('=') for item in _arg_options['--ymin'].replace('"', '').split(','))
      print(YMIN_OVERRIDE)

  if _arg_options['--ymax'] != -1.0:
    global DO_YMAX_OVERRIDE
    DO_YMAX_OVERRIDE = True
    try:
      YMAX_OVERRIDE['raw_data'] = float(_arg_options['--ymax'])
    except ValueError:
      YMAX_OVERRIDE = dict(item.split('=') for item in _arg_options['--ymax'].replace('"', '').split(','))
      print(YMAX_OVERRIDE)

  if _arg_options['--normalize_y']:
    DO_NORMALIZE_Y = _arg_options['--normalize_y']

  if _arg_options['--baseline']:
    baseline_df_file = _arg_options['--baseline']

    if baseline_df_file != dataset_filename:
      global BASELINE_DATASET_FILE
      HAVE_BASELINE = True

      print('Have baseline data: ', baseline_df_file)
      BASELINE_DATASET_FILE = os.path.basename(baseline_df_file)

  if _arg_options['--baseline_linestyle']:
    BASELINE_LINESTYLE = _arg_options['--baseline_linestyle']

  if _arg_options['--legend']:
    PLOT_LEGEND = True

  if _arg_options['--annotate_filenames']:
    ANNOTATE_DATASET_FILENAMES = True

  sort_timer_labels = None
  if _arg_options['--sort_timer_labels']:
    sort_timer_labels = _arg_options['--sort_timer_labels']
    if sort_timer_labels == 'None':
      sort_timer_labels = None

  number_plots = False
  if _arg_options['--number_plots']:
    number_plots = (_arg_options['--number_plots'].lower() == 'true')

  restrict_timer_labels = False
  if _arg_options['--restrict_timer_labels']:
    restrict_timer_labels = _arg_options['--restrict_timer_labels']

  timer_mapping_file = False
  if _arg_options['--timer_name_remapping']:
    timer_mapping_file = _arg_options['--timer_name_remapping']

  print('study: {study}\nscaling_type: {scaling}\ndataset: {data}'.format(study=study_type,
                                                                          scaling=scaling_study_type,
                                                                          data=dataset_filename))
  print('Max Nodes: {max}\tMin Nodes: {min}'.format(max=max_num_nodes, min=min_num_nodes))
  print('Max PPN: {max}\tMin PPN: {min}'.format(max=max_procs_per_node, min=min_procs_per_node))

  # TODO: Need to create a mapping that will glue timers labels together
  timer_name_mappings = {}
  if timer_mapping_file:
    if VERBOSITY & 64:
      print('Attempting to remap timer names using: ', timer_mapping_file)
    timer_name_mappings = read_timer_name_mappings(timer_mapping_file)
    if not timer_name_mappings:
      print('Failed to read timer names from file.')

  DATASET_FILE = os.path.basename(dataset_filename)

  if scaling_study_type == 'weak':
    dataset, driver_dataset = load_dataset(dataset_filename=dataset_filename,
                                           min_num_nodes=min_num_nodes,
                                           max_num_nodes=max_num_nodes,
                                           min_procs_per_node=min_procs_per_node,
                                           max_procs_per_node=max_procs_per_node,
                                           timer_name_mappings=timer_name_mappings)
    if HAVE_BASELINE:
      BASELINE_DATASET_DF, BASELINE_DRIVER_DF = load_dataset(dataset_filename=baseline_df_file,
                                                             min_num_nodes=min_num_nodes,
                                                             max_num_nodes=max_num_nodes,
                                                             min_procs_per_node=min_procs_per_node,
                                                             max_procs_per_node=max_procs_per_node,
                                                             timer_name_mappings=timer_name_mappings)
  else:
    dataset, driver_dataset = load_dataset(dataset_filename=dataset_filename,
                                           min_num_nodes=0,
                                           max_num_nodes=64,
                                           min_procs_per_node=min_procs_per_node,
                                           max_procs_per_node=max_procs_per_node,
                                           timer_name_mappings=timer_name_mappings)

    if HAVE_BASELINE:
      BASELINE_DATASET_DF, BASELINE_DRIVER_DF = load_dataset(dataset_filename=baseline_df_file,
                                                             min_num_nodes=min_num_nodes,
                                                             max_num_nodes=max_num_nodes,
                                                             min_procs_per_node=min_procs_per_node,
                                                             max_procs_per_node=max_procs_per_node,
                                                             timer_name_mappings=timer_name_mappings)

  ###########################################################################################################
  # restrict the labels if requested
  # restriction currently assumes that the timer labels are the same in the baseline
  restricted_timers = dataset['Timer Name'].unique().tolist()

  if restrict_timer_labels:
    if restrict_timer_labels == 'muelu_levels':
      restricted_timers = get_muelu_level_timers(timer_names=restricted_timers)
      if VERBOSITY & 64:
        print('Restricting timer labels with: ', restrict_timer_labels)
    elif restrict_timer_labels == 'spmv':
      restrict_timer_labels = r'(.*Operation Op\*x.*)|(.*OPT::Apply.*)'
      if VERBOSITY & 64:
        print('Restricting timer labels with: ', restrict_timer_labels)
      timer_label_re = re.compile(restrict_timer_labels)
      restricted_timers = [label for label in restricted_timers if timer_label_re.match(label)]
    else:
      if VERBOSITY & 64:
        print('Restricting timer labels with: ', restrict_timer_labels)
      timer_label_re = re.compile(restrict_timer_labels)
      restricted_timers = [label for label in restricted_timers if timer_label_re.match(label)]
    if VERBOSITY & 64:
      print('Analyzing Timers: ')
      print(restricted_timers)
    if len(restricted_timers) == 0:
      print('Timer label restriction produced no timers to analyze.')
      exit(-1)

  dataset = dataset[dataset['Timer Name'].isin(restricted_timers)]

  if HAVE_BASELINE:
    BASELINE_DATASET_DF = BASELINE_DATASET_DF[BASELINE_DATASET_DF['Timer Name'].isin(restricted_timers)]

  ###########################################################################################################
  # restrict the datasets for a study
  if study_type == 'muelu_constructor':
    total_time_key = '3 - Constructing Preconditioner'
    restriction_tokens = {'solver_name': 'Constructor',
                          'solver_attributes': '-Only',
                          'prec_name': 'MueLu'}

  elif study_type == 'linearAlg':
    total_time_key = '0 - Total Time'
    restriction_tokens = {'solver_name': 'LinearAlgebra',
                          'solver_attributes': '-Tpetra',
                          'prec_name': 'None'}
    dataset = correct_linearAlg_timer_labels(dataset)
    if HAVE_BASELINE:
      BASELINE_DATASET_DF = correct_linearAlg_timer_labels(BASELINE_DATASET_DF)

  elif study_type == 'muelu_prec':
    total_time_key = '5 - Solve'
    restriction_tokens = {'solver_name': 'CG',
                          'solver_attributes': '',
                          'prec_name': 'MueLu',
                          'prec_attributes': '-repartition'}

  elif study_type == 'solvers':
    total_time_key = '5 - Solve'
    restriction_tokens = {'prec_name': 'None'}
  elif study_type == 'none':
    total_time_key = None
    restriction_tokens = None
  else:
    raise ValueError('unknown study_type ({})'.format(study_type))

  ###########################################################################################################
  # sort the labels if desired
  if sort_timer_labels is not None:
    if VERBOSITY & 128:
      print(sort_timer_labels)
    ordered_timers = get_ordered_timers(dataset,
                                        rank_by_column_name=sort_timer_labels)
  else:
    if VERBOSITY & 128:
      print('No sort, ' + sort_timer_labels)
    ordered_timers = dataset['Timer Name'].unique().tolist()

  if len(ordered_timers) != len(set(ordered_timers)):
    print('Timer labels have duplicates after sorting and filtering!')
    exit(-1)

  ###########################################################################################################
  # analyze the timers
  if VERBOSITY & 256:
    print('Final set of timers to analyze:')
    print(ordered_timers)

  plot_dataset(dataset=dataset,
               driver_dataset=driver_dataset,
               ordered_timers=ordered_timers,
               total_time_key=total_time_key,
               scaling_type=scaling_study_type,
               restriction_tokens=restriction_tokens,
               number_plots=number_plots)


def do_tpetra_analysis(dataset):
  restricted_data = dataset[dataset['Timer Name'].str.match('(^([a-zA-Z]+::)+\d+)')]
  restricted_data.to_csv('lin-ops.csv')
  unique_timers = restricted_data['Timer Name'].unique()


def correct_linearAlg_timer_labels(dataset):
  # if MVT::MVScale1 and MVT::MVInit0 everything is fine
  # if you have MVT::MVScale and MVT::MVInit (unannotated) then relabel them
  # MVT::MVScale => MVT::MVScale1::1
  timer_labels = dataset['Timer Name'].unique()
  if ('MVT::MVScale' in timer_labels) and ('MVT::MVScale1:1' not in timer_labels):
    dataset[dataset['Timer name'] == 'MVT::MVScale', 'Timer name'] = 'MVT::MVScale1::1'

  return dataset


def get_muelu_level_timers(timer_names=[],
                           dataset=None):
  import re

  # examples:
  # MueLu: AmalgamationFactory: Build (level=[0-9]*)
  # MueLu: CoalesceDropFactory: Build (level=[0-9]*)
  # MueLu: CoarseMapFactory: Build (level=[0-9]*)
  # MueLu: FilteredAFactory: Matrix filtering (level=[0-9]*)
  # MueLu: NullspaceFactory: Nullspace factory (level=[0-9]*)
  # MueLu: UncoupledAggregationFactory: Build (level=[0-9]*)
  # MueLu: CoordinatesTransferFactory: Build (level=[0-9]*)
  # MueLu: TentativePFactory: Build (level=[0-9]*)
  # MueLu: Zoltan2Interface: Build (level=[0-9]*)
  # MueLu: SaPFactory: Prolongator smoothing (level=[0-9]*)
  # MueLu: SaPFactory: Fused (I-omega\*D\^{-1} A)\*Ptent (sub, total, level=1[0-9]*)
  # MueLu: RAPFactory: Computing Ac (level=[0-9]*)
  # MueLu: RAPFactory: MxM: A x P (sub, total, level=[0-9]*)
  # MueLu: RAPFactory: MxM: P' x (AP) (implicit) (sub, total, level=[0-9]*)
  # MueLu: RepartitionHeuristicFactory: Build (level=[0-9]*)
  # MueLu: RebalanceTransferFactory: Build (level=[0-9]*)
  # MueLu: RebalanceAcFactory: Computing Ac (level=[0-9]*)
  # MueLu: RebalanceAcFactory: Rebalancing existing Ac (sub, total, level=[0-9]*)
  # MueLu: Hierarchy: Setup(total, level=[0-1]*)
  #
  # Do not contain the text total
  # TpetraExt MueLu::SaP-[0-9]*: Jacobi All I&X
  # TpetraExt MueLu::SaP-[0-9]*: Jacobi All Multiply
  # TpetraExt MueLu::A\*P-[0-9]*: MMM All I&X
  # TpetraExt MueLu::A\*P-[0-9]*: MMM All Multiply
  # TpetraExt MueLu::R\*(AP)-implicit-[0-9]*: MMM All I&X
  # TpetraExt MueLu::R\*(AP)-implicit-[0-9]*: MMM All Multiply

  # for now, hard code these.
  level_timers = [
    r'(?P<label_prefix>MueLu: Hierarchy: Setup)\s*\(total, level=(?P<level_number>[0-9]*)\)(?P<label_suffix>.*)',
    r'(?P<label_prefix>MueLu: AmalgamationFactory: Build)\s*\(level=(?P<level_number>[0-9]*)\)(?P<label_suffix>.*)',
    r'(?P<label_prefix>MueLu: CoalesceDropFactory: Build)\s*\(level=(?P<level_number>[0-9]*)\)(?P<label_suffix>.*)',
    r'(?P<label_prefix>MueLu: CoarseMapFactory: Build)\s*\(level=(?P<level_number>[0-9]*)\)(?P<label_suffix>.*)',
    r'(?P<label_prefix>MueLu: FilteredAFactory: Matrix filtering)\s*\(level=(?P<level_number>[0-9]*)\)(?P<label_suffix>.*)',
    r'(?P<label_prefix>MueLu: NullspaceFactory: Nullspace factory)\s*\(level=(?P<level_number>[0-9]*)\)(?P<label_suffix>.*)',
    r'(?P<label_prefix>MueLu: UncoupledAggregationFactory: Build)\s*\(level=(?P<level_number>[0-9]*)\)(?P<label_suffix>.*)',
    r'(?P<label_prefix>MueLu: CoordinatesTransferFactory: Build)\s*\(level=(?P<level_number>[0-9]*)\)(?P<label_suffix>.*)',
    r'(?P<label_prefix>MueLu: TentativePFactory: Build)\s*\(level=(?P<level_number>[0-9]*)\)(?P<label_suffix>.*)',
    r'(?P<label_prefix>MueLu: Zoltan2Interface: Build)\s*\(level=(?P<level_number>[0-9]*)\)(?P<label_suffix>.*)',
    r'(?P<label_prefix>MueLu: SaPFactory: Prolongator smoothing)\s*\(level=(?P<level_number>[0-9]*)\)',
    r'(?P<label_prefix>MueLu: SaPFactory: Fused \(I-omega\*D\^{-1} A\)\*Ptent \(sub, total,)\s*level=(?P<level_number>[0-9]*)\)(?P<label_suffix>.*)',
    r'(?P<label_prefix>MueLu: RAPFactory: Computing Ac)\s*\(level=(?P<level_number>[0-9]*)\)(?P<label_suffix>.*)',
    r'(?P<label_prefix>MueLu: RAPFactory: MxM: A x P \(sub, total,)\s*level=(?P<level_number>[0-9]*)\)(?P<label_suffix>.*)',
    r'(?P<label_prefix>MueLu: RAPFactory: MxM: P\' x \(AP\) \(implicit\) \(sub, total,)\s*level=(?P<level_number>[0-9]*)\)(?P<label_suffix>.*)',
    r'(?P<label_prefix>MueLu: RepartitionHeuristicFactory: Build)\s*\(level=(?P<level_number>[0-9]*)\)(?P<label_suffix>.*)',
    r'(?P<label_prefix>MueLu: RebalanceTransferFactory: Build)\s*\(level=(?P<level_number>[0-9]*)\)(?P<label_suffix>.*)',
    r'(?P<label_prefix>MueLu: RebalanceAcFactory: Computing Ac)\s*\(level=(?P<level_number>[0-9]*)\)(?P<label_suffix>.*)',
    r'(?P<label_prefix>MueLu: RebalanceAcFactory: Rebalancing existing Ac \(sub, total,)\s*level=(?P<level_number>[0-9]*)\)(?P<label_suffix>.*)'
  ]
  # # Do not contain the text total
  # r'(?P<label_prefix>TpetraExt MueLu::SaP)-(?P<level_number>[0-9]*):\s*(?P<label_suffix>Jacobi All I&X)',
  # r'(?P<label_prefix>TpetraExt MueLu::SaP)-(?P<level_number>[0-9]*):\s*(?P<label_suffix>Jacobi All Multiply)',
  # r'(?P<label_prefix>TpetraExt MueLu::A\*P)-(?P<level_number>[0-9]*):\s*(?P<label_suffix>MMM All I&X)',
  # r'(?P<label_prefix>TpetraExt MueLu::A\*P)-(?P<level_number>[0-9]*):\s*(?P<label_suffix>MMM All Multiply)',
  # r'(?P<label_prefix>TpetraExt MueLu::R\*\(AP\)-implicit)-(?P<level_number>[0-9]*):\s*(?P<label_suffix>MMM All I&X)',
  # r'(?P<label_prefix>TpetraExt MueLu::R\*\(AP\)-implicit)-(?P<level_number>[0-9]*):\s*(?P<label_suffix>MMM All Multiply)'

  level_timers_re = []
  for re_str in level_timers:
    level_timers_re.append(re.compile(re_str))

  level_timer_map = {0: list(),
                     1: list(),
                     2: list(),
                     3: list(),
                     4: list(),
                     5: list(),
                     6: list(),
                     7: list(),
                     8: list(),
                     9: list(),
                     10: list(),
                     'total': list()}

  # populate the map's level specific timer names
  for level_timer_re in level_timers_re:
    for timer_name in timer_names:
      for m in [level_timer_re.search(timer_name)]:
        if m:
          level_timer_map[int(m.group('level_number'))].append(timer_name)

  # look for (total) labels
  for timer_name in timer_names:
    m = re.search(r'\(\s*total\s*\)', timer_name)
    if m:
      level_timer_map['total'].append(timer_name)

  # level timers:
  # always return the total labels first
  level_timer_names = level_timer_map['total']
  for level_id in level_timer_map:
    if level_id == 'all' or level_id == 'total':
      continue
    level_timer_names += level_timer_map[level_id]

  return level_timer_names


def adjust_timer_names(df, timer_name_mappings):
  for new_name, other_names in timer_name_mappings.items():
    for other_name in other_names:
      if other_name == new_name:
        continue
      print(new_name, other_name)
      df.loc[df['Timer Name'] == other_name, 'Timer Name'] = new_name
  return df


def read_timer_name_mappings(filename):
  timer_name_mappings = {}
  try:
    with open(filename, 'r') as fin:
      for line in fin:
        line = line.strip()
        timer_names = line.split(';')
        timer_name_mappings[timer_names[0]] = timer_names
  except:
    return timer_name_mappings

  return timer_name_mappings


###############################################################################
if __name__ == '__main__':
  main()
