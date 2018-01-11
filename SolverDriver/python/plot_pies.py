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
mpl.use('MacOSX')

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

  print(dataset['Timer Name'].unique())

  # grab the total time
  total_timings = dataset[ dataset['Timer Name'] == 'MueLu: Hierarchy: Setup (total)' ].copy()
  total_timings = total_timings.set_index(['num_nodes', 'decomp_label', 'threads_per_core'], verify_integrity=True)
  total_level_timings = dataset[ dataset['Timer Name'] == 'MueLu: Hierarchy: Setup (total, level=1)'].copy()
  total_level_timings = total_level_timings.set_index(['num_nodes', 'decomp_label', 'threads_per_core'], verify_integrity=True)

  dataset = dataset.set_index(['num_nodes', 'decomp_label', 'threads_per_core', 'Timer Name'], verify_integrity=True, drop=False)

  print(total_timings.index)
  dataset['Percent Hierarchy Total'] = np.nan
  dataset['Percent Hierarchy Level Total'] = np.nan
  dataset['Percent Group Total'] = np.nan

  # construct the ordered timers
  ordered_timers = []
  for parent, children in timer_groups.items():
    timing_group = [parent]
    if children:
      for child in children:
        timing_group.append(child)

    ordered_timers += timing_group

  # restrict the data to the timers we care about
  dataset = dataset[ dataset['Timer Name'].isin(restricted_timers) ]

  # construct the ordered timers
  for parent, children in timer_groups.items():
    timing_group = [parent]
    if children:
      for child in children:
        timing_group.append(child)
    group_idx = dataset['Timer Name'].isin(timing_group)
    dataset.loc[group_idx, 'Group Total Time'] = dataset.loc[group_idx].groupby(['num_nodes', 'decomp_label', 'threads_per_core'])['minT'].transform(lambda x: x.max())

  # add hierarchy total
  dataset['Hierarchy Total Time'] = dataset.groupby(['num_nodes', 'decomp_label', 'threads_per_core'])['minT'].transform(lambda x: total_timings['minT'].loc[x.name])
  # add level total
  dataset['Hierarchy Level Time'] = dataset.groupby(['num_nodes', 'decomp_label', 'threads_per_core'])['minT'].transform(lambda x: total_level_timings['minT'].loc[x.name])

  dataset['Percent Hierarchy Total'] = dataset['minT'] / dataset['Hierarchy Total Time']
  dataset['Percent Hierarchy Level Total'] = dataset['minT'] / dataset['Hierarchy Level Time']
  dataset['Percent Group Total'] = dataset['minT'] / dataset['Group Total Time']

  TABLE_ROW_NAMES = ['Percent Hierarchy Total', 'Percent Hierarchy Level Total', 'Percent Group Total']

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

      # add the percentages
      # plt.table expects a list of lists corresponding to r lists (rows)
      # with c columns in each sub list.
      labels = []
      table_rows = []
      for row_name in TABLE_ROW_NAMES:
        table_row = []
        tmp_df = ht_group.pivot(index='Timer Name', columns='decomp_label', values=row_name)
        for decomp_label in decomp_labels:
          for timer_name in ordered_timers:
            text_label = ''
            try:
              table_row.append('{:.1%}'.format(tmp_df.loc[timer_name, decomp_label]))
              text_label += '{:.1%}\n'.format(tmp_df.loc[timer_name, decomp_label])
            except KeyError:
              table_row.append(np.NaN)
              text_label += '\n'
            labels.append(text_label)

        table_rows.append(table_row)

      # the_table = plt.table(cellText=table_rows,
      #                       rowLabels=TABLE_ROW_NAMES,
      #                       colLabels=decomp_labels*len(ordered_timers),
      #                       loc='top')
      #
      # the_table.set_fontsize(14)
      rects = ax.patches

      # Now make some labels
      labels = []

      for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label, ha='center', va='bottom')

      ax.set_ylim([y_min, y_max])
      ax.set_yscale('log', basey=2)
      print(y_max)

      h.legend(title="Procs per Node x Cores per Proc",
               loc='upper right', ncol=ndecomps, fontsize=14)
      h.subplots_adjust(bottom=0.25)
      h.savefig(fname, format=IMG_FORMAT, dpi=IMG_DPI)
      plt.show(block=True)
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
