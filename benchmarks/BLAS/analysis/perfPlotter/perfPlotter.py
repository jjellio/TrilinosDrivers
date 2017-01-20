#!/usr/bin/env python

'''
  This script is designed for Python2.7
  On some systems, this entails using a provided 2.7 install,
  and creating a virtual environment that has the required modules.
'''

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools as itools

import string
import os.path
import sys

FIGURE_H_in = 5

SYMBOLS = []
SYMBOLS.append('s')
SYMBOLS.append('d')
SYMBOLS.append('>')
SYMBOLS.append('*')

USE_MICROSECONDS=True
COLORS = []
COLORS.append([86.0 / 255.0, 180.0 / 255.0, 233.0 / 255.0])
COLORS.append([213.0 / 255.0, 94.0 / 255.0, 0.0 / 255.0])
COLORS.append([204.0 / 255.0, 121.0 / 255.0, 167.0 / 255.0])
COLORS.append([0 / 255.0, 0 / 255.0, 167.0 / 255.0])


font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 12}

mpl.rc('font', **font)
pd.set_option('expand_frame_repr', False)


def get_df(prefix='tmp-data', blaslib='mkl', threading='intel', arch='hsw', HT=1,
           nameMask='BlasTests.{}_{}_{}-HT_{}_details.csv',
           add_index=True):
  fname = nameMask.format(blaslib, threading, arch, HT)
  if prefix != '':
    fname = '{}/{}'.format(prefix, fname)

  if os.path.isfile(fname):
    df = pd.read_csv(fname, quotechar='"', skipinitialspace=True)
  else:
    return (pd.DataFrame())

  df.loc[:, 'HT'] = HT
  df.loc[:, 'threading'] = threading
  df.loc[:, 'arch'] = arch

  df.loc[df['Label'] == 'A^t x X + 0*Y', 'Label'] = 'InnerProduct'
  df.loc[df['Label'] == 'neg A x X + 1*Y', 'Label'] = 'Update'
  df.loc[df['Label'] == 'innerProduct', 'Label'] = 'InnerProduct'
  if add_index:
    set_df_index(df)

  return (df)


def set_df_index(df):
  print df.columns
  # sort the dataframe
  df.sort_values(by=['Label', 'OMP_WAIT_POLICY', 'OMP_PLACES', 'n', 'np'], inplace=True)
  # set the index to be this and don't drop
  df.set_index(keys=['Label', 'OMP_WAIT_POLICY', 'OMP_PLACES', 'n', 'np'], drop=False, inplace=True)


def aggregate_trilinos_df (trilinos_df, include_network=False):

  # for this plotter, we focus on kernel performance not networking
  network_df = trilinos_df[trilinos_df['detail_label'] == 'Tpetra::Multiply::internal_reduce']

  gemm_df = trilinos_df[trilinos_df['detail_label'] == 'Tpetra::Multiply::gemm_device']
  pre_gemm_df = trilinos_df[trilinos_df['detail_label'] == 'Tpetra::Multiply::pre_gemm']
  post_gemm_df = trilinos_df[trilinos_df['detail_label'] == 'Tpetra::Multiply::post_gemm']
  #
  print gemm_df['time (ns)'].describe()
  print pre_gemm_df['time (ns)'].describe()
  print post_gemm_df['time (ns)'].describe()
  print network_df['time (ns)'].describe()

  if include_network:
    foo = gemm_df['time (ns)'].values + pre_gemm_df['time (ns)'].values + post_gemm_df['time (ns)'].values + network_df['time (ns)'].values
  else:
    foo = gemm_df['time (ns)'].values + pre_gemm_df['time (ns)'].values + post_gemm_df['time (ns)'].values


  print (len(foo))
  print gemm_df['time (ns)'].count ()
  gemm_df['time (ns)'] = foo

  return gemm_df

def main():
  unified_df = pd.DataFrame()
  unified_df = pd.concat([unified_df, get_df(HT=1)])
  unified_df = pd.concat([unified_df, get_df(HT=2)])
  unified_df = pd.concat([unified_df, get_df(HT=3)])
  unified_df = pd.concat([unified_df, get_df(HT=4)])

  nameMask = 'ortho_bench.{}_{}_{}-HT_{}_details.csv'
  trilinos_df = pd.DataFrame()
  trilinos_df = pd.concat([trilinos_df, get_df(HT=1, nameMask=nameMask, add_index=False)])
  trilinos_df = pd.concat([trilinos_df, get_df(HT=2, nameMask=nameMask, add_index=False)])
  trilinos_df = pd.concat([trilinos_df, get_df(HT=3, nameMask=nameMask, add_index=False)])
  trilinos_df = pd.concat([trilinos_df, get_df(HT=4, nameMask=nameMask, add_index=False)])

  # we have lots to plot
  # Label, n, wait policy, and places identify unique sets of NPs to compare
  unique_labels = unified_df['Label'].unique()
  unique_policy = unified_df['OMP_WAIT_POLICY'].unique()
  unique_places = unified_df['OMP_PLACES'].unique()
  unique_n = unified_df['n'].unique()

  # unified_df = unified_df[unified_df['np'] != 1]
  # trilinos_df = trilinos_df[trilinos_df['np'] != 1]
  trilinos_df = aggregate_trilinos_df(trilinos_df, include_network=True)

  for n in unique_n:
    if n == 1:
      continue
    for label in unique_labels:
      min_max_query = '(Label == \'{}\') & (n == {}) & (np != 1)'.format(label,n)
      tmp = unified_df.query(min_max_query)
      min1 = tmp.loc[:, 'time (ns)'].min()
      max1 = tmp.loc[:, 'time (ns)'].max()

      tmp = trilinos_df.query(min_max_query)
      min1 = min(min1, tmp.loc[:, 'time (ns)'].min())
      max1 = max(max1, tmp.loc[:, 'time (ns)'].max())

      for policy in unique_policy:
        for place in unique_places:
          query = '(Label == \'{}\') & (OMP_WAIT_POLICY == \'{}\') & (OMP_PLACES == \'{}\') & (n == {})'.format(label,
                                                                                                                policy,
                                                                                                                place,
                                                                                                                n)
          comparison_df = unified_df.query(query)
          trilinos_comp_df = trilinos_df.query(query)

          generateComparison(comparison_df, min1, max1, trilinos_comp_df)

# function for setting the colors of the box plots pairs
def setBoxColors(bp, numPlots, trilinos=False):
  from matplotlib.pyplot import setp

  for x in range(0, numPlots):
    if trilinos:
      my_color = 'magenta'
    else:
      my_color = COLORS[x]

    setp(bp['boxes'][x], color=my_color)
    setp(bp['caps'][x*2], color=my_color)
    setp(bp['caps'][x*2+1], color=my_color)
    setp(bp['whiskers'][x*2], color=my_color)
    setp(bp['whiskers'][x*2+1], color=my_color)
    setp(bp['fliers'][x], color=my_color, markeredgecolor=my_color)
    setp(bp['medians'][x], color=my_color)


def generateComparison (df,gmin,gmax, trilinos_comp_df):
  label  = df['Label'][0]
  policy = df['OMP_WAIT_POLICY'][0]
  place  = df['OMP_PLACES'][0]
  n      = df['n'][0]
  arch = df['arch'][0]
  blaslib = df['BlasLib'][0]
  threading = df['threading'][0]

  numProcs = df['np'].unique()
  myHTs = df['HT'].unique()
  numHTs = myHTs.size

  if USE_MICROSECONDS:
    gmin *= 1.e-3
    gmax *= 1.e-3
    df.loc[:, 'time (ns)'] *= 1.e-3
    trilinos_comp_df.loc[:, 'time (ns)'] *= 1.e-3
    ylabel = 'Time in microseconds'
  else:
    ylabel = 'Time in nanoseconds'

  print 'min:{}, max:{}'.format(gmin, gmax)

  FIGURE_W_in = (FIGURE_H_in * 4 / 3) * (numHTs)  # 4/3 aspect ratio, fattened for HTs
  h = plt.figure(figsize=[FIGURE_W_in, FIGURE_H_in])
  ax = plt.axes()
  plt.hold(True)

  position_idx = 1
  xtick_locs = []
  xtick_labels = []

  tpl_outlier_props = dict(marker='o', markersize=8, linestyle='none')
  trilinos_outlier_props = dict(marker='x', markersize=8, linestyle='none')

  for procs in numProcs:
    box_pair = []
    trilinos_box_pair = []
    box_locs = []
    thread_counts = []
    for HT in myHTs:
      box_pair.append( df[(df['HT'] == HT) & (df['np'] == procs)]['time (ns)'].tolist() )
      trilinos_box_pair.append( trilinos_comp_df[(trilinos_comp_df['HT'] == HT) & (trilinos_comp_df['np'] == procs)]['time (ns)'].tolist() )
      box_locs.append(position_idx+HT-1)
      thread_counts.append(df[(df['HT'] == HT) & (df['np'] == procs)]['OMP_NUM_THREADS'].unique()[0])

    # create a pair of boxplots [position_idx, position_idx+numHTs-1]
    bp = plt.boxplot(box_pair, positions=box_locs, widths=0.75, flierprops=tpl_outlier_props)
    setBoxColors(bp, numHTs)
    bp_tril = plt.boxplot(trilinos_box_pair, positions=box_locs, widths=0.75, flierprops=trilinos_outlier_props)
    setBoxColors(bp_tril, numHTs, trilinos=True)
    xtick_locs.append(position_idx + (numHTs-1)/2.0)
    tmp_str = '  /  '.join(str(x) for x in thread_counts)
    xtick_labels.append('({})\n{}'.format(tmp_str,procs))

    if procs != numProcs[-1]:
      plt.plot((position_idx+numHTs - 1.0/numHTs, position_idx+numHTs- 1.0/numHTs),
               (gmin, gmax), color='gray', linestyle='dotted')
    position_idx += (numHTs+0.5)

  # create custom xticks and labels
  plt.xticks(xtick_locs, xtick_labels)
  plt.xlim([xtick_locs[0]-(numHTs-1), position_idx])
  # add padding above and below the max/min values
  dy = np.abs(gmax-gmin)/70
  plt.ylim([gmin-dy, gmax+dy])
  plt.title('{}\nArch={},Blas={},Threads={}\nWait={}, Places={}, NumVectors={}'.format(label,
                                                                                       arch,
                                                                                       blaslib,
                                                                                       threading,
                                                                                       policy, place, n))
  plt.xlabel('\n(Number of Threads)\nNumber of MPI Processes')
  plt.ylabel(ylabel)

  handles = []
  handle_lables = []
  for HT in myHTs:
    # draw temporary red and blue lines and use them to create a legend
    h, = plt.plot([1, 1], marker='_', color=COLORS[HT-1])
    handles.append(h)
    handle_lables.append('{} HT'.format(HT))

  plt.legend(handles, handle_lables, handlelength=1, numpoints=1, loc='upper left', bbox_to_anchor=(0.75, 1.15),
             fancybox=True, shadow=True, ncol=numHTs)
  for h in handles:
    h.set_visible(False)

  ax.get_yaxis().get_major_formatter().set_useOffset(False)
  #ax.get_yaxis().get_major_formatter().set_scientific(False)
  plt.subplots_adjust(top=0.85)
  plt.tight_layout()
  plt.show()


if __name__ == "__main__":
  main()
