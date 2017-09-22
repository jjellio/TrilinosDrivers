#!/usr/bin/env python3
"""analysis.py

Usage:
  analysis.py -b baseline  -c comparables [-o OUTPUT]
  analysis.py (-h | --help)

Options:
  -h --help                     Show this screen.
  -b FILE --baseline=FILE       Use this file as the baseline
  -c FILES --comparables=FILES  YAMLS that can be compared to the baseline
  -o FILE --output-csv=FILE     Output file [default: all_data.csv]
"""

import glob
import numpy  as np
import pandas as pd
from   docopt import docopt
import yaml
from pathlib import Path
import re


def file_len(PathLibFilename):
  i = 0

  with PathLibFilename.open() as f:
    for i, l in enumerate(f):
      pass
  return i + 1


# this file can be refactored into a single parse YAML to CSV.
# it is a quick hack
def construct_dataframe(yaml_data):
  """Construst a pandas DataFrame from the timers section of the provided YAML data"""
  timers = yaml_data['Timer names']
  data = np.ndarray([len(timers), 8])

  ind = 0
  for timer in timers:
    t = yaml_data['Total times'][timer]
    c = yaml_data['Call counts'][timer]
    data[ind, 0:8] = [
      t['MinOverProcs'], c['MinOverProcs'],
      t['MeanOverProcs'], c['MeanOverProcs'],
      t['MaxOverProcs'], c['MaxOverProcs'],
      t['MeanOverCallCounts'], c['MeanOverCallCounts']
    ]
    ind = ind + 1

  df = pd.DataFrame(data,
                    index=timers,
                    columns=['minT', 'minC', 'meanT', 'meanC', 'maxT', 'maxC', 'meanCT', 'meanCC'])

  # df['Timer Name'] = df.index
  return df


def load_yaml(filename):
  with open(filename) as data_file:
    yaml_data = yaml.safe_load(data_file)
  return yaml_data


if __name__ == '__main__':
  # Process input
  options = docopt(__doc__)

  comparable_files  = options['--comparables']
  baseline_file = options['--baseline']
  output_csv   = options['--output-csv']

  print('baseline_file: {baseline}\n'
        'comparable_files: {comparable}\n'
        'output_csv: {output}\n'.format(baseline=baseline_file,
                                        output=output_csv,
                                        comparable=comparable_files))

  if isinstance(comparable_files, str):
    print('Globbing')
    comparable_files = glob.glob(comparable_files)

  print(comparable_files)

  baseline_df = construct_dataframe(load_yaml(baseline_file))

  unified_timer_names = set(baseline_df.index)

  comparable_file_mapping = {}
  short_name = chr(ord('A'))

  for comparable_file in comparable_files:
    comparable_file_mapping[short_name] = comparable_file
    short_name = chr(ord(short_name) + 1)

  for short_name in sorted(comparable_file_mapping.keys()):
    comparable_file = comparable_file_mapping[short_name]

    comparable_yaml = load_yaml(comparable_file)
    # add new timers
    unified_timer_names = unified_timer_names.union(set(comparable_yaml['Timer names']))
    # update the dataframe of baseline to use this new index
    baseline_df.reindex(list(unified_timer_names))
    # build a new data frame
    comparable_df = construct_dataframe(comparable_yaml)
    comparable_df = comparable_df.reindex(list(unified_timer_names))

    baseline_df = pd.merge(baseline_df, comparable_df,
                           left_index=True,
                           right_index=True,
                           how='outer',
                           suffixes=('', '_'+short_name))

  # df.filter(regex=("[A-Z]_.*"))
  timer_types = ['minT', 'maxT', 'meanT', 'meanCT']
  for timer_type in timer_types:
    output_columns = [timer_type]

    for short_name in sorted(comparable_file_mapping.keys()):
      comparable_file = comparable_file_mapping[short_name]
      lookup_column = '{timer}_{short_name}'.format(timer=timer_type,
                                                    short_name=short_name)

      new_colum = '{timer}_speedup_{short_name}'.format(timer=timer_type,
                                                        short_name=short_name)
      baseline_df[new_colum] = baseline_df[timer_type] / baseline_df[lookup_column]

      baseline_df[new_colum] = pd.Series([round(val, 3) for val in baseline_df[new_colum] ],
                                         index=baseline_df.index)

      output_columns.append(lookup_column)
      output_columns.append(new_colum)

    baseline_df.to_csv('{timer}_comparison.csv'.format(timer=timer_type),
                       index_label='Timer Name',
                       index=True,
                       columns=output_columns)

  for short_name in sorted(comparable_file_mapping.keys()):
    comparable_file = comparable_file_mapping[short_name]
    print('{short_name}: {file}'.format(short_name=short_name, file=comparable_file))
