#!/usr/bin/env python3
"""analysis.py

Usage:
  analysis.py --baseline=<FILE> --comparables=<FILES> [--output-csv=<FILE>] [--average=<averaging>] [-a <PATH>]
  analysis.py (-h | --help)

Options:
  -h              --help                  # Show this screen.
  --baseline=<FILE>       # Use this file as the baseline
  --comparables=<FILES>   # YAMLS that can be compared to the baseline
  --average=<averaging>   # Average the times using callcounts, numsteps, or none [default: none]
  --output-csv=<FILE>     # Output file [default: all_data.csv]
  -a PATH --affinity-dir=PATH   Path to directory with affinity CSV data [default: ../affinity]

Arguments:

      averaging: cc - call counts
                 ns - num_steps
                 none - do not average

"""

import glob
import os
import numpy       as np
import pandas      as pd
from   docopt import docopt
import yaml
from pathlib import Path
import re
import ScalingFilenameParser as SFP

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


def file_len(PathLibFilename):
  i = 0

  with PathLibFilename.open() as f:
    for i, l in enumerate(f):
      pass
  return i + 1


# affinity_files have the name:
# affinity_dir/Laplace3D-BS-1-1240x1240x1240_OpenMP-threads-16_np-2048_decomp-128x16x4x1_affinity.csv
# Tokenized:
# # affinity_dir/<problem_type>-BS-1-1240x1240x1240_OpenMP-threads-16_np-2048_decomp-128x16x4x1_affinity.csv
def parse_affinity_data(affinity_path, tokens):
  affinity_filename = '{path}/{name}'.format(path=affinity_path,
                                             name=SFP.build_affinity_filename(my_tokens))

  my_file_lookup = Path(affinity_filename)
  try:
    affinity_file_abs = my_file_lookup.resolve()
  except:
    print('Missing Affinity File: {}'.format(affinity_filename))
    return

  #if ~my_file.is_file():
  #  print('Missing Affinity File: {}'.format(affinity_filename))
  #  print(my_file.stat())
  #  return
  file_lines = file_len(affinity_file_abs)

  # 0,64,64,0,"nid02623","5-11-2017 21:02:30.900965",0,0|68|136|204
  expected_lines = tokens['num_nodes'] \
                 * tokens['procs_per_node'] \
                 * tokens['cores_per_proc'] \
                 * tokens['threads_per_core'] \
                 + 1
  if expected_lines != file_lines:
    import re

    print('{fname}: has {num} lines, expected {exp}'.format(fname=affinity_filename,
                                                            num=file_lines,
                                                            exp=expected_lines))
    #line_re = r'\d+,\d+,\d+,\d+,"\w+","[0-9- :.]+",\d+,\d+|\d+|\d+|\d\d\d'
    hostnames = 'unknown'
    timestamp = 'unknown'

  else:
    # use pandas to read the CSV
    df = pd.read_csv(affinity_file_abs,
                     parse_dates=True,
                     skipinitialspace=True,
                     low_memory=False)

    hostnames = ','.join(df['hostname'].unique())
    timestamp = df['Timestamp'].max()

  tokens['nodes'] = hostnames
  tokens['timestamp'] = timestamp


if __name__ == '__main__':
  # Process input
  from docopt import DocoptExit
#  try:
  options = docopt(__doc__)
#  except DocoptExit:
#    print(__doc__)
#    exit(0)

  comparable_files  = options['--comparables']
  affinity_dir = options['--affinity-dir']
  baseline_file = options['--baseline']
  output_csv   = options['--output-csv']
  averaging = options['--average']

  print('baseline_file: {baseline}\n'
        'comparable_files: {comparable}\n'
        'average: {avg}\n'
        'output_csv: {output}\n'.format(baseline=baseline_file,
                                        avg=averaging,
                                        output=output_csv,
                                        comparable=comparable_files))

  if isinstance(comparable_files, str) and '*' in comparable_files:
    print('Globbing')
    comparable_files = glob.glob(comparable_files)
  else:
    comparable_files = [comparable_files]

  print(comparable_files)

  baseline_df = construct_dataframe(load_yaml(baseline_file))

  unified_timer_names = set(baseline_df.index)

  comparable_file_mapping = {}
  short_name = chr(ord('A'))

  for comparable_file in comparable_files:
    print(comparable_file)

    comparable_file_mapping[short_name] = comparable_file
    short_name = chr(ord(short_name) + 1)

  for short_name in sorted(comparable_file_mapping.keys()):
    comparable_file = comparable_file_mapping[short_name]

    my_tokens = SFP.parseYAMLFileName(comparable_file)
    rebuilt_filename = SFP.rebuild_source_filename(my_tokens)

    if rebuilt_filename == os.path.basename(comparable_file):
      print("Rebuild OK: {} == {}".format(rebuilt_filename, comparable_file))
    else:
      print("Rebuild FAIL: {} != {}".format(rebuilt_filename, comparable_file))
      exit(-1)

    parse_affinity_data(affinity_path=affinity_dir, tokens=my_tokens)

    print(my_tokens)

    comparable_yaml = load_yaml(comparable_file)

    # add new timers
    unified_timer_names = unified_timer_names.union(set(comparable_yaml['Timer names']))
    # update the dataframe of baseline to use this new index
    baseline_df.reindex(list(unified_timer_names))
    # build a new data frame
    comparable_df = construct_dataframe(comparable_yaml)
    comparable_df = comparable_df.reindex(list(unified_timer_names))

    #comparable_df.to_csv('comparable-{}.csv'.format(short_name))
    # add the experiment's other data (cores, threads, timestamp, etc..)
    if averaging == 'cc':
      comparable_df['minT'] = comparable_df['minT'] / comparable_df['minC']
      comparable_df['maxT'] = comparable_df['maxT'] / comparable_df['maxC']
      comparable_df['meanT'] = comparable_df['meanT'] / comparable_df['meanC']
    if averaging == 'ns':
      comparable_df['minT'] = comparable_df['minT'] / my_tokens['numsteps']
      comparable_df['maxT'] = comparable_df['maxT'] / my_tokens['numsteps']
      comparable_df['meanT'] = comparable_df['meanT'] / my_tokens['numsteps']

    baseline_df = pd.merge(baseline_df, comparable_df,
                           left_index=True,
                           right_index=True,
                           how='outer',
                           suffixes=('', '_'+short_name))

  #baseline_df.to_csv('bs.csv')
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

    fname = '{timer}_comparison'.format(timer=timer_type)
    if averaging != 'none':
      fname += '_averaged_by_{avg}'.format(avg=averaging)
    baseline_df.to_csv('{fname}.csv'.format(fname=fname),
                       index_label='Timer Name',
                       index=True,
                       columns=output_columns)

  for short_name in sorted(comparable_file_mapping.keys()):
    comparable_file = comparable_file_mapping[short_name]
    print('{short_name}: {file}'.format(short_name=short_name, file=comparable_file))
