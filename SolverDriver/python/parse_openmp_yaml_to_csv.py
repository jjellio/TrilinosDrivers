#!/usr/bin/env python3
"""analysis.py

Usage:
  analysis.py -i INPUT... [-o OUTPUT] [-a MODE] [-d DISPLAY] [-s STYLE] [-t TOP]
  analysis.py (-h | --help)

Options:
  -h --help                     Show this screen.
  -i FILE --input-files=FILE    Input file
  -o FILE --output-file=FILE    Output file
  -a MODE --analysis=MODE       Mode [default: setup_timers]
  -d DISPLAY --display=DISPLAY  Display mode [default: print]
  -s STYLE --style=STYLE        Plot style [default: stack]
  -t TOP --top=TOP              Number of timers [default: 11]
"""

import glob
import numpy       as np
import pandas      as pd
from   docopt import docopt
import yaml
import re
import ScalingFilenameParser as SFP

# this file can be refactored into a single parse YAML to CSV.
# it is a quick hack

def construct_dataframe(yaml_data, extra_columns):
  """Construst a pandas DataFrame from the timers section of the provided YAML data"""
  timers = yaml_data['Timer names']
  data = np.ndarray([len(timers), 8+len(extra_columns)])

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

  df = pd.DataFrame(data, index=timers,
                      columns=['minT', 'minC', 'meanT', 'meanC', 'maxT', 'maxC', 'meanCT', 'meanCC']+extra_columns)
  df['Timer Name'] = df.index
  return df

def string_split_by_numbers(x):
  r = re.compile('(\d+)')
  l = r.split(x)
  return [int(x) if x.isdigit() else x for x in l]


if __name__ == '__main__':
  import os
  # Process input
  options = docopt(__doc__)

  input_files = options['--input-files']

  SCALING_TYPE = 'weak'

  if len(input_files) == 1:
    input_files = glob.glob(input_files[0])
    # Impose sorted order (similar to bash "sort -n"
    input_files = sorted(input_files, key=string_split_by_numbers)

  #df_cols = ['Experiment', 'Problem', 'Timer Name', 'num_nodes', 'procs_per_node', 'cores_per_proc', 'threads_per_core',
  #           'Num Threads', 'Max Aggregate Time']
  dataset = pd.DataFrame()
  df_idx = -1
  future_index = ['Experiment', 'Timer Name']

  for input_file in input_files:
    with open(input_file) as data_file:
      yaml_data = yaml.safe_load(data_file)

    my_tokens = SFP.parseYAMLFileName(input_file)
    rebuilt_filename = SFP.rebuild_source_filename(my_tokens)

    if rebuilt_filename == os.path.basename(input_file):
      print("Rebuild OK: {} == {}".format(rebuilt_filename, input_file))
    else:
      print("Rebuild FAIL: {} != {}".format(rebuilt_filename, input_file))
      exit(-1)

    print(my_tokens)

    experiment_id = "{solver_name}{solver_attributes}_{prec_name}{prec_attributes}".format(**my_tokens)

    timer_data = construct_dataframe(yaml_data, list(my_tokens.keys()))

    for key, value in my_tokens.items():
      timer_data[key] = my_tokens[key]

    timer_data['Max Aggregate Time'] = timer_data['maxT'] / timer_data['maxC']
    timer_data['Experiment'] = experiment_id

    if df_idx == -1:
      num_timers = timer_data['Timer Name'].nunique()
      num_files  = len(input_files)
      dataset = pd.DataFrame(columns=list(timer_data), index=np.arange(num_timers*num_files*4))
      future_index = list(my_tokens.keys()) + future_index

    for index, row in timer_data.iterrows():
      df_idx += 1
      dataset.loc[df_idx] = row[:] #new_data[key]
      # new_data = {'Timer Name' : index,
      #             'Experiment' : experiment_id,
      #             'Problem' : row['Problem'],
      #             'num_nodes' : row['num_nodes'],
      #             'procs_per_node' : row['procs_per_node'],
      #             'cores_per_proc' : row['cores_per_proc'],
      #             'threads_per_core': row['threads_per_core'],
      #             'Num Threads' : row['Num Threads'],
      #             'Max Aggregate Time' : row['Max Aggregate Time']}
      # for key in new_data.keys():
      #   dataset.loc[df_idx, key] = new_data[key]

  # drop over allocated data
  dataset = dataset[pd.notnull(dataset['Timer Name'])]
  # set the index, verify it, and sort
  dataset.set_index(keys=['Experiment',
                          'problem_type', 'problem_nx', 'problem_ny', 'problem_nz',
                          'Timer Name',
                          'num_nodes', 'procs_per_node', 'cores_per_proc', 'threads_per_core',
                          'omp_num_threads'],
                    drop=False, inplace=True, verify_integrity=True)

  dataset.sort_values(inplace=True,
                      by=['Experiment',
                          'problem_type', 'problem_nx', 'problem_ny', 'problem_nz',
                          'Timer Name',
                          'num_nodes', 'procs_per_node', 'cores_per_proc', 'threads_per_core',
                          'omp_num_threads'])
  # write the total dataset out, index=False, because we do not drop it above
  dataset.to_csv('all_data.csv', index=False)
