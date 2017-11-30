#!/usr/bin/env python3
"""analysis.py

Usage:
  analysis.py --input-files=FILES
              [--output-csv=FILE]
              [--affinity-dir=PATH]
              [--skip-affinity]
  analysis.py (-h | --help)

Options:
  -h --help                     Show this screen.
  -i FILE --input-files=FILE    Input file
  -o FILE --output-csv=FILE     Output file [default: all_data.csv]
  -a PATH --affinity-dir=PATH   Path to directory with affinity CSV data [default: ../affinity]
  --skip-affinity               Do not process affinity
"""

import glob
import numpy       as np
import pandas      as pd
import docopt as dpt
import yaml
from pathlib import Path
import re
import ScalingFilenameParser as SFP
import TeuchosTimerUtils as TTU

# this is used to preallocate the dataframe. It does not need to be perfect, simply large enough
MAX_NUM_TIMERS = 1024


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

  try:
    my_file_lookup = Path(affinity_filename)

    affinity_file_abs = my_file_lookup.resolve()
  except FileNotFoundError or RuntimeError:
    print('Missing Affinity File: {}'.format(affinity_filename))
    return

  if affinity_file_abs.is_file() is False:
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


def string_split_by_numbers(x):
  r = re.compile('(\d+)')
  l = r.split(x)
  return [int(x) if x.isdigit() else x for x in l]


if __name__ == '__main__':
  import os
  # Process input
  options = dpt.docopt(__doc__)
  print(dpt.__file__)

  input_files  = options['--input-files']
  affinity_dir = options['--affinity-dir']
  output_csv   = options['--output-csv']
  parse_affinity = options['--skip-affinity'] is False

  print(options)

  SCALING_TYPE = 'weak'

  if isinstance(input_files, str) and '*' in input_files:
    print('Globbing')
    input_files = glob.glob(input_files)
  else:
    input_files = [input_files]

  dataset = pd.DataFrame()
  df_idx = -1
  future_index = SFP.getIndexColumns(execspace_name='OpenMP')

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

    if parse_affinity:
      print('Parsing affinity')
      parse_affinity_data(affinity_path=affinity_dir, tokens=my_tokens)

    print(my_tokens)

    # parse the YAML timers
    timer_data = TTU.construct_dataframe(yaml_data, list(my_tokens.keys()))

    # add the experiment's other data (cores, threads, timestamp, etc..)
    for key, value in my_tokens.items():
      timer_data[key] = my_tokens[key]

    # pre allocate the dataframe, this could fail. Currently, we allocate the number of timers, in the first file
    # read, times 4, and then assume that will be true for all files read.
    # This just needs to be large enough to hold all data, then we delete the unused rows.
    # preallocating greatly speeds up the insertion phase.
    if df_idx == -1:
      num_timers = timer_data['Timer Name'].nunique()
      num_files  = len(input_files)

      dataset = pd.DataFrame(columns=list(timer_data), index=np.arange(MAX_NUM_TIMERS*num_files*4))

    for index, row in timer_data.iterrows():
      df_idx += 1
      dataset.loc[df_idx] = row[:]

  # drop over allocated data
  dataset = dataset[pd.notnull(dataset['Timer Name'])]

  index_columns = SFP.getIndexColumns(execspace_name='OpenMP')

  new_columns = list(dataset)
  index_columns = list(set(index_columns).intersection(set(new_columns)))

  # set the index, verify it, and sort
  dataset = dataset.set_index(keys=index_columns,
                              drop=False,
                              verify_integrity=True)

  dataset = dataset.sort_index()

  # write the total dataset out, index=False, because we do not drop it above
  dataset.to_csv(output_csv, index=False)
