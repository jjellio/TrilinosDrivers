#!/usr/bin/env python3
"""analysis.py

Usage:
  analysis.py --baseline=<FILE> --comparable=<FILES>
              [--output_csv=<FILE>]
              [--average=<averaging>]
              [--bl_affinity_dir=<PATH>]
              [--comparable_affinity_dir=<PATH>]
              [--bl_log_dir=<PATH>]
              [--comparable_log_dir=<PATH>]
              [--remove_string=<STRING>]
              [--total_time_key=<STRING>]
              [--write_percent_total]
              [--muelu_prof]
  analysis.py (-h | --help)

Options:
  -h, --help               Show this screen.
  --baseline=<FILE>       Use this file as the baseline
  --comparable=<FILES>   YAMLS that can be compared to the baseline
  --average=<averaging>   Average the times using callcounts, numsteps, or none [default: none]
  --output_csv=<FILE>     Output file [default: all_data.csv]
  --remove_string=STRING    remove the STRING from timer labels [default: _kokkos]
  --total_time_key=STRING   use this timer key to compute percent total time [default: MueLu: Hierarchy: Setup (total)]
  --bl_affinity_dir=PATH          Path to directory with affinity CSV data, relative to baseline [default: .]
  --comparable_affinity_dir=PATH  Path to directory with affinity CSV data, relative to comparable [default: ../affinity]
  --bl_log_dir=PATH          Path to directory with text logs, relative to baseline [default: .]
  --comparable_log_dir=PATH  Path to directory with text logs, relative to comparable [default: ../affinity]
  --write_percent_total     Write a column with percent total
  --muelu_prof              Do MueLu profile output

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
from tabulate import tabulate
import yaml
from pathlib import Path
import TeuchosTimerUtils as TTU

try:
  import ScalingFilenameParser as SFP
  __HAVE_SCALING_FILE_PARSER = True
except ImportError:
  __HAVE_SCALING_FILE_PARSER = False
  pass


def df_to_markdown(df):
  print(tabulate(df, headers='keys', tablefmt='simple', showindex=False))


def load_yaml(filename):
  with open(filename) as data_file:
    yaml_data = yaml.safe_load(data_file)
    # try to parse c++ mangled names if needed
    # demangeYAML_TimerNames(yaml_data)
    return yaml_data


def file_len(pathlib_filename):
  i = 0

  with pathlib_filename.open() as f:
    for i, l in enumerate(f):
      pass
  return i + 1


# affinity_files have the name:
# affinity_dir/Laplace3D-BS-1-1240x1240x1240_OpenMP-threads-16_np-2048_decomp-128x16x4x1_affinity.csv
# Tokenized:
# # affinity_dir/<problem_type>-BS-1-1240x1240x1240_OpenMP-threads-16_np-2048_decomp-128x16x4x1_affinity.csv
def parse_affinity_data(affinity_path, tokens):
  try:
    affinity_filename = '{path}/{name}'.format(path=affinity_path,
                                               name=SFP.build_affinity_filename(tokens))
  except:
    return

  my_file_lookup = Path(affinity_filename)
  try:
    affinity_file_abs = my_file_lookup.resolve()
  except:
    print('Missing Affinity File: {}'.format(affinity_filename))
    raise

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


def do_averaging(averaging, df, my_tokens):
  if averaging == 'cc':
    df['minT'] = df['minT'] / df['minC']
    df['maxT'] = df['maxT'] / df['maxC']
    df['meanT'] = df['meanT'] / df['meanC']
  elif averaging == 'ns':
    df['minT'] = df['minT'] / my_tokens['numsteps']
    df['maxT'] = df['maxT'] / my_tokens['numsteps']
    df['meanT'] = df['meanT'] / my_tokens['numsteps']
  else:
    return


def add_percent_total(total_time_key, df):
  if total_time_key:
    total_time_row = df.loc[total_time_key]
    timer_types = ['minT', 'maxT', 'meanT', 'meanCT']
    for timer_type in timer_types:
      df['perc_{timer}'.format(timer=timer_type)] = (df[timer_type] / total_time_row[timer_type])
      df['total_time_{timer}'.format(timer=timer_type)] = total_time_row[timer_type]


def construct_short_name_for_file(filename, affinity_dir):
  try:
    my_tokens = SFP.parseYAMLFileName(filename)

    # parse affinity information, because it contains timestamps
    try:
      parse_affinity_data(affinity_path=affinity_dir, tokens=my_tokens)

      short_name = str(my_tokens['num_nodes']) + 'x' + \
                   str(my_tokens['procs_per_node']) + 'x' + \
                   str(my_tokens['cores_per_proc']) + 'x' + \
                   str(my_tokens['threads_per_core'])

      short_name += '_' + pd.to_datetime(my_tokens['timestamp']).strftime('%b-%Y')
    except:
      short_name = 'A'

    return short_name
  except:
    pass

  # attempt try using the filenames
  try:
    short_name = os.path.basename(filename).split('_')[0]
    return short_name
  except:
    return 'A'


def construct_short_names(comparable_files, relative_affinity_dir, comparable_file_mapping={}):
  # construct a short name for this data

  if isinstance(comparable_files, str):
    affinity_dir = os.path.dirname(comparable_files) + '/' + relative_affinity_dir

    short_name = construct_short_name_for_file(affinity_dir=affinity_dir,
                                               filename=comparable_files)
    return short_name
  else:
    short_name_failed = False

    for comparable_file in comparable_files:
      print(comparable_file)
      affinity_dir = os.path.dirname(comparable_file) + '/' + relative_affinity_dir

      short_name = construct_short_name_for_file(affinity_dir=affinity_dir,
                                                 filename=comparable_file)

      if short_name in comparable_file_mapping:
        short_name_failed = True
        break
      else:
        comparable_file_mapping[short_name] = comparable_file

    if short_name_failed:
      print('Failed short naming')
      comparable_file_mapping.clear()
      short_name = chr(ord('A'))

      for comparable_file in comparable_files:
        print(comparable_file)

        comparable_file_mapping[short_name] = comparable_file

        short_name = chr(ord(short_name) + 1)
      print(comparable_file_mapping)


def remove_sparc_label_numbers(df):
  df['Timer Name'].replace(to_replace=r'^(\d+\.?)+:\s*', value='', inplace=True, regex=True)

  df.index = df['Timer Name'].tolist()
  return df


def main():
  # Process input
  options = docopt(__doc__)

  comparable_files  = options['--comparable']
  baseline_file     = options['--baseline']
  bl_affinity_dir         = options['--bl_affinity_dir']
  comparable_affinity_dir = options['--comparable_affinity_dir']
  output_csv        = options['--output_csv']
  remove_string     = options['--remove_string']
  averaging         = options['--average']
  write_percent_total = options['--write_percent_total']
  total_time_key    = options['--total_time_key']
  total_time_key    = ''

  DO_PERCENT_TOTAL = write_percent_total
  NUKE_SPARC_NUMBERS = True

  if remove_string == '':
    remove_string = None

  if total_time_key == '':
    total_time_key   = None
    DO_PERCENT_TOTAL = False

  ignored_labels = [ '0 - Total Time',
                     '1 - Reseting Linear System',
                     '2 - Adjusting Nullspace for BlockSize',
                     '3 - Constructing Preconditioner',
                     '4 - Constructing Solver',
                     '5 - Solve' ]

  print(options)

  # we allow globs on the file names, but it should return a single file
  if '*' in baseline_file:
    baseline_files = glob.glob(baseline_file)
    if len(baseline_files) != 1:
      print('Error locating a baseline file. Globbed the filename, but obtained multiple files.')
      print(baseline_file)
      print(baseline_files)
    baseline_file = baseline_files[0]

  print('baseline_file: {baseline}\n'
        'comparable_files: {comparable}\n'
        'average: {avg}\n'
        'remove_string: {suffix}\n'
        'output_csv: {output}\n'
        'total_time_key: {total_time_key}'.format(baseline=baseline_file,
                                                  avg=averaging,
                                                  output=output_csv,
                                                  suffix=remove_string,
                                                  comparable=comparable_files,
                                                  total_time_key=total_time_key))

  if isinstance(comparable_files, str):
    if '*' in comparable_files:
      print('Globbing comparison files')
      comparable_files = glob.glob(comparable_files)
    else:
      comparable_files = [comparable_files]

  print(comparable_files)

  bl_yaml = load_yaml(baseline_file)

  TTU.remove_timer_string(bl_yaml, string=remove_string)

  my_tokens = {}
  try:
    my_tokens = SFP.parseYAMLFileName(baseline_file)
  except:
    pass

  baseline_df = TTU.construct_dataframe(bl_yaml)
  baseline_df = remove_sparc_label_numbers(baseline_df)

  baseline_df.to_csv('baseline_df-a.csv', index_label='Original Timer')
  baseline_df = TTU.demange_muelu_timer_names_df(baseline_df)
  baseline_df.to_csv('baseline_df.csv', index_label='Demangled Timer')

  # remove ignored timers
  baseline_df = baseline_df.drop(ignored_labels, errors='ignore')

  # perform averaging
  do_averaging(averaging=averaging, df=baseline_df, my_tokens=my_tokens)

  # compute the percentage of total time
  if DO_PERCENT_TOTAL:
    add_percent_total(total_time_key=total_time_key, df=baseline_df)

  # track the global set of timer labels
  unified_timer_names = set(baseline_df.index)

  # construct a short name for this data
  comparable_file_mapping = {}

  construct_short_names(comparable_files=comparable_files,
                        comparable_file_mapping=comparable_file_mapping,
                        relative_affinity_dir=comparable_affinity_dir)

  baseline_short_name = construct_short_names(comparable_files=baseline_file,
                                              relative_affinity_dir=bl_affinity_dir)

  print(comparable_file_mapping)
  for short_name in sorted(comparable_file_mapping.keys()):
    comparable_file = comparable_file_mapping[short_name]

    # we need the tokens, because they contain num_steps, which is
    # an averaging option, since call counts is not appropriate in all cases
    my_tokens = {}
    try:
      my_tokens = SFP.parseYAMLFileName(comparable_file)
      rebuilt_filename = SFP.rebuild_source_filename(my_tokens)

      if rebuilt_filename == os.path.basename(comparable_file):
        print("Rebuild OK: {} == {}".format(rebuilt_filename, comparable_file))
      else:
        print("Rebuild FAIL: {} != {}".format(rebuilt_filename, comparable_file))
        exit(-1)
    except:
      pass

    # load the YAML data
    comparable_yaml = load_yaml(comparable_file)

    # remove a string from the timer labels (_kokkos)
    TTU.remove_timer_string(comparable_yaml, string=remove_string)

    # construct the dataframe
    comparable_df = TTU.construct_dataframe(comparable_yaml)

    comparable_df.to_csv('comparable_df-a.csv', index_label='Original Timer')
    comparable_df = remove_sparc_label_numbers(comparable_df)
    comparable_df.to_csv('comparable_df.csv', index_label='Demangled Timer')

    # drop the unwanted timers
    comparable_df = comparable_df.drop(ignored_labels, errors='ignore')

    # add new timers to the global set of parsed timers (required, because we do not always have the same
    # set of timers)
    unified_timer_names = unified_timer_names.union(set(comparable_df.index))

    # update the dataframe of baseline to use this new index
    baseline_df = baseline_df.reindex(list(unified_timer_names))

    # update the comparable datafrae
    comparable_df = comparable_df.reindex(list(unified_timer_names))

    # apply any averaging to the timers
    do_averaging(averaging=averaging, df=comparable_df, my_tokens=my_tokens)

    # optionally, compute the perct total time
    if DO_PERCENT_TOTAL:
      add_percent_total(total_time_key=total_time_key, df=comparable_df)

    # merge the new timer data into the baseline dataframe using columns "Foo_shortname"
    baseline_df = pd.merge(baseline_df, comparable_df,
                           left_index=True,
                           right_index=True,
                           how='outer',
                           suffixes=('', '_'+short_name))

  baseline_df.to_csv('happy.csv', index=True)
  #baseline_df = baseline_df.dropna(subset=['Timer Name'])
  baseline_df = baseline_df.dropna()
  baseline_df.to_csv('happy-d.csv', index=True)
  timer_types = ['minT', 'maxT', 'meanT', 'meanCT']
  for timer_type in timer_types:
    output_columns = ['Timer Name', timer_type]

    if DO_PERCENT_TOTAL:
      output_columns.append('perc_{timer}'.format(timer=timer_type))

    lookup_column = ''

    for short_name in sorted(comparable_file_mapping.keys()):
      lookup_column = '{timer}_{short_name}'.format(timer=timer_type,
                                                    short_name=short_name)

      speedup_column = '{timer}_speedup_{short_name}'.format(timer=timer_type,
                                                        short_name=short_name)
      baseline_df[speedup_column] = baseline_df[timer_type] / baseline_df[lookup_column]

      # round the speedup  ?
      baseline_df[speedup_column] = pd.Series([round(val, 3) for val in baseline_df[speedup_column] ],
                                              index=baseline_df.index)

      output_columns.append(lookup_column)
      output_columns.append(speedup_column)

      if DO_PERCENT_TOTAL:
        output_columns.append('perc_{timer}_{short_name}'.format(timer=timer_type,
                                                                 short_name=short_name))
        print(output_columns)
      if total_time_key is not None:
        output_columns.append('total_time_{timer}_{short_name}'.format(timer=timer_type,
                                                                       short_name=short_name))

    fname = '{timer}_comparison'.format(timer=timer_type)
    if averaging != 'none':
      fname += '_averaged_by_{avg}'.format(avg=averaging)

    # slice this data off and rank/sort
    data_slice = baseline_df[output_columns]

    data_slice.to_csv('{fname}.csv'.format(fname=fname),
                      index=False,
                      columns=output_columns)

    df_to_markdown(data_slice[output_columns])

    # data_slice.to_csv('{fname}.csv'.format(fname=fname),
    #                    index_label='Timer Name',
    #                    index=True,
    #                    columns=output_columns)

  for short_name in sorted(comparable_file_mapping.keys()):
    comparable_file = comparable_file_mapping[short_name]
    print('{short_name}: {file}'.format(short_name=short_name, file=comparable_file))


if __name__ == '__main__':
  main()
