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
  -h, --help                 Show this screen.
  --baseline=<FILE>          Use this file as the baseline
  --comparable=<FILES>       YAMLS that can be compared to the baseline
  --average=<averaging>      Average the times using callcounts, numsteps, or none [default: none]
  --output_csv=<FILE>        Output file [default: all_data.csv]
  --remove_string=STRING     remove the STRING from timer labels [default: _kokkos]
  --total_time_key=STRING    use this timer key to compute percent total time [default: MueLu: Hierarchy: Setup (total)]
  --bl_affinity_dir=PATH          Path to directory with affinity CSV data, relative to baseline [default: .]
  --comparable_affinity_dir=PATH  Path to directory with affinity CSV data, relative to comparable [default: ../affinity]
  --bl_log_dir=PATH          Path to directory with text logs, relative to baseline [default: .]
  --comparable_log_dir=PATH  Path to directory with text logs, relative to comparable [default: .]
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
import yaml
from pathlib import Path
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

  df['Timer Name'] = df.index
  return df


def df_to_markdown(df):
  from tabulate import tabulate
  print(tabulate(df, headers='keys', tablefmt='simple', showindex=False))


def demangeDF_TimerNames(df):
  df['Timer Name'].replace(to_replace=r'N\d+MueLu\d+', value='', inplace=True, regex=True)
  df['Timer Name'].replace(to_replace=r'I[d]*ixN\d+Kokkos\d+Compat\d+KokkosDeviceWrapperNodeINS\d+_\d+(OpenMPENS|SerialENS)\d+_\d+HostSpace[E]+', value='', inplace=True, regex=True)
  df.index = df['Timer Name'].tolist()

  return df


def demange_timer_name(timer_name):
  import re
  rebuilt_name = re.sub(r'N\d+MueLu\d+', '', timer_name)
  rebuilt_name = re.sub(r'I[d]*ixN\d+Kokkos\d+Compat\d+KokkosDeviceWrapperNodeINS\d+_\d+(OpenMPENS|SerialENS)\d+_\d+HostSpace[E]+', '', rebuilt_name)

  return rebuilt_name


def demangeYAML_TimerNames(yaml_data):
  import re
  # import copy

  # prior_timers = copy.deepcopy(yaml_data['Timer names'])

  for idx, timer_name in enumerate(yaml_data['Timer names']):
    rebuilt_name = demange_timer_name(timer_name)

    if rebuilt_name == timer_name:
      print('Identical: {}'.format(rebuilt_name))
      continue

    rename_teuchos_timers_yaml_key(old_key=timer_name,
                                   yaml_data=yaml_data,
                                   new_key=rebuilt_name,
                                   timer_index=idx)

    print(rebuilt_name)


def rename_teuchos_timers_yaml_key(old_key,
                                   yaml_data,
                                   new_key,
                                   timer_index):
  try:
    yaml_data['Total times'][new_key] = yaml_data['Total times'].pop(old_key)
    yaml_data['Call counts'][new_key] = yaml_data['Call counts'].pop(old_key)

    yaml_data['Timer names'][timer_index] = new_key
  except KeyError as e:
    print(
      'Attempting to remove suffix and prefix from timer names, but we have a collision: {} has already been mapped.'.format(
        new_key))
    raise e


def remove_timer_string(yaml_data,
                        string=None):

  import copy

  if not string:
    return

  # we modify the yaml_data, so make a copy
  prior_timers = copy.deepcopy(yaml_data['Timer names'])

  for idx, timer_name in enumerate(prior_timers):
    # remove the string from the timer label, then adjust the dict
    rebuilt_timer_name = timer_name.replace(string, '')

    # no modifications so skip to the next
    if rebuilt_timer_name == timer_name:
      continue

    rename_teuchos_timers_yaml_key(old_key=timer_name,
                                   yaml_data=yaml_data,
                                   new_key=rebuilt_timer_name,
                                   timer_index=idx)


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
  affinity_filename = '{path}/{name}'.format(path=affinity_path,
                                             name=SFP.build_affinity_filename(tokens))

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
  if averaging == 'ns':
    df['minT'] = df['minT'] / my_tokens['numsteps']
    df['maxT'] = df['maxT'] / my_tokens['numsteps']
    df['meanT'] = df['meanT'] / my_tokens['numsteps']


def add_percent_total(total_time_key, df):
  if total_time_key:
    total_time_row = df.loc[total_time_key]
    timer_types = ['minT', 'maxT', 'meanT', 'meanCT']
    for timer_type in timer_types:
      df['perc_{timer}'.format(timer=timer_type)] = (df[timer_type] / total_time_row[timer_type])
      df['total_time_{timer}'.format(timer=timer_type)] = total_time_row[timer_type]


def construct_short_name_for_file(filename, affinity_dir):
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


def construct_short_names(comparable_files, relative_affinity_dir, comparable_file_mapping={}):
  # import datetime

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
      comparable_file_mapping = {}
      short_name = chr(ord('A'))

      for comparable_file in comparable_files:
        print(comparable_file)

        comparable_file_mapping[short_name] = comparable_file

        short_name = chr(ord(short_name) + 1)


def relableTpetraExperimentTimers(df):
  # if we have MVT::MvInit and not MVT::MvInit0
  # then the rename MVT::MvInit to MVT::MvInit0, because it was a zero initialize
  # if we have MVT::MvScale and not MVT::MvScale1
  # rename to MvScale1, because that is what the first version of that experiment did.
  return df


def parse_teuchos_timers_from_log_file(YAML_filename, relative_log_dir):
  import os

  # parse the YAML filename
  my_tokens = SFP.parseYAMLFileName(YAML_filename)

  # weakscaling_openmp_Laplace3D-246x246x246_decomp-1x64x1x1_knl-quad-cache-onyx.log-ok
  logfile_fmt_str = 'weakscaling_{lexecspace_name}_' \
                    '{problem_type}-{problem_nx}x{problem_ny}x{problem_nz}_' \
                    'decomp-{num_nodes}x{procs_per_node}x{cores_per_proc}x{threads_per_core}_' \
                    '*.log-ok'
  # construct the log file's name
  logfile_name = logfile_fmt_str.format(**my_tokens)

  # form an absolute path
  log_dir = os.path.dirname(os.path.realpath(YAML_filename)) + '/' + relative_log_dir
  print('log_dir: ', os.path.dirname(os.path.realpath(YAML_filename)), 'relative: ', relative_log_dir)
  logfile_path = log_dir + '/' + logfile_name
  print(logfile_path)
  # glob the variable parts of the log name
  logfiles = glob.glob(logfile_path)
  print(logfiles)
  # we should find a single logfile
  if len(logfiles) != 1:
    print('Searched for logfiles, but found more than one. Move the similar log file away.')
    print('log files found: ', len(logfiles))
    for log in logfiles:
      print(log)
    exit(-1)

  # parse the timer names from the log file
  return gather_timer_name_sets_from_logfile(logfiles[0])


def gather_timer_name_sets_from_logfile(logfile):
  import re
  # we form N number of sets (lists technically)
  set_counter = 0
  sets = {}

  float_re_str = r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'
  timing_and_callcount_re_str = r'(\s+{float}\s+\({float}\))'.format(float=float_re_str)
  four_column_re_str = r'(?P<label>.+)' + timing_and_callcount_re_str + r'{4}'
  single_column_re_str = r'(?P<label>.+)' + timing_and_callcount_re_str + r'{1}'

  four_column_re = re.compile(four_column_re_str)
  single_column_re = re.compile(single_column_re_str)

  # track when we enter a Teuchos Timer output block
  in_teuchos_timer_output_block = False

  # loop line by line
  with open(logfile) as fptr:

    for line in fptr:
      # a region starts with the column headers, we assume these are not repeated within a region
      if re.match(r'^Timer Name\s*MinOverProcs\s*MeanOverProcs\s*MaxOverProcs\s*MeanOverCallCounts\s*$', line):
        # print('Region: ' + str(in_teuchos_timer_output_block) + '  ' + line)
        in_teuchos_timer_output_block = True
        # start a region, so start a new set for these timer labels
        sets[set_counter] = list()
        continue

      # if we are inside a timer block then match timer labels
      if in_teuchos_timer_output_block:
        # if inside a region, look for a line that is all '=', this marks
        # the end of the region
        endregion_m = re.match(r'^[=]+$', line)
        # match a label by matching FOUR pairs of 'timing (counts)'
        #four_column_label_m = re.match(r'(?P<label>.+)(\s+[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?\s+\([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?\)){4}',line)
        four_column_label_m = four_column_re.match(line)
        single_column_label_m = single_column_re.match(line)

        if endregion_m:
          in_teuchos_timer_output_block = False
          set_counter += 1
        elif four_column_label_m:

          label = four_column_label_m.group('label').strip()
          #print('matched 4: ', label)
          sets[set_counter].append(label)
        elif single_column_label_m:
          label = single_column_label_m.group('label').strip()
          #print('matched 1: ', label)
          sets[set_counter].append(label)
        else:
          continue

  return sets


def reconcile_timer_set_and_YAML(timer_list, sets):
  num_sets = len(sets)
  timer_set = set(timer_list)
  expected_labels = len(timer_set)
  total_missing = 0

  print('---------------------------------------------------')
  for set_id in sets:
    if any('MueLu' in x for x in sets[set_id]):
      print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
      print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
      print('Testing set:')
      print(sets[set_id])
      print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
      print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    else:
      print(sets[set_id])
      print('Skipping set, muelu not contained in labels')
      continue

    potential_set = set(sets[set_id])
    matching_labels = timer_set.intersection(potential_set)
    labels_in_yaml_not_in_stdout = timer_set.difference(potential_set)
    labels_in_stdout_not_in_yaml = potential_set.difference(timer_set)
    num_matches = len(matching_labels)
    num_missing = len(labels_in_stdout_not_in_yaml) + len(labels_in_yaml_not_in_stdout)

    if num_missing == 0:
      print('Missing NONE!')
    else:
      total_missing += num_missing
      print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
      print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
      print('Matched: ', num_matches)
      print('\n'.join(matching_labels))
      print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
      print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
      print("Labels in Stdout but not in YAML: ", len(labels_in_stdout_not_in_yaml))
      print('\n'.join(labels_in_stdout_not_in_yaml))
      print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
      print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
      print("Labels in YAML but not in Stdout: ", len(labels_in_yaml_not_in_stdout))
      print('\n'.join(labels_in_yaml_not_in_stdout))
      print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
      print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
      # df = pd.DataFrame()
      # unified_timers = list(timer_set.union(potential_set))
      # df['Timer Names'] = unified_timers
      # df.index = unified_timers

  print('---------------------------------------------------')
  print('total missing: ', total_missing)
  print('---------------------------------------------------')
  return total_missing


def main():
  # Process input
  from docopt import DocoptExit
#  try:
  options = docopt(__doc__)
#  except DocoptExit:
#    print(__doc__)
#    exit(0)

  comparable_files  = options['--comparable']
  baseline_file     = options['--baseline']
  bl_affinity_dir         = options['--bl_affinity_dir']
#    print(__doc__)
#    exit(0)

  comparable_files  = options['--comparable']
  baseline_file     = options['--baseline']
  bl_affinity_dir         = options['--bl_affinity_dir']
  comparable_affinity_dir = options['--comparable_affinity_dir']
  bl_log_dir         = options['--bl_log_dir']
  comparable_log_dir = options['--comparable_log_dir']
  output_csv        = options['--output_csv']
  remove_string     = options['--remove_string']
  averaging         = options['--average']
  write_percent_total = options['--write_percent_total']
  muelu_prof        = options['--muelu_prof']
  total_time_key    = options['--total_time_key']
  total_time_key    = ''

  DO_MUELU_COMP    = muelu_prof
  DO_PERCENT_TOTAL = write_percent_total

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

  baseline_file = glob.glob(baseline_file)[0]

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

  if isinstance(comparable_files, str) and '*' in comparable_files:
    print('Globbing')
    comparable_files = glob.glob(comparable_files)
  else:
    comparable_files = [comparable_files]

  print(comparable_files)

  sets = parse_teuchos_timers_from_log_file(YAML_filename=baseline_file,
                                            relative_log_dir=bl_log_dir)
  bl_yaml = load_yaml(baseline_file)
  num_missing = reconcile_timer_set_and_YAML(timer_list=bl_yaml['Timer names'], sets=sets)

  print('Missing Labels: ', num_missing)

  # print('---------------------------------------------------------------------')
  # import pprint
  # pprint.PrettyPrinter(indent=4).pprint(bl_yaml)
  # print('---------------------------------------------------------------------', )

  remove_timer_string(bl_yaml, string=remove_string)

  my_tokens = SFP.parseYAMLFileName(baseline_file)
  baseline_df = construct_dataframe(bl_yaml)

  baseline_df.to_csv('baseline_df-a.csv', index_label='Original Timer')
  baseline_df = demangeDF_TimerNames(baseline_df)
  baseline_df.to_csv('baseline_df.csv', index_label='Demangled Timer')

  # remove ignored timers
  baseline_df = baseline_df.drop(ignored_labels, errors='ignore')

  # perform averaging
  do_averaging(averaging=averaging, df=baseline_df, my_tokens=my_tokens)

  # compute the percentage of total time
  if DO_PERCENT_TOTAL:
    add_percent_total(total_time_key=total_time_key, df=baseline_df)

  # this will drastically reduce the number of timer labels
  if DO_MUELU_COMP:
    baseline_df = annotate_muelu_dataframe(df=baseline_df, total_time_key=total_time_key)

  # track the global set of timer labels
  unified_timer_names = set(baseline_df.index)

  # construct a short name for this data
  comparable_file_mapping = {}

  construct_short_names(comparable_files=comparable_files,
                        comparable_file_mapping=comparable_file_mapping,
                        relative_affinity_dir=comparable_affinity_dir)

  baseline_short_name = construct_short_names(comparable_files=baseline_file,
                                              relative_affinity_dir=bl_affinity_dir)

  for short_name in sorted(comparable_file_mapping.keys()):
    comparable_file = comparable_file_mapping[short_name]

    # we need the tokens, because they contain num_steps, which is
    # an averaging option, since call counts is not appropriate in all cases
    my_tokens = SFP.parseYAMLFileName(comparable_file)
    rebuilt_filename = SFP.rebuild_source_filename(my_tokens)

    if rebuilt_filename == os.path.basename(comparable_file):
      print("Rebuild OK: {} == {}".format(rebuilt_filename, comparable_file))
    else:
      print("Rebuild FAIL: {} != {}".format(rebuilt_filename, comparable_file))
      exit(-1)

    # load the YAML data
    comparable_yaml = load_yaml(comparable_file)

    # remove a string from the timer labels (_kokkos)
    remove_timer_string(comparable_yaml, string=remove_string)

    sets = parse_teuchos_timers_from_log_file(YAML_filename=comparable_file,
                                              relative_log_dir=comparable_log_dir)
    num_missing = reconcile_timer_set_and_YAML(timer_list=comparable_yaml['Timer names'], sets=sets)

    print('Missing Labels: ', num_missing)
    # construct the dataframe
    comparable_df = construct_dataframe(comparable_yaml)

    comparable_df.to_csv('comparable_df-a.csv', index_label='Original Timer')
    comparable_df = demangeDF_TimerNames(comparable_df)
    comparable_df.to_csv('comparable_df.csv', index_label='Demangled Timer')

    # print('---------------------------------------------------------------------')
    # import pprint
    # pprint.PrettyPrinter(indent=4).pprint(comparable_yaml)
    # print('---------------------------------------------------------------------', )

    # drop the unwanted timers
    comparable_df = comparable_df.drop(ignored_labels, errors='ignore')

    if DO_MUELU_COMP:
      comparable_df = annotate_muelu_dataframe(df=comparable_df, total_time_key=total_time_key)

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


  baseline_df.to_csv('happy.csv')
  timer_types = ['minT', 'maxT', 'meanT', 'meanCT']
  for timer_type in timer_types:
    output_columns = [timer_type]

    if DO_PERCENT_TOTAL:
      output_columns.append('perc_{timer}'.format(timer=timer_type))

    lookup_column = ''

    for short_name in sorted(comparable_file_mapping.keys()):
      comparable_file = comparable_file_mapping[short_name]
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
    data_slice = add_running_total_and_rank(data_slice, column_name=timer_type, prefix_label='bl_')

    if DO_MUELU_COMP:
      data_slice = add_running_total_and_rank(data_slice, column_name=lookup_column, prefix_label='')
      # data_slice = data_slice.sort_values(by=[lookup_column], ascending=True)
      # data_slice['running'] = data_slice[lookup_column].cumsum()
      data_slice['rank_delta'] = data_slice['rank'] - data_slice['bl_rank']
      output_columns.append('running')
      output_columns.append('rank')
      output_columns.append('rank_delta')
      data_slice['Timer Name'] = data_slice.index.values
      output_columns = ['Timer Name'] + output_columns

      data_slice = data_slice[data_slice['running'].notnull()]
      data_slice = data_slice.reset_index(drop=True)
      data_slice = data_slice.sort_index(ascending=True).reset_index()
      data_slice = data_slice.rename(columns={timer_type : baseline_short_name})
      output_columns[output_columns.index(timer_type)] = baseline_short_name

      output_columns.remove('rank')
      output_columns.remove('rank_delta')
      output_columns = ['rank', 'rank_delta'] + output_columns

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


def add_running_total_and_rank(df, column_name, prefix_label):
  df = df.sort_values(by=[column_name], ascending=True)
  running_label = '{prefix}running'.format(prefix=prefix_label)
  rank_label = '{prefix}rank'.format(prefix=prefix_label)

  df[running_label] = df[column_name].cumsum()
  df[rank_label] = df[running_label].rank(ascending=True)
  return df

  # df = df[df[running_label].notnull()]

  #data_slice = data_slice.sort_index(ascending=True).reset_index()


def annotate_muelu_dataframe (df, total_time_key):
  import re
  import os

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
    # r'(?P<label_prefix>MueLu: SaPFactory: Fused \(I-omega\*D\^{-1} A\)\*Ptent \(sub, total,)\s*level=(?P<level_number>[0-9]*)\)(?P<label_suffix>.*)',
    r'(?P<label_prefix>MueLu: RAPFactory: Computing Ac)\s*\(level=(?P<level_number>[0-9]*)\)(?P<label_suffix>.*)',
    # r'(?P<label_prefix>MueLu: RAPFactory: MxM: A x P \(sub, total,)\s*level=(?P<level_number>[0-9]*)\)(?P<label_suffix>.*)',
    # r'(?P<label_prefix>MueLu: RAPFactory: MxM: P\' x \(AP\) \(implicit\) \(sub, total,)\s*level=(?P<level_number>[0-9]*)\)(?P<label_suffix>.*)',
    r'(?P<label_prefix>MueLu: RepartitionHeuristicFactory: Build)\s*\(level=(?P<level_number>[0-9]*)\)(?P<label_suffix>.*)',
    r'(?P<label_prefix>MueLu: RebalanceTransferFactory: Build)\s*\(level=(?P<level_number>[0-9]*)\)(?P<label_suffix>.*)',
    r'(?P<label_prefix>MueLu: RebalanceAcFactory: Computing Ac)\s*\(level=(?P<level_number>[0-9]*)\)(?P<label_suffix>.*)',
    # r'(?P<label_prefix>MueLu: RebalanceAcFactory: Rebalancing existing Ac \(sub, total,)\s*level=(?P<level_number>[0-9]*)\)(?P<label_suffix>.*)',
    # r'(?P<label_prefix>TpetraExt MueLu::R\*\(AP\)-implicit)-(?P<level_number>[0-9]*):\s*(?P<label_suffix>MMM All Multiply)'
  ]

  # gather all timer labels
  timer_names = df.index.values

  df['level'] = int(-1)
  df['total'] = False

  level_timers_re = []
  for re_str in level_timers:
    level_timers_re.append(re.compile(re_str))

  for level_timer_re in level_timers_re:
    for timer_name in timer_names:

      m = re.search(r'\(\s*total\s*\)', timer_name)
      if m:
        df.loc[timer_name, 'total'] = True
      else:
        for m in [level_timer_re.search(timer_name)]:
          if m:
            df.loc[timer_name, 'level'] = int(m.group('level_number'))

  level_df = df[ df['level'] != -1]
  return level_df


if __name__ == '__main__':
  main()
