#!/usr/bin/env python3
import numpy as np
import yaml
import copy as cp

try:
  import ScalingFilenameParser as SFP
  __HAVE_SCALING_FILE_PARSER = True
except ImportError:
  __HAVE_SCALING_FILE_PARSER = False
  pass

try:
  import pandas as pd
  __HAVE_PANDAS = True
except ImportError:
  __HAVE_PANDAS = False
  pass


def demange_muelu_timer_names_df(df,
                                 set_index=True):
  df['Timer Name'].replace(to_replace=r'N\d+MueLu\d+', value='', inplace=True, regex=True)
  df['Timer Name'].replace(to_replace=r'I[d]*ixN\d+Kokkos\d+Compat\d+KokkosDeviceWrapperNodeINS\d+_\d+(OpenMPENS|SerialENS)\d+_\d+HostSpace[E]+', value='', inplace=True, regex=True)
  if set_index:
    df.index = df['Timer Name'].tolist()

  return df


def demangle_muelu_timer_name(timer_name):
  import re
  rebuilt_name = re.sub(r'N\d+MueLu\d+', '', timer_name)
  rebuilt_name = re.sub(r'I[d]*ixN\d+Kokkos\d+Compat\d+KokkosDeviceWrapperNodeINS\d+_\d+(OpenMPENS|SerialENS)\d+_\d+HostSpace[E]+', '', rebuilt_name)

  return rebuilt_name


def demange_muelu_timer_names_yaml(yaml_data,
                                   verbose=0):

  for idx, timer_name in enumerate(yaml_data['Timer names']):
    rebuilt_name = demangle_muelu_timer_name(timer_name)

    if rebuilt_name == timer_name:
      if verbose: print('Identical: {}'.format(rebuilt_name))
      continue

    rename_teuchos_timers_yaml_key(old_key=timer_name,
                                   yaml_data=yaml_data,
                                   new_key=rebuilt_name,
                                   timer_index=idx)

    if verbose: print(rebuilt_name)


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
  if not string:
    return

  # we modify the yaml_data, so make a copy
  prior_timers = cp.deepcopy(yaml_data['Timer names'])

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


# optionally take extra columns, this is helpful when parsing data into a large CSV
def construct_dataframe(yaml_data, extra_columns=[]):
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


# def construct_dataframe(yaml_data):
#   """Construst a pandas DataFrame from the timers section of the provided YAML data"""
#   timers = yaml_data['Timer names']
#   data = np.ndarray([len(timers), 8])
#
#   ind = 0
#   for timer in timers:
#     t = yaml_data['Total times'][timer]
#     c = yaml_data['Call counts'][timer]
#     data[ind, 0:8] = [
#       t['MinOverProcs'], c['MinOverProcs'],
#       t['MeanOverProcs'], c['MeanOverProcs'],
#       t['MaxOverProcs'], c['MaxOverProcs'],
#       t['MeanOverCallCounts'], c['MeanOverCallCounts']
#     ]
#     ind = ind + 1
#
#   df = pd.DataFrame(data,
#                     index=timers,
#                     columns=['minT', 'minC', 'meanT', 'meanC', 'maxT', 'maxC', 'meanCT', 'meanCC'])
#
#   df['Timer Name'] = df.index
#   return df


'''
  The following function shouldn't be here.
  Reading is standard, but writing is non-standard, as we carry metadata
'''
def load_yaml(filename):
  with open(filename) as data_file:
    yaml_data = yaml.safe_load(data_file)
    # try to parse c++ mangled names if needed
    # demangeYAML_TimerNames(yaml_data)
    return yaml_data


def write_yaml(yaml_data,
               filename=None,
               useFileParserFieldIfPresent=True,
               verbose=1):

  # delete the raw text blob if it exists, we make a copy
  # since dicts and lists are mutable (and passed by reference)
  local_yaml = yaml_data
  if local_yaml['raw_text']:
    local_yaml = deepcopy(yaml_data)
    del local_yaml['raw_text']

  # check if we parsed any special filename for this data
  # this only works with the experiment stuff from jjellio
  if useFileParserFieldIfPresent and yaml_data['experiment_file'] != '':
    filename = yaml_data['experiment_file']

  # dump the data to yaml, this seems fragile and the formatting is not consistent
  # with that produced by teuchos timers
  with open(filename, 'x') as yaml_file:
    yaml.dump(local_yaml,
              yaml_file,
              canonical=True,
              default_flow_style=False)

    if verbose > 0:
      print('Wrote: {filename}'.format(filename=filename))


def write_raw_unparsed_teuchos_table(yaml_data,
                                     write_to_stdout=True,
                                     filename=None,
                                     useFileParserFieldIfPresent=True,
                                     verbose=1):
  if write_to_stdout:
    print(''.join(yaml_data['raw_text']))

  if useFileParserFieldIfPresent and yaml_data['experiment_file'] != '':
    filename = yaml_data['experiment_file'].replace('.yaml', '.raw.txt')

  if filename is not None:
    f = open(filename, 'x')
    f.write(''.join(yaml_data['raw_text']))

    if verbose > 0:
      print('Wrote: {filename}'.format(filename=filename))

    f.close()


def yaml_dict_to_teuchos_table(yaml_data,
                               write_to_stdout=True,
                               filename=None,
                               useFileParserFieldIfPresent=True,
                               verbose=1):
  from tabulate import tabulate

  table_rows = []

  # header
  table_header = ['Timer Name']
  for stat_name in yaml_data['Statistics collected']:
    table_header.append(stat_name)
    table_header.append('')

  # data
  for timer_name in yaml_data['Timer names']:
    table_row = [ timer_name ]
    for stat_name in yaml_data['Statistics collected']:
      table_row.append('{}'.format(yaml_data['Total times'][timer_name][stat_name]))
      table_row.append('({})'.format(yaml_data['Call counts'][timer_name][stat_name]))

    table_rows.append(table_row)

  # create a tabular table using the simple mode
  table_str = tabulate(tabular_data=table_rows, headers=table_header, tablefmt='simple')
  # use a list, as we will rearrange the lines
  table_lines = table_str.splitlines()
  # grab the separator
  # massage the separator to match Tuechos'
  separator_line = table_lines[1].replace(' ','-')
  # figure out the line length
  line_width = len(separator_line.strip())

  # construct the region seperator
  region_line = '=' * line_width

  # construct the teuchos banner
  banner_line = 'TimeMonitor results over ' + str(yaml_data['Number of processes']) + ' processor'
  if yaml_data['Number of processes'] > 1:
    banner_line += 's'
  banner_line = (' ' * int((line_width - len(banner_line))/2 + 1)) + banner_line

  # construct the output
  massaged_output = [region_line, '', banner_line, '', table_lines[0]]

  # add the separators
  i = 0
  for line in table_lines[2:]:
    if i % 10 == 0:
      massaged_output.append(separator_line)
    massaged_output.append(line)
    i += 1
  # end the timing region
  massaged_output.append(region_line)

  if write_to_stdout:
    print('\n'.join(massaged_output))

  if useFileParserFieldIfPresent and yaml_data['experiment_file'] != '':
    filename = yaml_data['experiment_file'].replace('.yaml', '.txt')

  if filename is not None:
    f = open(filename, 'x')
    f.write('\n'.join(massaged_output))

    if verbose > 0:
      print('Wrote: {filename}'.format(filename=filename))

    f.close()


def gather_timer_name_sets_from_logfile(logfile):
  import re
  from copy import deepcopy

  PARSER_DEBUG = False

  # we form N number of sets (lists technically)
  set_counter = 0
  sets = {}

  yaml_four_data = { 'Timer names' : list(),
                     'Total times' : dict(),
                     'Call counts' : dict(),
                     'Number of processes' : int(0),
                     'Statistics collected' : ['MinOverProcs', 'MeanOverProcs', 'MaxOverProcs', 'MeanOverCallCounts'],
                     'raw_text' : [],
                     'experiment_file' : ''}

  yaml_single_data = { 'Timer names' : list(),
                       'Total times' : dict(),
                       'Call counts' : dict(),
                       'Number of processes' : int(0),
                       'Statistics collected' : ['Total'],
                       'raw_text': [],
                       'experiment_file' : ''}

  process_banner_re = re.compile(r'^\s*TimeMonitor results over\s+(?P<num_procs>\d+)\s+processor[s]?\s*$')
  blank_line_re = re.compile(r'^\s*$')
  timer_line_separator_re = re.compile(r'^[-]+$')

  float_re_str = r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'
  # timing_and_callcount_re_str = r'(\s+{float}\s+\({float}\))'.format(float=float_re_str)

  four_column_header_re = re.compile(r'^Timer Name\s*MinOverProcs\s*MeanOverProcs\s*MaxOverProcs\s*MeanOverCallCounts\s*$')
  single_column_header_re = re.compile(r'^Timer Name\s*Global time \(num calls\)\s*$')

  four_column_re_str = r'(?P<label>.+)' \
                     + r'(\s+(?P<MinOverProcsT>{float})\s+\((?P<MinOverProcsC>{float})\))'.format(float=float_re_str) \
                     + r'(\s+(?P<MeanOverProcsT>{float})\s+\((?P<MeanOverProcsC>{float})\))'.format(float=float_re_str) \
                     + r'(\s+(?P<MaxOverProcsT>{float})\s+\((?P<MaxOverProcsC>{float})\))'.format(float=float_re_str) \
                     + r'(\s+(?P<MeanOverCallCountsT>{float})\s+\((?P<MeanOverCallCountsC>{float})\))'.format(float=float_re_str)

  single_column_re_str = r'(?P<label>.+)' \
                       + r'(\s+(?P<TotalT>{float})\s+\((?P<TotalC>{float})\))'.format(float=float_re_str)

  four_column_re = re.compile(four_column_re_str)
  single_column_re = re.compile(single_column_re_str)

  # track when we enter a Teuchos Timer output block
  in_teuchos_four_timer_output_block = False
  in_teuchos_single_timer_output_block = False

  teuchos_num_procs = None
  timer_region_line_counter = 0

  timer_region_re = re.compile(r'^[=]+$')

  raw_text_lines = []

  '''
    --------------------------------------------------------------------------------
    ------  Laplace3D-BS-1-246x246x246_LinearAlgebra-Tpetra_numsteps-100_OpenMP-threads-8_np-8_decomp-1x8x8x1.yaml
    --------------------------------------------------------------------------------

  '''

  experiment_output_match_re = re.compile(r'^------  (?P<filename>[-_0-9A-Za-z]+\.yaml)\s*$')
  scaling_experiment_name = ''

  # loop line by line
  with open(logfile) as fptr:

    for line in fptr:
      '''
      ================================================================================

                      TimeMonitor results over 1 processor
      '''
      # test for the start or end
      if timer_region_line_counter == 0:
        timer_region_m = timer_region_re.match(line)
        if timer_region_m and (timer_region_line_counter == 0):
          timer_region_line_counter = 1
          raw_text_lines = [line]
          if PARSER_DEBUG: print('Matched START region: ', line)
          continue
        else:
          # if line_counter == 0, and we don't see a region, then skip all
          experiment_output_m = experiment_output_match_re.match(line)
          if experiment_output_m:
            scaling_experiment_name = experiment_output_m.group('filename')
            if PARSER_DEBUG: print('MATCHED scaling study experiment filename: ', scaling_experiment_name)
            continue

          if PARSER_DEBUG: print('Skipping: ', line)
          continue

      elif timer_region_line_counter == 1 or timer_region_line_counter == 3:
        blank_line_m = blank_line_re.match(line)
        if blank_line_m:
          raw_text_lines.append(line)
          if PARSER_DEBUG: print('Matched BLANK LINE region: ', line, ' line count: ', timer_region_line_counter)
          timer_region_line_counter += 1
          continue
        else:
          timer_region_line_counter = 0
          if PARSER_DEBUG: print('No BLANK LINE Skipping: ', line)
          continue

      elif timer_region_line_counter == 2:
        timer_process_banner_m = process_banner_re.match(line)
        if timer_process_banner_m:
          timer_region_line_counter += 1
          teuchos_num_procs = int(timer_process_banner_m.group('num_procs'))
          raw_text_lines.append(line)
          if PARSER_DEBUG: print('Matched BANNER region: ', line)
          continue
        else:
          # we expect the banner on line 1 of the region, if not abort
          timer_region_line_counter = 0
          if PARSER_DEBUG: print('No Banner Skipping: ', line)
          continue

      # we expect the column headers 4 lines into a region
      elif timer_region_line_counter == 4:
        # a region starts with the column headers, we assume these are not repeated within a region
        # Global time (num calls)
        if four_column_header_re.match(line):
          in_teuchos_four_timer_output_block = True
          print('Expecting four region block')
          # start a region, so start a new set for these timer labels
          sets[set_counter] = deepcopy(yaml_four_data)
          sets[set_counter]['Number of processes'] = teuchos_num_procs
          sets[set_counter]['experiment_file'] = scaling_experiment_name
          raw_text_lines.append(line)
          timer_region_line_counter += 1
          continue
        elif single_column_header_re.match(line):
          in_teuchos_single_timer_output_block = True
          print('Expecting single region block')
          # start a region, so start a new set for these timer labels
          sets[set_counter] = deepcopy(yaml_single_data)
          sets[set_counter]['Number of processes'] = teuchos_num_procs
          raw_text_lines.append(line)
          timer_region_line_counter += 1
          continue
        else:
          # abort, this is not a timer region
          print('Line: ', line, ' Did not match anything')
          teuchos_num_procs = 0
          timer_region_line_counter = 0
          continue

      if timer_line_separator_re.match(line):
        if PARSER_DEBUG: print('Matched a timer region line separator')
        raw_text_lines.append(line)
        continue
      # this means we are at the end of the region
      elif timer_region_re.match(line):
        in_teuchos_four_timer_output_block = False
        in_teuchos_single_timer_output_block = False
        timer_region_line_counter = 0
        scaling_experiment_name = ''
        if len(sets[set_counter]['Timer names']) == 0:
          # we didn't parse anything so delete the empty struct and continue
          sets[set_counter] = None
        else:
          raw_text_lines.append(line)
          sets[set_counter]['raw_text'] = raw_text_lines
          set_counter += 1
        if PARSER_DEBUG: print('Matched END region: ', line)
        continue

      timer_matches = None

      # if inside a four region block parse out the four timings
      if in_teuchos_four_timer_output_block:
        # match a label by matching FOUR pairs of 'timing (counts)'
        timer_matches = four_column_re.match(line)
        if timer_matches is None:
          print('In a four column block, and failed to match an ending or a timer set. line: ')
          print(line)
          print(timer_region_line_counter)
          exit(-1)
      # if inside a single region block parse out the single timing
      elif in_teuchos_single_timer_output_block:
        # match a label by matching FOUR pairs of 'timing (counts)'
        timer_matches = single_column_re.match(line)
        if timer_matches is None:
          print('In a single column block, and failed to match an ending or a timer set.')
          print(line)
          exit(-1)
      else:
        # we should be inside a timer block by now, if we aren't the logic above should
        # have kicked the parser back to looking for a start region
        print('Logic error, we have a region line count larger than 2, but are not in a timer region. Line count: ',
              timer_region_line_counter)
        print(line)
        exit(-1)

      if timer_matches is None:
        print('No timer matches, but we parsed everything! This is a bug')
        print(line)
        exit(-1)

      # parse the matches
      timer_name = timer_matches.group('label').strip()

      # could error check here, the timer name better not exist
      if timer_name in sets[set_counter]['Timer names']:
        print('Parsed a timer name, but it is already in the timer set, duplicates are not possible')
        print('This is an error')
        print(line)
        exit(-1)

      sets[set_counter]['Timer names'].append(timer_name)

      # add the dict for this timer
      sets[set_counter]['Total times'][timer_name] = dict()
      sets[set_counter]['Call counts'][timer_name] = dict()

      for stat in sets[set_counter]['Statistics collected']:
        keyT = stat + 'T'
        keyC = stat + 'C'

        sets[set_counter]['Total times'][timer_name][stat] = __make_int(float(timer_matches.group(keyT).strip()))

        sets[set_counter]['Call counts'][timer_name][stat] = __make_int(float(timer_matches.group(keyC).strip()))

      raw_text_lines.append(line)
      timer_region_line_counter += 1

  return sets


# attempt to detect integer values that are stored as floats
def __make_int(scalar):
  integer = int(scalar)

  if np.isclose(integer, scalar, rtol=0.0):
    return int(integer)
  else:
    return scalar
