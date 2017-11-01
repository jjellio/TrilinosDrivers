#!/usr/bin/env python3
import numpy as np
import yaml
from copy import deepcopy


def write_yaml(yaml_data,
               filename=None,
               verbose=1):

  local_yaml = yaml_data
  if local_yaml['raw_text']:
    local_yaml = deepcopy(yaml_data)
    del local_yaml['raw_text']

  with open(filename, 'x') as yaml_file:
    yaml.dump(local_yaml,
              yaml_file,
              canonical=True,
              default_flow_style=True)

    if verbose > 0:
      print('Wrote: {filename}'.format(filename=filename))


def write_raw_unparsed_teuchos_table(yaml_data,
                                     write_to_stdout=True,
                                     filename=None,
                                     verbose=1):
  if write_to_stdout:
    print(''.join(yaml_data['raw_text']))

  if filename is not None:
    f = open(filename, 'x')
    f.write(''.join(yaml_data['raw_text']))

    if verbose > 0:
      print('Wrote: {filename}'.format(filename=filename))

    f.close()


def yaml_dict_to_teuchos_table(yaml_data,
                               write_to_stdout=True,
                               filename=None,
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
                     'raw_text' : []}

  yaml_single_data = { 'Timer names' : list(),
                       'Total times' : dict(),
                       'Call counts' : dict(),
                       'Number of processes' : int(0),
                       'Statistics collected' : ['Total'],
                       'raw_text': []}

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
          # start a region, so start a new set for these timer labels
          sets[set_counter] = deepcopy(yaml_four_data)
          sets[set_counter]['Number of processes'] = teuchos_num_procs
          raw_text_lines.append(line)
          timer_region_line_counter += 1
          continue
        elif single_column_header_re.match(line):
          in_teuchos_single_timer_output_block = True
          # start a region, so start a new set for these timer labels
          sets[set_counter] = deepcopy(yaml_single_data)
          sets[set_counter]['Number of processes'] = teuchos_num_procs
          raw_text_lines.append(line)
          timer_region_line_counter += 1
          continue
        else:
          # abort, this is not a timer region
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
