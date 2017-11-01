#!/usr/bin/env python3
"""analysis.py

Usage:
  analysis.py --logs=<FILES>
  analysis.py (-h | --help)

Options:
  -h, --help       Show this screen.
  --logs=<FILES>   The files to parse

"""

from docopt import docopt
import glob
import TeuchosTimerUtils as ttu


def main():
  # Process input
  options = docopt(__doc__)

  log_files_  = options['--logs']
  log_files = []

  # figure out the log files
  if isinstance(log_files_, str):
    log_files_ = [log_files_]

  for log_file in log_files_:
    if '*' in log_file:
      globbed_files = glob.glob(log_file)
      log_files += globbed_files
  else:
    log_files.append(log_file)

  for log_file in log_files:
    timer_sets = ttu.gather_timer_name_sets_from_logfile(logfile=log_file)
    print('Found ', len(timer_sets), ' timer sets')
    for timer_set_id in timer_sets:
      filename = '{filename}_teuchos-parsed-formatted-{id}.txt'.format(filename=log_file, id=timer_set_id)
      ttu.yaml_dict_to_teuchos_table(timer_sets[timer_set_id],
                                     write_to_stdout=False,
                                     filename=filename)
      filename = '{filename}_teuchos_timers-{id}.txt'.format(filename=log_file, id=timer_set_id)
      ttu.write_raw_unparsed_teuchos_table(timer_sets[timer_set_id],
                                           write_to_stdout=False,
                                           filename=filename)

      filename = '{filename}_teuchos_timers-{id}.yaml'.format(filename=log_file, id=timer_set_id)
      ttu.write_yaml(timer_sets[timer_set_id],
                     filename=filename)


if __name__ == '__main__':
  main()
