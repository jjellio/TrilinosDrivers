#!/usr/bin/env python3
"""analysis.py

Usage:
  analysis.py --log=FILE
  analysis.py (-h | --help)

Options:
  -h --help  Show this screen.
"""

import docopt as dpt
import TeuchosTimerUtils as TTU


if __name__ == '__main__':

  # Process input
  options = dpt.docopt(__doc__)
  print(dpt.__file__)

  logfile = options['--log']

  print(options)

  yaml_sets = TTU.gather_timer_name_sets_from_logfile(logfile)

  for i, yaml_data in enumerate(yaml_sets):
    df = TTU.construct_dataframe(yaml_data)
    fname='{}-{}.csv'.format(logfile, i)
    df.to_csv(fname, index_label='timer_name', index=True)
    print('Wrote: ', fname)
