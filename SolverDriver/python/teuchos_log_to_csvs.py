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

  yaml_sets = TTU.gather_timer_name_sets_from_logfile(logfile)
  num_sets = len(yaml_sets)

  for i, yaml_data in yaml_sets.items():

    df = TTU.construct_dataframe(yaml_data)
    if num_sets == 1:
      fname='{}.csv'.format(logfile)
    else:
      fname = '{}-{}.csv'.format(logfile, i)

    df.rename(columns={'Timer Name': 'timer_name'}, inplace=True)

    df.to_csv(fname, index=False, columns=['timer_name',
                                           'minT',
                                           'minC',
                                           'meanT',
                                           'meanC',
                                           'maxT',
                                           'maxC',
                                           'meanCT',
                                           'meanCC'])
    print('Wrote: ', fname)
    i+=1
