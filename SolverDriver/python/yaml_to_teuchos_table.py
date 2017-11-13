#!/usr/bin/env python3
"""analysis.py

Usage:
  analysis.py --files=<FILES> [--demangle]
  analysis.py (-h | --help)

Options:
  -h, --help       Show this screen.
  --files=<FILES>  YAML files to parse
  --demangle       Demangle the timer labels [default: False]

"""

from docopt import docopt
import glob
import TeuchosTimerUtils as ttu


def main():
  # Process input
  options = docopt(__doc__)

  yaml_files_  = options['--files']
  demangle = options['--demangle']
  yaml_files = []

  # figure out the log files
  if isinstance(yaml_files_, str):
    yaml_files_ = [yaml_files_]

  for yaml_file in yaml_files_:
    if '*' in yaml_file:
      globbed_files = glob.glob(yaml_file)
      yaml_files += globbed_files
  else:
    yaml_files.append(yaml_file)

  for yaml_file in yaml_files:

    yaml_data = ttu.load_yaml(yaml_file)
    if demangle:
      ttu.demange_muelu_timer_names_yaml(yaml_data)

    filename = '{filename}_teuchos-parsed-formatted.txt'.format(filename=yaml_file)
    try:
      ttu.yaml_dict_to_teuchos_table(yaml_data,
                                     write_to_stdout=False,
                                     filename=filename)
    except:
      print(yaml_data)


if __name__ == '__main__':
  main()
