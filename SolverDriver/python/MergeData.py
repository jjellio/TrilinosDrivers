#!/usr/bin/env python3
"""MergeData.py

Usage:
  MergeData.py --new_data=new_data.csv --original_data=all_data.csv --output=new_all_data.csv --execution_space=OpenMP
  MergeData.py (-h | --help)

Options:
  -h --help                          Show this screen.
  -n FILE --new_data=FILE                 The new data to replace the old [default: new_data.csv]
  -d FILE --original_data=FILE            The existing data [default: all_data.csv]
  -o FILE --output=FILE                   The output filename  [default: new_all_data.csv]
  -e NAME --execution_space=NAME          The type of data to expect [default: OpenMP]
"""

from docopt import docopt
import pandas as pd
import ScalingFilenameParser as SFP
import numpy as np

def load_dataset(dataset_name, execspace_name):
  print('Reading {}'.format(dataset_name))

  # use low_memory=False, as this will usually get the types correct for the data.
  # e.g., ints vs floats
  dataset = pd.read_csv(dataset_name, low_memory=False)

  print('Read csv complete')

  dataset.fillna(value='None', inplace=True)

  # integral_columns = SFP.getIndexColumns(execspace_name='OpenMP')
  # non_integral_names = ['Timer Name',
  #                          'problem_type',
  #                          'solver_name',
  #                          'solver_attributes',
  #                          'prec_name',
  #                          'prec_attributes',
  #                          'execspace_name',
  #                          'execspace_attributes']
  # integral_columns = list(set(integral_columns).difference(set(non_integral_names)))
  # dataset[integral_columns] = dataset[integral_columns].astype(np.int32)

  # # this is a hack to correct a mistake in the dataset. For whatever reason, the number of threads
  # # was set incorrectly. This was detected because the the number of active threads is detected at runtime
  # # and this value does not match the expected number of threads based on the queued parameters.
  # dataset['omp_num_threads'] = dataset['threads_per_core'] * dataset['cores_per_proc']
  # dataset['execspace_attributes'] = "-threads-" + dataset['omp_num_threads'].map(str)

  # set the index, verify it, and sort
  dataset.set_index(keys=SFP.getIndexColumns(execspace_name=execspace_name),
                    drop=False, inplace=True, verify_integrity=True)
  print('Verified index')

  dataset.sort_index(inplace=True)
  print('Sorted')

  return dataset


def main():
  # Process input
  options = docopt(__doc__)
  print(options)

  new_data_csv = options['--new_data']
  original_data_csv = options['--original_data']
  output_csv = options['--output']
  exec_space = options['--execution_space']

  new_data_df = load_dataset(new_data_csv, execspace_name=exec_space)
  original_data_df  = load_dataset(original_data_csv, execspace_name=exec_space)

  #new_df = original_data_df.copy()

  original_data_df.update(new_data_df)
  original_data_df.sort_index(inplace=True)
  original_data_df.to_csv(output_csv, index=False)


if __name__ == '__main__':
  main()