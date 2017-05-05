#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd
import ScalingFilenameParser as SFP

MIN_NUM_NODES = 1
MAX_NUM_NODES = 512

if __name__ == '__main__':
  # write the total dataset out, index=False, because we do not drop it above
  dataset = pd.read_csv('analysis.csv', low_memory=False)

  dataset.set_index(keys=SFP.getIndexColumns(execspace_name='OpenMP'),
                    drop=False, inplace=True, verify_integrity=True)

  dataset.sort_values(inplace=True,
                      by=SFP.getIndexColumns(execspace_name='OpenMP'))

  restriction_query = '(problem_type != \"Elasticity3D\") & ' \
                      '(solver_name == \"Constructor\") & ' \
                      '(num_nodes >= {min_num_nodes}) & ' \
                      '(num_nodes <= {max_num_nodes})' \
                      ''.format(min_num_nodes=MIN_NUM_NODES,
                                max_num_nodes=MAX_NUM_NODES)

  dataset = dataset.query(restriction_query)

  # enforce all plots use the same num_nodes. E.g., the axes will be consistent
  my_nodes = np.array(list(map(int, dataset['num_nodes'].unique())))
  my_num_nodes = dataset['num_nodes'].nunique()
  #
  my_ticks = np.arange(start=1, stop=my_num_nodes + 1, step=1, dtype=int)
  max_num_nodes = np.max(my_nodes)

  expected_data_points = my_num_nodes
  print(my_nodes)

  ######################################################################################################################
  # analysis pass
  dataset = dataset[~dataset['Timer Name'].isin(['0 - Total Time',
                                                  '1 - Reseting Linear System',
                                                  '2 - Adjusting Nullspace for BlockSize',
                                                  '3 - Constructing Preconditioner',
                                                  '4 - Constructing Solver',
                                                  '5 - Solve'])]

  omp_groupby_columns = SFP.getMasterGroupBy(execspace_name='OpenMP', scaling_type='weak')
  omp_groupby_columns.remove('procs_per_node')
  omp_groupby_columns.remove('cores_per_proc')
  #omp_groupby_columns.append('num_nodes')

  master_analysis_df = pd.DataFrame(columns=dataset.columns.values)

  print('Decomp Rating: A = 4x16, B = 8x8, C = 16x4, D=32x2')
  num_ht_benefit = dataset[dataset['threads_per_core'] > 1]['threads_per_core'].count()
  num_ht_nobenefit = dataset[dataset['threads_per_core'] == 1]['threads_per_core'].count()
  row_count = dataset['threads_per_core'].count()
  print('   HT Benefits: {:>6} ({:2.4f}%)'.format(num_ht_benefit, num_ht_benefit/row_count*100))
  print('No HT Benefits: {:>6} ({:2.4f}%)'.format(num_ht_nobenefit, num_ht_nobenefit / row_count * 100))

  unique_procs_per_node = np.sort(dataset['procs_per_node'].unique())
  proc_per_node_count = dataset.groupby('procs_per_node')['procs_per_node'].count()

  for idx in unique_procs_per_node:
    print('Procs per Node frequency: {:>4} = {:<5} ({:2.4f}%)'.format(int(idx),
                                                                proc_per_node_count[idx],
                                                                proc_per_node_count[idx]/row_count*100))

  timer_name_index = omp_groupby_columns.index('Timer Name')

  timer_names = dataset['Timer Name'].unique()
  max_timer_name_len = int(len(max(timer_names, key=len)))

  study_groups = dataset.groupby(omp_groupby_columns, as_index=False) #['Max Aggregate Time'].min()

  row_fmt = '{padding:<{padding_width}}: {problem_type: <12} : ' \
            '{problem_nx: <6} ({num_nodes: >4}) |' \
            ' {decomp_rating} ({procs_per_node: >2}x{cores_per_proc: >2}) |' \
            ' {ht_benefit: >3} (HTs={threads_per_core}) |' \
            ' {timing:1.6e}'

  header_row_fmt = '{padding:<{padding_width}}: {problem_type: <12} : ' \
            '{problem_nx: <6} ({num_nodes}) |' \
            ' {decomp_rating} ({procs_per_node}x{cores_per_proc}) |' \
            ' {ht_benefit} (HTs) |' \
            ' {timing}'

  print(header_row_fmt.format(padding='Timer Name',
                       padding_width=max_timer_name_len,
                       problem_type='Problem',
                       problem_nx='Size',
                       num_nodes='Nodes',
                       decomp_rating='Rating',
                       procs_per_node='procs_per_node',
                       cores_per_proc='cores_per_proc',
                       ht_benefit='HT Benefit',
                       threads_per_core='',
                       timing='Timing (s)'))

  iter_count = 0
  for group_name, group in study_groups:

    # for a specific group of data, compute the scaling terms, which are things like min/max
    # this also flattens the timer creating a 'fat_timer_name'
    # essentially, this function computes data that is relevant to a group, but not the whole
    #print(group)
    my_tokens = SFP.getTokensFromDataFrameGroupBy(group)
    simple_fname = SFP.getScalingFilename(my_tokens, weak=True)
    simple_title = SFP.getScalingTitle(my_tokens, weak=True)
    simple_fname = '{}_best'.format(simple_fname)
    simple_title = 'BEST: {}'.format(simple_title)

    #print("Rebuilt filename: {}".format(SFP.rebuild_source_filename(my_tokens)))

    print('{timer:<{padding_width}}'.format(timer=group_name[timer_name_index],
                                            padding_width=max_timer_name_len))
    for index, row in group.sort_values(by='num_nodes').iterrows():
      my_dict = row.to_dict()
      for key, value in my_dict.items():
        if value is not None:
          try:
            my_dict[key] = int(value)
          except ValueError:
            continue

      if row['procs_per_node'] == 4:
        my_dict['decomp_rating'] = 'A'
      elif row['procs_per_node'] == 8:
        my_dict['decomp_rating'] = 'B'
      elif row['procs_per_node'] == 16:
        my_dict['decomp_rating'] = 'C'
      elif row['procs_per_node'] == 32:
        my_dict['decomp_rating'] = 'D'

      if row['threads_per_core'] > 1:
        my_dict['ht_benefit'] = 'YES'
      else:
        my_dict['ht_benefit'] = 'NO'

      my_dict['timing'] = row['Max Aggregate Time']

      my_dict['padding'] = ''
      my_dict['padding_width'] = max_timer_name_len
      print(row_fmt.format(**my_dict))

    iter_count += 1
    print('{foo:{fill}<{width}}'.format(foo='', fill='-', width=40))
    #print('Complete: {:3.2f}'.format(iter_count/len(study_groups)*100.0))