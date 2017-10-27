#!/usr/bin/env python3

class ScalingFileNameParser:
  '''Tokenize and support the complex filenames that arrise from scaling studies'''

  fmt_strs = {}
  re_strs = {}
  expected_tokens = []

  def __init__(self):
    import re
    import sys
    # TODO: Generalize the tokens They can all be written as {Primrary}{Primrary_Attributes}
    #       From this generalization, then you can parse out specifics from the attributes

    # at first glance, this looks a nightmare.
    # it is actually a neat python3 feature.
    #
    # The gist:
    #  1) Use named matches with python regular expressions. Then 'tokens' below contain {identifier}, which is used
    #     as a format string.
    #  2) Regular expressions pieces are defined using named matches that have the same token names.
    #  3) You can then match the regex, grab the matches as a dict
    #     and then use the format strings below to build strings from the matches
    #

    # example file
    # Brick3D-BS-1-246x246x246_CG_MueLu-repartition_numsteps-1_OpenMP-threads-16_np-2048_decomp-512x4x16x1.yaml

    # tokens:
    #         {problem_type}-BS-{problem_bs}-{problem_nx}x{problem_ny}x{problem_nz}
    #         {solver_name}{solver_attributes}
    #         {prec_name}{prec_attributes}
    #          numsteps-{numsteps}
    #         {execspace}
    #          np-{num_mpi_procs}
    #          decomp-{num_nodes}x{procs_per_node}x{cores_per_proc}x{threads_per_core}
    self.fmt_strs['timer_token']     = "{flat_timer_name}"
    self.fmt_strs['problem_token']   = "{problem_type}-BS-{problem_bs}-{problem_nx}x{problem_ny}x{problem_nz}"
    self.fmt_strs['solver_token']    = "{solver_name}{solver_attributes}"
    self.fmt_strs['prec_token']      = "{prec_name}{prec_attributes}"
    self.fmt_strs['steps_token']     = "numsteps-{numsteps}"
    self.fmt_strs['execspace_token'] = "{execspace_name}{execspace_attributes}"
    self.fmt_strs['execspace_name']  = "{execspace_name}"
    self.fmt_strs['np_token']        = "np-{num_mpi_procs}"
    self.fmt_strs['decomp_token']    = "decomp-{num_nodes}x{procs_per_node}x{cores_per_proc}x{threads_per_core}"
    # for weak scaling, problem size changes, but problem type does not
    # solver/prec/execspace are the same
    # np changes, decomp will have the same procs/cores/threads but not nodes
    self.fmt_strs['weak_prob_token'] = "{problem_type}-BS-{problem_bs}" \
                                       "-{min_problem_nx}x{min_problem_ny}x{min_problem_nz}" \
                                       "-{max_problem_nx}x{max_problem_ny}x{max_problem_nz}"

    self.fmt_strs['scaling_np_token'] = "np-{min_num_mpi_procs}-{max_num_mpi_procs}"
    self.fmt_strs['scaling_decomp_token'] = "decomp-{min_num_nodes}-{max_num_nodes}" \
                                  "-{procs_per_node}x{cores_per_proc}x{threads_per_core}"

    # regex definitions
    self.re_strs['problem_type'] = r'(?P<problem_type>[a-zA-Z0-9]+)-BS-(?P<problem_bs>\d+)'
    self.re_strs['problem_size'] = r'(?P<problem_nx>\d+)x(?P<problem_ny>\d+)x(?P<problem_nz>\d+)'
    self.re_strs['problem_token'] = r'{0}-{1}'.format(self.re_strs['problem_type'], self.re_strs['problem_size'])

    self.re_strs['solver_name'] = r'(?P<solver_name>[a-zA-Z0-9]+)'
    self.re_strs['solver_attributes'] = r'(?P<solver_attributes>(-[a-zA-Z0-9]+)*)?'
    # self.re_strs['solver_attributes'] = r'(?P<solver_attributes>-[a-zA-Z0-9]+)*'
    self.re_strs['solver_token'] = r'{0}{1}'.format(self.re_strs['solver_name'], self.re_strs['solver_attributes'])

    self.re_strs['prec_name'] = r'(?P<prec_name>[a-zA-Z0-9]+)'
    self.re_strs['prec_attributes'] = r'(?P<prec_attributes>(-[a-zA-Z0-9]+)*)?'
    # self.re_strs['prec_attributes'] = r'(?P<prec_attribute1>-[a-zA-Z0-9]+)?(?P<prec_attribute2>-[a-zA-Z0-9]+)?(?P<prec_attribute3>-[a-zA-Z0-9]+)?'
    self.re_strs['prec_token'] = r'{0}{1}'.format(self.re_strs['prec_name'], self.re_strs['prec_attributes'])

    self.re_strs['numsteps'] = r'numsteps-(?P<numsteps>\d+)'
    self.re_strs['execspace'] = r'(?P<execspace_name>OpenMP|Cuda|Serial)(?P<execspace_attributes>(-[a-zA-Z0-9]+)*)?'
    self.re_strs['np'] = r'np-(?P<num_mpi_procs>\d+)'
    self.re_strs['decomp'] = r'decomp-(?P<num_nodes>\d+)x' \
                             r'(?P<procs_per_node>\d+)x' \
                             r'(?P<cores_per_proc>\d+)x' \
                             r'(?P<threads_per_core>\d+)'

    self.re_strs['yaml_file'] = r'{problem_token}' \
                                r'_{solver_token}' \
                                r'(_{prec_token})?' \
                                r'_{numsteps}' \
                                r'_{execspace}' \
                                r'_{np}_' \
                                r'{decomp}'.format(**self.re_strs)

    self.re_strs['execspace_openmp'] = r'-threads-(?P<omp_num_threads>\d+)'
    self.re_strs['execspace_cuda']   = r'-(?P<cuda_device_name>Tesla-P100-SXM2-16GB|Tesla-K80)'

    # the above leads to this regex, perhaps easier to read from here going upwards.
    # the above statement defines the filename as pieces. The pieces are defined above it.
    self.filename_re         = re.compile(self.re_strs['yaml_file'])
    self.execspace_openmp_re = re.compile(self.re_strs['execspace_openmp'])
    self.execspace_cuda_re   = re.compile(self.re_strs['execspace_cuda'])

    self.scaling_dict_terms = {  # track min / max problem (weak scaling)
      'min_problem_nx': sys.maxsize,
      'min_problem_ny': sys.maxsize,
      'min_problem_nz': sys.maxsize,
      'max_problem_nx': int(0),
      'max_problem_ny': int(0),
      'max_problem_nz': int(0),
      # min/max num nodes
      'min_num_nodes': sys.maxsize,
      'max_num_nodes': int(0),
      # min/max procs
      'min_num_mpi_procs': sys.maxsize,
      'max_num_mpi_procs': int(0)}

  # this function parses a filename and gathers all of the attributes
  # The parsing needs to be written in a more general form
  # The general structure is, FEATURE[-ATTRIBUTE[-ATTRIBUTE]]_FEATURE2[-ATTRIBUTE...
  # The logic used to parse the execution space is a model of how to do this.
  # First, parse FEATURE[-anything but underscore]. Use 'FEATURE' to determine how to parse
  # the other stuff. E.g., if FEATURE=OpenMP, then parse the number of OpenMP threads.
  # I.e., the attribute parsing is determined by the Feature.
  def parseYAMLFileName(self, filename):
    import os
    # parse the decomp token
    yaml_filename = os.path.basename(filename)

    print(yaml_filename)
    # first, match the general tokens (problem, solver+attributes)
    yaml_matches = self.filename_re.match(os.path.basename(yaml_filename))

    my_tokens = yaml_matches.groupdict()

    execspace_dict = None

    # track a lowercase version of this
    my_tokens['lexecspace_name'] = my_tokens['execspace_name'].lower()

    # from the general tokens, gather specific information
    if my_tokens['execspace_name'] == 'OpenMP':
      execspace_matches = self.execspace_openmp_re.match(my_tokens['execspace_attributes'])
      execspace_dict = execspace_matches.groupdict()
    elif my_tokens['execspace_name'] == 'Serial':
      # nothing
      pass
    elif my_tokens['execspace_name'] == 'Cuda':
      print(my_tokens['execspace_attributes'])
      execspace_matches = self.execspace_cuda_re.match(my_tokens['execspace_attributes'])
      execspace_dict = execspace_matches.groupdict()
    else:
      print("Unknown Execspace Name: '{}' This is a bug. "
            "execspace_name is a predefined match in execspace re.".format(my_tokens['execspace_name']))

    if execspace_dict:
      my_tokens.update(execspace_dict)

    for key, value in my_tokens.items():
      if value is not None:
        try:
          my_tokens[key] = int(value)
        except ValueError:
          continue

    # return a dict of tokens
    return my_tokens

  def getColumnsDTypes(self, execspace_name='unknown'):
    import numpy as np
    # this would be easier with a refactor
    # e.g., problem_name, problem_attributes, solver_name, solver_attributes, ...
    if execspace_name == 'OpenMP' or execspace_name == 'Serial':
      return {
              'Timer Name' : 'str',
              'problem_type' : 'str',
              'problem_nx' : 'int32',
              'problem_ny' : 'int32',
              'problem_nz' : 'int32',
              'problem_bs' : 'int32',
              'solver_name' : 'str',
              'solver_attributes' : 'str',
              'execspace_name' : 'str',
              'execspace_attributes' : 'str',
              'prec_name' : 'str',
              'prec_attributes' : 'str',
              'numsteps' : 'int32',
              'num_mpi_procs' : 'int32',
              # order matters here, we want things sorted how we group/analyze data
              'num_nodes' : 'int32',
              'procs_per_node' : 'int32',
              'cores_per_proc' : 'int32',
              'threads_per_core' : 'int32',
              # these are unique to the execspace
              'omp_num_threads' : 'int32',
              'nodes' : 'str',
              'timestamp': 'str',
              # teuchos stuff
              'minT'  : 'float64',
              'minC'  : 'float64',
              'meanT' : 'float64',
              'meanC' : 'float64',
              'maxT'  : 'float64',
              'maxC'  : 'float64',
              'meanCT' : 'float64',
              'meanCC' : 'float64'}
    elif execspace_name == 'Cuda':
      return {'Timer Name' : 'str',
              'problem_type' : 'str',
              'problem_nx' : 'int32',
              'problem_ny' : 'int32',
              'problem_nz' : 'int32',
              'problem_bs' : 'int32',
              'solver_name' : 'str',
              'solver_attributes' : 'str',
              'execspace_name' : 'str',
              'execspace_attributes' : 'str',
              'prec_name' : 'str',
              'prec_attributes' : 'str',
              'numsteps' : 'int32',
              'num_mpi_procs' : 'int32',
              # order matters here, we want things sorted how we group/analyze data
              'num_nodes' : 'int32',
              'procs_per_node' : 'int32',
              'cores_per_proc' : 'int32',
              'threads_per_core' : 'int32',
              # these are unique to the execspace
              'cuda_device_name' : 'str',
              'nodes' : 'str',
              'timestamp': 'str',
              # teuchos stuff
              'minT'  : 'float64',
              'minC'  : 'float64',
              'meanT' : 'float64',
              'meanC' : 'float64',
              'maxT'  : 'float64',
              'maxC'  : 'float64',
              'meanCT' : 'float64',
              'meanCC' : 'float64'}

  def getIndexColumns(self, execspace_name='unknown'):
    # this would be easier with a refactor
    # e.g., problem_name, problem_attributes, solver_name, solver_attributes, ...
    if execspace_name == 'OpenMP' or execspace_name == 'Serial':
      return ['Timer Name',
              'problem_type',
              'problem_nx',
              'problem_ny',
              'problem_nz',
              'problem_bs',
              'solver_name',
              'solver_attributes',
              'execspace_name',
              'execspace_attributes',
              'prec_name',
              'prec_attributes',
              'numsteps',
              'num_mpi_procs',
              # order matters here, we want things sorted how we group/analyze data
              'num_nodes',
              'procs_per_node',
              'cores_per_proc',
              'threads_per_core',
              # these are unique to the execspace
              'omp_num_threads',
              'nodes',
              'timestamp']
    elif execspace_name == 'Cuda':
      return ['Timer Name',
              'problem_type',
              'problem_nx',
              'problem_ny',
              'problem_nz',
              'problem_bs',
              'solver_name',
              'solver_attributes',
              'execspace_name',
              'execspace_attributes',
              'prec_name',
              'prec_attributes',
              'numsteps',
              'num_mpi_procs',
              # order matters here, we want things sorted how we group/analyze data
              'num_nodes',
              'procs_per_node',
              'cores_per_proc',
              'threads_per_core',
              # these are unique to the execspace
              'cuda_device_name',
              'nodes',
              'timestamp']

  def getMasterGroupBy(self, execspace_name='unknown', scaling_type='unknown'):
    # this would be easier with a refactor
    # e.g., problem_name, problem_attributes, solver_name, solver_attributes, ...
    if execspace_name == 'OpenMP' or execspace_name == 'Serial':
      if scaling_type == 'strong':
        return ['Timer Name',
                'problem_type',
                'problem_nx',
                'problem_ny',
                'problem_nz',
                'problem_bs',
                'solver_name',
                'solver_attributes',
                'execspace_name',
                'prec_name',
                'prec_attributes',
                'numsteps',
                # number of MPI procs is not constant
                # order matters here, we want things sorted how we group/analyze data
                'procs_per_node',
                'cores_per_proc']
      elif scaling_type == 'weak':
        return ['Timer Name',
                'problem_type',
                # problem size is not constant
                'problem_bs',
                'solver_name',
                'solver_attributes',
                'execspace_name',
                'prec_name',
                'prec_attributes',
                'numsteps',
                # number of MPI procs is not constant
                # order matters here, we want things sorted how we group/analyze data
                'procs_per_node',
                'cores_per_proc']
    elif execspace_name == 'Cuda':
      if scaling_type == 'strong':
        return ['Timer Name',
                'problem_type',
                'problem_nx',
                'problem_ny',
                'problem_nz',
                'problem_bs',
                'solver_name',
                'solver_attributes',
                'execspace_name',
                # don't group by execspace attributes
                'prec_name',
                'prec_attributes',
                'numsteps',
                # number of MPI procs is not constant
                'procs_per_node',
                'cores_per_proc',
                'threads_per_core']
      if scaling_type == 'weak':
        return ['Timer Name',
                'problem_type',
                'problem_bs',
                'solver_name',
                'solver_attributes',
                'execspace_name',
                # don't group by execspace attributes
                'prec_name',
                'prec_attributes',
                'numsteps',
                # number of MPI procs is not constant
                'procs_per_node',
                'cores_per_proc',
                'threads_per_core']

  def getDecompLabel(self, yaml_matches):
    return "{num_nodes}\n({num_mpi_procs})".format(**yaml_matches)

  def rebuild_source_filename(self, my_tokens):
    # reconstruct the file name these tokens were parsed from
    if my_tokens['prec_name'] is None:
      data_orig_filename_fmt  = "{problem_token}" \
                                "_{solver_token}" \
                                "_{steps_token}" \
                                "_{execspace_token}" \
                                "_{np_token}" \
                                "_{decomp_token}.yaml".format(**self.fmt_strs)
    else:
      data_orig_filename_fmt  = "{problem_token}" \
                                "_{solver_token}" \
                                "_{prec_token}" \
                                "_{steps_token}" \
                                "_{execspace_token}" \
                                "_{np_token}" \
                                "_{decomp_token}.yaml".format(**self.fmt_strs)

    return data_orig_filename_fmt.format(**my_tokens)

  def build_affinity_filename(self, my_tokens):
    # reconstruct the file name these tokens were parsed from
    data_orig_filename_fmt  = "{problem_token}" \
                              "_{execspace_token}" \
                              "_{np_token}" \
                              "_{decomp_token}_affinity.csv".format(**self.fmt_strs)
    return data_orig_filename_fmt.format(**my_tokens)

  def updateScalingTerms(self, my_tokens, scaling_dict_terms):
    if not scaling_dict_terms:
      # extract scaling measures that are needed for filename construction
      scaling_dict_terms['min_num_mpi_procs'] = int(my_tokens['num_mpi_procs'])
      scaling_dict_terms['max_num_mpi_procs'] = int(my_tokens['num_mpi_procs'])

      scaling_dict_terms['min_num_nodes'] = int(my_tokens['num_nodes'])
      scaling_dict_terms['max_num_nodes'] = int(my_tokens['num_nodes'])

      scaling_dict_terms['min_procs_per_node'] = int(my_tokens['procs_per_node'])
      scaling_dict_terms['max_procs_per_node'] = int(my_tokens['procs_per_node'])

      scaling_dict_terms['min_cores_per_proc'] = int(my_tokens['cores_per_proc'])
      scaling_dict_terms['max_cores_per_proc'] = int(my_tokens['cores_per_proc'])

      scaling_dict_terms['min_threads_per_core'] = int(my_tokens['threads_per_core'])
      scaling_dict_terms['max_threads_per_core'] = int(my_tokens['threads_per_core'])

      scaling_dict_terms['min_problem_nx'] = int(my_tokens['problem_nx'])
      scaling_dict_terms['max_problem_nx'] = int(my_tokens['problem_nx'])

      scaling_dict_terms['min_problem_ny'] = int(my_tokens['problem_ny'])
      scaling_dict_terms['max_problem_ny'] = int(my_tokens['problem_ny'])

      scaling_dict_terms['min_problem_nz'] = int(my_tokens['problem_nx'])
      scaling_dict_terms['max_problem_nz'] = int(my_tokens['problem_nz'])
    else:
      # extract scaling measures that are needed for filename construction
      scaling_dict_terms['min_num_mpi_procs'] = min(scaling_dict_terms['min_num_mpi_procs'],
                                                    int(my_tokens['num_mpi_procs']))
      scaling_dict_terms['max_num_mpi_procs'] = max(scaling_dict_terms['max_num_mpi_procs'],
                                                    int(my_tokens['num_mpi_procs']))

      scaling_dict_terms['min_num_nodes'] = min(scaling_dict_terms['min_num_nodes'],
                                                int(my_tokens['num_nodes']))
      scaling_dict_terms['max_num_nodes'] = max(scaling_dict_terms['max_num_nodes'],
                                                int(my_tokens['num_nodes']))

      scaling_dict_terms['min_procs_per_node'] = min(scaling_dict_terms['min_procs_per_node'],
                                                     int(my_tokens['procs_per_node']))
      scaling_dict_terms['max_procs_per_node'] = max(scaling_dict_terms['max_procs_per_node'],
                                                     int(my_tokens['procs_per_node']))

      scaling_dict_terms['min_cores_per_proc'] = min(scaling_dict_terms['min_cores_per_proc'],
                                                     int(my_tokens['cores_per_proc']))
      scaling_dict_terms['max_cores_per_proc'] = max(scaling_dict_terms['max_cores_per_proc'],
                                                     int(my_tokens['cores_per_proc']))

      scaling_dict_terms['min_threads_per_core'] = min(scaling_dict_terms['min_threads_per_core'],
                                                       int(my_tokens['threads_per_core']))
      scaling_dict_terms['max_threads_per_core'] = max(scaling_dict_terms['max_threads_per_core'],
                                                       int(my_tokens['threads_per_core']))

      scaling_dict_terms['min_problem_nx'] = min(scaling_dict_terms['min_problem_nx'],
                                                 int(my_tokens['problem_nx']))
      scaling_dict_terms['max_problem_nx'] = max(scaling_dict_terms['max_problem_nx'],
                                                 int(my_tokens['problem_nx']))

      scaling_dict_terms['min_problem_ny'] = min(scaling_dict_terms['min_problem_ny'],
                                                 int(my_tokens['problem_ny']))
      scaling_dict_terms['max_problem_ny'] = max(scaling_dict_terms['max_problem_ny'],
                                                 int(my_tokens['problem_ny']))

      scaling_dict_terms['min_problem_nz'] = min(scaling_dict_terms['min_problem_nz'],
                                                 int(my_tokens['problem_nx']))
      scaling_dict_terms['max_problem_nz'] = max(scaling_dict_terms['max_problem_nz'],
                                                 int(my_tokens['problem_nz']))
      scaling_dict_terms.update(my_tokens)
    #
    # # construct a label for this data
    # xlabel_str = "{num_nodes}\n({num_mpi_procs})".format(**my_tokens)
    # xlabel_strs.append(xlabel_str)
    # del my_tokens['num_mpi_procs']
    # del my_tokens['num_nodes']
    #
    # if scaling_type == "weak":
    #   del my_tokens['problem_nx']
    #   del my_tokens['problem_ny']
    #   del my_tokens['problem_nz']

  def getScalingTerms(self, dataframe_group):
    import math
    import re

    scaling_dict_terms = {}
    for my_tokens in dataframe_group.to_dict(orient='records'):
      self.updateScalingTerms(my_tokens, scaling_dict_terms)

    for key, value in scaling_dict_terms.items():
      if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        scaling_dict_terms[key] = None
      elif value is not None:
        try:
          scaling_dict_terms[key] = int(value)
        except ValueError:
          continue

    timer_name = scaling_dict_terms['Timer Name']
    flat_timer_name = re.sub(r'[: &]', '-', timer_name)
    flat_timer_name = re.sub(r"[',%#@!^{}()/\\\"*?<>|]", '', flat_timer_name)
    flat_timer_name = re.sub('=+', '-', flat_timer_name)
    flat_timer_name = re.sub(r'(?P<symbol>-|_){2,}', '\g<symbol>', flat_timer_name)
    scaling_dict_terms['flat_timer_name'] = flat_timer_name
    return scaling_dict_terms

  def getScalingFilename(self, my_tokens, weak=False, strong=False, onnode=False,
                      composite=False,
                      skip_none_values=True,
                      timer_name=True,
                      solver_name=True, solver_attributes=True,
                      prec_name=True, prec_attributes=True):

    if weak is False and strong is False and onnode is False:
      print("getScalingFilename was called with all possible scaling study flags set to false")

    my_fmt_string = ""
    my_solver_token = ""

    if weak:
      my_fmt_string = "weak"
    elif strong:
      my_fmt_string = "strong"
    elif onnode:
      my_fmt_string = "onnode"

    if timer_name:
      my_fmt_string += "_{flat_timer_name}"

    if solver_name is True and (skip_none_values is True and my_tokens['solver_name'] is not None
                                and my_tokens['solver_name'] != 'None'):
      my_solver_token += "_{solver_name}"

      if solver_attributes is True and (skip_none_values is True and my_tokens['solver_attributes'] is not None
                                        and my_tokens['solver_attributes'] != 'None'):
        my_solver_token += "{solver_attributes}"

    if prec_name is True and (skip_none_values is True and my_tokens['prec_name'] is not None
                              and my_tokens['prec_name'] != 'None'):
      my_solver_token += "_{prec_name}"

      if prec_attributes is True and (skip_none_values is True and my_tokens['prec_attributes'] is not None
                                      and my_tokens['prec_attributes'] != 'None'):
        my_solver_token += "{prec_attributes}"

    if my_tokens['execspace_name'] == 'OpenMP' or my_tokens['execspace_name'] == 'Serial':
      if weak:
        my_fmt_string += "_{weak_prob_token}".format(**self.fmt_strs)
        my_fmt_string += my_solver_token
        #my_fmt_string += "_{execspace_name}"
        my_fmt_string += "_{steps_token}".format(**self.fmt_strs)
        if composite == False:
          my_fmt_string += "_{scaling_np_token}" \
                           "_{scaling_decomp_token}".format(**self.fmt_strs)
        else:
          my_fmt_string += "_{min_num_nodes}-{max_num_nodes}-composite"
      elif strong or onnode:
        my_fmt_string += "_{problem_token}".format(**self.fmt_strs)
        my_fmt_string += my_solver_token
        my_fmt_string += "_{steps_token}".format(**self.fmt_strs)
        #my_fmt_string += "_{execspace_name}"
        if composite == False:
          my_fmt_string += "_{scaling_np_token}" \
                           "_{scaling_decomp_token}".format(**self.fmt_strs)
        else:
          my_fmt_string += "_{min_num_nodes}-{max_num_nodes}-composite"
    elif my_tokens['execspace_name'] == 'Cuda':
      if weak:
        my_fmt_string += "_{weak_prob_token}".format(**self.fmt_strs)
        my_fmt_string += my_solver_token
        my_fmt_string += "_{steps_token}".format(**self.fmt_strs)
        my_fmt_string += "_{execspace_name}"
        if composite == False:
          my_fmt_string += "_{scaling_np_token}" \
                           "_{scaling_decomp_token}".format(**self.fmt_strs)
        else:
          my_fmt_string += "_{min_num_nodes}-{max_num_nodes}-composite"
      elif strong:
        my_fmt_string += "_{problem_token}".format(**self.fmt_strs)
        my_fmt_string += my_solver_token
        my_fmt_string += "_{steps_token}".format(**self.fmt_strs)
        my_fmt_string += "_{execspace_name}"
        if composite == False:
          my_fmt_string += "_{steps_token}" \
                           "_{scaling_np_token}" \
                           "_{scaling_decomp_token}".format(**self.fmt_strs)
        else:
          my_fmt_string += "_{min_num_nodes}-{max_num_nodes}-composite"
      elif onnode:
        print("onnode tile was requested for the Cuda execution space, which is not currently defined")
        raise ValueError
        my_fmt_string += "UNDEFINED"

    return my_fmt_string.format(**my_tokens)

  def getScalingTitle(self, my_tokens, weak=False, strong=False, onnode=False,
                      composite=False,
                      skip_none_values=True,
                      solver_name=True, solver_attributes=True,
                      prec_name=True, prec_attributes=True):

    if weak is False and strong is False and onnode is False:
      print("getScalingTitle was called with all possible scaling study flags set to false")

    my_fmt_string = ""

    if weak:
      my_fmt_string = "Weak Scaling"
    elif strong:
      my_fmt_string = "Strong Scaling"
    elif onnode:
      my_fmt_string = "On Node Scaling"

    my_fmt_string += " {Timer Name}\n"

    if solver_name is True and (skip_none_values is False and my_tokens['solver_name'] is not None):
      my_fmt_string += "{solver_name}"

    if solver_attributes is True and (skip_none_values is False and my_tokens['solver_attributes'] is not None):
      my_fmt_string += "{solver_attributes}\n"

    if prec_name is True and (skip_none_values is False and my_tokens['prec_name'] is not None):
      my_fmt_string += "{prec_name}"

    if prec_attributes is True and (skip_none_values is False and my_tokens['prec_attributes'] is not None):
      my_fmt_string += "{prec_attributes}\n"

    if my_tokens['execspace_name'] == 'OpenMP' or my_tokens['execspace_name'] == 'Serial':
      if weak:
        my_fmt_string += "{problem_type} {min_problem_nx}x{min_problem_ny}x{min_problem_nz}" \
                         " to {max_problem_nx}x{max_problem_ny}x{max_problem_nz}\n" \
                         "Total MPI Processes: {min_num_mpi_procs}-{max_num_mpi_procs}"
        my_fmt_string += "({procs_per_node} procs per node)" if composite == False else ""
      elif strong:
        my_fmt_string +=  "{problem_type} {problem_nx}x{problem_ny}x{problem_nz}" \
                          " {min_num_nodes}-{max_num_nodes} Nodes ({min_num_mpi_procs}-{max_num_mpi_procs}" \
                          " MPI Processes)"
        my_fmt_string +=  "\n{procs_per_node} procs per node, {cores_per_proc} cores per proc" if composite == False else ""
      elif onnode:
        my_fmt_string +=  "{problem_type} {problem_nx}x{problem_ny}x{problem_nz}," \
                          " On one Node with {procs_per_node} MPI Processes per Node"
        my_fmt_string +=  "\nWith {min_cores_per_proc}-{max_cores_per_proc} cores per proc," \
                          " and {min_threads_per_core}-{max_threads_per_core} thread(s) per core" \
                          if composite == False else ""
    elif my_tokens['execspace_name'] == 'Cuda':
      if weak:
        my_fmt_string += "{problem_type} {min_problem_nx}x{min_problem_ny}x{min_problem_nz}," \
                         " to {max_problem_nx}x{max_problem_ny}x{max_problem_nz}\n" \
                         "Using Cuda Execution Space and {procs_per_node} MPI Processes per Node"
      elif strong:
        my_fmt_string += "{problem_type} {problem_nx}x{problem_ny}x{problem_nz}," \
                         " Using Cuda Execution Space and\n{procs_per_node} MPI Processes per Node" \
                         " using {min_num_nodes}-{max_num_nodes} Nodes ({min_num_mpi_procs}-{max_num_mpi_procs}" \
                         " MPI Processes)"
      elif onnode:
        print("onnode tile was requested for the Cuda execution space, which is not currently defined")
        raise ValueError
        my_fmt_string += "UNDEFINED"

    return my_fmt_string.format(**my_tokens)

############################
# private
_myFileNameParser = ScalingFileNameParser()
############################
# public interface
############################


def parseYAMLFileName(filename):
  return _myFileNameParser.parseYAMLFileName(filename)


def getScalingTitle(my_tokens, weak=False, strong=False, onnode=False,
                    composite=False,
                    skip_none_values=True,
                    solver_name=True, solver_attributes=True,
                    prec_name=True, prec_attributes=True):
  return _myFileNameParser.getScalingTitle(my_tokens,
                                           weak=weak,
                                           strong=strong,
                                           onnode=onnode,
                                           composite=composite,
                                           skip_none_values=skip_none_values,
                                           solver_name=solver_name,
                                           solver_attributes=solver_attributes,
                                           prec_name=prec_name,
                                           prec_attributes=prec_attributes)


def getScalingFilename(my_tokens, weak=False, strong=False, onnode=False,
                       composite=False,
                       skip_none_values=True,
                       solver_name=True, solver_attributes=True,
                       prec_name=True, prec_attributes=True):
  return _myFileNameParser.getScalingFilename(my_tokens,
                                              weak=weak,
                                              strong=strong,
                                              onnode=onnode,
                                              composite=composite,
                                              skip_none_values=skip_none_values,
                                              solver_name=solver_name,
                                              solver_attributes=solver_attributes,
                                              prec_name=prec_name,
                                              prec_attributes=prec_attributes)


def rebuild_source_filename(my_tokens):
  return _myFileNameParser.rebuild_source_filename(my_tokens)


def build_affinity_filename(my_tokens):
  return _myFileNameParser.build_affinity_filename(my_tokens)


def getTokensFromDataFrameGroupBy(dataframe_group):
  return _myFileNameParser.getScalingTerms(dataframe_group)


def getIndexColumns(execspace_name):
  return _myFileNameParser.getIndexColumns(execspace_name=execspace_name)


def getColumnsDTypes(execspace_name):
  return _myFileNameParser.getColumnsDTypes(execspace_name=execspace_name)


def getMasterGroupBy(execspace_name, scaling_type):
  return _myFileNameParser.getMasterGroupBy(execspace_name=execspace_name, scaling_type=scaling_type)