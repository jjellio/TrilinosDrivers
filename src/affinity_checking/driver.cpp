#include "affinity_check.hpp"

#include <string>
#include <iostream>

#include <mpi.h>

int main (int argc, char** argv)
{
  using PerfUtils::write_affinity_csv;
  using PerfUtils::gather_affinity_pthread;
  using PerfUtils::print_affinity;
  using process_affinity_map_t = PerfUtils::cpu_map_type;

  MPI_Init (&argc, &argv);


  std::stringstream oss;
  process_affinity_map_t aff;

  PerfUtils::gather_affinity_pthread (aff);

//  PerfUtils::print_affinity (oss, aff);
//  std::cout << oss.str () << std::endl;
//
//  std::vector<int> local_displacements;
//  std::vector<int> local_tids;
//  std::vector<int> local_affinities;
//
//  PerfUtils::cpu_map_to_vector(aff, local_displacements, local_tids, local_affinities);
//
//  process_affinity_map_t aff2;
//  PerfUtils::vector_to_cpu_map(aff2, local_displacements, local_tids, local_affinities);
//
//  oss.str ("");
//  PerfUtils::print_affinity (oss, aff2);
//  std::cout << oss.str () << std::endl;

  PerfUtils::analyze_node_affinities (aff);

  write_affinity_csv ("affinity_details.csv");

  MPI_Finalize();
}
