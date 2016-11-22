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

  PerfUtils::print_affinity (oss, aff);
  std::cout << oss.str () << std::endl;

  write_affinity_csv ("test.csv");
}
