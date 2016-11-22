#include "mpi_local_ranks.hpp"


namespace PerfUtils {
/*
 * API usage:
 * get information about how many processes are on a single node
      MPI_Comm nodeComm;
      get_node_mpi_comm (&nodeComm);
      const int local_rank = get_local_rank (nodeComm);
      const int local_procs = get_local_size (nodeComm);
 */

//
// MPI 3.0 call that can identify mpi processes that have access to
// the same shared memory.
// Modified from:
// http://stackoverflow.com/questions/35626377/get-nodes-with-mpi-program-in-c/35629629
//
void get_node_mpi_comm (MPI_Comm * nodeComm)
{
  int globalRank, localRank;
  MPI_Comm masterComm;

  MPI_Comm_rank( MPI_COMM_WORLD, &globalRank);
  MPI_Comm_split_type( MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, globalRank,
                       MPI_INFO_NULL, nodeComm );
  MPI_Comm_rank( *nodeComm, &localRank);
  MPI_Comm_split( MPI_COMM_WORLD, localRank, globalRank, &masterComm );

  MPI_Comm_free( &masterComm );
}


int get_local_rank (MPI_Comm nodeComm)
{
  int local_rank;
  MPI_Comm_rank (nodeComm, &local_rank);
  return local_rank;
}

int get_local_size (MPI_Comm nodeComm)
{
  int local_size;
  MPI_Comm_size (nodeComm, &local_size);
  return local_size;
}

std::string getHostname ()
{
  char name[MPI_MAX_PROCESSOR_NAME];
  int len;
  MPI_Get_processor_name( name, &len );
  std::string this_host (name);
  return this_host;
}

} // end namespace PerfUtils
