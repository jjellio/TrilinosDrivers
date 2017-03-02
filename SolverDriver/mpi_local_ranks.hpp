#ifndef PerfUtils_mpi_local_ranks_HPP
#define PerfUtils_mpi_local_ranks_HPP

#include <mpi.h>
#include <string>
#include <Teuchos_Comm.hpp>
#include <Teuchos_OpaqueWrapper.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_HashUtils.hpp>

namespace PerfUtils {

typedef Teuchos::Comm<int> teuchos_comm_type;

/*
 * API usage:
 * get information about how many processes are on a single node
      MPI_Comm nodeComm;
      get_node_mpi_comm (&nodeComm);
      const int local_rank = get_local_rank (nodeComm);
      const int local_procs = get_local_size (nodeComm);
 */

Teuchos::RCP<teuchos_comm_type> getNodeLocalComm_mpi3 (const Teuchos::RCP<const teuchos_comm_type> comm);
Teuchos::RCP<teuchos_comm_type> getNodeLocalComm (const Teuchos::RCP<const teuchos_comm_type> comm);

Teuchos::RCP<teuchos_comm_type> getNodeComm (
  const Teuchos::RCP<const teuchos_comm_type> comm,
  Teuchos::RCP<const teuchos_comm_type> nodeLocalcomm = Teuchos::null );
std::string getHostname ();


bool compareComm (const Teuchos::RCP<const teuchos_comm_type> comm1, const Teuchos::RCP<const teuchos_comm_type> comm2);


void get_node_mpi_comm (MPI_Comm* comm);
int get_local_rank (MPI_Comm nodeComm);
int get_local_size (MPI_Comm nodeComm);

} // end namespace PerfUtils

#endif // PerfUtils_mpi_local_ranks_HPP
