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
//void getNodeComm (MPI_Comm* nodeComm)
//{
//  int globalRank, localRank;
//
//  MPI_Comm_rank( MPI_COMM_WORLD, &globalRank);
//  MPI_Comm_split_type( MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, globalRank,
//                       MPI_INFO_NULL, nodeComm );
//
//}

Teuchos::RCP<teuchos_comm_type>
getNodeLocalComm_mpi3 (const Teuchos::RCP<const teuchos_comm_type> comm)
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::Comm;
  using Teuchos::OpaqueWrapper;
  using Teuchos::opaqueWrapper;
  using Teuchos::mpiErrorCodeToString;

  typedef int ordinal_type;
  using Teuchos::MpiComm;


  const auto globalRank = comm->getRank();
  MPI_Comm nodeComm;

  // ideally this would be supported by Teuchos, but I am not adding that functionality,
  // rebuilding trilinos, and hoping something isn't broken. To hell with that.
  // The fact you cannot easily add functionality is an AMAZING flaw in Trilinos' software stack.
  int splitReturn = MPI_Comm_split_type(
                      *(reinterpret_cast<const MpiComm<ordinal_type>&>(*comm).getRawMpiComm () ),
                      MPI_COMM_TYPE_SHARED,
                      globalRank,
                       MPI_INFO_NULL,
                       &nodeComm );

  TEUCHOS_TEST_FOR_EXCEPTION(
    splitReturn != MPI_SUCCESS,
    std::logic_error,
    "Teuchos::MpiComm::MPI_Comm_split_type: Failed to create communicator with type = MPI_COMM_TYPE_SHARED "
    << "and key " << globalRank << ".  MPI_Comm_split_type failed with error \""
    << mpiErrorCodeToString (splitReturn) << "\".");
  if (nodeComm == MPI_COMM_NULL) {
    return (RCP< Comm<ordinal_type> >());
  } else {
    RCP<const OpaqueWrapper<MPI_Comm> > wrapped =
      opaqueWrapper<MPI_Comm> (nodeComm, Teuchos::details::safeCommFree);
    // Since newComm's raw MPI_Comm is the result of an
    // MPI_Comm_split, its messages cannot collide with those of any
    // other MpiComm.  This means we can assign its tag without an
    // MPI_Bcast.
    return (rcp (new MpiComm<ordinal_type> (wrapped, Teuchos::MpiComm<ordinal_type>::minTag_)));
  }

}

bool
compareComm (const Teuchos::RCP<const teuchos_comm_type> comm1,
             const Teuchos::RCP<const teuchos_comm_type> comm2)
{
  typedef int ordinal_type;
  using Teuchos::MpiComm;

  int result;
  int rc = -1;
  rc = MPI_Comm_compare (
    *(reinterpret_cast<const MpiComm<ordinal_type>&>(*comm1).getRawMpiComm () ),
    *(reinterpret_cast<const MpiComm<ordinal_type>&>(*comm2).getRawMpiComm () ),
    &result);

  // IDENT means the communicators are actually the same objects
  // CONGRUENT means they are different objects but contain the same processes
  switch (result) {
    case MPI_IDENT :
      std::cout << "IDENTICAL" << std::endl;
      return (true);
    case MPI_CONGRUENT:
      std::cout << "CONGRUENT" << std::endl;
      return (true);
    case MPI_SIMILAR:
      std::cout << "SIMILAR" << std::endl;
      return (false);
    case MPI_UNEQUAL:
      std::cout << "UNEQUAL" << std::endl;
      return (false);
    default:
      std::cout << "ERROR" << std::endl;
      return (false);
  }

  return (false);
}


Teuchos::RCP<teuchos_comm_type>
getNodeLocalComm (const Teuchos::RCP<const teuchos_comm_type> comm)
{
  using Teuchos::RCP;
  using Teuchos::rcp;

  const std::string myHost = getHostname ();
  const auto myColor = Teuchos::hashCode(myHost);
  const int myKey   = comm->getRank ();

  RCP<teuchos_comm_type> nodeComm = comm->split (myColor, myKey);
  return (nodeComm);
}

// A communicator for a process that lives on a unique machine
Teuchos::RCP<teuchos_comm_type>
getNodeComm (
  const Teuchos::RCP<const teuchos_comm_type> comm,
  Teuchos::RCP<const teuchos_comm_type> nodeLocalcomm)
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::Comm;
  using Teuchos::MpiComm;
  using Teuchos::OpaqueWrapper;
  using Teuchos::opaqueWrapper;
  using Teuchos::mpiErrorCodeToString;

  typedef int ordinal_type;

  if ( nodeLocalcomm.is_null() )
  {
    nodeLocalcomm = getNodeLocalComm (comm);
  }

  const int myColor = nodeLocalcomm->getRank () == 0 ? 0 : -1;
  const int myKey   = comm->getRank ();

  RCP<teuchos_comm_type> nodeComm = comm->split (myColor, myKey);
  return (nodeComm);
}


void get_node_mpi_comm (MPI_Comm* comm)
{

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
  return (this_host);
}

} // end namespace PerfUtils
