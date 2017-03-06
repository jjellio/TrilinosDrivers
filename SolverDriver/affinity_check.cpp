#include "affinity_check.hpp"


// whether to use GNU Pthread to get affinity
// if undef, use linux syscall
#define USE_PTHREAD_AFFINITY


// this is needed to enable the GNU extensions, e.g., pthread_foo_np
#if !defined(_GNU_SOURCE) && defined(USE_PTHREAD_AFFINITY)
  #define _GNU_SOURCE
#endif

#include <pthread.h>

#include <sys/types.h>   // for pid_t
#include <unistd.h>      // syscall
#include <sys/syscall.h> // SYS_gettid


#include <set>       // set
#include <algorithm> // set_difference
#include <iostream>  // endl

// use OpenMP
// TODO: Make this more generic without depending on Kokkos
// Can C++11 support everything?  (No,still uses sched_getAffinity)
#include <omp.h>

#include <chrono>
#include <ctime>

#include <fstream>

#include "mpi_local_ranks.hpp"



namespace PerfUtils {


// local function

std::string get_process_affinity_csv_banner ();
std::string get_process_affinity_csv_string (const cpu_map_type& a, MPI_Comm& nodeComm);
void vector_gatherv (const std::vector<int>& local_vec, std::vector<int>& global_vec, MPI_Comm& nodeComm);

// this uses a **LINUX** sycall, and will likely not work on no Linux OSes
void gather_affinity_linux_syscall (cpu_map_type& cpu_map)
{
  cpu_map.clear ();

  omp_lock_t map_lock;
  omp_init_lock(&map_lock);

  // still enter a parallel region. This makes an implicit assumption
  // that the underlying thread model will support gettid()
  #pragma omp parallel
  {
    // get the logical thread ID
    int tid = omp_get_thread_num ();

    // for this thread, query its cpuset
    cpu_set_t my_cpuset;
    CPU_ZERO(&my_cpuset);

    // this requries the posix thread API
    pid_t thread = syscall(SYS_gettid);

    int s = sched_getaffinity (thread, sizeof(cpu_set_t), &my_cpuset);

    if (s != 0)
      do { errno = s; perror("sched_getaffinity"); exit(EXIT_FAILURE); } while (0);

    // for each entry in the CPUset, see which execution units this thread is allowed
    // to execute on.
    for (int j = 0; j < CPU_SETSIZE; j++)
    {
      if (CPU_ISSET(j, &my_cpuset))
      {
        // if a thread is allowed, add this cpu-id to the map under
        // this logical thread-id. This is *not* one to one, as threads may
        // be bound at the core or socket level
        // the cpu map is also shared, so use a lock to modify it.
        omp_set_lock(&map_lock);
          cpu_map.insert( std::make_pair(tid,j) );
        omp_unset_lock(&map_lock);
      }
    }
  }

  omp_destroy_lock(&map_lock);
}

Teuchos::RCP<const cpu_map_type> gather_affinity_pthread ()
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using std::endl;

  RCP<cpu_map_type> this_cpu_map = rcp (new cpu_map_type());
  //gather_affinity_linux_syscall (*this_cpu_map);
  gather_affinity_pthread(*this_cpu_map);;

  return (this_cpu_map);
}

#ifdef USE_PTHREAD_AFFINITY
// This uses a GNU extension to pthreads, again not portable ... ugh
void gather_affinity_pthread (cpu_map_type& cpu_map)
{
  cpu_map.clear ();

  omp_lock_t map_lock;
  omp_init_lock(&map_lock);

  // open a parallel region, this will activate whatever threads have been created
  // This assumes kokkos has been initialized, if not this call will spawn
  // and bind threads.
  #pragma omp parallel
  {
    // get the logical thread ID
    int tid = omp_get_thread_num ();

    // for this thread, query its cpuset
    cpu_set_t my_cpuset;
    CPU_ZERO(&my_cpuset);

    // this requries the posix thread API
    pthread_t thread = pthread_self ();
    int s = pthread_getaffinity_np(thread, sizeof(cpu_set_t), &my_cpuset);

    if (s != 0)
      do { errno = s; perror("pthread_getaffinity"); exit(EXIT_FAILURE); } while (0);

    // for each entry in the CPUset, see which execution units this thread is allowed
    // to execute on.
    for (int j = 0; j < CPU_SETSIZE; j++)
    {
      if (CPU_ISSET(j, &my_cpuset))
      {
        // if a thread is allowed, add this cpu-id to the map under
        // this logical thread-id. This is *not* one to one, as threads may
        // be bound at the core or socket level
        // the cpu map is also shared, so use a lock to modify it.
        omp_set_lock(&map_lock);
          cpu_map.insert( std::make_pair(tid,j) );
        omp_unset_lock(&map_lock);
      }
    }
  }

  omp_destroy_lock(&map_lock);
}
#endif


bool check_exclusive_affinity (const cpu_map_type& a, const cpu_map_type& b)
{

  using std::set;
  using std::set_intersection;
  typedef set<cpu_map_type::value_type> cpu_set_type;

  cpu_map_type result;
  cpu_set_type s1 (a.cbegin(), a.cend());
  cpu_set_type s2 (b.cbegin(), b.cend());

  set_intersection (s1.begin(), s1.end(),
                    s2.begin(), s2.end(),
                    std::inserter (result, result.end()));

  set_intersection (s2.begin(), s2.end(),
                    s1.begin(), s1.end(),
                    std::inserter (result, result.end()));

  return (result.empty ());

}

/*! \brief Compare two process affinity maps
 *
 * A cpu map is a multiset, with threads potentially mapped to the same hardware.
 * E.g., if threads are bound to cores and hardware threads are enabled, then
 * the logical threads mapping to each hyper thread will both be mapped to the same
 * physical execution units. If threads are bound to hardware threads, then these
 */
bool compare_affinity (const cpu_map_type& a, const cpu_map_type& b)
{
  if (a.size () != b.size ())
    return (false);

  using std::set;
  using std::set_difference;
  typedef set<cpu_map_type::value_type> cpu_set_type;

  // there are multiple ways to test that the sets are equal
  // This way simply finds elements that are in set1 but not in set2
  // Then, the elements that are in set2 but not in set1.
  // Another test would be to ensure that the cardinality of the intersection
  // is the same as the cardinality of each set (e.g., they are all the same)
  cpu_map_type result;
  cpu_set_type s1 (a.cbegin(), a.cend());
  cpu_set_type s2 (b.cbegin(), b.cend());

  set_difference (s1.begin(), s1.end(),
                  s2.begin(), s2.end(),
                  std::inserter (result, result.end()));

  set_difference (s2.begin(), s2.end(),
                  s1.begin(), s1.end(),
                  std::inserter (result, result.end()));


  return result.empty ();
}

void print_affinity (std::stringstream& oss, const cpu_map_type& a)
{
  auto iter = a.cbegin ();
  int tid = -1;
  int tid_count = 0;
  for(; iter != a.end (); iter++)
  {
    if (tid != iter->first)
    {
      if (tid_count > 0)
        oss << "}" << std::endl;

      oss << iter->first << " bound to {";
      tid_count = 0;
    }

    if (tid_count > 0)
      oss << ",";

    oss << iter->second;

    tid = iter->first;
    tid_count++;
  }

  if (tid >= 0)
  {
    oss << "}" << std::endl;
  }
}


bool analyze_node_affinities (cpu_map_type& local_cpu_map)
{
  bool good_affinity = true;
  // obtain a node communicator
  MPI_Comm nodeComm;
  get_node_mpi_comm (&nodeComm);

  int nodeCommSize;
  int nodeRank;
  MPI_Comm_size(nodeComm, &nodeCommSize);
  MPI_Comm_rank(nodeComm, &nodeRank);

  std::vector<int> local_displacements;
  std::vector<int> local_tids;
  std::vector<int> local_affinities;


  cpu_map_to_vector (local_cpu_map,
                     local_displacements,
                     local_tids,
                     local_affinities);

  // we expect all processes to have the same type of bindings.
  // e.g., all should have the same size affinity maps
  std::vector<int> remote_displacements(local_displacements.size(), -1);
  std::vector<int> remote_tids(local_tids.size(), -1);
  std::vector<int> remote_affinities(local_affinities.size(), -1);

  // this is noisy on a node, but all communication is local
  for(int remoteRank=0; remoteRank < nodeCommSize; ++remoteRank)
  {
    if (remoteRank == nodeRank)
    {
      // we are the sender
      MPI_Bcast(&local_displacements[0], local_displacements.size(), MPI_INT, nodeRank, nodeComm);
      MPI_Bcast(&local_tids[0]         , local_tids.size()         , MPI_INT, nodeRank, nodeComm);
      MPI_Bcast(&local_affinities[0]   , local_affinities.size()   , MPI_INT, nodeRank, nodeComm);
    }
    else
    {
      // we are the receiver
      MPI_Bcast(&remote_displacements[0], remote_displacements.size(), MPI_INT, remoteRank, nodeComm);
      MPI_Bcast(&remote_tids[0]         , remote_tids.size()         , MPI_INT, remoteRank, nodeComm);
      MPI_Bcast(&remote_affinities[0]   , remote_affinities.size()   , MPI_INT, remoteRank, nodeComm);
      cpu_map_type remote_cpu_map;
      vector_to_cpu_map (remote_cpu_map, remote_displacements, remote_tids, remote_affinities);
      if (! check_exclusive_affinity (local_cpu_map, remote_cpu_map) )
      {
        std::stringstream ss;
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        ss << "Rank: " << rank << ", Local Rank: " << nodeRank << ". Bad affinity detected"
            << std::endl
            << "Local Affinity: " << std::endl;
        print_affinity(ss, local_cpu_map);
        ss << "Remote Affinity: (localRank = " << remoteRank << "):" << std::endl;
        print_affinity(ss, remote_cpu_map);
        std::cerr << ss.str ();

        good_affinity = false;
      }
    }
  }

  return(good_affinity);
}


// get the local affinity information
std::string getProcessAffinitCSV_str (
    const Teuchos::RCP<const teuchos_comm_type>& globalComm,
    const cpu_map_type& a,
    const Teuchos::RCP<const teuchos_comm_type>& nodeComm)
{
  using oss_t = std::ostringstream;

  // output data as
  // MPI_comm_rank, MPI_Comm_size, local_size, Local_rank, hostname, Timestamp, thread_id, affinity set
  const int myRank = globalComm->getRank();
  const int mySize = globalComm->getSize();
  const int local_rank = nodeComm->getRank();
  const int local_procs = nodeComm->getSize ();
  const std::string hostname = getHostname ();
  const std::string timestamp = getCurrentTimeString ();

  oss_t oss;

  auto iter = a.cbegin ();
  int tid = -1;
  int tid_count = 0;
  for(; iter != a.end (); iter++)
  {
    if (tid != iter->first)
    {
      if (tid_count > 0)
        oss << std::endl;

      tid = iter->first;

      // start a new row
      oss   << myRank
          << ","
            << mySize
          << ","
            << local_procs
          << ","
            << local_rank
          << ","
            << "\""
            << hostname
            << "\""
          << ","
            << "\""
            << timestamp
            << "\""
          << ","
            << tid
          << ",";

      tid_count = 0;
    }

    // use something to separate cpu_sets (something that is not a CSV separator)
    if (tid_count > 0)
      oss << "|";

    oss << iter->second;

    tid_count++;
  }

  if (tid > 0)
  {
    oss << std::endl;
  }

  return (oss.str ());
}

// get the local affinity information
void writeAffinityCSV (const std::string filename,
                       const Teuchos::RCP<const teuchos_comm_type>& globalComm,
                       Teuchos::RCP<Teuchos::FancyOStream>& pOut,
                       Teuchos::RCP<const teuchos_comm_type>& localNodeComm)
{
  using Teuchos::RCP;
  using Teuchos::Comm;
  using Teuchos::FancyOStream;
  typedef int ordinal_type;
  using Teuchos::MpiComm;

  FancyOStream& out = *pOut;

  // get a communicator for processes on this node
  if (localNodeComm.is_null())
    localNodeComm = getNodeLocalComm(globalComm);

  const int localCommSize = localNodeComm->getSize ();
  const int localRank = localNodeComm->getRank ();

  // first collect the affinity information
  cpu_map_type cpu_map;

  // there is not a clear portable way to do this.
  // Option 1, uses a GNU extension to pthreads
  // option 2, makes a linux system call
  #ifdef USE_PTHREAD_AFFINITY
    gather_affinity_pthread (cpu_map);
  #else
    gather_affinity_linux_syscall (cpu_map);
  #endif

  // each process gets their affinity as a string
  const std::string process_affintiy_str =  getProcessAffinitCSV_str (globalComm,cpu_map,localNodeComm);

  // on rank 0, gather all strings and write the file
  const int myRank = globalComm->getRank();
  const int mySize = globalComm->getSize();

  // share the length of each string with rank 0
  std::vector<int> string_sizes (mySize);
  string_sizes.reserve (mySize);
  string_sizes.resize (mySize);
  std::vector<int> char_displacements (mySize);
  char_displacements.reserve (mySize);
  char_displacements.resize (mySize);

  const char * my_c_str = process_affintiy_str.c_str ();
  // we do not need to send the null terminator, but we should append
  // it to the gathered large string
  int my_str_length = std::char_traits<char>::length (my_c_str);

  globalComm->gather(sizeof(int),
                     reinterpret_cast<char*>(&my_str_length),
                     sizeof(int), reinterpret_cast<char*>(&string_sizes[0]),
                     0);

  int total_chars = 0;
  for (size_t i=0; i < string_sizes.size (); ++i)
  {
    char_displacements[i] = total_chars;
    total_chars += string_sizes[i];
  }

  // allocate storage
  std::vector<char> collected_strings (total_chars+1);
  collected_strings.reserve (total_chars+1);
  collected_strings.resize (total_chars+1);
  collected_strings[total_chars] = '\0'; // add the terminator

  MPI_Gatherv(
    my_c_str,
    my_str_length,
    MPI_CHAR,
    &collected_strings[0],
    &string_sizes[0],
    &char_displacements[0],
    MPI_CHAR,
    0,
    *(reinterpret_cast<const MpiComm<ordinal_type>&>(*globalComm).getRawMpiComm () )
  );

  if (myRank == 0) {
    using std::ofstream;

    ofstream csv_file;
    csv_file.open (filename);

    std::string all_data (&collected_strings[0]);
    csv_file << get_process_affinity_csv_banner ()
             << std::endl
             << all_data;
    csv_file.close();
  }

}

bool
gatherAllAffinities (Teuchos::RCP<teuchos_comm_type>& globalComm,
                     Teuchos::RCP<Teuchos::FancyOStream>& pOut)
{
  using Teuchos::RCP;
  using Teuchos::Comm;
  using Teuchos::FancyOStream;

  FancyOStream& out = *pOut;

  // get a communicator for processes on this node
  RCP<teuchos_comm_type> localNodeComm = getNodeLocalComm(globalComm);

  const int localCommSize = localNodeComm->getSize ();
  const int localRank = localNodeComm->getRank ();

  std::vector<int> local_displacements;
  std::vector<int> local_tids;
  std::vector<int> local_affinities;


  // gather this process' affinity
  cpu_map_type this_cpu_map;
  gather_affinity_pthread (this_cpu_map);

  // serialize the cpu_map
  cpu_map_to_vector (this_cpu_map,
                     local_displacements,
                     local_tids,
                     local_affinities);

  // we expect all processes to have the same type of bindings.
  // e.g., all should have the same size affinity maps
  std::vector<int> remote_displacements(local_displacements.size(), -1);
  std::vector<int> remote_tids(local_tids.size(), -1);
  std::vector<int> remote_affinities(local_affinities.size(), -1);


  bool good_affinity = true;
  // this is noisy on a node, but all communication is local
  for(int remoteRank=0; remoteRank < localCommSize; ++remoteRank)
  {
    if (remoteRank == localRank)
    {
      // we are the sender
      localNodeComm->broadcast(localRank,
                               local_displacements.size() * sizeof(int),
                               reinterpret_cast<char *>(&local_displacements[0]));

      localNodeComm->broadcast(localRank,
                               local_tids.size() * sizeof(int),
                               reinterpret_cast<char *>(&local_tids[0]));

      localNodeComm->broadcast(localRank,
                               local_affinities.size() * sizeof(int),
                               reinterpret_cast<char *>(&local_affinities[0]));

    }
    else
    {
      // we are the receiver
      localNodeComm->broadcast(remoteRank,
                               local_displacements.size() * sizeof(int),
                               reinterpret_cast<char *>(&local_displacements[0]));

      localNodeComm->broadcast(remoteRank,
                               local_tids.size() * sizeof(int),
                               reinterpret_cast<char *>(&local_tids[0]));

      localNodeComm->broadcast(remoteRank,
                               local_affinities.size() * sizeof(int),
                               reinterpret_cast<char *>(&local_affinities[0]));

      // reconstruct the cpu_map
      cpu_map_type remote_cpu_map;
      vector_to_cpu_map (remote_cpu_map, remote_displacements, remote_tids, remote_affinities);

      if (! check_exclusive_affinity (this_cpu_map, remote_cpu_map) )
      {

        std::stringstream ss;

        ss  << "Rank: " << globalComm->getRank () << ", Local Rank: " << localRank << ". Bad affinity detected"
            << std::endl
            << "Local Affinity: " << std::endl;

        print_affinity(ss, this_cpu_map);
        ss << "Remote Affinity: (localRank = " << remoteRank << "):" << std::endl;
        print_affinity(ss, remote_cpu_map);
        out << ss.str ();

        good_affinity = false;
      }
    }

  }

  return(good_affinity);
}


// convert the map into a list of cpu_ids
// record the indices where different thread's affinities begin.
void cpu_map_to_vector (const cpu_map_type& cpu_map,
                        std::vector<int>& displacements,
                        std::vector<int>& tids,
                        std::vector<int>& affinities)
{
  int prior_tid = -1;
  // maps have a guaranteed ordering
  for (const auto& kv : cpu_map) {
    const auto tid = kv.first;
    const auto cpu_id = kv.second;

    // if this is a new tid, then store its displacement in the affinity vector
    if (tid != prior_tid)
    {
      const int displacement = affinities.size ();
      if (displacement > 0)
        displacements.push_back (displacement);
      tids.push_back (tid);
      prior_tid = tid;
    }

    affinities.push_back (cpu_id);
  }
}

// convert the map into a list of cpu_ids
// record the indices where different thread's affinities begin.
void vector_to_cpu_map (cpu_map_type& cpu_map,
                        const std::vector<int>& displacements,
                        const std::vector<int>& tids,
                        const std::vector<int>& affinities)
{
  cpu_map.clear ();
  int didx = 0;
  for(int idx=0; idx < affinities.size(); ++idx)
  {
    if (idx == displacements[didx])
      ++didx;

    int tid = tids[didx];

    cpu_map.insert( std::make_pair(tid, affinities[idx]) );

  }
}

//
//// convert the map into a list of cpu_ids
//// record the indices where different thread's affinities begin.
//void gather_nodal_affinties (const cpu_map_type& cpu_map,
//                                   node_affinity_type& node_map,
//                                   MPI_Comm& nodeComm)
//{
//  std::vector<int> local_displacements;
//  std::vector<int> local_tids;
//  std::vector<int> local_affinities;
//
//  std::vector<int> node_displacements;
//  std::vector<int> node_tids;
//  std::vector<int> node_affinities;
//
//  cpu_map_to_vector (cpu_map,
//                     local_displacements,
//                     local_tids,
//                     local_affinities);
//
//  vector_gatherv (local_displacements, node_displacements, nodeComm);
//  vector_gatherv (local_tids, node_tids, nodeComm);
//  vector_gatherv (local_affinities, node_affinities, nodeComm);
//
//
//
//}

//void vector_gatherv (const std::vector<int>& local_vec, std::vector<int>& global_vec, MPI_Comm& nodeComm)
//{
//  int commSize;
//  MPI_Comm_size (nodeComm, &commSize);
//
//  std::vector<int> vec_sizes;
//  vec_sizes.reserve (commSize);
//  vec_sizes.resize (commSize);
//  std::vector<int> vec_displ;
//  vec_displ.reserve (commSize);
//  vec_displ.resize (commSize);
//
//  int vec_size = local_vec.size ();
//  int global_size = 0;
//
//  // gather the sizes from our neighbors
//  MPI_Gather(
//    &vec_size,
//    1,
//    MPI_INT,
//    &vec_sizes[0],
//    1,
//    MPI_INT,
//    0,
//    MPI_COMM_WORLD);
//
//  for (int i=0; i < vec_sizes.size (); ++i)
//  {
//    vec_displ[i] = global_size;
//    global_size += vec_sizes[i];
//  }
//
//  global_vec.reserve (global_size);
//  global_vec.resize (global_size);
//
//  // gather the data
//  MPI_Gatherv(
//    &local_vec[0],
//    vec_size,
//    MPI_INT,
//    &global_vec[0],
//    &vec_sizes[0],
//    &vec_displ[0],
//    MPI_INT,
//    0,
//    MPI_COMM_WORLD
//  );
//}

//void analyze_node_affinities (cpu_map_type& local_cpu_map,
//                              std::vector<cpu_map_type>& node_cpu_map,
//                              MPI_Comm& nodeComm)
//{
//
//  // convert the map into a vector and record the offsets for each thread's entries
//  std::vector<int> displacements;
//  std::vector<int> affinities;
//  cpu_map_to_vector (local_cpu_map, displacements, affinities);
//
//  // gather this information at a node level
//  int nodeRank;
//  MPI_Comm_size (nodeComm, &nodeRank);
//  int nodeSize;
//  MPI_Comm_size (nodeComm, &nodeSize);
//
//
//  int myLengths[2];
//  myLengths[0] = displacements.size ();
//  myLengths[2] = affinities.size ();
//
//  std::vector<int> node_displ_aff_lengths;
//  node_displ_aff_lengths.reserve (nodeSize*2);
//  node_displ_aff_lengths.resize (nodeSize*2);
//
//  MPI_Gather(
//    myLengths,
//    2,
//    MPI_INT,
//    &node_displ_aff_lengths[0],
//    2,
//    MPI_INT,
//    0,
//    MPI_COMM_WORLD);
//
//  std::vector<int> node_displacements;
//  std::vector<int> node_affinities;
//
//  int total_node_displ_size = 0;
//  int total_node_affinity_size = 0;
//
//  for (int i=0; i < node_displ_aff_lengths.size (); ++i)
//  {
//    char_displacements[i] = total_chars;
//    total_chars += string_sizes[i];
//  }
//
//  std::vector<int> node_displacements;
//  std::vector<int> node_affinities;
//}

// get the local affinity information
void write_affinity_csv (const std::string filename)
{
  // obtain a node communicator
  MPI_Comm nodeComm;
  get_node_mpi_comm (&nodeComm);

  // first collect the affinity information
  cpu_map_type cpu_map;

  // there is not a clear portable way to do this.
  // Option 1, uses a GNU extension to pthreads
  // option 2, makes a linux system call
  #ifdef USE_PTHREAD_AFFINITY
    gather_affinity_pthread (cpu_map);
  #else
    gather_affinity_linux_syscall (cpu_map);
  #endif

  // each process gets their affinity as a string
  const std::string process_affintiy_str = get_process_affinity_csv_string (cpu_map, nodeComm);

  // on rank 0, gather all strings and write the file
  int myRank;
  MPI_Comm_rank (MPI_COMM_WORLD, &myRank);
  int mySize;
  MPI_Comm_size (MPI_COMM_WORLD, &mySize);

  // share the length of each string with rank 0
  std::vector<int> string_sizes (mySize);
  string_sizes.reserve (mySize);
  string_sizes.resize (mySize);
  std::vector<int> char_displacements (mySize);
  char_displacements.reserve (mySize);
  char_displacements.resize (mySize);

  const char * my_c_str = process_affintiy_str.c_str ();
  // we do not need to send the null terminator, but we should append
  // it to the gathered large string
  int my_str_length = std::char_traits<char>::length (my_c_str);

  MPI_Gather(
    &my_str_length,
    1,
    MPI_INT,
    &string_sizes[0],
    1,
    MPI_INT,
    0,
    MPI_COMM_WORLD);

  int total_chars = 0;
  for (size_t i=0; i < string_sizes.size (); ++i)
  {
    char_displacements[i] = total_chars;
    total_chars += string_sizes[i];
  }

  // allocate storage
  std::vector<char> collected_strings (total_chars+1);
  collected_strings.reserve (total_chars+1);
  collected_strings.resize (total_chars+1);
  collected_strings[total_chars] = '\0'; // add the terminator

  MPI_Gather(
    &my_str_length,
    1,
    MPI_INT,
    &string_sizes[0],
    1,
    MPI_INT,
    0,
    MPI_COMM_WORLD);

  MPI_Gatherv(
    my_c_str,
    my_str_length,
    MPI_CHAR,
    &collected_strings[0],
    &string_sizes[0],
    &char_displacements[0],
    MPI_CHAR,
    0,
    MPI_COMM_WORLD
  );

  if (myRank == 0) {
    using std::ofstream;

    ofstream csv_file;
    csv_file.open (filename);

    std::string all_data (&collected_strings[0]);
    csv_file << get_process_affinity_csv_banner ()
             << std::endl
             << all_data;
    csv_file.close();
  }

}

std::string get_process_affinity_csv_banner ()
{
  return "MPI_comm_rank, MPI_Comm_size, local_size, local_rank, hostname, Timestamp, thread_id, affinity set";
}

// get the local affinity information
std::string get_process_affinity_csv_string (const cpu_map_type& a, MPI_Comm& nodeComm)
{
  using oss_t = std::ostringstream;

  // output data as
  // MPI_comm_rank, MPI_Comm_size, local_size, Local_rank, hostname, Timestamp, thread_id, affinity set
  int myRank;
  int mySize;
  MPI_Comm_rank (MPI_COMM_WORLD, &myRank);
  MPI_Comm_size (MPI_COMM_WORLD, &mySize);
  const int local_rank = get_local_rank (nodeComm);
  const int local_procs = get_local_size (nodeComm);
  const std::string hostname = getHostname ();
  const std::string timestamp = getCurrentTimeString ();

  oss_t oss;

  auto iter = a.cbegin ();
  int tid = -1;
  int tid_count = 0;
  for(; iter != a.end (); iter++)
  {
    if (tid != iter->first)
    {
      if (tid_count > 0)
        oss << std::endl;

      tid = iter->first;

      // start a new row
      oss   << myRank
          << ","
            << mySize
          << ","
            << local_procs
          << ","
            << local_rank
          << ","
            << "\""
            << hostname
            << "\""
          << ","
            << "\""
            << timestamp
            << "\""
          << ","
            << tid
          << ",";

      tid_count = 0;
    }

    // use something to separate cpu_sets (something that is not a CSV separator)
    if (tid_count > 0)
      oss << "|";

    oss << iter->second;

    tid_count++;
  }

  if (tid > 0)
  {
    oss << std::endl;
  }

  return oss.str ();
}

// C++11's standard output lacks precision finer than a second.
// This method from stackexchange seems to be the least convoluted
// http://stackoverflow.com/questions/12835577/how-to-convert-stdchronotime-point-to-calendar-datetime-string-with-fraction
//
std::string getCurrentTimeString ()
{
  using clock = std::chrono::system_clock;
  using time_point = clock::time_point;
  using oss_t = std::ostringstream;

  time_point current = std::chrono::system_clock::now();

  // Convert std::chrono::system_clock::time_point to std::time_t
  time_t tt = clock::to_time_t(current);
  // Convert std::time_t to std::tm (popular extension)
  std::tm tm = std::tm{0};
  gmtime_r(&tt, &tm);

  oss_t oss;

  // Output month
  oss << tm.tm_mon + 1 << '-';
  // Output day
  oss << tm.tm_mday << '-';
  // Output year
  oss << tm.tm_year+1900 << ' ';
  // Output hour
  if (tm.tm_hour <= 9)
    oss << '0';
  oss << tm.tm_hour << ':';
  // Output minute
  if (tm.tm_min <= 9)
    oss << '0';
  oss << tm.tm_min << ':';
  // Output seconds with fraction
  //   This is the heart of the question/answer.
  //   First create a double-based second
  std::chrono::duration<double> sec = current -
                                 clock::from_time_t(tt) +
                                 std::chrono::seconds(tm.tm_sec);
  //   Then print out that double using whatever format you prefer.
  if (sec.count() < 10)
     oss << '0';
  oss << std::fixed << sec.count();

  return oss.str ();
}


#undef _GNU_SOURCE

} // end namespace PerfUtils
