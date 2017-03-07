#ifndef PerfUtils_affinity_check_HPP
#define PerfUtils_affinity_check_HPP

#include <map> // multimap
#include <string>
#include <sstream>  // ostringstream
#include <vector>

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_Comm.hpp>

namespace PerfUtils {

typedef std::multimap<int,int> cpu_map_type;
typedef std::map<int,cpu_map_type> node_affinity_type;
typedef Teuchos::Comm<int> teuchos_comm_type;

void writeAffinityCSV (const std::string filename,
                       const Teuchos::RCP<const teuchos_comm_type>& globalComm,
                       Teuchos::RCP<Teuchos::FancyOStream>& pOut,
                       Teuchos::RCP<const teuchos_comm_type>& localNodeComm);

//void gather_affinity (std::vector<int>& cpus)
//{
//  cpus.clear ();
//
//  cpu_set_t my_cpuset;
//
//  CPU_ZERO(&my_cpuset);
//
//  int s = sched_getaffinity(0, sizeof(cpu_set_t), &my_cpuset);
//
//  if (s != 0)
//    do { errno = s; perror("sched_getaffinity"); exit(EXIT_FAILURE); } while (0);
//
//  for (int j = 0; j < CPU_SETSIZE; j++)
//      if (CPU_ISSET(j, &my_cpuset))
//        cpus.push_back (j);
//
//  std::sort (cpus.begin (), cpus.end ());
//}

Teuchos::RCP<const cpu_map_type> gather_affinity_pthread ();

void gather_affinity_pthread (cpu_map_type& cpu_map);

/*! \brief Compare two process affinity maps
 *
 * A cpu map is a multiset, with threads potentially mapped to the same hardware.
 * E.g., if threads are bound to cores and hardware threads are enabled, then
 * the logical threads mapping to each hyper thread will both be mapped to the same
 * physical execution units. If threads are bound to hardware threads, then these
 */
bool compare_affinity (const cpu_map_type& a, const cpu_map_type& b);

void print_affinity (std::stringstream& oss, const cpu_map_type& a);
//void print_affinity (Teuchos::FancyOStream& oss, const cpu_map_type& a);

// collective! gather each local rank's affinity and test that
// there is no overlap. returns true if affinity is exclusive
bool analyze_node_affinities (cpu_map_type& local_cpu_map);

std::string getCurrentTimeString ();

void write_affinity_csv (const std::string filename);

void cpu_map_to_vector (const cpu_map_type&,
                        std::vector<int>&,
                        std::vector<int>&,
                        std::vector<int>&);
void vector_to_cpu_map (cpu_map_type&,
                        const std::vector<int>&,
                        const std::vector<int>&,
                        const std::vector<int>&);

} // end namespace PerfUtils

#endif // PerfUtils_affinity_check_HPP
