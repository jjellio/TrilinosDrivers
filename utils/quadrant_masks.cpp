#include <iostream>
#include <cmath>
#include <sstream>
#include <deque>
#include <vector>

const int QUADRANT_0[] = {0, 1,12,13,20,21,28,29,36,37,42,43,50,51,58,59, 66,67};  // 18
const int QUADRANT_1[] = {2, 3, 6, 7,14,15,22,23,30,31,38,39,44,45,52,53, 60,61};  // 18
const int QUADRANT_2[] = {4, 5,10,11,18,19,26,27,34,35,48,49,56,57,64,65       };  // 16
const int QUADRANT_3[] = {8, 9,16,17,24,25,32,33,40,41,46,47,54,55,62,63       };  // 16

const int EXCLUDED_CORES[] = {66, 67, 60, 61};
const int NUM_EXCLUDED_CORES = 4;
const int NUM_CORES=68;

void add_quadrant(const int * quadrant, const int core_count, std::deque< std::deque<int> >& quadrants) {
  std::deque<int> quad;
  for (int q=0; q < core_count; ++q) {
    int core = quadrant[q];
    for(int ex=0; ex < NUM_EXCLUDED_CORES; ++ex) {
      if (core == EXCLUDED_CORES[ex]) {
        core = -1;
        break;
      }
    }
    if (core != -1)
      quad.push_back(core);
    
  }
  
  quadrants.push_back(quad);
}

std::deque<int> get_cores (const int num_cores, std::deque< std::deque<int> >& quadrants) {

  while (! quadrants.empty ()) {
  
    if (quadrants.front().size () < num_cores) {
      quadrants.pop_front();
      continue;      
    }

    std::deque<int> cores;
    while ((cores.size() != num_cores) && (quadrants.front().size () > 0)) {
      const auto& core = quadrants.front().front();
      cores.push_back(core);
      quadrants.front().pop_front();
    }
    
    if (cores.size() == num_cores)
      return cores;
      
    quadrants.pop_front();
  }
  std::cerr << "Unable to select a cpuset that does not span quadrants!" << std::endl;
  // failure
  exit(-1);
}

int main (int argc, char** argv) {

int ppn = 32;
int cpp = -1;
int tpc = 4;

if (argc == 4) {
  ppn = std::atoi(argv[1]);
  cpp = std::atoi(argv[2]);
  tpc = std::atoi(argv[3]);
}
else {
  cpp = 64 / ppn;
}

const int procs_per_node = ppn;
const int cores_per_proc = cpp;
const int threads_per_core = tpc;
std::deque< std::deque<int> > quadrants;
std::vector< std::deque<int> > tasks;

add_quadrant(QUADRANT_0, 18, quadrants);
add_quadrant(QUADRANT_1, 18, quadrants);
add_quadrant(QUADRANT_2, 16, quadrants);
add_quadrant(QUADRANT_3, 16, quadrants);

for (int proc=0; proc < procs_per_node; ++proc) {
  tasks.push_back(get_cores (cores_per_proc, quadrants));
}

using std::stringstream;
using std::endl;

stringstream details_ss;
stringstream bitmask_ss;
int t=0;
stringstream hexout;

for (const auto& task : tasks) {
  if (t>0) {
    bitmask_ss << ",";
  }
  details_ss << "Task : " << ++t << " : ";
  
  char bitmask[69];
	for (int c=0; c < 68; ++c) bitmask[c] = '0';
	bitmask[68] = '\0';
	
  for (const auto& core : task) {
    details_ss << core << ",";
    bitmask[67-core] = '1';
  }

	stringstream current_mask;
  stringstream hexstr;
  stringstream zeros;
  current_mask << "0x";
  
  for (int i=0; i < 68; i += 4) {
    int h = 0;
	  if (bitmask[i+3] == '1')
	    h += 1;
	  if (bitmask[i+2] == '1')
	    h += 2;
	  if (bitmask[i+1] == '1')
	    h += 4;
	  if (bitmask[i] == '1')
	    h += 8;
		hexstr << std::hex << h;
		zeros << "0";
  }
  
  switch (threads_per_core) {
  case 1:
    current_mask
       << zeros.str()
       << zeros.str()
       << zeros.str()
       << hexstr.str();
       break;
  case 2:
    current_mask 
       << zeros.str()
       << hexstr.str()
       << zeros.str()
       << hexstr.str();
       break;
  case 3:
    current_mask
       << zeros.str()
       << hexstr.str()
       << hexstr.str()
       << hexstr.str();
       break;
  case 4:
    current_mask
       << hexstr.str()
       << hexstr.str()
       << hexstr.str()
       << hexstr.str();
       break;
  }
  
  details_ss << current_mask.str() << endl;
  bitmask_ss << current_mask.str();

}

std::cout << bitmask_ss.str ();
std::cerr << details_ss.str();

return(0);
}

