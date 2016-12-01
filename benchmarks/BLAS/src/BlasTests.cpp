#include <cstdio>
#include <cstdlib>  // malloc
#include <cstdint>   // int32_t
#include <chrono>    // timers
#include <random>    // random number generators
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <iostream>

#ifdef HAVE_MPI
  #include <mpi.h>
//  #include "mpi_local_ranks.hpp"
#endif

#include <omp.h>

#ifdef FORTRAN_NEEDS_UNDERSCORE
  #define DGEMV dgemv_
  #define ILAVER ilaver_
#else
  #define DGEMV dgemv
  #define ILAVER ilaver
#endif

extern "C" {
  void dgemv(char*, int32_t*, int32_t*, const double *, const double *, int32_t*, const double *, int32_t*, const double *, double *, int32_t*);
  void dgemv_(char*, int32_t*, int32_t*, const double *, const double *, int32_t*, const double *, int32_t*, const double *, double *, int32_t*);

  void ilaver(int* major,int* minor,int* patch);
  void ilaver_(int* major,int* minor,int* patch);
}


struct experiment_pack {

  experiment_pack (const int n, const int m, const int numProcs, const int numTrials)
  : n(n),
    m(m),
    numProcs(numProcs),
    numTrials (numTrials),
    update_sec (0.0),
    update_ops (0.0),
    update_gflops (0.0),
    inner_product_sec (0.0),
    inner_product_ops (0.0),
    inner_product_gflops (0.0),
    normalized_update_time (0.0),
    normalized_inner_product_time (0.0)
  {}

  int n;
  int m;
  int numProcs;
  int numTrials;
  double update_sec;
  double update_ops;
  double update_gflops;
  double inner_product_sec;
  double inner_product_ops;
  double inner_product_gflops;
  double normalized_update_time;
  double normalized_inner_product_time;
  std::string OMP_WAIT_POLICY;
  std::string OMP_PLACES;

  std::vector<double> inner_product_timings_ns;
  std::vector<double> update_timings_ns;
};


double inner_product_dgemv_ops (double m, double n);
double update_dgemv_ops (double m, double n);

void randomize2D (double * u, int i_max, int j_max, std::random_device& rd)
{
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(-1, 1);

  int i,j;
  for(i = 0; i < i_max; i++){
     for(j = 0; j < j_max; j++){
        u[j + i * j_max] = dis(gen);
     }
  }
}

void randomize1D (double * u, int i_max, std::random_device& rd)
{
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(-1, 1);

  int i;
  for(i = 0; i < i_max; i++){
    u[i] = dis(gen);
  }
}

void psuedoOrtho (experiment_pack& ex, double * A, double * x_inner, double * y_inner, double * x_update, double * y_update)
{
  using std::chrono::steady_clock;

  char trans='N';
  int32_t inc_x=1,inc_y=1;

  int l;

  double alpha_inner[] = {1.0, 1.0};
  double beta_inner [] = {0.0, 0.0};

  double alpha_update[] = {-1.0, -1.0};
  double beta_update [] = { 1.0,  1.0};


  auto timer_inner = steady_clock::duration::zero ();
  auto timer_update = steady_clock::duration::zero ();

  ex.inner_product_timings_ns.clear ();
  ex.update_timings_ns.clear ();

  #ifdef HAVE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
  #endif

  for (l=0; l<ex.numTrials; l++)
  {
   // time it
   {
    trans='T';
    auto start = steady_clock::now();
    DGEMV (&trans, &ex.m, &ex.n, alpha_inner, A, &ex.m, x_inner, &inc_x, beta_inner, y_inner, &inc_y );
    auto stop = steady_clock::now();
    auto timing = stop - start;

    // record the specific timing (no averaging)
    ex.inner_product_timings_ns.push_back(std::chrono::duration_cast<std::chrono::nanoseconds> (timing).count ());

    timer_inner += timing;
   }

   #ifdef HAVE_MPI
     MPI_Barrier(MPI_COMM_WORLD);
   #endif
   {
    trans='N';
    auto start = steady_clock::now();
    DGEMV (&trans, &ex.m, &ex.n, alpha_update, A, &ex.m, x_update, &inc_x, beta_update, y_update, &inc_y );
    auto stop = steady_clock::now();
    auto timing = stop - start;

    // record the specific timing (no averaging)
    ex.update_timings_ns.push_back(std::chrono::duration_cast<std::chrono::nanoseconds> (timing).count ());

    timer_update += timing;
   }
   #ifdef HAVE_MPI
     MPI_Barrier(MPI_COMM_WORLD);
   #endif
  }

  timer_inner /= (double) ex.numTrials;
  timer_update /= (double) ex.numTrials;
  // convert to seconds
  typedef std::chrono::duration<double> double_sec_type;
  ex.inner_product_sec = double_sec_type(timer_inner).count();
  ex.inner_product_ops = inner_product_dgemv_ops(ex.m,ex.n);
  ex.inner_product_gflops = ex.inner_product_ops / ex.inner_product_sec * 1.e-9;

  ex.update_sec = double_sec_type(timer_update).count();
  ex.update_ops = update_dgemv_ops(ex.m,ex.n);
  ex.update_gflops = ex.update_ops / ex.update_sec * 1.e-9;

  // query the OMP env
  if(const char* env_omp_wait = std::getenv("OMP_WAIT_POLICY"))
  {
    ex.OMP_WAIT_POLICY = std::string(env_omp_wait);
  }
  else
  {
    std::cerr << "Failed to look up OMP_WAIT_POLICY" << std::endl;
    exit (-1);
  }

  if(const char* env_omp_places = std::getenv("OMP_PLACES"))
  {
    ex.OMP_PLACES = std::string(env_omp_places);
  }
  else
  {
    std::cerr << "Failed to look up OMP_PLACES" << std::endl;
    exit (-1);
  }
}


double inner_product_dgemv_ops (double m, double n)
{
  // assume beta == 0 is optimized away
  // assume alpha == 1 is optimized away
  // we compute (n x m) dot (m x 1)
  // For each entry in the resulting nx1 matrix
  // we must compute m-1 additions and m multiplies
  // The total ops per cell is 2m-1
  // The total ops with alpha and beta optimized out is
  // n * (2m-1)
  return (double) ((n)*(2*m - 1));
}

double update_dgemv_ops (double m, double n)
{
  // assume beta == 1 is optimized away (no penalty for multiply)y
  // we compute (m x n) dot (n x 1)
  // For each entry in the resulting mx1 vector
  // we must compute n-1 additions and n multiplies
  // The total ops per cell is 2n-1
  // The total ops to form the first result is m*(2n-1)
  // Assume, alpha == -1 transforms the resulting scale + add into subtraction
  // We assume then m ops to add the two vectors
  // Total ops:
  //            m*(2n-1) + m
  // If the negation + addition is two ops, then
  //            m*(2n-1) + 2m
  return (double) ((m)*(2*n - 1) + m);
}

std::string getLapackVersion ()
{
  int major, minor, patch;

  ILAVER (&major, &minor, &patch);

  std::stringstream ss;
  ss << major << "." << minor << "." << patch;

  return ss.str ();
}

#ifdef MKL_LIB

  #include <mkl_version.h>

  std::string getBlasLibVersion ()
  {
    /*
        #define __INTEL_MKL_BUILD_DATE 20151022

        #define __INTEL_MKL__ 11
        #define __INTEL_MKL_MINOR__ 3
        #define __INTEL_MKL_UPDATE__ 1
     */

    std::stringstream ss;

    ss << "MKL " << __INTEL_MKL__ << "." << __INTEL_MKL_MINOR__ << "." << __INTEL_MKL_UPDATE__ << "." << __INTEL_MKL_BUILD_DATE;
    return ss.str ();
  }
#elif OPENBLAS_LIB

  #include <openblas_config.h>

  std::string getBlasLibVersion ()
  {
    /*
        #define OPENBLAS_VERSION " OpenBLAS 0.2.19.dev "
     */

    std::stringstream ss;

    ss << OPENBLAS_VERSION;
    return ss.str ();
  }
#elif LIBSCI_LIB
  // No API for libsci... Compilation should pass the version
  // -DLIBSCI_VERSION=\"${LIBSCI_VERSION}\"
  std::string getBlasLibVersion ()
  {
    char * libsci_ver = LIBSCI_VERSION;

    std::stringstream ss;
    ss << "Libsci " << libsci_ver;
    return ss.str ();
  }

#else
  #warning("No define is specified for the blas lib being linked. Please Define one of: -DMKL_LIB, -DOPENBLAS_LIB")
  std::string getBlasLibVersion ()
  {
    return "UNKNOWN";
  }
#endif


std::string getDateTime ()
{
  using std::chrono::system_clock;

  auto current_time_point = system_clock::now();

  auto current_ctime = system_clock::to_time_t(current_time_point);
  std::tm now_tm = *std::localtime(&current_ctime);

  char s[1000];
  std::strftime(s, 1000, "%c", &now_tm);

  return std::string(s);
}

void gatherHostnames (std::vector<std::string>& hostnames)
{
  char name[MPI_MAX_PROCESSOR_NAME];
  int len;
  MPI_Get_processor_name( name, &len );


  int rank;
  int numProcs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

  typedef struct CStrings {
    char cstr[MPI_MAX_PROCESSOR_NAME];
  } CStrings_t;

  std::vector<CStrings_t> tmp_cstrings;

  if (rank == 0)
  {
    tmp_cstrings.reserve (numProcs);
    tmp_cstrings.resize (numProcs);

    hostnames.clear ();
    hostnames.reserve (numProcs);
    hostnames.resize (numProcs);
  }

  MPI_Gather(name,  MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
             &tmp_cstrings[0], MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
             0, MPI_COMM_WORLD);

  if (rank == 0)
  {
    for (int i=0; i < numProcs; ++i)
    {
      hostnames[i] = tmp_cstrings[i].cstr;
    }
  }
}

void descriptive_stats (std::vector<double>& all_timings,
                        FILE * aggr_fptr,
                        const experiment_pack& ex,
                        FILE * detail_fptr,
                        const std::string& label,
                        const std::vector<double>& local_data,
                        const std::string& timestamp,
                        const std::vector<std::string>& hostnames)
{
  int rank;
  int numProcs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

  std::fill(all_timings.begin(), all_timings.end(), 0.0);

  const int numTrials = local_data.size ();

  MPI_Gather(&local_data[0],  numTrials, MPI_DOUBLE,
             &all_timings[0], numTrials, MPI_DOUBLE,
             0, MPI_COMM_WORLD);


  const std::string blasLib = getBlasLibVersion ();
  const std::string lapackVersion = getLapackVersion ();
  if (rank == 0)
  {

    for (int r=0; r < numProcs; ++r)
    {
      for (int j=0; j < numTrials; ++j)
      {
        // fprintf(detail_fptr, "Label, hostname, rank, np, OMP_NUM_THREADS, OMP_WAIT_POLICY, OMP_PLACES, m (local), n, time (ns), Date\n");
        fprintf(detail_fptr,
         "\"%s\", \"%s\", %d, %d, %d, \"%s\", \"%s\", %d, %d, %10.0f, \"%s\", \"%s\", \"%s\"\n",
         label.c_str(),
         hostnames[r].c_str (),
         r,
         numProcs,
         omp_get_max_threads (),
         ex.OMP_WAIT_POLICY.c_str(),
         ex.OMP_PLACES.c_str(),
         ex.m,
         ex.n,
         all_timings[r*numTrials + j],
         timestamp.c_str (),
         blasLib.c_str (),
         lapackVersion.c_str ()
        );
      }
    }
  }

  // Always do this after writing the details, because we modify the data (sort)

  // Create a summary of the data.. this may have no value with the detailed output,
  // since that data enables the calculation of these values
  if (rank == 0)
  {
    // sort the times
    std::sort (all_timings.begin(), all_timings.end());

    double stats_min = all_timings[0];
    double stats_max = all_timings.back ();
    double stats_mean = std::accumulate( all_timings.begin(), all_timings.end(), 0.0)/all_timings.size();
    auto num_samples = all_timings.size ();
    size_t first_quartile_idx = (num_samples / 4) - 1;
    size_t second_quartile_idx = (num_samples / 4)*2 - 1;
    size_t third_quartile_idx = (num_samples / 4)*3 - 1;

    fprintf(aggr_fptr,
     "\"%s\", %d, %d, \"%s\", \"%s\", %d, %d, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, %10.6f, \"%s\", \"%s\", \"%s\"\n",
      label.c_str(),
      numProcs,
      omp_get_max_threads (),
      ex.OMP_WAIT_POLICY.c_str(),
      ex.OMP_PLACES.c_str(),
      ex.m,
      ex.n,
      stats_min,
      stats_max,
      stats_mean,
      all_timings[second_quartile_idx],
      all_timings[first_quartile_idx],
      all_timings[third_quartile_idx],
      timestamp.c_str (),
      blasLib.c_str (),
      lapackVersion.c_str ()
      );
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char *argv[])
{

#ifdef HAVE_MPI
  MPI_Init (&argc, &argv);
#endif
  int rank = 0;
  int commSize = 1;

#ifdef HAVE_MPI
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &commSize);
#endif

  const int numTrials = 10;
  const int numProcs = commSize;
  const int n_min = 1;
  const int n_max = 10;
  const int m_max = 144*144*144*5;
  const int m = m_max / numProcs;
  std::random_device rd;

  if (m_max % numProcs != 0)
  {
    fprintf(stderr, "ERROR, m_max does not divided evenly among numProcs, m=%d, np=%d",m_max, numProcs);
    exit (-1);
  }

  // Ortho inner products:
  // y = alpha * A^T * x + beta * y
  // A = m x n
  // x = m x 1
  // y = n x 1
  // alpha = 1, beta = 0
  //
  // Then the updates to remove the components
  // y = alpha * A * z + beta * y
  // A = m x n
  // z = n x 1
  // y = m x 1
  // alpha = -1, beta = 1
  double * y_inner;
  double * A;
  double * x_inner;

  double * y_update;
  double * x_update;
  // posix_memalign(void **memptr, size_t alignment, size_t size)

  if (posix_memalign (reinterpret_cast<void **> (&A), 128, sizeof(double) * m * n_max) != 0){
    fprintf(stderr,"Out of Memory!!\nFailed to alloc %ld x %d x %d = %ld",
            sizeof(double), m, n_max, sizeof(double) * m * n_max );exit(1);
  }

  if (posix_memalign (reinterpret_cast<void **> (&x_inner), 128, sizeof(double) * m) != 0){
    fprintf(stderr,"Out of Memory!!\nFailed to alloc %ld x %d x %d = %ld",
            sizeof(double), m, 1, sizeof(double) * m * 1 );exit(1);
  }

  if (posix_memalign (reinterpret_cast<void **> (&y_inner), 128, sizeof(double) * n_max) != 0){
    fprintf(stderr,"Out of Memory!!\nFailed to alloc %ld x %d x %d = %ld",
            sizeof(double), n_max, 1, sizeof(double) * n_max * 1 );exit(1);
  }

  if (posix_memalign (reinterpret_cast<void **> (&x_update), 128, sizeof(double) * n_max) != 0){
    fprintf(stderr,"Out of Memory!!\nFailed to alloc %ld x %d = %ld",
            sizeof(double), n_max, sizeof(double) * n_max);exit(1);
  }

  if (posix_memalign (reinterpret_cast<void **> (&y_update), 128, sizeof(double) * m) != 0){
    fprintf(stderr,"Out of Memory!!\nFailed to alloc %ld x %d = %ld",
            sizeof(double), m, sizeof(double) * m  );exit(1);
  }

  randomize2D (A, n_max, m, rd);
  randomize1D (x_inner, m, rd);
  randomize1D (y_inner, n_max, rd);
  randomize1D (y_update, m, rd);
  randomize1D (x_update, n_max, rd);

  std::vector<experiment_pack> experiments;
  const std::string update_label        = "neg A x X + 1*Y";
  const std::string inner_product_label = "A^t x X + 0*Y";


  for (int n=n_min; n <= n_max; ++n)
  {
    experiments.push_back (experiment_pack(n, m, numProcs, numTrials));
  }

  // randomize the order, but this keeps the pattern of inner product/squash
  //std::random_shuffle(experiments.begin(), experiments.end());

  // run the experiments
  for(auto& ex : experiments)
  {
    psuedoOrtho (ex, A, x_inner, y_inner, x_update, y_update);
  }

//  // post process the data
//  for(auto& ex : experiments)
//  {
//    if (ex.numProcs == 1)
//    {
//      // numProcs == 1 solves the full problem
//      // numProcs == 2 solves exactly half, so if the kernel scales
//      // then time_np1 approx time_np2*2
//      // Normalize this as norma_time = time_np_x * x / time_np1
//      for(auto& ex2 : experiments)
//      {
//        if (ex.n == ex2.n)
//        {
//          // n and m define comparable results
//          ex2.normalized_inner_product_time = (ex2.inner_product_sec * ex2.numProcs) / ex.inner_product_sec;
//          ex2.normalized_update_time = (ex2.update_sec * ex2.numProcs) / ex.update_sec;
//        }
//      }
//    }
//  }

  FILE * aggr_fptr   = NULL;
  FILE * detail_fptr = NULL;
  std::vector<double> all_timings;
  std::vector<std::string> hostnames;
  std::string timeString = getDateTime ();

  // print things
  if (rank == 0)
  {
    std::cout << getLapackVersion () << std::endl;

    aggr_fptr = fopen( "aggregate.csv", "w+");
    if (! aggr_fptr)
    {
      fprintf(stderr, "Failed to open file for aggregate writing");
      exit (-1);
    }

    detail_fptr = fopen( "details.csv", "w+");
    if (! detail_fptr)
    {
      fprintf(stderr, "Failed to open file for detail writing");
      exit (-1);
    }


    all_timings.reserve (numProcs * numTrials);
    all_timings.resize (numProcs * numTrials);

    fprintf(aggr_fptr,   "Label, np, OMP_NUM_THREADS, OMP_WAIT_POLICY, OMP_PLACES, m (local), n, Min proc time (ns), Max proc time (ns), Average proc time (ns), Median (ns), 1st Quartile (ns), 3rd Quartile (ns), Date, BlasLib, LapackVersion\n");
    fprintf(detail_fptr, "Label, hostname, rank, np, OMP_NUM_THREADS, OMP_WAIT_POLICY, OMP_PLACES, m (local), n, time (ns), Date, BlasLib, LapackVersion\n");
  }

  // gather the hostnames that map to the ranks
  gatherHostnames (hostnames);

  #ifdef HAVE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
  #endif

  for(auto& ex : experiments)
  {
    // construct an aggregate output and detailed output
    descriptive_stats (all_timings, aggr_fptr, ex, detail_fptr, inner_product_label, ex.inner_product_timings_ns, timeString, hostnames);
    descriptive_stats (all_timings, aggr_fptr, ex, detail_fptr, update_label, ex.update_timings_ns, timeString, hostnames);
  }
  // print things
  if (rank == 0)
  {
    fclose( aggr_fptr );
    fclose( detail_fptr );
  }

  free (y_inner);
  free (A);
  free (x_inner);
  free (y_update);
  free (x_update);

  #ifdef HAVE_MPI
    MPI_Finalize ();
  #endif

  return 0;
}

// void main(int argc, char *argv[]) __attribute__((weak, alias("MAIN__")));
