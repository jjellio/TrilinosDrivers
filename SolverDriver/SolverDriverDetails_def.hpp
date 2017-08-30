// @HEADER
//
// ***********************************************************************
//
//        MueLu: A package for multigrid based preconditioning
//                  Copyright 2012 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact
//                    James Elliott   (jjellio@sandia.gov)
//                    Tim Fuller      (tjfulle@sandia.gov)
//
// ***********************************************************************
//
// @HEADER

#include "SolverDriverDetails_decl.hpp"

extern bool __KokkosSparse_Impl_USE_PURE_OPENMP_SPMV;

template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
SolverDriverDetails<Scalar,LocalOrdinal,GlobalOrdinal,Node>::SolverDriverDetails (
  int argc, char* argv[],
  Teuchos::CommandLineProcessor& clp,
  Teuchos::RCP<Xpetra::Parameters>& xpetraParams):
  orig_A_           (Teuchos::null),
  orig_coordinates_ (Teuchos::null),
  orig_nullspace_   (Teuchos::null),
  orig_X_           (Teuchos::null),
  orig_B_           (Teuchos::null),
  orig_map_         (Teuchos::null),
  comm_             (Teuchos::null),
  nodeComm_         (Teuchos::null),
  nodeLocalComm_    (Teuchos::null),
  pOut_             (Teuchos::null),
  driverPL_         (Teuchos::null),
  galeriPL_         (Teuchos::null),
  xpetraParams_     (xpetraParams),
  configPL_         (Teuchos::null),
  my_cpu_map_       (Teuchos::null)
{
  using CLP = Teuchos::CommandLineProcessor;
  using Teuchos::GlobalMPISession;
  using GaleriParams = Galeri::Xpetra::Parameters<GlobalOrdinal>;
  using Teuchos::RCP;
  using Teuchos::rcpFromRef;
  using Teuchos::FancyOStream;
  using Teuchos::ParameterList;
  using Teuchos::parameterList;
  using std::cout;
  using std::endl;

  GlobalOrdinal nx = 100, ny = 100, nz = 100;
  GaleriParams galeriParameters(clp, nx, ny, nz, "Laplace3D"); // manage parameters of the test case

  std::string xmlFileName = "driver.xml"; clp.setOption("xml", &xmlFileName, "read parameters from a file");
  useSmartSolverLabels_ = true;           clp.setOption("smartLabels", "nosmartLabels",
                                            &useSmartSolverLabels_,
                                            "set solver labels to match the solver name");
  checkConvergence_ = false;              clp.setOption("checkConvergence", "nocheckConvergence",
                                            &checkConvergence_,
                                            "Report if the solver converged.");
  cores_per_proc_=-1;                  clp.setOption("cores_per_proc", &cores_per_proc_,
                                            "How many physical cores are allocated to this process (required). "
                                            "For Serial, this should be 1.");
  threads_per_core_=-1;                clp.setOption("Threads_per_core", &threads_per_core_,
                                            "How many hardware threads are allowed per core (required). "
                                            "For Serial, this could be one or 2. E.g., does this process exclusively own the core,"
                                            "or does it share the core with another process. E.g., 4 MPI procs per KNL core.");

  std::string spmv_backend = "kokkos_teams";
  clp.setOption("spmv_backend", &spmv_backend,
                "The backend used for the SpMV: kokkos_teams, omp_flat, omp_nested");

  std::string reportSolversAndExit = "";
  clp.setOption("reportSolversAndExit", &reportSolversAndExit,
                "report the solvers supported by the SolverFactory and exit (SolverFactory=Belos|MueLu)");

  clp.recogniseAllOptions(true);
  switch (clp.parse(argc, argv)) {
    case CLP::PARSE_HELP_PRINTED:        exit(EXIT_SUCCESS);
    case CLP::PARSE_ERROR:
    case CLP::PARSE_UNRECOGNIZED_OPTION: exit(EXIT_FAILURE);
    case CLP::PARSE_SUCCESSFUL:          break;
  }
  {
    setenv("KOKKOS_SPARSE_CRS_SPMV", spmv_backend.c_str(), 1); // does overwrite
  }

  comm_ = Teuchos::DefaultComm<int>::getComm();
  nodeLocalComm_ = PerfUtils::getNodeLocalComm_mpi3(comm_);
  nodeComm_      = PerfUtils::getNodeComm(comm_, nodeLocalComm_);

for (int r=0; r < comm_->getSize (); ++r)
{
  if (r == comm_->getRank ()) {
    std::cout << "rank: " << comm_->getRank ()
              <<  std::endl
              << "  Node: " << PerfUtils::getHostname ()
              << endl
              << "local rank: " << nodeLocalComm_->getRank () << " / " << nodeLocalComm_->getSize ()
              <<  std::endl
              << "Node rank: " << nodeComm_->getRank () << " / " << nodeComm_->getSize ()
              << endl;
  }
  comm_->barrier ();
}

  // Instead of checking each time for rank, create a rank 0 stream
  pOut_ = fancyOStream(rcpFromRef(cout));
  // Instead of checking each time for rank, create a rank 0 stream
  pOut_->setOutputToRootOnly(0);
  FancyOStream& out = *pOut_;

  // this needs to be here, because we don't know the template types in main ()
  if (reportSolversAndExit == "Belos") {
    reportBelosSolvers();
    exit(EXIT_SUCCESS);
  }

  if (cores_per_proc_ == -1 || threads_per_core_ == -1) {
    out << "ERROR, --core_per_proc= and --threads_per_core= *must* be set." << endl;
    exit(EXIT_FAILURE);
  }

  timerReportParams_ = parameterList();
  timerReportParams_->set("YAML style",                "compact");         // "spacious" or "compact"
  timerReportParams_->set("How to merge timer sets",   "Union");
  timerReportParams_->set("alwaysWriteLocal",          false);
  timerReportParams_->set("writeGlobalStats",          true);
  timerReportParams_->set("writeZeroTimers",           true);


  driverPL_ = parameterList ();
  // the parameter list that will drive this study
  Teuchos::updateParametersFromXmlFileAndBroadcast(xmlFileName, driverPL_.ptr(), *comm_);

  // Retrieve matrix parameters (they may have been changed on the command line)
  // [for instance, if we changed matrix type from 2D to 3D we need to update nz]
  galeriPL_ = parameterList(galeriParameters.GetParameterList());


  #ifdef HAVE_MUELU_OPENMP
  {
    std::string node_name = Node::name();
    if(!comm_->getRank() && !node_name.compare("OpenMP/Wrapper"))
      out << "OpenMP Max Threads = "
          << omp_get_max_threads()
          << std::endl;
  }
  #endif

  // ===========================================================================
  // Report the types
  // ===========================================================================
  configPL_ = parameterList ("Runtime Information");
  createConfigParameterList(*configPL_);

  // =========================================================================
  // Problem construction
  // =========================================================================
  createLinearSystem(*xpetraParams_, *galeriPL_);

  // construct the file tokens used for output
  if (comm_->getRank() == 0)
    setFileTokens ();

  // =========================================================================
  // Thread Affinity construction
  // =========================================================================
  gatherAffinityInfo ();

  configPL_->print(out);

  Teuchos::TimeMonitor::summarize(out, false, true, true, Teuchos::ECounterSetOp::Union);

  if (comm_->getRank () == 0 )
    NO::execution_space::print_configuration(out, true);
}


template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void
SolverDriverDetails<Scalar,LocalOrdinal,GlobalOrdinal,Node>::createConfigParameterList (Teuchos::ParameterList& configPL)
{
  using Teuchos::ParameterList;
  // gather information about MPI
  {
    ParameterList& mpiPL = configPL.sublist("MPI", false, "Parallel partitioning information");

    const int nodeCommSize = nodeLocalComm_->getSize();
    const int nodeCommRank = nodeLocalComm_->getRank();

    //const bool rc = PerfUtils::compareComm(localComm, localComm2);

    mpiPL.set("Comm Size", comm_->getSize());
    mpiPL.set("Node Comm Size", nodeCommSize);
  }

  ParameterList& versionPL = configPL.sublist("Version Information", false, "Version Information");
  versionPL.set("Trilinos Major", int(TRILINOS_MAJOR_VERSION));
  versionPL.set("Trilinos Minor", int(TRILINOS_MAJOR_MINOR_VERSION));
  versionPL.set("Trilinos", TRILINOS_VERSION_STRING);

  ParameterList& ptypePL = configPL.sublist("Primitive Types", false, "Primitive Type Information");
  ptypePL.set("Scalar", Teuchos::demangleName(typeid(SC).name()));
  ptypePL.set("LocalOrdinal", Teuchos::demangleName(typeid(LocalOrdinal).name()));
  ptypePL.set("GlobalOrdinal", Teuchos::demangleName(typeid(GlobalOrdinal).name()));
  ptypePL.set("Node", Teuchos::demangleName(typeid(Node).name()));

  ParameterList& sizesPL = configPL.sublist("Primitive Type Sizes", false, "Primitive Type Size Information");
  sizesPL.set("Scalar",       int(sizeof(SC)));
  sizesPL.set("LocalOrdinal", int(sizeof(LO)));
  sizesPL.set("GlobalOrdinal",int(sizeof(GO)));

  ParameterList& typePL = configPL.sublist("Types", false, "Type Information");
  {

    #ifdef XPETRA_MATRIX_SHORT
    typePL.set("Matrix", Teuchos::demangleName(typeid(Matrix).name()), "Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>");
    #endif

    #ifdef XPETRA_CRSMATRIX_SHORT
    typePL.set("CrsMatrix", Teuchos::demangleName(typeid(CrsMatrixWrap).name()), "Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>");
    #endif

    #ifdef XPETRA_VECTOR_SHORT
    typePL.set("Vector", Teuchos::demangleName(typeid(Vector).name()), "Xpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>");
    #endif

    #ifdef XPETRA_MULTIVECTOR_SHORT
    typePL.set("MultiVector", Teuchos::demangleName(typeid(MultiVector).name()), "Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>");
    #endif

    typePL.set("Hierarchy", Teuchos::demangleName(typeid(Hierarchy).name()));

  }

  // track load imbalance
  ParameterList& loadPL = configPL.sublist("Rank Loads", false, "Per Rank Load");
  {
    const GO my_rows = orig_X_->getLocalLength ();
    const GO total_rows = orig_X_->getGlobalLength ();
    std::vector<GO> rank_rows(comm_->getSize());
    rank_rows.resize(comm_->getSize());

    Teuchos::gather<int, GO> (&my_rows, 1, rank_rows.data(), 1, 0, *comm_);

    for (size_t r=0; r < rank_rows.size(); ++r) {
      std::stringstream rank_str;
      rank_str << r;

      loadPL.set(rank_str.str(), rank_rows[r]);
    }
  }

}

template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void
SolverDriverDetails<Scalar,LocalOrdinal,GlobalOrdinal,Node>::gatherAffinityInfo ()
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ArrayRCP;
  using Teuchos::TimeMonitor;
  using Teuchos::Time;
  using Teuchos::FancyOStream;
  using Teuchos::OSTab;
  using Teuchos::ParameterList;
  using std::string;
  using std::endl;

  //return;

  FancyOStream& out = *pOut_;

  out << "========================================================"
      << endl;
  out << "Gathering Affinity and Process Mapping Information"
      << endl;
  OSTab tab (out);

  RCP<Time> tm =  TimeMonitor::getNewTimer("Driver: 2 - Gather Thread Affinity");
  Teuchos::TimeMonitor affinityTimer ( *tm );

  std::ostringstream oss;
  oss << getProblemFileToken() << "_"
      << getNumThreadsFileToken() << "_"
      << getDecompFileToken()    << "_"
      << AFFINITY_MAP_CSV_STR;

  const std::string filename = oss.str ();


  auto localComm2 = PerfUtils::getNodeLocalComm_mpi3(comm_);
  my_cpu_map_    = PerfUtils::gather_affinity_pthread();

  const int nodeCommSize = nodeLocalComm_->getSize();
  const int nodeCommRank = nodeLocalComm_->getRank();

  const bool comms_congruent = PerfUtils::compareComm(nodeLocalComm_, localComm2);
  out << "MPI-3 and MPI-2 implementation of local node communicators are congruent: "
      << (comms_congruent ? "OK" : "ERROR!!!")
      << endl;
  nodeLocalComm_ = localComm2;

  PerfUtils::writeAffinityCSV(filename, comm_, pOut_, nodeLocalComm_);
  out << "Wrote Affinities: " << filename << endl;

  // describe the environment as best we can
  if(comm_->getRank () == 0)
  {
    out << "========================================================"
        << endl
        << "Parallel Environment appears to be:"
        << endl;

    Teuchos::OSTab tab (out);

    out << "MPI Global Comm Size: "
        << comm_->getSize()
        << endl
        << "Number of Physical Nodes: "
        << nodeComm_->getSize ()
        << endl
        << "Processes per Node: "
        << nodeLocalComm_->getSize ()
        << endl
        << "Thread mapping on rank 0: "
        << endl;
    std::stringstream oss;
    PerfUtils::print_affinity (oss, *my_cpu_map_);
    out << oss.str ();

  }

#ifdef KOKKOS_HAVE_CUDA
// first query this process's device
int myDevice = -1;
cudaError_t myDevice_rc = cudaGetDevice (&myDevice);
std::vector<int> localNodeDeviceIDs (nodeLocalComm_->getSize ());
localNodeDeviceIDs.reserve (nodeLocalComm_->getSize ());
localNodeDeviceIDs.resize (nodeLocalComm_->getSize ());

nodeLocalComm_->gather ( int(sizeof(int)), reinterpret_cast<char *> (&myDevice), int(sizeof(int)), reinterpret_cast<char *> (&localNodeDeviceIDs[0]), 0);

if (nodeLocalComm_->getRank () == 0) {
  int nDevices = -1;
  std::ostringstream ss;

  if (comm_->getRank () == 0)
    ss << "Cuda: " << endl;

  cudaGetDeviceCount(&nDevices);
  ss << "  Node: " << PerfUtils::getHostname ()
     << endl
     << "  nDevices: " << nDevices
     << endl
     << "  nDevices == Process per Node : "
     << ( nodeLocalComm_->getSize () == nDevices ? "OK" : "ERROR!")
     << endl;
  
  std::ostringstream oss;
  for(int i=0; i < nDevices; ++i) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    ss  << "    Device [ " << i << " ]: " << prop.name
        << endl;
  }

  std::map<int,int> dupeDevices;
  for(int i=0; i < localNodeDeviceIDs.size (); ++i) {
    dupeDevices[ localNodeDeviceIDs[i] ]++;
  }
  ss << "  Duplicates Detected: " << ( dupeDevices.size () == nodeLocalComm_->getSize () ? "OK" : "ERROR")
     << endl;

  ss << "  Local Rank to Device Mapping :" << endl;
  for(int i=0; i < localNodeDeviceIDs.size (); ++i) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, localNodeDeviceIDs[i]);
    ss  << "    Local Rank[ " << i << " ]: " << "Device[ " << localNodeDeviceIDs[i] << " ]"
        << endl;
  }
  

  const std::string nodeDevStr = ss.str ();
  const int nodeDevStr_sz = nodeDevStr.length () + 1;
  
  for (int r=0; r < nodeComm_->getSize (); ++r) {
    if (r == nodeComm_->getRank () && nodeComm_->getRank () == 0) {
      out << ss.str ();
    } else if (r != nodeComm_->getRank () && nodeComm_->getRank () == 0) {
      int remote_sz = -1;
      Teuchos::receive<int, int>( *nodeComm_, r, int(sizeof(remote_sz)), &remote_sz);

      char * remote_str = new char[remote_sz];
      Teuchos::receive<int, char> ( *nodeComm_, r, remote_sz, remote_str);

      out << remote_str;
    } else if (r == nodeComm_->getRank () && nodeComm_->getRank () != 0) {
      int remote_sz = nodeDevStr_sz;
      Teuchos::send<int, int> ( *nodeComm_, int(sizeof(remote_sz)),  &nodeDevStr_sz, 0);

      Teuchos::send<int,char> (*nodeComm_, remote_sz, nodeDevStr.c_str(), 0);
    }
  }
}
#endif

  comm_->barrier();
}

template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void
SolverDriverDetails<Scalar,LocalOrdinal,GlobalOrdinal,Node>::createLinearSystem(
    const Xpetra::Parameters& xpetraParameters,
    Teuchos::ParameterList& matrixPL)
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ArrayRCP;
  using Teuchos::TimeMonitor;
  using Teuchos::Time;
  using Teuchos::FancyOStream;
  using Teuchos::OSTab;
  using Teuchos::ParameterList;
  using std::string;
  using std::endl;

  FancyOStream& out = *pOut_;

  RCP<Time> tm =  TimeMonitor::getNewTimer("Driver: 1 - Matrix Build");
  Teuchos::TimeMonitor linearSystemCreationTimer ( *tm );

  const bool readMatrixFile = matrixPL.isParameter("matrixFile");
  const bool readRHSFile = matrixPL.isParameter("rhsFile");

  RCP<Matrix>      A ;
  RCP<MultiVector> coordinates;
  RCP<MultiVector> nullspace;
  RCP<MultiVector> X;
  RCP<MultiVector> B;
  RCP<Map>         map;

  out << "========================================================"
      << endl;
  out << "Constructing Linear System"
      << endl;
  OSTab tab (out);

  if (!readMatrixFile) {
    xpetraParameters.describe(out,DESCRIBE_VERB_LEVEL);
    matrixPL.print(out);

    // Galeri will attempt to create a square-as-possible distribution of subdomains di, e.g.,
    //                                 d1  d2  d3
    //                                 d4  d5  d6
    //                                 d7  d8  d9
    //                                 d10 d11 d12
    // A perfect distribution is only possible when the #processors is a perfect square.
    // This *will* result in "strip" distribution if the #processors is a prime number or if the factors are very different in
    // size. For example, np=14 will give a 7-by-2 distribution.
    // If you don't want Galeri to do this, specify mx or my on the galeriList.
    const string matrixType = matrixPL.get<string>("matrixType");
    out << "MatrixName: " << matrixType
        << endl;

    // Create map and coordinates
    // In the future, we hope to be able to first create a Galeri problem, and then request map and coordinates from it
    // At the moment, however, things are fragile as we hope that the Problem uses same map and coordinates inside
    if (matrixType == "Laplace1D") {
      map = Galeri::Xpetra::CreateMap<LO, GO, Node>(xpetraParameters.GetLib(), "Cartesian1D", this->comm_, matrixPL);
      this->orig_coordinates_ = Galeri::Xpetra::Utils::CreateCartesianCoordinates<SC,LO,GO,Map,MultiVector>("1D", map, matrixPL);

    } else if (matrixType == "Laplace2D" || matrixType == "Star2D" ||
               matrixType == "BigStar2D" || matrixType == "Elasticity2D") {
      map = Galeri::Xpetra::CreateMap<LO, GO, Node>(xpetraParameters.GetLib(), "Cartesian2D", this->comm_, matrixPL);
      coordinates = Galeri::Xpetra::Utils::CreateCartesianCoordinates<SC,LO,GO,Map,MultiVector>("2D", map, matrixPL);

    } else if (matrixType == "Laplace3D" || matrixType == "Brick3D" || matrixType == "Elasticity3D") {
      map = Galeri::Xpetra::CreateMap<LO, GO, Node>(xpetraParameters.GetLib(), "Cartesian3D", this->comm_, matrixPL);
      coordinates = Galeri::Xpetra::Utils::CreateCartesianCoordinates<SC,LO,GO,Map,MultiVector>("3D", map, matrixPL);
    }

    // Expand map to do multiple DOF per node for block problems
    if (matrixType == "Elasticity2D")
      map = Xpetra::MapFactory<LO,GO,Node>::Build(map, 2);
    if (matrixType == "Elasticity3D")
      map = Xpetra::MapFactory<LO,GO,Node>::Build(map, 3);

    out << "Processor subdomains in x direction: " << matrixPL.get<GO>("mx") << endl
        << "Processor subdomains in y direction: " << matrixPL.get<GO>("my") << endl
        << "Processor subdomains in z direction: " << matrixPL.get<GO>("mz") << endl
        << "========================================================"
        << endl;

    if (matrixType == "Elasticity2D" || matrixType == "Elasticity3D") {
      // Our default test case for elasticity: all boundaries of a square/cube have Neumann b.c. except left which has Dirichlet
      matrixPL.set("right boundary" , "Neumann");
      matrixPL.set("bottom boundary", "Neumann");
      matrixPL.set("top boundary"   , "Neumann");
      matrixPL.set("front boundary" , "Neumann");
      matrixPL.set("back boundary"  , "Neumann");
    }

    typedef Galeri::Xpetra::Problem<Map,CrsMatrixWrap,MultiVector> galeri_problem_type;

    RCP<galeri_problem_type> Pr = Galeri::Xpetra::BuildProblem<SC,LO,GO,Map,CrsMatrixWrap,MultiVector>(matrixType, map, matrixPL);
    A = Pr->BuildMatrix();

    if (matrixType == "Elasticity2D" ||
        matrixType == "Elasticity3D") {
      nullspace = Pr->BuildNullspace();
      A->SetFixedBlockSize((matrixType == "Elasticity2D") ? 2 : 3);
      out << "Setting Matrix FixedBlockSize to " << A->GetFixedBlockSize()
          << endl;
    }

  } else {
    out << "Matrix reading is not implemented yet."
        << endl;
    exit(-1);
  }

  X = VectorFactory::Build(map);
  B = VectorFactory::Build(map);

  if ( !readRHSFile ) {
    typedef Teuchos::ScalarTraits<SC> STS;
    SC zero = STS::zero(), one = STS::one();

    // we set seed for reproducibility
    Utilities::SetRandomSeed(*(this->comm_));
    X->randomize();
    A->apply(*(X), *(B), Teuchos::NO_TRANS, one, zero);

    Teuchos::Array<typename STS::magnitudeType> norms(1);
    B->norm2(norms);
    B->scale(one/norms[0]);

  } else {
    out << "RHS reading is not implemented yet."
        << endl;
    exit(-1);
  }

  this->comm_->barrier();
  tm = Teuchos::null;

  {
    out << "Describing build structures"
        << endl;
    OSTab tab1 (out);

    A->describe (out, DESCRIBE_VERB_LEVEL);
    X->describe (out, DESCRIBE_VERB_LEVEL);
    map->describe (out, DESCRIBE_VERB_LEVEL);
  }

  out << "Galeri complete."
      << endl
      << "========================================================"
      << endl;

  //out << "calling create_block_partitioning" << std::endl;
 
  //A->getLocalMatrix ().graph.create_block_partitioning(omp_get_max_threads());
  // make the pointers const. The problem should not be changed during execution
  orig_A_           = A;
  orig_coordinates_ = coordinates;
  orig_nullspace_   = nullspace;
  orig_X_           = X;
  orig_B_           = B;
  orig_map_         = map;

//    // update the configPL with descriptions
//    ParameterList& descriptPL = configPL_->sublist("Type Descriptions");
//    {
//      std::stringstream ss;
//      ss << orig_A_;
//      std::string data = ss.str();
//      RCP<ParameterList> matPL = Teuchos::getParametersFromYamlString(data);
//
//      descriptPL.set(matPL->name(), *matPL);
//    }
//    {
//      std::stringstream ss;
//      ss << orig_B_;
//      std::string data = ss.str();
//      RCP<ParameterList> mvPL = Teuchos::getParametersFromYamlString(data);
//      descriptPL.set(mvPL->name(), *mvPL);
//    }
//    {
//      std::stringstream ss;
//      ss << orig_map_;
//      std::string data = ss.str();
//      RCP<ParameterList> mapPL = Teuchos::getParametersFromYamlString(data);
//      descriptPL.set(mapPL->name(), *mapPL);
//    }

}

template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void
SolverDriverDetails<Scalar,LocalOrdinal,GlobalOrdinal,Node>::setFileTokens ()
{
  setProblemFileToken();
  setDecompFileToken();
  setNumThreadsFileToken();
}


template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
GlobalOrdinal
SolverDriverDetails<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getGaleriProblemDim (const GlobalOrdinal n)
{
  typedef typename std::make_unsigned<GO>::type GO_unsigned;
  // check if n is the unsigned max, or if n = -1, both mean this value was not used
  if (n == std::numeric_limits<GO_unsigned>::max() || n == -1) {
    return (1);
  } else {
    return (n);
  }
}

template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
std::string
SolverDriverDetails<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getProblemFileToken () {
  return (problemFileToken_);
}

template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void
SolverDriverDetails<Scalar,LocalOrdinal,GlobalOrdinal,Node>::setProblemFileToken ()
{
  using std::string;
  std::stringstream ss;

  // galeri assumes these are signed... e.g., -1 is assigned!
  const GO nx = galeriPL_->get<GO>("nx");
  const GO ny = galeriPL_->get<GO>("ny");
  const GO nz = galeriPL_->get<GO>("nz");

  // Laplace3D-bs-1-XxYxZ
  //
  ss << galeriPL_->get<string>("matrixType")
     << "-BS-" << orig_A_->GetFixedBlockSize ()
     << "-"
     << getGaleriProblemDim(nx) << "x"
     << getGaleriProblemDim(ny) << "x"
     << getGaleriProblemDim(nz);

  problemFileToken_ = ss.str();
}

template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
std::string
SolverDriverDetails<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getDecompFileToken () {
  return (decompFileToken_);
}

// num_nodes x procs_per_node x cores_per_proc x thread_per_core
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void
SolverDriverDetails<Scalar,LocalOrdinal,GlobalOrdinal,Node>::setDecompFileToken () {
  std::stringstream ss;

  ss << "decomp-"
     << nodeComm_->getSize ()
     << "x"
     << nodeLocalComm_->getSize ()
     << "x"
     << cores_per_proc_
     << "x"
     << threads_per_core_;

  decompFileToken_ = ss.str();
}


template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
std::string
SolverDriverDetails<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getNumThreadsFileToken () {
  return (numThreadsFileToken_);
}

template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void
SolverDriverDetails<Scalar,LocalOrdinal,GlobalOrdinal,Node>::setNumThreadsFileToken () {
  std::stringstream ss;

  const std::string node_name = Node::name();

  if (node_name == "OpenMP/Wrapper") {
    ss << "OpenMP-threads-";

      #ifdef HAVE_MUELU_OPENMP
      ss << omp_get_max_threads ();
      #else
      ss << "asdasd";
      #endif
      *pOut_ << ss.str ();
  }
  else if (node_name == "Serial/Wrapper") {
    ss << "Serial";
  }
  else if (node_name == "Cuda/Wrapper") {
    ss << "Cuda-";

    #ifdef KOKKOS_HAVE_CUDA
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::string name = prop.name;
    std::replace( name.begin(), name.end(), ' ', '-'); // replace all 'x' to 'y'
    ss << name;
    #endif
  }
  else {
    *pOut_ << "node_name = " << node_name << ", is unknown" << std::endl;
  }

  ss << "_np-" << comm_->getSize ();


  numThreadsFileToken_ = ss.str();
}

template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
std::string
SolverDriverDetails<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getTimeStepFileToken (const int numsteps) {
  std::stringstream ss;
  ss << "numsteps-" << numsteps;
  return (ss.str());
}



template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void
SolverDriverDetails<Scalar,LocalOrdinal,GlobalOrdinal,Node>::performLinearAlgebraExperiment (Teuchos::ParameterList& runParamList, const int runID)
{


  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::rcp_const_cast;
  using Teuchos::ArrayRCP;
  using Teuchos::TimeMonitor;
  using Teuchos::Time;
  using Teuchos::FancyOStream;
  using Teuchos::OSTab;
  using Teuchos::ParameterList;
  using Teuchos::parameterList;
  using std::string;
  using std::endl;
  using ss_t = std::stringstream;

  typedef MultiVector MV;
  typedef Belos::MultiVecTraits<SC,MV> MVT;
  typedef Belos::OperatorTraits<SC,MV,Matrix> OPT;

  FancyOStream& out = *pOut_;


  // determine solver/preconditioner configurations
  auto defaultDriverPL = getDefaultLinearAlgebraExperimentParameters ();

  // apply defaults
  runParamList.setParametersNotAlreadySet(*defaultDriverPL);

  const int    pseudoTimesteps   = runParamList.get<int>(PL_KEY_TIMESTEP);
  const bool   do_deep_copies    = runParamList.get<bool>(PL_KEY_TIMESTEP_DEEPCOPY);
  const bool construction_only   = runParamList.get<bool>(PL_KEY_CONSTRUCTOR_ONLY);
  const string experimentType    = runParamList.get<string>(PL_KEY_EXPERIMENT_TYPE);

  // Timer IO file tokens
  const string problemFileToken   = getProblemFileToken ();
  const string numStepsFileToken  = getTimeStepFileToken (pseudoTimesteps);
  const string execSpaceFileToken = getNumThreadsFileToken ();
  const string decompFileToken    = getDecompFileToken ();

  std::ostringstream oss;
  oss << problemFileToken   << "_"
      << "LinearAlgebra-" << Xpetra::toString(xpetraParams_->GetLib()) << "_"
      << numStepsFileToken  << "_"
      << execSpaceFileToken << "_"
      << decompFileToken    << ".yaml";

  const std::string fileName = oss.str ();


  out << std::string(80,'-')
      << endl
      << "------  " << fileName
      << endl
      << std::string(80,'-')
      << endl;



  typedef Teuchos::ScalarTraits<SC> STS;
  const SC zero = STS::zero();
  const SC one  = STS::one();
  // define counters for the main phases
  RCP<Time> globalTime_       = TimeMonitor::getNewTimer(TM_LABEL_GLOBAL);
  RCP<Time> copyTime_         = TimeMonitor::getNewTimer(TM_LABEL_COPY);
  RCP<Time> applyTime_        = TimeMonitor::getNewTimer("OPT::Apply");
  RCP<Time> norm2Time_        = TimeMonitor::getNewTimer("MVT::Norm2");
  RCP<Time> scaleTime_        = TimeMonitor::getNewTimer("MVT::MvScale");
  std::vector<int> block_sizes;
  RCP<MultiVector> Q= Teuchos::null;
  constexpr int num_Q = 100+1;

  std::vector<SC> lclSum;
  std::vector<SC> gblSum;

  for (size_t i=0; i < num_Q; ++i) {
    lclSum.push_back(SC(i));
    gblSum.push_back(zero);
  }

  Q   = MultiVectorFactory::Build(orig_X_->getMap(), num_Q, false);

  block_sizes.push_back(int(1));
  block_sizes.push_back(int(2));
  block_sizes.push_back(int(3));
  block_sizes.push_back(int(4));
  block_sizes.push_back(int(5));
  block_sizes.push_back(int(6));
  block_sizes.push_back(int(7));
  block_sizes.push_back(int(8));
  block_sizes.push_back(int(9));
  block_sizes.push_back(int(10));
  block_sizes.push_back(int(15));
  block_sizes.push_back(int(20));
  block_sizes.push_back(int(25));
  block_sizes.push_back(int(30));
  block_sizes.push_back(int(35));
  block_sizes.push_back(int(40));
  block_sizes.push_back(int(45));
  block_sizes.push_back(int(50));
  block_sizes.push_back(int(60));
  block_sizes.push_back(int(70));
  block_sizes.push_back(int(80));
  block_sizes.push_back(int(90));
  block_sizes.push_back(int(100));

  // Timestep loop around here
  // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // Run starts here.
  // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  for (int i=0; i < pseudoTimesteps; ++i)
  {

    // We keep two copies of the matrix.
    // The original, and the one Muelu can repartition
    // This allows measuring repartitioning costs, without
    // paying for matrix assembly again.
    RCP<Matrix> A = Teuchos::null;
    RCP<MultiVector> coordinates = Teuchos::null;
    RCP<MultiVector> nullspace= Teuchos::null;
    RCP<MultiVector> X= Teuchos::rcp_const_cast<MultiVector> (orig_X_);
    RCP<const MultiVector> B = orig_B_;
    RCP<MultiVector> R0 = Teuchos::null;
    RCP<const Map>   map= Teuchos::null;
    RCP<Map>        mapT= Teuchos::null;

    OSTab tab (out);

    this->comm_->barrier();
    // start the clock
    Teuchos::TimeMonitor glb_tm (*globalTime_);

    // ===========================================================================
    // reset the matrix, initial guess and RHS.
    // ===========================================================================
    {
      out << "Deep Copying Vectors and Matrices"
          << endl;

      // start the clock
      Teuchos::TimeMonitor tm (*copyTime_);

      Q->putScalar(one);
      X->putScalar(zero);
//      X   = MultiVectorFactory::Build(orig_X_->getMap(), orig_X_->getNumVectors(), true);
//      B   = MultiVectorFactory::Build(orig_B_->getMap(), orig_B_->getNumVectors(), false);
//      *B  = *orig_B_;
    }
    // barrier after the timer scope, this allows the timers to track variations
    comm_->barrier();

    // apply
    {
      // start the clock
      Teuchos::TimeMonitor tm (*applyTime_);

      OPT::Apply(*orig_A_, *B, *X, Belos::NOTRANS)
      //orig_A_->apply(*(B), *(X), Teuchos::NO_TRANS, one, zero);
    }
    // barrier after the timer scope, this allows the timers to track variations
    comm_->barrier();

    {
      // start the clock
      Teuchos::TimeMonitor tm (*norm2Time_);
      std::vector<typename STS::magnitudeType> norms (1);

      //Teuchos::Array<typename STS::magnitudeType> norms(1);
      MVT::MvNorm(*X, norms, Belos::TwoNorm);
      //X->norm2(norms);
    }
    // barrier after the timer scope, this allows the timers to track variations
    comm_->barrier();

    {
      // start the clock
      Teuchos::TimeMonitor tm (*scaleTime_);
      MVT::MvScale(*X, SC(1.00253));
    }
    // barrier after the timer scope, this allows the timers to track variations
    comm_->barrier();

    {
      using ss_t = std::stringstream;
      ss_t timerLabel;

      // mimic classical gram schmidt ortho passes
      std::vector<int> indices (1);


      for (const auto& num_vectors : block_sizes) {
      //for (int i=1; i < (num_Q-1); ++i) {
        indices.resize(num_vectors);
        for(int i=0; i < num_vectors; ++i) {
          indices[i] = i;
        }


        Teuchos::RCP<const MV> Q_prev;
        Teuchos::RCP<MV> Q_prev_nonconst;
        {
          timerLabel.str("");
          timerLabel << "MVT::CloneView::" << num_vectors;
          RCP<Time> the_timer   = TimeMonitor::getNewTimer(timerLabel.str());

          // start the clock
          Teuchos::TimeMonitor tm (*the_timer);

          // view vectors 0,..,num_vectors-1
          Q_prev = MVT::CloneView( *Q, indices);
        }
        comm_->barrier();


        Teuchos::RCP<MV> Q_new;
        {
          std::vector<int> view_indices (1);
          view_indices[0] = num_vectors;

          timerLabel.str("");
          timerLabel << "MVT::CloneViewNonConst::" << num_vectors+1;
          RCP<Time> the_timer   = TimeMonitor::getNewTimer(timerLabel.str());

          // start the clock
          Teuchos::TimeMonitor tm (*the_timer);

          // view the i+1 vector
          Q_new = MVT::CloneViewNonConst( *Q, view_indices );
          // view the 0,...,i vectors in non-const
          Q_prev_nonconst = MVT::CloneViewNonConst( *Q, indices );
        }
        comm_->barrier();

        {
          timerLabel.str("");
          timerLabel << "MVT::MvInit::" << num_vectors;
          RCP<Time> the_timer   = TimeMonitor::getNewTimer(timerLabel.str());

          // start the clock
          Teuchos::TimeMonitor tm (*the_timer);

          MVT::MvInit (*Q_prev_nonconst, one);
        }
        comm_->barrier();


        Teuchos::RCP< Teuchos::SerialDenseMatrix<int,SC> > Z;
        {
          timerLabel.str("");
          timerLabel << "MVT::InnerProduct::" << num_vectors;

          RCP<Time> the_timer   = TimeMonitor::getNewTimer(timerLabel.str());


          Z = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,SC> (num_vectors, 1) );

          // start the clock
          Teuchos::TimeMonitor tm (*the_timer);

          MVT::MvTransMv(STS::one(), *Q_prev, *Q_new, *Z);
        }
        comm_->barrier();

        {
          timerLabel.str("");
          timerLabel << "MVT::Update::" << num_vectors;

          RCP<Time> the_timer   = TimeMonitor::getNewTimer(timerLabel.str());


          // start the clock
          Teuchos::TimeMonitor tm (*the_timer);

          MVT::MvTimesMatAddMv( -STS::one(), *Q_prev, *Z, STS::one(), *Q_new );
        }
        comm_->barrier();

        {
          timerLabel.str("");
          timerLabel << "Teuchos::reduceAll::" << num_vectors;
          RCP<Time> the_timer   = TimeMonitor::getNewTimer(timerLabel.str());

          // start the clock
          Teuchos::TimeMonitor tm (*the_timer);

          Teuchos::reduceAll<int, SC> (*comm_, Teuchos::REDUCE_SUM, num_vectors, lclSum.data(), gblSum.data());
        }
        comm_->barrier();

      }
    }
    // barrier after the timer scope, this allows the timers to track variations
    comm_->barrier();

  }
  writeTimersForFunAndProfit (fileName);
}


template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void
SolverDriverDetails<Scalar,LocalOrdinal,GlobalOrdinal,Node>::performSolverExperiment (Teuchos::ParameterList& runParamList, const int runID)
{


  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::rcp_const_cast;
  using Teuchos::ArrayRCP;
  using Teuchos::TimeMonitor;
  using Teuchos::Time;
  using Teuchos::FancyOStream;
  using Teuchos::OSTab;
  using Teuchos::ParameterList;
  using Teuchos::parameterList;
  using std::string;
  using std::endl;
  using ss_t = std::stringstream;

  FancyOStream& out = *pOut_;


  // determine solver/preconditioner configurations
  auto defaultDriverPL = getDefaultSolverExperimentParameters ();

  // apply defaults
  runParamList.setParametersNotAlreadySet(*defaultDriverPL);

  RCP<ParameterList> solverPL_ = getSolverParameters(runParamList);
  RCP<ParameterList> precPL_   = getPreconditionerParameters(runParamList);

  const string precName          = runParamList.get<string>(PL_KEY_PREC_NAME);
  const string precFactoryName   = runParamList.get<string>(PL_KEY_PREC_FACTORY);
  const string solverName        = runParamList.get<string>(PL_KEY_SOLVER_NAME);
  const string solverFactoryName = runParamList.get<string>(PL_KEY_SOLVER_FACTORY);
  const int    pseudoTimesteps   = runParamList.get<int>(PL_KEY_TIMESTEP);
  const bool   copy_R0           = runParamList.get<bool>(PL_KEY_COPY_R0);
  const bool   do_deep_copies    = runParamList.get<bool>(PL_KEY_TIMESTEP_DEEPCOPY);
  const bool construction_only   = runParamList.get<bool>(PL_KEY_CONSTRUCTOR_ONLY);
  const string experimentType    = runParamList.get<string>(PL_KEY_EXPERIMENT_TYPE);

  // Timer IO file tokens
  const string solverFileToken   = runParamList.get<string>("solverFileToken");
  const string precFileToken     = runParamList.get<string>("precFileToken");
  const string problemFileToken   = getProblemFileToken ();
  const string numStepsFileToken  = getTimeStepFileToken (pseudoTimesteps);
  const string execSpaceFileToken = getNumThreadsFileToken ();
  const string decompFileToken    = getDecompFileToken ();

  std::ostringstream oss;
  oss << problemFileToken   << "_"
      << solverFileToken    << "_"
      << precFileToken      << "_"
      << numStepsFileToken  << "_"
      << execSpaceFileToken << "_"
      << decompFileToken    << ".yaml";

  const std::string fileName = oss.str ();

  const bool havePrec   = (precName != "None");
  const bool haveSolver = (solverName != "None");
  std::string solver_smart_label;


  out << std::string(80,'-')
      << endl
      << "------  " << fileName
      << endl
      << std::string(80,'-')
      << endl;

  // massage the solver label
  if (solverFactoryName == "Belos" && useSmartSolverLabels_) {
    std::stringstream ss;
    ss << "Belos:" << solverName;
    if (solverPL_->isParameter("Num Blocks")) {
      ss << "[" << solverPL_->get<int>("Num Blocks") << "]";
    }
    solver_smart_label = ss.str ();
    solverPL_->set(PL_KEY_BELOS_TIMER_LABEL, solver_smart_label);
  }

  RCP<const ParameterList> solverPL = solverPL_;
  RCP<const ParameterList> precPL   = precPL_;

  RCP<BelosSolverManager> solver;
  RCP<BelosLinearProblem> belosProblem;
  RCP<BelosOperatorT> belosPrec;
  RCP<BelosOperatorT> belosOp;
  RCP<Hierarchy>  H;


  // define counters for the main phases
  RCP<Time> globalTime_       = TimeMonitor::getNewTimer(TM_LABEL_GLOBAL);
  RCP<Time> copyTime_         = TimeMonitor::getNewTimer(TM_LABEL_COPY);
  RCP<Time> nullspaceTime_    = TimeMonitor::getNewTimer(TM_LABEL_NULLSPACE);
  RCP<Time> precSetupTime_    = TimeMonitor::getNewTimer(TM_LABEL_PREC_SETUP);
  RCP<Time> solverSetupTime_  = TimeMonitor::getNewTimer(TM_LABEL_SOLVER_SETUP);
  RCP<Time> solveTime_        = TimeMonitor::getNewTimer(TM_LABEL_SOLVE);

  // Timestep loop around here
  // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // Run starts here.
  // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  for (int i=0; i < pseudoTimesteps; ++i)
  {

    // We keep two copies of the matrix.
    // The original, and the one Muelu can repartition
    // This allows measuring repartitioning costs, without
    // paying for matrix assembly again.
    RCP<Matrix> A = Teuchos::null;
    RCP<MultiVector> coordinates = Teuchos::null;
    RCP<MultiVector> nullspace= Teuchos::null;
    RCP<MultiVector> X= Teuchos::null;
    RCP<MultiVector> B= Teuchos::null;
    RCP<MultiVector> R0 = Teuchos::null;
    RCP<const Map>   map= Teuchos::null;
    RCP<Map>        mapT= Teuchos::null;

    // destroy the objects created last pass, if needed
    solver        = Teuchos::null;
    belosProblem  = Teuchos::null;
    belosPrec     = Teuchos::null;
    belosOp       = Teuchos::null;
    H             = Teuchos::null;

    OSTab tab (out);

    this->comm_->barrier();
    // start the clock
    Teuchos::TimeMonitor glb_tm (*globalTime_);

    // ===========================================================================
    // reset the matrix, initial guess and RHS.
    // ===========================================================================
    {
      out << "Deep Copying Vectors and Matrices"
          << endl;

      // start the clock
      Teuchos::TimeMonitor tm (*copyTime_);

      RCP<Node> dupeNode = rcp (new Node());

      // copying does not set the fixed block size
      A = MatrixFactory2::BuildCopy(orig_A_);

      if (orig_A_->GetFixedBlockSize() > 1)
        A->SetFixedBlockSize( orig_A_->GetFixedBlockSize() );

      if (! orig_coordinates_.is_null() ) {
        coordinates  = MultiVectorFactory::Build(orig_coordinates_->getMap(), orig_coordinates_->getNumVectors(), false);
        *coordinates = *orig_coordinates_;
      }

      if (! orig_nullspace_.is_null() ) {
        nullspace  = MultiVectorFactory::Build(orig_nullspace_->getMap(), orig_nullspace_->getNumVectors(), false);
        *nullspace = *orig_nullspace_;
      }

      X   = MultiVectorFactory::Build(orig_X_->getMap(), orig_X_->getNumVectors(), true);
      B   = MultiVectorFactory::Build(orig_B_->getMap(), orig_B_->getNumVectors(), false);
      *B  = *orig_B_;

      if (copy_R0) {
        R0  = MultiVectorFactory::Build(orig_B_->getMap(), orig_B_->getNumVectors(), false);
        *R0 = *orig_B_;
      }

      mapT= Xpetra::clone(*orig_map_,dupeNode);
      map = mapT;;
/*

      A = Teuchos::rcp_const_cast<Matrix> (orig_A_);
      B = Teuchos::rcp_const_cast<MultiVector> (orig_B_);
      X = Teuchos::rcp_const_cast<MultiVector> (orig_X_);
      X->putScalar(0.0);
      nullspace = Teuchos::rcp_const_cast<MultiVector> (orig_nullspace_);
      coordinates = Teuchos::rcp_const_cast<MultiVector> (orig_coordinates_);
      map = orig_map_;
*/
    }
    // barrier after the timer scope, this allows the timers to track variations
    comm_->barrier();


    // ===========================================================================
    // adjust/compute the nullspace
    // ===========================================================================
    {

      int blkSize = A->GetFixedBlockSize ();
      if ((havePrec && precName == "MueLu") || (solverName == "MueLu")) {
        out << "Checking MueLu BlockSize and Matrix BlockSize"
            << endl;

        // this is a bit convoluted. Need to think carefully about the logic
        RCP<const ParameterList> pl = (solverName == "MueLu") ? solverPL : precPL;
        int tmpBlockSize = -1;

        if (pl->isSublist("Matrix")) {
          // Factory style parameter list
          const Teuchos::ParameterList& operatorList = pl->sublist("Matrix");
          if (operatorList.isParameter("PDE equations"))
            tmpBlockSize = operatorList.get<int>("PDE equations");

          if (tmpBlockSize != blkSize) {
            Teuchos::ParameterList& tmpPL = rcp_const_cast<ParameterList>(pl)->sublist("Matrix", true);
            out << "Setting 'PDE equations' to " << blkSize
                << ", because it was not set and the matrix has blockSize=" << blkSize << endl;
            tmpPL.set("PDE equations", blkSize);
          }

        } else if (pl->isParameter("number of equations")) {
          // Easy style parameter list
          tmpBlockSize = pl->get<int>("number of equations");

          if (tmpBlockSize != blkSize) {
            out << "Setting 'number of equations' to " << blkSize
                << ", because it was not set and the matrix has blockSize=" << blkSize << endl;
            rcp_const_cast<ParameterList>(pl)->set("number of equations", blkSize);
          }

        } else {
          out << "Setting 'number of equations' to " << blkSize
              << ", because it was not set and the matrix has blockSize=" << blkSize << endl;
          rcp_const_cast<ParameterList>(pl)->set("number of equations", blkSize);
        }
      }

      // adjust/compute the nullspace.
      if (nullspace.is_null() && ((havePrec && precName == "MueLu") || (solverName == "MueLu"))) {
        out << "Adjusting Nullspace for BlockSize"
            << endl;

        // start the clock
        Teuchos::TimeMonitor tm (*nullspaceTime_);

        nullspace = MultiVectorFactory::Build(map, blkSize);
        for (int i = 0; i < blkSize; i++) {
          RCP<const Map> domainMap = A->getDomainMap();
          GO             indexBase = domainMap->getIndexBase();

          ArrayRCP<SC> nsData = nullspace->getDataNonConst(i);
          for (int j = 0; j < nsData.size(); j++) {
            GO GID = domainMap->getGlobalElement(j) - indexBase;

            if ((GID-i) % blkSize == 0)
              nsData[j] = Teuchos::ScalarTraits<SC>::one();
          }
        }
      }
    }
    // barrier after the timer scope, this allows the timers to track variations
    comm_->barrier();

    // ===========================================================================
    // Preconditioner construction
    // ===========================================================================
    {
      if ((havePrec && precName == "MueLu") || solverName == "MueLu")
      {
        out << "Constructing the Preconditioner MueLu"
            << endl;

        // this is a bit convoluted. Need to think carefully about the logic
        RCP<const ParameterList> pl = (solverName == "MueLu") ? solverPL : precPL;

        // start the clock
        Teuchos::TimeMonitor tm (*precSetupTime_);

        A->SetMaxEigenvalueEstimate(-STS::one());

        H = MueLu::CreateXpetraPreconditioner(A, *pl, coordinates);
      }
    }
    // barrier after the timer scope, this allows the timers to track variations
    comm_->barrier();

    // =========================================================================
    // Solver construction
    // =========================================================================;
    if (haveSolver)
    {
      out << "Constructing the Solver"
          << endl;

      // start the clock
      Teuchos::TimeMonitor tm (*solverSetupTime_);
      if (solverFactoryName == "MueLu") {
        // nothing to do?
        //
      } else if (solverFactoryName == "Belos") {

        // Turns a Xpetra::Matrix object into a Belos operator
        belosOp   = Teuchos::rcp(new BelosXpetraOp (A));
        // Construct a Belos LinearProblem object
        belosProblem = rcp(new BelosLinearProblem (belosOp, X, B));

        if (havePrec) {
          // Turns a MueLu::Hierarchy object into a Belos operator
          H->IsPreconditioner(true);
          belosPrec = Teuchos::rcp(new BelosMueLuOp (H));

          belosProblem->setRightPrec(belosPrec);
        }

        // the initial guess is zero...
        if (copy_R0)
          belosProblem->setInitResVec(R0);

        if (useSmartSolverLabels_)
          belosProblem->setLabel (solver_smart_label);

        bool set = belosProblem->setProblem();
        if (set == false) {
          out << "ERROR:  Belos::LinearProblem failed to set up correctly!" << std::endl;

          TEUCHOS_TEST_FOR_EXCEPTION(set == false, std::runtime_error,
              "ERROR:  Belos::LinearProblem failed to set up correctly!");
        }

        // Create an iterative solver manager
        BelosSolverFactory factory;
        solver = factory.create(solverName, rcp_const_cast<ParameterList>(solverPL));
        solver->setProblem (belosProblem);
      }
    }
    // barrier after the timer scope, this allows the timers to track variations
    comm_->barrier();


    // =========================================================================
    // Solver execution
    // =========================================================================
    if (! construction_only)
    {
      out << "Solving the linear system" << endl;

      if (solverFactoryName == "MueLu") {
        // TODO Fix this. Muelu needs to accept this in a parameter list or something
        std::pair<LO,SC> mueluPair(LO(200), SC(1.e-15));
        // start the clock
        Teuchos::TimeMonitor tm (*solveTime_);

        H->IsPreconditioner(false);
        H->Iterate(*B, *X, mueluPair);

      } else if (solverFactoryName == "Belos") {

        // Perform solve
        Belos::ReturnType ret = Belos::Unconverged;
        {
          // start the clock
          Teuchos::TimeMonitor tm (*solveTime_);
          // solve the linear system
          ret = solver->solve();
        }

        // Get the number of iterations for this solve.
        out << "Number of iterations performed for this solve: " << solver->getNumIters() << std::endl;
        // Check convergence
        if (checkConvergence_) {
          if (ret != Belos::Converged)
            out << std::endl << "ERROR:  Belos did not converge! " << std::endl;
          else
            out << std::endl << "SUCCESS:  Belos converged!" << std::endl;
        }

      } else {
        throw MueLu::Exceptions::RuntimeError("Unknown solver Factory: \"" + solverFactoryName + "\"");
      }
    }
    // barrier after the timer scope, this allows the timers to track variations
    comm_->barrier();
  }// timestep loop

  // report the solver's effective parameters
  if (haveSolver)
  {
    out << "Effective Solver parameters from final solve: "
        << endl;
    OSTab tab (out);
    if (solverFactoryName == "Belos") {
      // solver
      if (! solver.is_null())
        solver->getCurrentParameters ()->print(out);
    }
    else {
      out << "Solvers from: " << solverFactoryName << ", are not yet supported for description," << endl;
    }
  }
  // describe the solver
  if (haveSolver)
  {
    out << "Solver description from solver in use for last solve "
        << endl;
    OSTab tab (out);
    if (solverFactoryName == "Belos") {
      // solver
      if (! solver.is_null())
        solver->describe(out, Teuchos::VERB_EXTREME);
    }
    else if (solverFactoryName == "MueLu") {
      if (! H.is_null())
        H->describe(out, Teuchos::VERB_EXTREME);
    }
    else {
      out << "Solvers from: " << solverFactoryName << ", are not yet supported for description," << endl;
    }
  }
  // describe the prec
  if (havePrec)
  {
    out << "Preconditioner description from solver in use for last solve "
        << endl;
    OSTab tab (out);

    if (precFactoryName == "MueLu") {
      if (! H.is_null())
        H->describe(out, Teuchos::VERB_EXTREME);
    }
    else {
      out << "NOT SUPPORTED YET!" << endl;
    }
  }

  writeTimersForFunAndProfit (fileName);
}

template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void
SolverDriverDetails<Scalar,LocalOrdinal,GlobalOrdinal,Node>::performRun(Teuchos::ParameterList& runParamList, const int runID)
{

  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::rcp_const_cast;
  using Teuchos::ArrayRCP;
  using Teuchos::TimeMonitor;
  using Teuchos::Time;
  using Teuchos::FancyOStream;
  using Teuchos::OSTab;
  using Teuchos::ParameterList;
  using Teuchos::parameterList;
  using std::string;
  using std::endl;
  using ss_t = std::stringstream;

  Teuchos::TimeMonitor::clearCounters();

  FancyOStream& out = *pOut_;

  out << "Preparing for run #" << runID
      << endl;

  #ifdef HAVE_MUELU_OPENMP
  {
    std::string node_name = Node::name();
    if(!comm_->getRank() && !node_name.compare("OpenMP/Wrapper"))
      out << "OpenMP Max Threads = "
          << omp_get_max_threads()
          << std::endl;
  }
  #endif


  // determine what we are doing
  const string experimentType    = runParamList.isParameter(PL_KEY_EXPERIMENT_TYPE)
                                  ? runParamList.get<string>(PL_KEY_EXPERIMENT_TYPE)
                                  : PL_DEFAULT_EXPERIMENT_TYPE;

  if (experimentType == EXPERIMENT_TYPE_SOLVER) {
    performSolverExperiment(runParamList, runID);
  } else if (experimentType == EXPERIMENT_TYPE_LINEAR_ALGEBRA) {
    performLinearAlgebraExperiment(runParamList, runID);
  } else {

  }

}


template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void
SolverDriverDetails<Scalar,LocalOrdinal,GlobalOrdinal,Node>::writeTimersForFunAndProfit (const std::string& filename)
{
  using Teuchos::TimeMonitor;
  using Teuchos::FancyOStream;

  FancyOStream& out = *pOut_;

  const std::string filter = "";

  std::ios_base::fmtflags ff(out.flags());

  // screen output
  out << std::fixed;
  timerReportParams_->set("Report format", "Table");
  TimeMonitor::report(comm_.ptr(), out, filter, timerReportParams_);

  std::ostream* os = pOut_.get ();
  std::ofstream fptr;
  // only one worker writes a file
  if (comm_->getRank() == 0)
  {

    fptr.open(filename, std::ofstream::out);
    os = &fptr;
  }

  // "Table" or "YAML"
  out << std::scientific;
  timerReportParams_->set("Report format", "YAML");
  TimeMonitor::report(comm_.ptr(), *os, filter, timerReportParams_);

  out << std::setiosflags(ff);
  // close the file
  if (comm_->getRank() == 0)
  {
    fptr.close ();
  }
}

template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<const Teuchos::ParameterList>
SolverDriverDetails<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getDefaultLinearAlgebraExperimentParameters () const
{
  using Teuchos::RCP;
  using Teuchos::parameterList;
  using Teuchos::ParameterList;

  RCP<ParameterList> defaults = parameterList();

  // ExperimentType
  defaults->set(PL_KEY_EXPERIMENT_TYPE, PL_DEFAULT_EXPERIMENT_TYPE);

  // Fake timestepping e.g., repeat the whole experiment in a loop
  defaults->set(PL_KEY_TIMESTEP, PL_DEFAULT_TIMESTEP);

  // whether to compute or construct only
  defaults->set(PL_KEY_CONSTRUCTOR_ONLY, PL_DEFAULT_CONSTRUCTOR_ONLY);

  // whether the experiment should deep copy between timesteps
  defaults->set(PL_KEY_TIMESTEP_DEEPCOPY, PL_DEFAULT_TIMESTEP_DEEPCOPY);


  return (defaults);
}


template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<const Teuchos::ParameterList>
SolverDriverDetails<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getDefaultSolverExperimentParameters () const
{
  using Teuchos::RCP;
  using Teuchos::parameterList;
  using Teuchos::ParameterList;

  RCP<ParameterList> defaults = parameterList();

  // ExperimentType
  defaults->set(PL_KEY_EXPERIMENT_TYPE, PL_DEFAULT_EXPERIMENT_TYPE);

  // SolverName
  defaults->set(PL_KEY_SOLVER_NAME, PL_DEFAULT_SOLVER_NAME);

  // Solver Factory
  defaults->set(PL_KEY_SOLVER_FACTORY, PL_DEFAULT_SOLVER_FACTORY);

  // Preconditioner (None)
  defaults->set(PL_KEY_PREC_NAME, PL_DEFAULT_PREC_NAME);
  // Prec Factory = ""
  defaults->set(PL_KEY_PREC_FACTORY, PL_DEFAULT_PREC_FACTORY);

  // Fake timestepping e.g., repeat the whole experiment in a loop
  defaults->set(PL_KEY_TIMESTEP, PL_DEFAULT_TIMESTEP);

  defaults->set(PL_KEY_COPY_R0, PL_DEFAULT_COPY_R0);

  // whether to compute or construct only
  defaults->set(PL_KEY_CONSTRUCTOR_ONLY, PL_DEFAULT_CONSTRUCTOR_ONLY);

  // whether the experiment should deep copy between timesteps
  defaults->set(PL_KEY_TIMESTEP_DEEPCOPY, PL_DEFAULT_TIMESTEP_DEEPCOPY);


  return (defaults);
}
/*
 * Expects to see:
 *    Preconditioner        : Name of a preconditioner
 *    PreconditionerFactory : MueLu currently is the only one supported, this means we will pass this parameter list directory to MueLu
 *    PreconditionerParams  : A sublist name or XML file to load.
 *
 *    runPL is also expected to contain the defaults
 */

template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<Teuchos::ParameterList>
SolverDriverDetails<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getPreconditionerParameters(Teuchos::ParameterList& runPL)
{
  using std::string;
  using Teuchos::RCP;
  using Teuchos::ParameterList;
  using Teuchos::parameterList;

  std::string precFileToken;
  RCP<ParameterList> precPL = Teuchos::null;

  // check for the preconditioner name, it is required if you expect preconditioning
  const string precName = runPL.get<string>(PL_KEY_PREC_NAME);

  precFileToken = precName;

  runPL.set("precFileToken", precFileToken);

  if (precName == "None")
    return (precPL);

  // check for a parameter token
  if (! runPL.isParameter(PL_KEY_PREC_CONFIG))
    return (precPL);

  const string precConfig = runPL.get<string>(PL_KEY_PREC_CONFIG);

  // if XML, then read and broadcast
  if (isXML(precConfig))
  {
    *pOut_ << "Loading Preconditioner configuration: " << precConfig << std::endl;

    precPL = parameterList();

    Teuchos::updateParametersFromXmlFileAndBroadcast(precConfig, precPL.ptr(), *(this->comm_));

    precFileToken = getBasename(precConfig, ".xml");
  }
  else if( runPL.isSublist(precConfig) ) {

    *pOut_ << "Loading Preconditioner configuration from sublist: " << precConfig << std::endl;

    const ParameterList& t = runPL.sublist(precConfig);
    precPL = parameterList (t);


    precFileToken = getBasename(precConfig);

  }// check in the master PL
  else if( driverPL_->isSublist(precConfig) ) {

    *pOut_ << "Loading Preconditioner configuration from sublist: " << precConfig << std::endl;

    const ParameterList& t = driverPL_->sublist(precConfig);
    precPL = parameterList (t);

    precFileToken = getBasename(precConfig);

  }
  else {

    *pOut_ << "FAILED: Loading Preconditioner configuration: " << precConfig << ", is not a XML or a sublist." << std::endl;
  }

  runPL.set("precFileToken", precFileToken);

  return (precPL);
}


template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<Teuchos::ParameterList>
SolverDriverDetails<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getSolverParameters(Teuchos::ParameterList& runPL)
{
  using std::string;
  using Teuchos::RCP;
  using Teuchos::ParameterList;
  using Teuchos::parameterList;

  std::string solverFileToken;
  RCP<ParameterList> solverPL = Teuchos::null;

  // check for the preconditioner name, it is required if you expect preconditioning
  const string solverName = runPL.get<string>(PL_KEY_SOLVER_NAME);

  solverFileToken = solverName;

  // check for a parameter token
  if (! runPL.isParameter(PL_KEY_SOLVER_CONFIG))
    return (solverPL);

  const string solverConfig = runPL.get<string>(PL_KEY_SOLVER_CONFIG);

  // if XML, then read and broadcast
  if (isXML(solverConfig))
  {
    *pOut_ << "Loading Solver configuration: " << solverConfig << std::endl;

    solverPL = parameterList();

    Teuchos::updateParametersFromXmlFileAndBroadcast(solverConfig, solverPL.ptr(), *(this->comm_));

    solverFileToken = getBasename(solverConfig, ".xml");

  }
  else if( runPL.isSublist(solverConfig) ) {

    *pOut_ << "Loading Solver configuration from sublist: " << solverConfig << std::endl;

    const ParameterList& t = runPL.sublist(solverConfig);
    solverPL = parameterList (t);

    solverFileToken = getBasename(solverConfig);
  }
  else if( driverPL_->isSublist(solverConfig) ) {

    *pOut_ << "Loading Solver configuration from sublist: " << solverConfig << std::endl;

    const ParameterList& t = driverPL_->sublist(solverConfig);
    solverPL = parameterList (t);

    solverFileToken = getBasename(solverConfig);
  }
  else {

    *pOut_ << "FAILED: Loading Solver configuration: " << solverConfig << ", is not a XML or a sublist." << std::endl;
  }

  runPL.set("solverFileToken", solverFileToken);
  return (solverPL);
}



template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
std::string
SolverDriverDetails<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getBasename(const std::string& path, const std::string ext)
{
  std::string name;
  using std::cout;
  using std::string;
  using std::endl;

  // use the XML filename as the solverFileToken
  #ifdef _WIN32
    const string sep = "\\";
  #else
    const string sep = "/";
  #endif

  size_t i = path.rfind(sep, path.length());
  size_t j = path.rfind(ext, path.length());

  // get the location *before* the ext if it exists
  j = (j == std::string::npos || j<1) ? 0 : j-1;

  if (i == std::string::npos) {
    // no path
    int count =  j+1;
    if (count > 0)
      name = path.substr(0, count);

  }
  else {
    int count =  int(j)- int(i);
    if (count > 0)
      name = path.substr(i+1, count);
  }

  return (name);
}

// http://stackoverflow.com/questions/874134/find-if-string-ends-with-another-string-in-c

template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
bool
SolverDriverDetails<Scalar,LocalOrdinal,GlobalOrdinal,Node>::isXML(const std::string& value) const
{
  using std::string;
  using std::equal;

    const string ending = ".xml";
    if (ending.size() > value.size()) return (false);
    return (equal(ending.rbegin(), ending.rend(), value.rbegin()));
}



template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
int
SolverDriverDetails<Scalar,LocalOrdinal,GlobalOrdinal,Node>::run() {
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ArrayRCP;
  using Teuchos::TimeMonitor;
  using Teuchos::Time;
  using Teuchos::FancyOStream;
  using Teuchos::ParameterList;
  using std::string;
  using ss_t = std::stringstream;
  using std::endl;

  FancyOStream& out = *pOut_;

  comm_->barrier();
  //RCP<Time> globalTimeMonitor = TimeMonitor::getNewTimer("Driver: S - Global Time");
  Xpetra::UnderlyingLib lib = xpetraParams_->GetLib();

  using CItor=ParameterList::ConstIterator;

  for(CItor it = driverPL_->begin(); it != driverPL_->end(); ++it)
  {
    if (driverPL_->entry(it).isList()) {
      out << "Sublist: " << driverPL_->name(it) << endl << driverPL_->sublist(driverPL_->name(it), true);
    }
  }

  int runID = 0;
  string runLabel = "run";

  while(driverPL_->isSublist(runLabel))
  {

    using Teuchos::parameterList;
    ParameterList& runParamList = driverPL_->sublist(runLabel);

    out << "+++++"
        << endl
        << "================================== ++++ Run " << runID << " ++++ =================================="
        << endl
        << "+++++"
        << endl;

    performRun(runParamList, runID);

    ++runID;
    ss_t ss;
    ss << "run" << runID;
    runLabel = ss.str ();
  }

  return (EXIT_SUCCESS);
}

  /// \brief report the linear solvers available in Belos
  ///
  /// Report the names and parameter options available for each
  /// solver.

template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void
SolverDriverDetails<Scalar,LocalOrdinal,GlobalOrdinal,Node>::reportBelosSolvers ()
{
  using std::string;
  using std::endl;
  using Teuchos::Array;
  using Belos::SolverFactory;
  using Teuchos::RCP;
  using Teuchos::ParameterList;
  using Teuchos::parameterList;

  typedef Teuchos::Array<string> string_array_type;

  const std::string banner (80, '-');

  BelosSolverFactory factory;

  string_array_type supportedSolvers = factory.supportedSolverNames ();

  string_array_type::iterator it;
  for (it = supportedSolvers.begin (); it != supportedSolvers.end (); ++it)
  {
    const string& solverName = *it;
    RCP<ParameterList> solverParams = parameterList ();

    RCP<BelosSolverManager> aSolver = factory.create ( solverName, solverParams);

    *pOut_ << banner
           << std::endl
           << solverName
           << std::endl;

    RCP< const ParameterList > solverParams1 = aSolver->getValidParameters ();
    solverParams1->print(*pOut_);
  }
}

