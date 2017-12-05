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
#ifndef SolverDriverDetails_decl_HPP
#define SolverDriverDetails_decl_HPP

#include <omp.h>


#include <cstdio>
#include <iomanip>
#include <string>
#include <iostream>
#include <unistd.h>
#include <algorithm>
#include <stdexcept>
#include <utility>

#include <Teuchos_ArrayRCPDecl.hpp>
#include <Teuchos_BLAS_types.hpp>
#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_ENull.hpp>
#include <Teuchos_ParameterEntry.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_PerformanceMonitorBase.hpp>
#include <Teuchos_TestForException.hpp>
#include <Teuchos_Time.hpp>
#include <Teuchos_VerbosityLevel.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <Teuchos_StandardCatchMacros.hpp>
#include <Teuchos_Comm.hpp>

// Xpetra
#include <Xpetra_Map.hpp>
#include <Xpetra_Matrix.hpp>
#include <Xpetra_MultiVector.hpp>
#include <Xpetra_Parameters.hpp>
#include <Xpetra_Vector.hpp>
#include <Xpetra_IO.hpp>
#include <Xpetra_BlockedVector.hpp>
#include <Xpetra_CrsMatrixWrap.hpp>
#include <Xpetra_MapFactory.hpp>
#include <Xpetra_MatrixFactory.hpp>
#include <Xpetra_VectorFactory.hpp>
#include <Xpetra_ImportFactory.hpp>

// Galeri
#include <Galeri_Problem.hpp>
#include <Galeri_XpetraParameters.hpp>
#include <Galeri_XpetraProblemFactory.hpp>
#include <Galeri_XpetraUtils.hpp>
#include <Galeri_XpetraMaps.hpp>
// use eti version
#include <BelosOperatorT.hpp>
#include <BelosSolverManager.hpp>
#include <BelosTypes.hpp>
#include <BelosXpetraAdapterOperator.hpp>
#include <BelosMultiVecTraits.hpp>

#include <MueLu.hpp>
#include <MueLu_BaseClass.hpp>
#include <MueLu_Exceptions.hpp>
#include <MueLu_Hierarchy_decl.hpp>
#include <MueLu_UtilitiesBase_decl.hpp>

#ifdef HAVE_MUELU_EXPLICIT_INSTANTIATION
#include <MueLu_ExplicitInstantiation.hpp>
#endif
#include <MueLu_Level.hpp>
#include <MueLu_MutuallyExclusiveTime.hpp>
#include <MueLu_ParameterListInterpreter.hpp>
#include <MueLu_Utilities.hpp>

#ifdef HAVE_MUELU_BELOS
#include <BelosConfigDefs.hpp>
#include <BelosSolverFactory.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosXpetraAdapter.hpp>     // => This header defines Belos::XpetraOp
#include <BelosMueLuAdapter.hpp>      // => This header defines Belos::MueLuOp
#endif

#include <Teuchos_Array.hpp> // used for belos factory query

#include <MueLu_CreateXpetraPreconditioner.hpp>
#include <Trilinos_version.h>
#include "mpi_local_ranks.hpp"
#include "affinity_check.hpp"



template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
class SolverDriverDetails
{
  typedef Scalar        SC;
  typedef LocalOrdinal  LO;
  typedef GlobalOrdinal GO;
  typedef Node          NO;
  typedef Teuchos::ScalarTraits<SC> STS;

  typedef Xpetra::MultiVector<SC,LO,GO,NO>        MultiVector;
  typedef Xpetra::MultiVectorFactory<SC,LO,GO,NO> MultiVectorFactory;
  typedef Xpetra::Vector<SC,LO,GO,NO>             Vector;
  typedef Xpetra::VectorFactory<SC,LO,GO,NO>      VectorFactory;
  typedef Xpetra::Matrix<SC,LO,GO,NO>             Matrix;
  typedef Xpetra::MatrixFactory2<SC,LO,GO,NO>     MatrixFactory2;
  typedef Xpetra::CrsMatrixWrap<SC,LO,GO,NO>      CrsMatrixWrap;
  typedef Xpetra::Map<LO, GO, NO>                 Map;
  typedef MueLu::UtilitiesBase<SC,LO,GO,NO>       Utilities;
  typedef MueLu::Hierarchy<SC,LO,GO,NO>           Hierarchy;


  typedef Belos::OperatorT<MultiVector> BelosOperatorT;
  typedef Belos::MueLuOp <SC,LO,GO,NO>  BelosMueLuOp;
  typedef Belos::XpetraOp<SC,LO,GO,NO>  BelosXpetraOp;
  typedef Belos::LinearProblem<SC, MultiVector, BelosOperatorT> BelosLinearProblem;
  typedef Belos::SolverManager<SC,MultiVector,BelosOperatorT>   BelosSolverManager;
  typedef Belos::SolverFactory<SC,MultiVector,BelosOperatorT>   BelosSolverFactory;

  typedef PerfUtils::cpu_map_type cpu_map_type;

  typedef std::vector<std::string> proc_status_column_type;
  typedef std::map< std::string, proc_status_column_type> proc_status_table_type;

  #if defined (HAVE_MUELU_AMGX) and defined (HAVE_MUELU_TPETRA)
    typedef MueLu::AMGXOperator<SC,LO,GO,NO> amgx_op_type;
  #endif

public:
  SolverDriverDetails (
    int argc, char* argv[],
    Teuchos::CommandLineProcessor& clp,
    Teuchos::RCP<Xpetra::Parameters>& xpetraParams);

  virtual ~SolverDriverDetails () {}

public:
  int run();

private:
  Teuchos::RCP<const Matrix>      orig_A_ ;
  Teuchos::RCP<const MultiVector> orig_coordinates_;
  Teuchos::RCP<const MultiVector> orig_nullspace_;
  Teuchos::RCP<const MultiVector> orig_X_;
  Teuchos::RCP<const MultiVector> orig_B_;
  Teuchos::RCP<const Map>         orig_map_;
  Teuchos::RCP<const Teuchos::Comm<int> > comm_;
  Teuchos::RCP<const Teuchos::Comm<int> > nodeComm_;
  Teuchos::RCP<const Teuchos::Comm<int> > nodeLocalComm_;
  Teuchos::RCP<Teuchos::FancyOStream> pOut_;
  Teuchos::RCP<Teuchos::ParameterList> driverPL_;
  Teuchos::RCP<Teuchos::ParameterList> galeriPL_;
  Teuchos::RCP<Xpetra::Parameters> xpetraParams_;
  Teuchos::RCP<Teuchos::ParameterList> configPL_;
  Teuchos::RCP<Teuchos::ParameterList> timerReportParams_;
  bool useSmartSolverLabels_;
  bool checkConvergence_;
  int cores_per_proc_;
  int threads_per_core_;

  std::string decompFileToken_;
  std::string problemFileToken_;
  std::string numThreadsFileToken_;

  Teuchos::RCP<const cpu_map_type> my_cpu_map_;

  // This is odd, but in c++11, you can define these this way
  // and it silences the ISO warning.
  // static constexpr const <- this matters
  static constexpr const char* PL_KEY_SOLVER_NAME        = "Solver";
  static constexpr const char* PL_DEFAULT_SOLVER_NAME    = "Cg";

  static constexpr const char* PL_KEY_SOLVER_FACTORY     = "SolverFactory";
  static constexpr const char* PL_DEFAULT_SOLVER_FACTORY = "Belos";

  static constexpr const char* PL_KEY_SOLVER_CONFIG      = "SolverParams";

  static constexpr const char* PL_KEY_PREC_NAME          = "Preconditioner";
  static constexpr const char* PL_DEFAULT_PREC_NAME      = "None";

  static constexpr const char* PL_KEY_PREC_FACTORY       = "PreconditionerFactory";
  static constexpr const char* PL_DEFAULT_PREC_FACTORY   = "MueLu";

  static constexpr const char* PL_KEY_PREC_CONFIG        = "PreconditionerParams";

  // driver controls
  static constexpr const char* PL_KEY_TIMESTEP           = "Pseudo Timesteps";
  static const int             PL_DEFAULT_TIMESTEP;

  static constexpr const char* PL_KEY_COPY_R0            = "Set Initial Residual";
  static const bool            PL_DEFAULT_COPY_R0;

  static constexpr const char* PL_KEY_TIMESTEP_DEEPCOPY  = "Deep Copy each Timestep";
  static const bool            PL_DEFAULT_TIMESTEP_DEEPCOPY;

  static constexpr const char* PL_KEY_BELOS_TIMER_LABEL  = "Timer Label";

  // Solver | LinearAlg
  static constexpr const char* PL_KEY_EXPERIMENT_TYPE         = "ExperimentType";
  static constexpr const char* EXPERIMENT_TYPE_SOLVER         = "Solver";
  static constexpr const char* EXPERIMENT_TYPE_LINEAR_ALGEBRA = "Linear Algebra";

  static constexpr const char* PL_DEFAULT_EXPERIMENT_TYPE = EXPERIMENT_TYPE_SOLVER;

  static constexpr const char* PL_KEY_CONSTRUCTOR_ONLY     = "Construction Only";
  static const bool            PL_DEFAULT_CONSTRUCTOR_ONLY;


  static constexpr const char* TM_LABEL_GLOBAL       = "0 - Total Time";
  static constexpr const char* TM_LABEL_COPY         = "1 - Reseting Linear System";
  static constexpr const char* TM_LABEL_NULLSPACE    = "2 - Adjusting Nullspace for BlockSize";
  static constexpr const char* TM_LABEL_PREC_SETUP   = "3 - Constructing Preconditioner";
  static constexpr const char* TM_LABEL_SOLVER_SETUP = "4 - Constructing Solver";
  static constexpr const char* TM_LABEL_SOLVE        = "5 - Solve";

  static constexpr const char* TM_LABEL_APPLY        = "2 - Apply";



  static constexpr const char* AFFINITY_MAP_CSV_STR  = "affinity.csv";

  const Teuchos::EVerbosityLevel DESCRIBE_VERB_LEVEL = Teuchos::EVerbosityLevel::VERB_LOW;


  void createConfigParameterList (Teuchos::ParameterList& configPL);

  void gatherAffinityInfo ();

  void createLinearSystem(
      const Xpetra::Parameters& xpetraParameters,
      Teuchos::ParameterList& matrixPL);

  void setFileTokens ();

  GO getGaleriProblemDim (const GO n);

  std::string getProblemFileToken ();

  void setProblemFileToken ();

  std::string getDecompFileToken ();

  // num_nodes x procs_per_node x cores_per_proc x thread_per_core
  void setDecompFileToken ();

  std::string getNumThreadsFileToken ();

  void setNumThreadsFileToken ();

  std::string getTimeStepFileToken (const int numsteps);

  void performRun(Teuchos::ParameterList& runParamList, const int runID);
  void performSolverExperiment(Teuchos::ParameterList& runParamList, const int runID);
  void performLinearAlgebraExperiment(Teuchos::ParameterList& runParamList, const int runID);

  void writeTimersForFunAndProfit (const std::string& filename);

  Teuchos::RCP<const Teuchos::ParameterList>
  getDefaultSolverExperimentParameters () const;

  Teuchos::RCP<const Teuchos::ParameterList>
  getDefaultLinearAlgebraExperimentParameters () const;

  /*
   * Expects to see:
   *    Preconditioner        : Name of a preconditioner
   *    PreconditionerFactory : MueLu currently is the only one supported, this means we will pass this parameter list directory to MueLu
   *    PreconditionerParams  : A sublist name or XML file to load.
   *
   *    runPL is also expected to contain the defaults
   */
  Teuchos::RCP<Teuchos::ParameterList>
  getPreconditionerParameters(Teuchos::ParameterList& runPL);

  Teuchos::RCP<Teuchos::ParameterList>
  getSolverParameters(Teuchos::ParameterList& runPL);

  std::string getBasename(const std::string& path, const std::string ext="");

  // http://stackoverflow.com/questions/874134/find-if-string-ends-with-another-string-in-c
  bool isXML(const std::string& value) const;

  /// \brief report the linear solvers available in Belos
  ///
  /// Report the names and parameter options available for each
  /// solver.
  void
  reportBelosSolvers ();

  void track_memory_usage(proc_status_table_type& region_table,
                          Teuchos::FancyOStream& out);
};


template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
const int  SolverDriverDetails<Scalar,LocalOrdinal,GlobalOrdinal,Node>::PL_DEFAULT_TIMESTEP = 50;

template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
const bool SolverDriverDetails<Scalar,LocalOrdinal,GlobalOrdinal,Node>::PL_DEFAULT_COPY_R0  = false;

template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
const bool SolverDriverDetails<Scalar,LocalOrdinal,GlobalOrdinal,Node>::PL_DEFAULT_TIMESTEP_DEEPCOPY = true;

template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
const bool SolverDriverDetails<Scalar,LocalOrdinal,GlobalOrdinal,Node>::PL_DEFAULT_CONSTRUCTOR_ONLY = false;

#endif // SolverDriverDetails_decl_HPP
