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

#include <cstdio>
#include <iomanip>
#include <string>
#include <iostream>
#include <unistd.h>
#include <stdexcept>
#include <utility>

#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_ENull.hpp>
#include <Teuchos_ParameterEntry.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_TestForException.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <Teuchos_StandardCatchMacros.hpp>
#include <Teuchos_Comm.hpp>


// Xpetra
#include <Xpetra_Map.hpp> // needed for lib enum

#include <TpetraCore_config.h>
#ifdef HAVE_MUELU_EXPLICIT_INSTANTIATION
#include <MueLu_ExplicitInstantiation.hpp>
#endif

#ifdef HAVE_MUELU_OPENMP
#include <omp.h>
#endif

#include "SolverDriverDetails_decl.hpp"

#include <Teuchos_YamlParser_decl.hpp>
namespace Teuchos {

TEUCHOSPARAMETERLIST_LIB_DLL_EXPORT
RCP<ParameterList> getParametersFromYamlString(const std::string &yamlStr)
{
  Teuchos::RCP<Teuchos::ParameterList> params = YAMLParameterList::parseYamlText(yamlStr);
  return (params);
}

}

template<class Node>
int
runDriver(const std::string& SC_type, const std::string& LO_type, const std::string& GO_type,
    int argc, char* argv[],
    Teuchos::CommandLineProcessor& clp,
    Teuchos::RCP<Xpetra::Parameters>& xpetraParams)
{

//  #define _mueluMacroExists_(MueLuSC,MueLuLO,MueLuGO) defined(HAVE_MUELU_INST_##MueLuSC##_##MueLuLO##_##MueLuGO)

  #if \
    defined(HAVE_MUELU_INST_DOUBLE_INT_INT) || \
    defined(HAVE_MUELU_INST_DOUBLE_INT_LONGINT) || \
    defined(HAVE_MUELU_INST_DOUBLE_INT_LONGLONGINT) || \
    defined(HAVE_MUELU_INST_DOUBLE_LONGINT_INT) || \
    defined(HAVE_MUELU_INST_DOUBLE_LONGINT_LONGINT) || \
    defined(HAVE_MUELU_INST_DOUBLE_LONGINT_LONGLONGINT) || \
    defined(HAVE_MUELU_INST_DOUBLE_LONGLONGINT_INT) || \
    defined(HAVE_MUELU_INST_DOUBLE_LONGLONGINT_LONGINT) || \
    defined(HAVE_MUELU_INST_DOUBLE_LONGLONGINT_LONGLONGINT)
    #define _HAVE_DOUBLE_
  #endif

  #if \
    defined(HAVE_MUELU_INST_FLOAT_INT_INT) || \
    defined(HAVE_MUELU_INST_FLOAT_INT_LONGINT) || \
    defined(HAVE_MUELU_INST_FLOAT_INT_LONGLONGINT) || \
    defined(HAVE_MUELU_INST_FLOAT_LONGINT_INT) || \
    defined(HAVE_MUELU_INST_FLOAT_LONGINT_LONGINT) || \
    defined(HAVE_MUELU_INST_FLOAT_LONGINT_LONGLONGINT) || \
    defined(HAVE_MUELU_INST_FLOAT_LONGLONGINT_INT) || \
    defined(HAVE_MUELU_INST_FLOAT_LONGLONGINT_LONGINT) || \
    defined(HAVE_MUELU_INST_FLOAT_LONGLONGINT_LONGLONGINT)
    #define _HAVE_FLOAT_
  #endif

  #if \
    defined(HAVE_MUELU_INST_COMPLEX_INT_INT) || \
    defined(HAVE_MUELU_INST_COMPLEX_INT_LONGINT) || \
    defined(HAVE_MUELU_INST_COMPLEX_INT_LONGLONGINT) || \
    defined(HAVE_MUELU_INST_COMPLEX_LONGINT_INT) || \
    defined(HAVE_MUELU_INST_COMPLEX_LONGINT_LONGINT) || \
    defined(HAVE_MUELU_INST_COMPLEX_LONGINT_LONGLONGINT) || \
    defined(HAVE_MUELU_INST_COMPLEX_LONGLONGINT_INT) || \
    defined(HAVE_MUELU_INST_COMPLEX_LONGLONGINT_LONGINT) || \
    defined(HAVE_MUELU_INST_COMPLEX_LONGLONGINT_LONGLONGINT)
    #define _HAVE_COMPLEX_
  #endif

  if(SC_type == "double") {

    #if defined(_HAVE_DOUBLE_)
      typedef double ScalarType;

      if (LO_type == "int" && GO_type == "int") {
        #ifdef HAVE_MUELU_INST_DOUBLE_INT_INT
          typedef SolverDriverDetails<ScalarType,int,int,Node> sd_t;
          sd_t sd(argc,argv, clp, xpetraParams);
          return (sd.run ( ));
        #else
          throw MueLu::Exceptions::RuntimeError("ScalarType=" + SC_type + ", LocalOrdinal=" + LO_type + ", GlobalOrdinal=" + GO_type +" is disabled (must be compiled)");
        #endif
      } else if (LO_type == "int" && GO_type == "long_long") {
        #ifdef HAVE_MUELU_INST_DOUBLE_INT_LONGLONGINT
          typedef SolverDriverDetails<ScalarType,int,long long int,Node> sd_t;
          sd_t sd(argc,argv, clp, xpetraParams);
          return (sd.run ( ));
        #else
          throw MueLu::Exceptions::RuntimeError("ScalarType=" + SC_type + ", LocalOrdinal=" + LO_type + ", GlobalOrdinal=" + GO_type +" is disabled (must be compiled)");
        #endif
        } else {
          throw MueLu::Exceptions::RuntimeError("ScalarType=" + SC_type + ", LocalOrdinal=" + LO_type + ", GlobalOrdinal=" + GO_type +" is disabled (must be compiled or is not implemented in driver)");
        }


    #else
      throw MueLu::Exceptions::RuntimeError("ScalarType=" + SC_type + " is disabled");
    #endif

  } else if (SC_type == "float") {

    #if defined(_HAVE_FLOAT_)
      throw MueLu::Exceptions::RuntimeError("ScalarType=" + SC_type + " is not implemented in the driver");
    #else
        throw MueLu::Exceptions::RuntimeError("ScalarType=" + SC_type + " is disabled (must be compiled)");
    #endif

  } else if (SC_type == "complex") {

    #if defined(_HAVE_COMPLEX_)
      throw MueLu::Exceptions::RuntimeError("ScalarType=" + SC_type + " is not implemented in the driver");
    #else
        throw MueLu::Exceptions::RuntimeError("ScalarType=" + SC_type + " is disabled (must be compiled)");
    #endif

  } else {
      throw MueLu::Exceptions::RuntimeError("Unknown ScalarType: " + SC_type );
  }
}

int
runDriver(const std::string& nodeName, const std::string& SC_type, const std::string& LO_type, const std::string& GO_type,
    int argc, char* argv[],
    Teuchos::CommandLineProcessor& clp,
    Teuchos::RCP<Xpetra::Parameters>& xpetraParams)
{
  if(nodeName == "openmp") {

    #if defined(HAVE_MUELU_OPENMP) && defined(KOKKOS_HAVE_OPENMP)
      typedef Kokkos::Compat::KokkosOpenMPWrapperNode Node;
      return (runDriver<Node>(SC_type, LO_type, GO_type, argc, argv, clp, xpetraParams));
    #else
      throw MueLu::Exceptions::RuntimeError("OpenMP node type is disabled");
    #endif

  } else if (nodeName == "cuda") {

    #if defined(HAVE_TPETRA_INST_CUDA) && defined(KOKKOS_HAVE_CUDA)
      typedef Kokkos::Compat::KokkosCudaWrapperNode Node;
      return runDriver<Node>(SC_type, LO_type, GO_type, argc, argv, clp, xpetraParams);
    #else
      throw MueLu::Exceptions::RuntimeError("Cuda node type is disabled");
    #endif

  } else if (nodeName == "serial") {

    #if defined(HAVE_TPETRA_INST_SERIAL) && defined(KOKKOS_HAVE_SERIAL)
      typedef Kokkos::Compat::KokkosSerialWrapperNode Node;
      return runDriver<Node>(SC_type, LO_type, GO_type, argc, argv, clp, xpetraParams);
    #else
      throw MueLu::Exceptions::RuntimeError("Serial node type is disabled");
    #endif

  } else if (nodeName == "") {

      typedef KokkosClassic::DefaultNode::DefaultNodeType Node;
      return (runDriver<Node>(SC_type, LO_type, GO_type, argc, argv, clp, xpetraParams));

  } else {
      throw MueLu::Exceptions::RuntimeError("Unknown node type: " + nodeName );
  }
}


int main(int argc, char* argv[]) {
  bool success = false;
  bool verbose = true;

  //try {
    const bool throwExceptions = false;

    using CLP = Teuchos::CommandLineProcessor;
    using XpetraParams = Xpetra::Parameters;
    using Teuchos::RCP;
    using Teuchos::FancyOStream;
    using Teuchos::fancyOStream;
    using Teuchos::rcpFromRef;
    using Teuchos::Comm;
    using Teuchos::GlobalMPISession;
    using Teuchos::ParameterList;
    using std::cout;


    // =========================================================================
    // Parameters initialization
    // =========================================================================

    CLP clp(throwExceptions);


    RCP<XpetraParams> xpetraParams = rcp(new XpetraParams(clp));

    std::string node    = "";          clp.setOption("node",         &node,    "node type (serial | openmp | cuda)");
    std::string LO_type = "int";       clp.setOption("LocalOrdinal", &LO_type, "local ordinal type  (int | long | long_long)");
    std::string GO_type = "long_long"; clp.setOption("GlobalOrdinal",&GO_type, "global ordinal type (int | long | long_long)");
    std::string SC_type = "double";    clp.setOption("ScalarType",   &SC_type, "scalar type (float | double | complex)");

    clp.recogniseAllOptions(false);
    switch (clp.parse(argc, argv, NULL)) {
      case Teuchos::CommandLineProcessor::PARSE_ERROR:               return EXIT_FAILURE;
      case Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED:
      case Teuchos::CommandLineProcessor::PARSE_UNRECOGNIZED_OPTION:
      case Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL:          break;
    }
    // =========================================================================
    // MPI initialization using Teuchos
    // =========================================================================
    // do this so argc/argv will have kokkos items removed
    //Kokkos::initialize(argc, argv);
    GlobalMPISession mpiSession(&argc, &argv, NULL);


    Xpetra::UnderlyingLib lib = xpetraParams->GetLib();

    if (lib == Xpetra::UseEpetra) {
      #ifdef HAVE_MUELU_EPETRA
      {
        typedef SolverDriverDetails<double,int,int,Xpetra::EpetraNode> sd_t;
        sd_t sd(argc, argv, clp, xpetraParams);
        return (sd.run());
      }
      #else
        throw MueLu::Exceptions::RuntimeError("Epetra is not available");
      #endif
    }

    if (lib == Xpetra::UseTpetra) {
      #ifdef HAVE_MUELU_TPETRA
        return (runDriver(node, SC_type, LO_type, GO_type, argc, argv, clp, xpetraParams));
      #else
        throw MueLu::Exceptions::RuntimeError("Tpetra is not available");
      #endif
    }


  return ( success ? EXIT_SUCCESS : EXIT_FAILURE );
}
