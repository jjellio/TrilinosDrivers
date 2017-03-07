TRILINOS_MAKEFILE :=/share/install/trilinos-12.11_gcc-6.1.0-openmpi-2.0.0/include/Makefile.export.Trilinos
include $(TRILINOS_MAKEFILE)

arch=snb
execSpace=openmp

BINARY_NAME :=Solver_driver.$(arch).$(execSpace).x

print-%  : ; @echo $* = $($*)


all: $(BINARY_NAME)


$(BINARY_NAME): SolverDriver.cpp SolverDriverDetails.hpp mpi_local_ranks.hpp mpi_local_ranks.cpp
	$(Trilinos_CXX_COMPILER) -g $(Trilinos_INCLUDE_DIRS) $(Trilinos_TPL_INCLUDE_DIRS) $(Trilinos_CXX_COMPILER_FLAGS) SolverDriver.cpp mpi_local_ranks.cpp /share/src/Trilinos/packages/teuchos/parameterlist/src/Teuchos_YamlParser.cpp -o $(BINARY_NAME) \
	$(Trilinos_LIBRARY_DIRS) $(Trilinos_LIBRARIES) $(Trilinos_TPL_LIBRARY_DIRS) $(Trilinos_TPL_LIBRARIES)


clean:
	-rm -f $(BINARY_NAME)

