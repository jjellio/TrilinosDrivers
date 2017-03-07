#!/bin/bash

eti_dir="eti"
mkdir -p $eti_dir


for LO in INT LONGINT LONGLONGINT; do
for GO in INT LONGINT LONGLONGINT; do
for NO in OPENMP SERIAL CUDA; do

lo=""
go=""
no=""
galeri_map_guard="0"

if [ "$LO" == "INT" ]; then
        lo=int;
elif [ "$LO" == "LONGINT" ]; then
        lo="long int";
elif [ "$LO" == "LONGLONGINT" ]; then
        lo="long long int";
else
        lo=error_me
fi

if [ "$GO" == "INT" ]; then
        go=int;
elif [ "$GO" == "LONGINT" ]; then
        go="long int";
elif [ "$GO" == "LONGLONGINT" ]; then
        go="long long int";
else
        go=error_me
fi

if [ "$NO" == "OPENMP" ]; then
        no="Kokkos::Compat::KokkosOpenMPWrapperNode";
elif [ "$NO" == "SERIAL" ]; then
        no="Kokkos::Compat::KokkosSerialWrapperNode";
elif [ "$NO" == "CUDA" ]; then
        no="Kokkos::Compat::KokkosCudaWrapperNode";
else
        no=error_me
fi


for S in DOUBLE FLOAT; do

s=""

if [ "$S" == "DOUBLE" ]; then
	s=double;
elif [ "$S" == "FLOAT" ]; then
	s=float;
else
	s=error_me
fi

muelu_def="HAVE_MUELU_INST_${S}_${LO}_${GO}"
tpetra_def="HAVE_TPETRA_INST_${NO}"
eti_name="SolverDriverDetails_${S}_${LO}_${GO}_${NO}.cpp"
eti_fname="${eti_dir}/${eti_name}"

cat > $eti_fname <<- EOM
#include <TpetraCore_config.h>
#include <MueLu_ExplicitInstantiation.hpp>
#if defined(${muelu_def}) && defined(${tpetra_def})
	#include "SolverDriverDetails_def.hpp"
	template class SolverDriverDetails<${s},${lo},${go},${no}>;
#endif

EOM

echo "${eti_fname}"

#galeri_map_guard="${galeri_map_guard} || ( defined(HAVE_MUELU_INST_${S}_${LO}_${GO}) && defined(HAVE_TPETRA_INST_${NO}) )"


done

#eti_name="Galeri_Xpetra_Map_${LO}_${GO}_${NO}.cpp"
#eti_fname="${eti_dir}/${eti_name}"
#
#cat > $eti_fname <<- EOM
##include <TpetraCore_config.h>
##include <MueLu_ExplicitInstantiation.hpp>
##if ${galeri_map_guard}
#  #include "Galeri_XpetraMaps_def.hpp"
#  template class Galeri::Xpetra::Map<${lo},${go},${no}>;
#
##endif
#
#EOM
#
#echo "${eti_fname}"



done
done
done


