eti_dir="eti"
mkdir $eti_dir

for S in DOUBLE FLOAT; do
for LO in INT LONGINT LONGLONGINT; do
for GO in INT LONGINT LONGLONGINT; do
for NO in OPENMP SERIAL CUDA; do

s=""
lo=""
go=""
no=""


if [ "$S" == "DOUBLE" ]; then
	s=double;
elif [ "$S" == "FLOAT" ]; then
	s=float;
else
	s=error_me
fi

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

muelu_def="HAVE_MUELU_INST_${S}_${LO}_${GO}"
tpetra_def="HAVE_TPETRA_INST_${NO}"
eti_fname="${eti_dir}/SolverDriverDetails_${s}_${lo}_${go}_${no}.cpp"

cat > $eti_fname <<- EOM
#include "SolverDriverDetails.hpp"
#if defined(${muelu_def}) && defined(${tpetra_def})
	template class SolverDriverDetails<${s},${lo},${go},${no}>;
#endif

EOM

echo "${eti_fname}"

done
done
done




