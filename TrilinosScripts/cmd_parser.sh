#!/bin/bash

# --arch=hsw|knl|hsw-knl
# --execSpace=serial|openmp|serial-openmp
# --blasThreaded=serial|openmp
# --prec
# --hugepages=2|4|8|16|32|64|128|256|512|none

function print_vars {
  echo "arch:         ${ARG_arch}"
  echo "execSpace:    ${ARG_execSpace}"
  echo "prec:         ${ARG_prec}"
  echo "hugepages:    ${ARG_hugepages}"
  echo "blasThreaded: ${ARG_blasThreaded}"
  echo "buildID      ${ARG_buildID}"
}


function parse_cmd {
# read the options
TEMP=`getopt -o a::e::pg::b::i::h --long arch::,execSpace::,prec,hugepages::,blasThreaded::,buildID::,help -n 'configure' -- "$@"`
eval set -- "$TEMP"

shopt -s extglob

# extract options and their arguments into variables.
while true ; do
    case "$1" in
        -h|--help)
            echo "Usage:";
            echo "      --arch="`echo ${ARG_arch_valid} | tr -d '(@)'`", arch optimization flags, hsw-knl compiles both execution paths";
            echo "      --execSpace="`echo ${ARG_execSpace_valid} | tr -d '(@)'`;
            echo "      --blasThreaded="`echo ${ARG_blasThreaded_valid} | tr -d '(@)'`", whether to use threaded or sequential BLAS libraries";
            echo "      --hugepages="`echo ${ARG_hugepages_valid} | tr -d '(@)'`", build with hugepage support";
            echo "      --prec , enable -fp-model precise";
            echo "      --buildID=<string>, string (no whitespace) that will be added to build install path."
            exit 0 ;;
        -a|--arch)
            case "$2" in
                ${ARG_arch_valid})
                   ARG_arch=$2;
                   shift 2 ;;
                *) 
                   echo "$1 $2";
                   echo "Invalid arch requested: $2, valid options are ${ARG_arch_valid}";
                   exit 1 ;;
            esac ;;
        -e|--execSpace)
            case "$2" in
                ${ARG_execSpace_valid})
                   ARG_execSpace=$2; 
                   shift 2 ;;
                *) 
                   echo "$1 $2";
                   echo "Invalid execSpace requested: $2, valid options are ${ARG_execSpace_valid}";
                   exit 1 ;;
            esac ;;
        -p|--prec) 
            ARG_prec=true;
            shift ;;
        -g|--hugepages)
            case "$2" in
                ${ARG_hugepages_valid})
                   ARG_hugepages=$2;
                   shift 2 ;;
                *) 
                   echo "$1 $2";
                   echo "Invalid hugepages requested: $2, valid options are ${ARG_hugepages_valid}";
                   exit 1 ;;
            esac ;;
        -i|--buildID)
            ARG_buildID="${2}"
            shift 2 ;;
        -b|--blasThreaded)
            case "$2" in
                ${ARG_blasThreaded_valid})
                   ARG_blasThreaded=$2;
                   shift 2 ;;
                *) 
                   echo "$1 $2";
                   echo "Invalid blasThreaded requested: $2, valid options are ${ARG_blasThreaded_valid}";
                   exit 1 ;;
            esac ;;
        --) shift ; break ;;
        *) 
        echo "$1 $2";
        echo "Internal error!" ; exit 1 ;;
    esac
done

shopt -u extglob
}

input_args=("$@") 
# initial values
ARG_arch=knl
ARG_execSpace=openmp
ARG_prec="false"
ARG_hugepages=none
ARG_blasThreaded=openmp
ARG_buildID=""

ARG_arch_valid='@(hsw|knl|hsw-knl|none)'
ARG_execSpace_valid='@(serial|openmp)'
ARG_hugepages_valid='@(2|4|8|16|32|64|128|256|512)'
ARG_blasThreaded_valid='@(threaded|sequential)'

parse_cmd "${input_args[@]}";

