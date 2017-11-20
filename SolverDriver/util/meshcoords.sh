#!/bin/bash
#
# This is to find the node IDs, node names (in the form of c#-#c#s#n#)
# and their dragonfly topology coordinates for the nodes allocated for
# the specified batch job or the nodes whose topology coordinates are
# specified.
#
# Please note that this uses a simplistic way of determining the XYZ
# coordinates on Edison and Cori as we know how topology geometry is set there.
# This method may not be valid on other Cray XC machines.
#
# wyang at NERSC
#
# 20160902: Created for edison
# 20160917: Revised
# 20161005: Revised for edison and cori phase 2
# 20161020: Added the option for specifying topology coords
# 20161101: Revised

 set -e

 if [[ "${NERSC_HOST}" == "edison" ]]; then

   COLS=8    # cabinet columns
   mom_node=edimom02
#  export ESWRAP_LOGIN=${mom_node}

 elif [[ "${NERSC_HOST}" == "cori" ]]; then

   COLS=12   # cabinet columns
   mom_node=cmom01

 else

   echo "ERROR: topology for ${NERSC_HOST} unknown"
   exit

 fi

######################################################################
# Functions
######################################################################

 print_usage() {
   echo
   echo "Usage: $(basename $0) [[-j] JobID] | [[-X TOPOXs] [-Y TOPOYs] [-Z TOPOZs]]"
   echo
   echo "   (1) [-j] JobID"
   echo "       to show node info for the nodes used by job JobID"
   echo "       JobID can be either <job_id> for an entire batch job"
   echo "       or <job_id>.<step_id> for a certain step (srun) in a batch job"
   echo
   echo "   (2) [-X TOPOXs] [-Y TOPOYs] [-Z TOPOZs]"
   echo "       to show node info for the specified topology coords"
   echo "       TOPOXs, TOPOYs, and TOPOZs are comma-separated values or ranges"
   echo "       (e.g., '-X 3,5,9-11 -Y 2')"
   echo "       Note that this option prints out info for compute nodes only."
 }

 ranges_from_arg() {
   local ent=$1
   local arr_tmp=( )
   local a
   local i=0
   shift

   for a in $(echo $1 | sed 's/,/ /g')
   do
     if [[ $a =~ - ]]; then
       arr_tmp[$((i++))]="${ent}.ge.${a%-*}.and.${ent}.le.${a#*-}"
     else
       arr_tmp[$((i++))]="${ent}.eq.${a}"
     fi
   done
   echo ${arr_tmp[@]}
 }

 combine2() {
   local arr1=( $1 )
   local arr2=( $2 )
   local arrf=( )
   local a1
   local a2
   local i=0

   if [[ ${#arr1[@]} -eq 0 ]]; then
     echo ${arr2[@]}
   else

     if [[ ${#arr2[@]} -eq 0 ]]; then
       echo ${arr1[@]}
     else

       for a1 in ${arr1[@]}
       do
         for a2 in ${arr2[@]}
         do
           arrf[$((i++))]=${a1}.and.${a2}
         done
       done

       echo ${arrf[@]}

     fi

   fi
 }

 nids_from_cnselect() {
   local copts="$1"
   local nodes=""

   if [[ $(hostname) == nid* || $(hostname) == *mom* ]]; then    # assume MOM nodes' hostname contains the word

     nodes=$(cnselect $copts)

   else

     nodes=$(ssh ${mom_node} 'cnselect '"${copts}" 2>/dev/null)

   fi

   echo $nodes
 }

 print_nodeinfo() {
   local nodes=$1

   echo "  NID   NODENAME      TYPE   CORES   X  Y  Z"

   if [[ $(hostname) == nid* || $(hostname) == *mom* ]]; then    # assume MOM nodes' hostname contains the word

     xtprocadmin -a cu -n "${nodes}" 2>/dev/null | \
       awk -v COLS=$COLS '($1 !~ /NID/) {split($3,crgsn,/[csn-]/);
         x = int(crgsn[2] / 2) + crgsn[3] * int(COLS / 2);
         y = crgsn[4] + (crgsn[2] % 2) * 3; z = crgsn[5];
         printf "%5d  %-13s %-9s %3d  %2d %2d %2d\n", $1, $3, $4, $5, x, y, z}'

   else

     ssh ${mom_node} 'xtprocadmin -a cu -n '"${nodes}" 2>/dev/null | \
       awk -v COLS=$COLS '($1 !~ /NID/) {split($3,crgsn,/[csn-]/);
         x = int(crgsn[2] / 2) + crgsn[3] * int(COLS / 2);
         y = crgsn[4] + (crgsn[2] % 2) * 3; z = crgsn[5];
         printf "%5d  %-13s %-9s %3d  %2d %2d %2d\n", $1, $3, $4, $5, x, y, z}'

   fi
 }

######################################################################
# Main
######################################################################

 using_topo_coords=0
 using_jobid=0

 while getopts ":j:X:Y:Z:" opt
 do
   case $opt in
     j  ) jobid=$OPTARG; using_jobid=1;;
     X  ) tx_a=( $(ranges_from_arg x_coord $OPTARG) ); using_topo_coords=1;;
     Y  ) ty_a=( $(ranges_from_arg y_coord $OPTARG) ); using_topo_coords=1;;
     Z  ) tz_a=( $(ranges_from_arg z_coord $OPTARG) ); using_topo_coords=1;;
     \? ) print_usage; exit;;
   esac
 done

 shift $(($OPTIND - 1))

 if [[ "$#" -eq 1 ]]; then    # backward compatibility for using a job ID
   jobid=$1
   using_jobid=1
 fi

 if [[ $using_jobid -eq 1 ]]; then

   if [[ $jobid =~ '.' ]]; then
     sopts="-np"
   else
     sopts="-npX"
   fi

   nodes=$(sacct -o nodelist "${sopts}" -j ${jobid} | \
           sed -e 's/nid0*\[//' -e 's/\]|*//' -e 's/|//' -e 's/\<0*//g')

 elif [[ $using_topo_coords -eq 1 ]]; then

   tmp=( $(combine2 "${tx_a[*]}" "${ty_a[*]}") )
   copts=$(combine2 "${tmp[*]}"  "${tz_a[*]}" | sed 's/  */.or./g')
   nodes=$(nids_from_cnselect $copts)

 fi

 if [[ -n "${nodes}" ]]; then

   print_nodeinfo ${nodes}

 fi
