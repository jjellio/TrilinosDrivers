#!/bin/bash

ortho_passes=1
max_iters=200
conv_tol="0.0"
RS_SET="10 15 20 25 30 35 40 50 60 70 80 90 100"

run_count=26;
DRIVER_ENTRY="gmres_driver.stubs"

for RS in ${RS_SET}; do
for ortho in "DGKS"; do

solver_name="GMRES-${RS}-${ortho}-${ortho_passes}"

FILE="${solver_name}.xml"

cat > $FILE <<- EOM
<ParameterList name="${solver_name}">
  <Parameter name="Block Size"            type="int"  value="1" />
  <Parameter name="Maximum Iterations"    type="int"  value="${max_iters}" />
  <Parameter name="Num Blocks"            type="int"  value="${RS}" />
  <Parameter name="Maximum Restarts"      type="int"  value="1000" />
  <Parameter name="Convergence Tolerance" type="double"  value="${conv_tol}" />

  <Parameter name="Orthogonalization"     type="string"  value="${ortho}" />
  <ParameterList name="${ortho}">  
    <Parameter name="maxNumOrthogPasses"  type="int"  value="${ortho_passes}" />
  </ParameterList>
    <Parameter name="Verbosity"             type="int"  value="0" />
<!--
  <Parameter name="Output Frequency"  type="int"  value="10" />
  <Parameter name="Output Style"      type="int"  value="1" />
  <Parameter name="Verbosity"         type="int"  value="33" />
-->
</ParameterList>
EOM

cat >> $DRIVER_ENTRY <<- EOM

<!-- ----------------------------------------------------------------------------------------------- -->
  <ParameterList name="run${run_count}">
    <Parameter name="Pseudo Timesteps"          type="int"      value="2" />
    <Parameter name="Set Initial Residual"      type="bool"     value="true" />

    <!-- Define the solver to be used -->
    <Parameter name="Solver"                    type="string"   value="GMRES"/>
    <Parameter name="SolverParams"              type="string"   value="gold/GMRES/${FILE}"/>
    <Parameter name="SolverFactory"             type="string"   value="Belos"/>

    <!-- Add a preconditioner to the solver -->
    <Parameter name="Preconditioner"            type="string"   value="None"/>
  </ParameterList>
<!-- ----------------------------------------------------------------------------------------------- -->

EOM

if [ "${run_count}" == "" ]; then
  run_count=1;
else
  let run_count=run_count+1
fi

done
done
