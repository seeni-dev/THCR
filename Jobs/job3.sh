#script for running init.sh
cd THCR

echo "Working in " `pwd`
sh init.sh
echo job1 done

echo "Running Driver "
python Driver.py
echo  "Driver Ran"

touch RanFlag.file