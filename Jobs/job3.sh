#script for running init.sh
cd THCR

echo "Working in " `pwd`
#if restore is set to True dont run init.sh
restore=$(head  -1 conf.py)
if [ "$restore" == "restore=True" ]; then
    echo "This pass is being run with restoring so skipping init.sh";
else
sh init.sh
fi


echo job1 done

echo "Running Driver "
python Driver.py
echo  "Driver Ran"
