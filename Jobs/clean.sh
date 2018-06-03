#This script for cleaning all input adn output job files
echo "Cleaning the below run files"
ls *.sh.[oe]*
rm *.sh.[oe]*

echo "cleaning pycache files "
rm  ../__pycache__/ -rf
rm ../*/__pycache__/ -rf
