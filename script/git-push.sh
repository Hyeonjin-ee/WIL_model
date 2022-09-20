#~bin/bash
Ym="model/"

GitDir="/root/data/model_test/WIL_model"
FileDir="$GitDir/model/"
today=`date`


cd $GitDir/

git pull origin main
git pull origin main --allow-unrelated-histories
git rm --cached ./model/*.bin
git add model/*.bin
git commit -m "update model $today"
git push origin main



