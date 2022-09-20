#~bin/bash
Ym="model/"

GitDir="/root/data/model_test/WIL_model"
FileDir="$GitDir/model/"

cd $GitDir/

git pull origin main
git pull origin main --allow-unrelated-histories
git add model/
git commit -m "update model"
git push origin main

if [$? -eq 0 ];then
echo "> push complete"
fi

