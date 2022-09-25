#~/bin/bash
/opt/conda/bin/python /root/data/model_test/WIL_model/exe/preprocessing.py >> /var/log/pre.log 2>&1
rm -rf /root/data/model_test/WIL_model/data/*.json
rm -rf /root/data/model_test/WIL_model/data/txtdata/*.txt
