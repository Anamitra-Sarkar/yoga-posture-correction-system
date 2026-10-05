#!/bin/bash
cd /home/anamitra/yoga_posture_workspace/modal; export PATH=$PATH:~/.local/bin
until [ "$(for k in 0 1 2 3 4 all; do modal volume ls asanaai-data runs/cueT_mlp_$( [ $k = all ] && echo all || echo fold$k ) 2>/dev/null | grep -c mlp_3head_model_v2.pth; done | paste -sd+ | bc)" = "6" ]; do sleep 20; done
sleep 10
echo "##### TARGET-POSE MLP: out-of-fold on the 422 independent Commons photos"
modal run asanaai_train2.py --stage oof --tag cueT_mlp_fold --csv-rel cueT_commons 2>&1 | grep -E "^==|^   " | cut -c1-200
echo "##### TARGET-POSE MLP: held-out public photos (all-data model)"
modal run asanaai_train2.py --stage pub --tag cueT_mlp_all 2>&1 | grep -E "^==|^   " | cut -c1-200
