#!/bin/bash
# Target-pose ST-GCN: windows (24 videos, targets only) -> v2 augmentation (mirror + photo-hold clips) -> 3 held-out-video folds + all
cd /home/anamitra/yoga_posture_workspace/modal; export PATH=$PATH:~/.local/bin
T="downward_dog,warrior_2,tree_pose,triangle,seated_easy_pose,child_pose,corpse,plank"
modal run asanaai_train2.py --stage windows --targets "$T" --out-prefix cueT > final_win1.log 2>&1 || exit 1
modal run asanaai_train2.py --stage windows2 --targets "$T" --in-prefix cueT --out-prefix cueT2 > final_win2.log 2>&1 || exit 1
for f in f0 f1 f2 all; do
  nohup modal run --detach asanaai_train2.py --stage stgcn --feats-tag cueT2_$f --tag cueT2_stgcn_$f > train_cueT2_stgcn_$f.log 2>&1 &
  sleep 4
done
echo LAUNCHED
