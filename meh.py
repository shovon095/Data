sudo apt install -y cuda-drivers
Reading package lists... Done
Building dependency tree
Reading state information... Done
cuda-drivers is already the newest version (530.30.02-1).
You might want to run 'apt --fix-broken install' to correct these.
The following packages have unmet dependencies:
 nvidia-dkms-535 : Depends: nvidia-kernel-common-535 (<= 535.146.02-1) but it is not going to be installed
                   Depends: nvidia-kernel-common-535 (>= 535.146.02) but it is not going to be installed
 nvidia-driver-535 : Depends: nvidia-kernel-common-535 (<= 535.146.02-1) but it is not going to be installed
                     Depends: nvidia-kernel-common-535 (>= 535.146.02) but it is not going to be installed
 nvidia-kernel-common-530 : Depends: nvidia-kernel-common-535 but it is not going to be installed
E: Unmet dependencies. Try 'apt --fix-broken install' with no packages (or specify a solution).


