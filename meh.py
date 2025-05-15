 sudo apt update
Hit:2 https://download.docker.com/linux/ubuntu bionic InRelease
Hit:3 http://ppa.launchpad.net/graphics-drivers/ppa/ubuntu bionic InRelease
Hit:4 http://security.ubuntu.com/ubuntu bionic-security InRelease
Hit:5 http://archive.ubuntu.com/ubuntu bionic InRelease
Hit:6 http://archive.ubuntu.com/ubuntu bionic-updates InRelease
Get:7 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  InRelease [1,581 B]
Get:1 https://international.download.nvidia.com/dgxstation/repos/bionic bionic InRelease [7,114 B]
Err:1 https://international.download.nvidia.com/dgxstation/repos/bionic bionic InRelease
  The following signatures couldn't be verified because the public key is not available: NO_PUBKEY 208CE844D9F220AD
Get:8 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  Packages [1,373 kB]
Fetched 1,382 kB in 2s (862 kB/s)
Reading package lists... Done
Building dependency tree
Reading state information... Done
10 packages can be upgraded. Run 'apt list --upgradable' to see them.
W: An error occurred during the signature verification. The repository is not updated and the previous index files will be used. GPG error: https://international.download.nvidia.com/dgxstation/repos/bionic bionic InRelease: The following signatures couldn't be verified because the public key is not available: NO_PUBKEY 208CE844D9F220AD
W: Failed to fetch http://international.download.nvidia.com/dgxstation/repos/bionic/dists/bionic/InRelease  The following signatures couldn't be verified because the public key is not available: NO_PUBKEY 208CE844D9F220AD
W: Some index files failed to download. They have been ignored, or old ones used instead.
shouvon@dgx1-DGX-Station:~$ sudo apt install -y nvidia-driver-515
Reading package lists... Done
Building dependency tree
Reading state information... Done
Some packages could not be installed. This may mean that you have
requested an impossible situation or if you are using the unstable
distribution that some required packages have not yet been created
or been moved out of Incoming.
The following information may help to resolve the situation:

The following packages have unmet dependencies:
 nvidia-driver-515 : Depends: libnvidia-gl-515 (= 515.105.01-0ubuntu1) but it is not going to be installed
                     Depends: libnvidia-encode-515 (= 515.105.01-0ubuntu1) but it is not going to be installed
                     Depends: nvidia-utils-515 (= 515.105.01-0ubuntu1) but it is not going to be installed
                     Depends: xserver-xorg-video-nvidia-515 (= 515.105.01-0ubuntu1) but it is not going to be installed
                     Depends: libnvidia-cfg1-515 (= 515.105.01-0ubuntu1) but it is not going to be installed
                     Depends: libnvidia-fbc1-515 (= 515.105.01-0ubuntu1) but it is not going to be installed
E: Unable to correct problems, you have held broken packages.


