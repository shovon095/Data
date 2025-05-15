wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
--2025-05-14 22:13:47--  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
Resolving developer.download.nvidia.com (developer.download.nvidia.com)... 23.221.22.197, 23.221.22.207
Connecting to developer.download.nvidia.com (developer.download.nvidia.com)|23.221.22.197|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 1650 (1.6K) [application/octet-stream]
Saving to: ‘3bf863cc.pub’

3bf863cc.pub                                100%[===========================================================================================>]   1.61K  --.-KB/s    in 0s

2025-05-14 22:13:47 (316 MB/s) - ‘3bf863cc.pub’ saved [1650/1650]

shouvon@dgx1-DGX-Station:~$ sudo bash -c 'echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
shouvon@dgx1-DGX-Station:~$ sudo apt update
Get:1 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  InRelease [1,581 B]
Hit:2 http://archive.ubuntu.com/ubuntu bionic InRelease
Hit:3 http://archive.ubuntu.com/ubuntu bionic-updates InRelease
Hit:4 https://download.docker.com/linux/ubuntu bionic InRelease
Err:1 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  InRelease
  The following signatures couldn't be verified because the public key is not available: NO_PUBKEY A4B469963BF863CC
Hit:6 http://ppa.launchpad.net/graphics-drivers/ppa/ubuntu bionic InRelease
Hit:7 http://security.ubuntu.com/ubuntu bionic-security InRelease
Get:5 https://international.download.nvidia.com/dgxstation/repos/bionic bionic InRelease [7,114 B]
Err:5 https://international.download.nvidia.com/dgxstation/repos/bionic bionic InRelease
  The following signatures couldn't be verified because the public key is not available: NO_PUBKEY 208CE844D9F220AD
Reading package lists... Done
W: GPG error: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  InRelease: The following signatures couldn't be verified because the public key is not available: NO_PUBKEY A4B469963BF863CC
E: The repository 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  InRelease' is not signed.
N: Updating from such a repository can't be done securely, and is therefore disabled by default.
N: See apt-secure(8) manpage for repository creation and user configuration details.
W: An error occurred during the signature verification. The repository is not updated and the previous index files will be used. GPG error: https://international.download.nvidia.com/dgxstation/repos/bionic bionic InRelease: The following signatures couldn't be verified because the public key is not available: NO_PUBKEY 208CE844D9F220AD
