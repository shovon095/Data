wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
--2025-05-14 22:23:32--  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
Resolving developer.download.nvidia.com (developer.download.nvidia.com)... 23.221.22.197, 23.221.22.207
Connecting to developer.download.nvidia.com (developer.download.nvidia.com)|23.221.22.197|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 4332 (4.2K) [application/x-deb]
Saving to: ‘cuda-keyring_1.0-1_all.deb’

cuda-keyring_1.0-1_all.deb                  100%[===========================================================================================>]   4.23K  --.-KB/s    in 0s

2025-05-14 22:23:33 (434 MB/s) - ‘cuda-keyring_1.0-1_all.deb’ saved [4332/4332]

shouvon@dgx1-DGX-Station:~$ sudo dpkg -i cuda-keyring_1.0-1_all.deb
Selecting previously unselected package cuda-keyring.
(Reading database ... 236541 files and directories currently installed.)
Preparing to unpack cuda-keyring_1.0-1_all.deb ...
Unpacking cuda-keyring (1.0-1) ...
Setting up cuda-keyring (1.0-1) ...
shouvon@dgx1-DGX-Station:~$ sudo apt update
E: Conflicting values set for option Signed-By regarding source https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /: /usr/share/keyrings/cuda-archive-keyring.gpg !=
E: The list of sources could not be read.
