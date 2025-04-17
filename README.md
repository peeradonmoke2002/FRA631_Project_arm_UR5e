# FRA631 Project Dual arm UR5
## UR5 Information

| Arm       | IP Address       |
|-----------|------------------|
| Left Arm  | 192.168.200.10   |
| Right Arm | 192.168.200.20   |


---

step 1 crate venv and activate it

For Windows
```bash
set-executionpolicy -Scope CurrentUser -ExecutionPolicy Unrestricted
python -m venv venv
venv/Scripts/Activate.ps1
```

For Linux or Mac

```bash
python3 -m venv venv
source venv/bin/activate
```


> [!WARNING]
> Make sure to activate the venv before installing the requirements
> ```bash
# for windows
> venv/Scripts/Activate.ps1
# for linux or mac
> source venv/bin/activate
> ```
> it shoud be like this
> ![bash venv](./images/bash_venv.png)

step 2 install requirements

If you are developer 
```bash
./install_requirements.sh dev
```

If you are install in robot
```bash
./install_requirements.sh robot
```


## Note
For update submodules

```bash
git submodule update --remote
```

