# FRA631 Project arm UR5e
## UR5 Information

| Arm       | IP Address       |
|-----------|------------------|
| Left Arm  | 192.168.200.10   |



---

## Installation

>[!CAUTION]
> This repository supports only Python 3.11 and does not support Python 3.12 or above.
> Please make sure you are using python 3.11

step 1 create venv and activate it

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
> # for windows
> venv/Scripts/Activate.ps1
> # for linux or mac
> source venv/bin/activate
> ```
> it should be like this
> ![bash venv](./images/bash_venv.png)

step 2 install requirements

If you are developer 
```bash
pip install -r dev-requirements.txt 
```

If you are installing on a robot
```bash
pip install -r robot-requirements.txt
```




