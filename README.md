# FRA631 Project Dual arm UR5
## UR5 Information

| Arm       | IP Address       |
|-----------|------------------|
| Left Arm  | 192.168.200.10   |
| Right Arm | 192.168.200.20   |


---

step 1 crate venv and activate it
```bash
set-executionpolicy -Scope CurrentUser -ExecutionPolicy Unrestricted
python -m venv venv
venv/Scripts/Activate.ps1
```
> [!WARNING]
> Make sure to activate the venv before installing the requirements
> ```bash
> venv/Scripts/Activate.ps1
> ```
> it shoud be like this
> ![bash venv](./images/bash_venv.png)

step 2 install requirements
```bash
pip3 install -r .\requirements.txt
```
