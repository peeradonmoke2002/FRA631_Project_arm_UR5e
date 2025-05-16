# FRA631 Project: Automatic Robotic Arm (UR5e) System for Box Detection and Sorting

## UR5 Robot Information

| Arm      | IP Address       |
| -------- | ---------------- |
| **Left** | `192.168.200.10` |

---

## Installation Guide

>[!CAUTION]
> ⚠️ **Supported Python Version:**
> This repository **requires Python 3.11**.
> Python 3.12 and newer are **not supported**.
> Please ensure your environment uses Python 3.11.

### Step 1: Create and Activate a Virtual Environment

**For Windows:**

```powershell
set-executionpolicy -Scope CurrentUser -ExecutionPolicy Unrestricted
python -m venv venv
venv\Scripts\Activate.ps1
```

**For Linux or macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

> **Important:**
> Always activate the virtual environment before installing any requirements.
>
> * **Windows:** `venv\Scripts\Activate.ps1`
> * **Linux/macOS:** `source venv/bin/activate`
>
> Your prompt should now look like this:
> ![Activated venv example](./images/bash_venv.png)

---

### Step 2: Install Requirements

* **For development:**

  ```bash
  pip install -r dev-requirements.txt
  ```

* **For deployment on the robot:**

  ```bash
  pip install -r robot-requirements.txt
  ```


  More information Plsese visit -> https://hcilab.net/uncategorized/dual-arm-robotic-handover-system-fra-631-foundation-robotic/
