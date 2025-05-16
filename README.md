# FRA631 Project: Automatic Robotic Arm (UR5e) System for Box Detection and Sorting

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation Guide](#installation-guide)
3. [UR5e Robot Information](#ur5e-robot-information)
4. [More Information](#more-information)

---

## Project Overview

This project is organized into several key modules:

* **UR5e\_Calibration:**
  Handles calibration between the UR5e robot arm and the camera to ensure precise positioning.  
  *See the [`README.md`](./UR5e_Calibration/README.md) in the `UR5e_Calibration` folder for more details.*

* **UR5e\_Trajectory\_Planning:**
  Responsible for generating smooth and efficient robot trajectories.  
  *See the [`README.md`](./UR5e_Trajectory_Planning/README.md) in the `UR5e_Trajectory_Planning` folder for more details.*

* **UR5e\_Task\_Planning:**
  Manages high-level task planning, enabling the UR5e to perform autonomous tasks.  
  *See the [`README.md`](./UR5e_Task_Planning/README.md) in the `UR5e_Task_Planning` folder for more details.*

---

## Installation Guide

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
>   Your terminal prompt should look like this:
>   ![Activated venv example](./images/bash_venv.png)

---

### Step 2: Install Requirements

* **For RealSense camera packages:**
  Please visit and follow the instructions in the official repository:
  [Intel RealSense SDK (librealsense)](https://github.com/IntelRealSense/librealsense?tab=readme-ov-file)

* **For development:**

  ```bash
  pip install -r dev-requirements.txt
  ```

* **For deployment on the robot:**

  ```bash
  pip install -r robot-requirements.txt
  ```

Details and instructions for running each process can be found in the `README.md` file of each subfolder.

---






## UR5e Robot Information

| Arm       | IP Address       |
| --------- | ---------------- |
| **Left**  | `192.168.200.10` |
| **Right** | `192.168.200.20` |




---

## More Information
This project is part of the Dual Arm Robotic Handover System (FRA 631) at KMUTT university, Thailand.
For more information, please visit:
[Dual Arm Robotic Handover System (FRA 631)](https://hcilab.net/uncategorized/dual-arm-robotic-handover-system-fra-631-foundation-robotic/)


