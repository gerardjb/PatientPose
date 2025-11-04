# PatientPose - using mediapipe tools to develop patient movement profiles
## Instructions for installation
- If you have not already done so, install miniconda or a similar environment and package manager that comes with a git client installation
- In a terminal manager, navigate to the directory where you would like to make the project installation with a command such as
```bash
cd path\to\your\preferred\installation\location
```
- Once there, use git to clone the project to your preferred location:
```bash
git clone https://github.com/gerardjb/PatientPose
```
  * Next, make a dedicated environment for the project with calls similar to:
  ```bash
  conda create -n patient_pose python=3.12
  conda activate patient_pose
  ```
  * Note that as of this writing (20251103), mediapipe does not support pyhton > 3.12
  * Finally, use pip to install the project using either a system install:
  ```bash
  pip install .[test]
  ```
  * or a local install if you'd like to make edits to the codebase with:
  ```bash
  pip install -e .[test]
  ```
- In each of the pip installs, I've included the optional test suite which includes a pytest unit test - once the install is done, you can immedialtely use pytest to make sure everything is running properly as:
```bash
pytest
``` 
  * If you see green text with "4 passed", your installation is good to go. If not, you can send the error codes my way.
## Running the script
- I'm keeping project scripts in the "scripts" directory in the project root. So, to call any scripts from the command line, go there:
```bash
cd scripts
```
- Then call any scripts with commands like:
```bash
python the_script_you_want.py
```
  * For example, the sample_patient_processing.py will (barring future deprecation which will be a TODO here) create a de-identified version of the input video (removal of features near all detected faces) and a csv that captures the keypoints across the pose and hand landmark detector models.
  * All such files will be populated to the "results" subdirectory of the project root for further inspection if desired.
- For now, you can test a local file using the same flags as in the previous version, such as:
```bash
python sample_patient_processing.py --filename "your_file.extension"
```
## Updating the version of the repo on your local machine
Git allows you to update to the latest version of the codebase as long as you're in the project root as:
```bash
git pull
```
- As long as you haven't made any cahnges to the codebase in the version you have on your machine, this should update your version without errors
- If it doesn't work...see me, and we'll talk. :)
