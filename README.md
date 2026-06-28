# Isaac Lab Task Workspace

## Overview

This repository is a small Isaac Lab workspace for custom tasks and shared debugging tools.
It keeps task code outside of the core Isaac Lab repository while leaving room for more tasks later.

**Layout:**

- `custom_tasks/object_in_bowl` contains the current custom Isaac Lab task package.
- `scripts` contains repo-level training, play, and smoke-test helpers.
- `tools` contains shared debugging utilities, including the browser debug viewer.

**Keywords:** extension, workspace, isaaclab, reinforcement-learning

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda or uv installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e custom_tasks/object_in_bowl/source/object_in_bowl
    ```

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: If the task name changes, it may be necessary to update the search pattern `"Isaac-Object-In-Bowl"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME> --max_steps=25
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME> --max_steps=25
            ```

        For this project, the first headless smoke test should be:

        ```bash
        python scripts/random_agent.py --task=Isaac-Object-In-Bowl-Franka-v0 --num_envs=1 --headless --max_steps=25
        ```

    - Inspecting the task with the debug viewer:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python tools/debug_viewer.py \
            --task=Isaac-Object-In-Bowl-Franka-v0 \
            --viewer-mode setup \
            --headless \
            --serve \
            --host=0.0.0.0 \
            --port=8080
        ```

        To inspect a trained RSL-RL checkpoint instead of zero/random setup actions:

        ```bash
        python tools/debug_viewer.py \
            --task=Isaac-Object-In-Bowl-Franka-v0 \
            --viewer-mode policy \
            --run-dir logs/rsl_rl/object_in_bowl_franka/<run-name> \
            --headless \
            --serve \
            --host=0.0.0.0 \
            --port=8080
        ```

        You can also pass an exact checkpoint with `--checkpoint <path-to-model.pt>`.
        If you use `--dump-json`, keep those dumps out of git because they can include
        local run/checkpoint paths.

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in
`custom_tasks/object_in_bowl/source/object_in_bowl/object_in_bowl/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to `custom_tasks/object_in_bowl/source`.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/custom_tasks/object_in_bowl/source/object_in_bowl"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```
