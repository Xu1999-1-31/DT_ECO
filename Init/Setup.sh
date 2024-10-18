#!/bin/bash

# mo_gymnasium Installation Path
MO_GYM_PATH=$(python -c "import mo_gymnasium; print(mo_gymnasium.__path__[0])")

if [ -z "$MO_GYM_PATH" ]; then
    echo "mo_gymnasium not installed in current environment"
else
    echo "mo_gymnasium installation path: $MO_GYM_PATH"

    # Check if ../dt_eco1 directory exists
    if [ -d "../dt_eco1" ]; then
        # Create the envs directory if it does not exist
        if [ ! -d "$MO_GYM_PATH/envs" ]; then
            mkdir -p "$MO_GYM_PATH/envs"
            echo "Created envs directory at $MO_GYM_PATH/envs"
        fi

        # Copy ../dt_eco1 to $MO_GYM_PATH/envs
        cp -r ../dt_eco1 "$MO_GYM_PATH/envs"
        echo "Copied ../dt_eco1 to $MO_GYM_PATH/envs"

        # Check if the __init__.py file exists in the envs directory
        if [ -f "$MO_GYM_PATH/envs/__init__.py" ]; then
            # Rename the original __init__.py to __init__.py.bak
            mv "$MO_GYM_PATH/envs/__init__.py" "$MO_GYM_PATH/envs/__init__.py.bak"
            echo "Renamed original __init__.py in "$MO_GYM_PATH/envs/" to __init__.py.bak"
        fi

        # Check if the __init__.py file exists in the current directory
        if [ -f "./__init__.py" ]; then
            # Copy __init__.py to the envs directory
            cp ./__init__.py "$MO_GYM_PATH/envs"
            echo "Copied __init__.py to $MO_GYM_PATH/envs"
        else
            echo "__init__.py not found in current directory"
        fi
    else
        echo "../dt_eco1 directory does not exist"
    fi
fi