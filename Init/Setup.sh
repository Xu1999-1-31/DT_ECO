#!/bin/bash

# mo_gymnasium Installation Path
MO_GYM_PATH=$(python -c "import mo_gymnasium; print(mo_gymnasium.__path__[0])")

if [ -z "$MO_GYM_PATH" ]; then
    echo "mo_gymnasium not installed in current environment"
else
    echo "mo_gymnasium installation path: $MO_GYM_PATH"
fi
