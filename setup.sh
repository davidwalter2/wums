export WUMS_BASE=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
export PYTHONPATH="${WUMS_BASE}:$PYTHONPATH"

echo "Created environment variable WUMS_BASE=${WUMS_BASE}"