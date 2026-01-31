#!/bin/bash

SVC_NAME="github-runner"
SVC_DESCRIPTION="GitHub Actions Runner"
RUNNER_ROOT=$(pwd)

CONF_PATH="/etc/supervisor/conf.d/${SVC_NAME}.conf"
SVC_CMD=$1
arg_2=${2}

if [ "$(id -u)" -ne 0 ]; then
    echo "Failed: This script requires to run with sudo / root." >&2
    exit 1
fi

function failed() {
    local error=${1:-Undefined error}
    echo "Failed: $error" >&2
    exit 1
}

function install() {
    echo "Creating Supervisor config in ${CONF_PATH}"
    
    run_as_user=${arg_2:-${SUDO_USER:-root}}
    echo "Run as user: ${run_as_user}"

    cat <<EOF > "${CONF_PATH}"
[program:${SVC_NAME}]
command=${RUNNER_ROOT}/bin/Runner.Listener run --startuptype service
directory=${RUNNER_ROOT}
user=${run_as_user}
autostart=true
autorestart=true
startretries=3
stdout_logfile=${RUNNER_ROOT}/_diag/runner.out.log
stderr_logfile=${RUNNER_ROOT}/_diag/runner.err.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=5
environment=HOME="/home/${run_as_user}",USER="${run_as_user}",http_proxy="${http_proxy:-}",https_proxy="${https_proxy:-}"
EOF

    chmod 644 "${CONF_PATH}"
    
    mkdir -p "${RUNNER_ROOT}/_diag"
    chown -R ${run_as_user} "${RUNNER_ROOT}/_diag"

    supervisorctl update || failed "failed to update supervisor"
    echo "Service installed and added to Supervisor."
}

function start() {
    supervisorctl start ${SVC_NAME}
}

function stop() {
    supervisorctl stop ${SVC_NAME}
}

function status() {
    supervisorctl status ${SVC_NAME}
    echo "--- Latest Logs ---"
    tail -n 20 "${RUNNER_ROOT}/_diag/runner.out.log"
}

function uninstall() {
    stop
    rm -f "${CONF_PATH}"
    supervisorctl update
    echo "Service uninstalled."
}

case $SVC_CMD in
   "install") install;;
   "start") start;;
   "stop") stop;;
   "status") status;;
   "uninstall") uninstall;;
   *) 
      echo "Usage: ./supervisvc.sh [install, start, stop, status, uninstall]"
      exit 1
      ;;
esac