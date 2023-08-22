#!/bin/bash

WORKSPACE=/opt/tiger/placeholder
echo "==> Go to '${WORKSPACE}'"
cd ${WORKSPACE}

# Run the environment configuration script
source ./arnold_before.sh || true

# If the cluster is in US, then proxy is not needed.
if [[ $ARNOLD_MONITOR_CLUSTER != cloudnative-maliva ]]; then
    export http_proxy=http://sys-proxy-rd-relay.byted.org:3128 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=.byted.org
fi

pip install -r ./requirements.txt

if [[ $ARNOLD_MONITOR_CLUSTER != cloudnative-maliva ]]; then
    unset http_proxy && unset https_proxy && unset no_proxy
fi

# Write your ssh keys
echo $SSH_LOGIN_KEYS >> ~/.arnold_ssh/authorized_keys;
echo $SSH_LOGIN_KEYS >> ~/.ssh/authorized_keys;

# Run the command
echo "==> Run command '${@}'"
${@}
