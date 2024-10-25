#!/bin/bash
#
#
nohup prefect server start &> prefect_server.log &
nohup prefect agent start -p 'default-agent-pool' &> prefect_agent.log &

echo "Services started"
