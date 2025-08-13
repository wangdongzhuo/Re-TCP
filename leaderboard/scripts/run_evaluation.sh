#!/bin/bash
export CARLA_ROOT=/home/ly/WDZ/TCP/carla
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner
export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=2000
export TM_PORT=8000
export DEBUG_CHALLENGE=0
export REPETITIONS=3 # multiple evaluation runs
#export REPETITIONS=1 # multiple evaluation runs
export RESUME=True

# TCP_Ablation_Baseline evaluation
export ROUTES=leaderboard/data/evaluation_routes/routes_town05_long.xml
#export ROUTES=leaderboard/data/evaluation_routes/routes_longest6.xml
#export ROUTES=leaderboard/data/evaluation_routes/routes_lav_valid.xml
#export ROUTES=leaderboard/data/validation_routes/routes_town05_short.xml
export TEAM_AGENT=team_code/tcp_agent_selftry.py #第三个创新
#export TEAM_AGENT=team_code/tcp_agent.py #第三个创新
#export TEAM_CONFIG=/home/ly/WDZ/TCP/log/TCP_Baseline+SE+CBAM_198K/best_epoch=32-val_loss=0.757.ckpt #198K
export TEAM_CONFIG=/home/ly/WDZ/TCP/log/TCP_Baseline+SE+CBAM_410K/best_epoch=56-val_loss=0.463.ckpt #410K
#export TEAM_CONFIG=/home/ly/WDZ/TCP/log/TCP/best_epoch=53-val_loss=0.688.ckpt #CBAM-only
#export TEAM_CONFIG=/home/ly/WDZ/TCP/log/TCP_Baseline_198K/best_epoch=50-val_loss=0.708.ckpt #Baseline
export CHECKPOINT_ENDPOINT=7-3Town5Long410_ours_30.json
export SCENARIOS=leaderboard/data/scenarios/all_towns_traffic_scenarios_no256.json
#export SCENARIOS=leaderboard/data/scenarios/longest6_eval_scenarios.json
#export SCENARIOS=leaderboard/data/scenarios/all_towns_traffic_scenarios.json
export SAVE_PATH=data/7-3Town5Long410_ours_30/

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT}


