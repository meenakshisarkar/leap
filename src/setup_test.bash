#Code developed by Meenakshi Sarkar, IISc for the project pgvg
export IMG_H=64
export IMG_W=64
export N_SAMPLES=20
export PAST_TS=5
# export PAST_TS=10  
export FUTURE_TS=10
export TEST_FUTURE_TS=20 
export C_DIM=3 
# export H_DIM=128 #svg paper
export H_DIM=256
export HIDDEN_DIM_PRIOR=512
export HIDDEN_DIM_PRED=512
export hidden_dim_priorA=32 

export A_DIM=2
export alpha_dim=8
export BATCH_SZ=64
export EPOCHS=300
export EPOCHS_CRITIC=1000
export EPOCHS_PGVG=1500
export GPU_ID=0
export FILTERS=32
export BETA=0.0001 #for bair 0.00001 for svg-leap
export A_BETA=0.0001 #for the action loss
export GAMMA=0.0001
export RELOAD_CKPT=False
# export G_TRAIN=True
export G_TRAIN=False
export C_TRAIN=False
export h_dimA=16
export A_RNN=32
export CKPT_G=285
export CKPT_P=299
export CKPT_C=191
export TEST_G=True
# export TEST_G=False
export LATENT_DIM=128 

export DATASET='roam'
# export model_name='svg'
export model_name='svg-leap'
# export model_name='causal-leap'
# export DATASET='kth'
export DATAPATH="../../../catkin_ws/RoAM_dataset/processed/train"
# python main_run.py | tee -a log.txt