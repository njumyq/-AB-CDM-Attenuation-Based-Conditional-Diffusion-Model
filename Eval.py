from DiffusionGuidence.TrainCondition import eval

def main(model_config=None):
    modelConfig = {
        "state": "eval", 
        "epoch": 300,  
        "batch_size": 4, 
        "T": 500,
        "channel": 48,
        "channel_mult": [1, 2, 2, 2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,    
        "multiplier": 2.5,
        "beta_1": 1e-4,
        "beta_T": 0.028,
        "grad_clip": 1.0, 

        "test_data_path": "./data/rock_500^3_2_100^3.h5",
        "save_dir": "./CheckpointsCondition/", 
        "test_load_weight": "/ckpt_299.pt", 
        "output_dir": "./runout/", 

    }
    if model_config is not None:
        modelConfig = model_config

    eval(modelConfig)

if __name__ == '__main__':
    main()
