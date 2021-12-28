import wandb
from tune_model_parameters_experiment import train

if __name__ == '__main__':
    sweep_config = {
        'name': 'test',
        'method': 'grid',
        'parameters': {
            'vector_size': {
                'values': [5, 30, 70]
            },
            'epochs': {
                'values': [10, 50, 100]
            },
            'dm': {
                'values': [0, 1]
            }
        }
    }

    # vector_size=10,
    # window=4,
    # min_count=1,
    # workers=4,
    # dm=1,
    # epochs=30
    sweep_id = wandb.sweep(sweep_config, project="test")
    wandb.agent(sweep_id, function=train)
