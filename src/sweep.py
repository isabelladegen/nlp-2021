import wandb
from tune_model_parameters_experiment import train
from src.evaluate import SCORE_F1

if __name__ == '__main__':
    sweep_config_grid = {
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

    sweep_config_bayes = {
        'name': 'test',
        'method': 'bayes',
        'metric': {
            'goal': 'maximize',
            'name': SCORE_F1
        },
        'parameters': {
            'vector_size': {
                'values': [20, 30, 70, 80, 100]
            },
            'epochs': {
                'values': [40, 50, 100, 150, 200]
            },
            'dm': {
                'values': [0, 1]
            },
        },
        'early_terminate': {
            'type': 'hyperband',
            's': 2,
            'eta': 3,
            'max_iter': 27
        }
    }

    sweep_id = wandb.sweep(sweep_config_bayes, project="test")
    wandb.agent(sweep_id, function=train)
