# copy the files from results folder, runs folder, and config.py file to the experiments folder and put it under a new file named [budget, prior_max_ply, opening_replacement_max_ply, pop_size_repertoires, pop_size_opponents, n_generations, tournament_size, crossover_rate, opponent_mutation_strength, novelty_weight, opponent_crossover_rate, opponent_mutation_rate... all possible variables] where this array should fetch values from the config.py [25, ....]

import os
import shutil
import src.config as config

def compile_data():
    # Create the experiments folder if it doesn't exist
    if not os.path.exists('experiments'):
        os.makedirs('experiments')

    # Create a new folder with the name based on the config variables
    folder_name = (
        f"{config.BUDGET}_{config.PRIOR_MAX_PLY}_{config.OPENING_REPLACEMENT_MAX_PLY}_"
        f"{config.POP_SIZE_REPERTOIRES}_{config.POP_SIZE_OPPONENTS}_{config.N_GENERATIONS}_"
        f"{config.TOURNAMENT_SIZE}_{config.CROSSOVER_RATE}_{config.MUTATION_RATE}_"
        f"{config.MUTATION_RETRIES}_{config.HOF_SIZE}_{config.OPPONENT_MUTATION_STRENGTH}_"
        f"{config.NOVELTY_WEIGHT}_{config.OPPONENT_CROSSOVER_RATE}_{config.OPPONENT_MUTATION_RATE}_"
        f"{config.LAMBDA_WEIGHT}_{config.MAIN_LAMBDA}"
    )
    folder_path = os.path.join('experiments', folder_name)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Copy the whole folders of results and runs folders to the new folder
    for folder in ['results', 'runs']:
        folder_destination = os.path.join(folder_path, folder)
        os.makedirs(folder_destination, exist_ok=True)
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                shutil.copy(file_path, folder_destination)
    
    # Copy the config.py file to the new folder
    shutil.copy('src/config.py', folder_path)

if __name__ == "__main__":
    compile_data()