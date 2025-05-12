import pathlib

### Task parameters
DATA_DIR = 'RLBench_ACT/Datasets'
SIM_TASK_CONFIGS = {
    'sim_transfer_cube_scripted':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_transfer_cube_human':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_human',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'change_channel': {
        'dataset_dir': DATA_DIR + '/change_channel',
        'num_episodes': 5,
        'episode_len': 50,
        'camera_names': ['top']
    },

    'reach_target': {
        'dataset_dir': DATA_DIR + '/reach_target',
        'num_episodes': 50,
        'episode_len': 50,
        'camera_names': ['top']
    },
    
    # RLBench
    'sorting_program5':{ 
        'dataset_dir': DATA_DIR + '/sorting_program5/variation0',
        'episode_len': 90,
        'num_episodes': 50,
        'num_variation': 1,
        'camera_names': ['wrist'],
    },
    
    'push_button':{ 
        'dataset_dir': DATA_DIR + '/push_button/variation0', 
        'episode_len': 30, 
        'num_episodes': 50,
        'num_variation': 1,
        'camera_names': ['wrist']
    },
    
    'basketball_in_hoop':{ 
        'dataset_dir': DATA_DIR + '/basketball_in_hoop/variation0', 
        'episode_len': 59,
        'num_episodes': 50,
        'num_variation': 1,
        'camera_names': ['wrist']
    },
    'beat_the_buzz':{
            'dataset_dir': DATA_DIR + '/basketball_in_hoop/variation0',
            'episode_len': 59,
            'num_episodes': 50,
            'num_variation': 1,
            'camera_names': ['wrist']
        },
    
    'phone_on_base':{ 
        'dataset_dir': DATA_DIR + '/phone_on_base/variation0', 
        'episode_len': 60, 
        'num_episodes': 50,
        'num_variation': 1,
        'camera_names': ['wrist']
    },
    
    'light_bulb_out':{ 
        'dataset_dir': DATA_DIR + '/light_bulb_out/variation0', 
        'episode_len': 60, 
        'num_episodes': 50,
        'num_variation': 1,
        'camera_names': ['wrist']
    },
    
    'meat_on_grill':{ 
        'dataset_dir': DATA_DIR + '/meat_on_grill/variation0', 
        'episode_len': 70, 
        'num_episodes': 50,
        'num_variation': 1,
        'camera_names': ['wrist']
    },

    'block_pyramid': {
        'dataset_dir': DATA_DIR + '/block_pyramid',
        'num_episodes': 50,
        'episode_len': 50,
        'camera_names': ['top']
    },

    'close_laptop_lid': {
    'dataset_dir': DATA_DIR + '/close_laptop_lid',
    'num_episodes': 50,
    'episode_len': 50,
    'camera_names': ['top']
    },
    'wipe_desk': {
        'dataset_dir': DATA_DIR + '/wipe_desk',
        'num_episodes': 50,
        'episode_len': 50,
        'camera_names': ['top']
    },
}

### Simulation envs fixed constants
DT = 0.02
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]

XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083
MASTER_GRIPPER_JOINT_CLOSE = -0.6842
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2
