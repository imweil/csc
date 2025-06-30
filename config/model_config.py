policy_config = {

    "idle_worker_input_dim": 12,
    "idle_worker_hidden_dim": 64,
    "idle_worker_num_layers": 4,
    "idle_worker_num_heads": 8,

    "around_worker_input_dim": 12,
    "around_worker_hidden_dim": 64,
    "around_worker_num_layers": 4,
    "around_worker_num_heads": 8,

    "around_task_input_dim": 15,
    "around_task_hidden_dim": 64,
    "around_task_num_layers": 4,
    "around_task_num_heads": 8,

    "optimal_task_input_dim": 15,
    "optimal_task_hidden_dim": 64,
    "optimal_task_num_layers": 4,
    "optimal_task_num_heads": 8,

    "dropout": 0.1,

    "mlp_hidden_dim": 12
}

value_config = {

    "idle_worker_input_dim": 12,
    "idle_worker_hidden_dim": 64,
    "idle_worker_num_layers": 4,
    "idle_worker_num_heads": 8,

    "around_worker_input_dim": 12,
    "around_worker_hidden_dim": 64,
    "around_worker_num_layers": 4,
    "around_worker_num_heads": 8,

    "around_task_input_dim": 15,
    "around_task_hidden_dim": 64,
    "around_task_num_layers": 4,
    "around_task_num_heads": 8,

    "optimal_task_input_dim": 15,
    "optimal_task_hidden_dim": 64,
    "optimal_task_num_layers": 4,
    "optimal_task_num_heads": 8,

    "dropout": 0.1,

    "mlp_hidden_dim": 12
}

vae_config = {

    #   encoder

    "idle_worker_input_dim": 13,
    "idle_worker_hidden_dim": 64,
    "idle_worker_num_layers": 4,
    "idle_worker_num_heads": 8,

    "around_worker_input_dim": 12,
    "around_worker_hidden_dim": 64,
    "around_worker_num_layers": 4,
    "around_worker_num_heads": 8,

    "around_task_input_dim": 15,
    "around_task_hidden_dim": 64,
    "around_task_num_layers": 4,
    "around_task_num_heads": 8,

    "optimal_task_input_dim": 15,
    "optimal_task_hidden_dim": 64,
    "optimal_task_num_layers": 4,
    "optimal_task_num_heads": 8,

    "dropout": 0.1,

    "concat_dim": 88,

    #   decoder

    "decoder_latent_dim": 88,

    "decoder_hidden_dim": 88,
    "decoder_output_dim": 1,

}


learning_config = {

    "actor_lr": 0.0001,
    "critic_lr": 0.0001,

    "vae_lr": 0.0001,

    "reward_lr": 0.0001,

    "cmi_lr": 0.0001,

    "gamma": 0.99,
    "eps": 0.2,
    "lmbda": 0.95,

    "eta":0.1,

    "epochs": 5

}
