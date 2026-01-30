from NN import *
from Player import *
from Train import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def copy_player(policy_nn, critic_nn, policy_player, side):
    policy_nn_copy = PolicyNetwork().to(device)
    policy_nn_copy.load_state_dict(policy_nn.state_dict())

    critic_nn_copy = CriticNetwork().to(device)
    critic_nn_copy.load_state_dict(critic_nn.state_dict())

    policy_player_copy = NNPlayer(side, policy_nn_copy, critic_nn_copy)

    return policy_nn_copy, critic_nn_copy, policy_player_copy

print(torch.cuda.is_available())

policy_nn = PolicyNetwork().to(device)

# print("loaded policy checkpoint")
# policy_state_dict = torch.load("policy_checkpoint.pt")
# policy_nn.load_state_dict(policy_state_dict)

critic_nn = CriticNetwork().to(device)

# print("loaded critic checkpoint")
# critic_state_dict = torch.load("critic_checkpoint.pt")
# critic_nn.load_state_dict(critic_state_dict)

policy_player = NNPlayer(0, policy_nn, critic_nn)

policy_nn_copy, critic_nn_copy, policy_player_copy = copy_player(policy_nn, critic_nn, policy_player, 1)


# To display progress
# show(policy_player, policy_player_copy, 50)

epochs = 20
for epoch in range(epochs):
    
    print(f"epoch {epoch}")
    train(policy_player, policy_player_copy, 500, 1, 0.1)
    win_rate = pit(policy_player, policy_player_copy, 50)
    print(win_rate)
    if win_rate[0] - 5 >= win_rate[1]:
        torch.save(policy_nn.state_dict(), f"policy_checkpoint.pt")
        torch.save(critic_nn.state_dict(), f"critic_checkpoint.pt")
        print("Performed better than before, updating agent.")
        policy_nn_copy, critic_nn_copy, policy_player_copy = copy_player(policy_nn, critic_nn, policy_player, 1)
    else:
        print("Performed worse, Keep training.")
        # policy_nn, critic_nn, policy_player = copy_player(policy_nn_copy, critic_nn_copy, policy_player_copy)