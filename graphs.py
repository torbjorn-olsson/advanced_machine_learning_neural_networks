import pickle
import matplotlib.pyplot as pp

with open(f'rev{1000}', 'rb') as f:
    rewards1 = pickle.load(f)

with open(f'revavg{1000}', 'rb') as f:
    avg_rewards1 = pickle.load(f)

with open(f'rev{10000}', 'rb') as f:
    rewards2 = pickle.load(f)

with open(f'revavg{10000}', 'rb') as f:
    avg_rewards2 = pickle.load(f)

with open(f'rev{200000}', 'rb') as f:
    rewards3 = pickle.load(f)

with open(f'revavg{200000}', 'rb') as f:
    avg_rewards3 = pickle.load(f)


pp.subplot(2, 3, 1)

pp.tight_layout(pad=3.5)

pp.plot(rewards1)
pp.title('Rewards: 1a')
pp.xlabel('Episode')
pp.ylabel('Reward')

pp.subplot(2, 3, 4)
pp.title('Avg. rewards: 1a')
pp.plot(avg_rewards1)
pp.xlabel('Episode / 10')
pp.ylabel('Avg. reward')


pp.subplot(2, 3, 2)
pp.plot(rewards2)
pp.title('Rewards: 2a')
pp.xlabel('Episode')
pp.ylabel('Reward')

pp.subplot(2, 3, 5)
pp.title('Avg. rewards: 2a')
pp.plot(avg_rewards2)
pp.xlabel('Episode / 10')
pp.ylabel('Avg. reward')


pp.subplot(2, 3, 3)
pp.plot(rewards3)
pp.title('Rewards: 1c')
pp.xlabel('Episode')
pp.ylabel('Reward')

pp.subplot(2, 3, 6)
pp.title('Avg. rewards: 1c')
pp.plot(avg_rewards3)
pp.xlabel('Episode / 10')
pp.ylabel('Avg. reward')

pp.show()