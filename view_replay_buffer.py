import pickle

buffer_path = 'models/replay_buffer.pkl'

with open(buffer_path, 'rb') as f:
    replay_buffer = pickle.load(f)

print(f"Số lượng trải nghiệm trong buffer: {len(replay_buffer)}")
print("Một số trải nghiệm đầu tiên:")
for i, exp in enumerate(replay_buffer[:5]):
    state, action, reward, next_state, done = exp
    print(f"Trải nghiệm {i+1}:")
    print(f"  State: {state}")
    print(f"  Action: {action}")
    print(f"  Reward: {reward}")
    print(f"  Next state: {next_state}")
    print(f"  Done: {done}\n")
