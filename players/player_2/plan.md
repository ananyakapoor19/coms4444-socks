
# 2026-09-20 4:30pm

# Task: implement distribution-aware sock selection policy for player 2
Implement the sock selection policy described below 


    
- then, for our current `offered` socks for this turn, we 

### Window Size calculation
- Compute mean and std of the window size for the last self.running_window_size values
- Once the global history exceeds self.running_window_size, compute both metrics using Welford's formula 

### Sock selection policy
- compute the pair-wise embarassment scores of the socks we just pulled using `offered`
    - this will help us gauge the embarassment of our sock distribution over time
    - With the socks that are left we calculate all possible actions: discard all, discard first one and return the others to the drawer, discard second and return, discard two and leave others, etc.
- Select the pair with the minimum embarrassment and update global distribution metrics. The possible actions to chose from for the discard policy are the ones that branch from this chosen pair.

### sock discard policy
- After choosing the first two socks, we evaluate the rest of the possible actions by calculating the new metrics of those selected actions and we choose the one with the least std. 
- Return the selection where the ones we wear are the first two selected ones, and the ones discarded (if any) are the ones discarded in the selected action


### Where to implement
Make these chnages in players/player_2/player.py. You are allowed to create helper functions and edit other existing files. The policy entry point is in select_socks(). 

### Implementation write-up
Create an .md file outlining what changes you made, including what we originally had implemented incorrectly that you fixed