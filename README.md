# POLICY EVALUATION

## AIM
To develop a Python program to evaluate the given policy.

## PROBLEM STATEMENT
The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.

#### States
The environment has 7 states:

Two Terminal States: G: The goal state & H: A hole state.
Five Transition states / Non-terminal States including S: The starting state.
#### Actions
The agent can take two actions:

R: Move right.
L: Move left.
#### Transition Probabilities
The transition probabilities for each action are as follows:

50% chance that the agent moves in the intended direction.
33.33% chance that the agent stays in its current state.
16.66% chance that the agent moves in the opposite direction.
For example, if the agent is in state S and takes the "R" action, then there is a 50% chance that it will move to state 4, a 33.33% chance that it will stay in state S, and a 16.66% chance that it will move to state 2.

#### Rewards
The agent receives a reward of +1 for reaching the goal state (G). The agent receives a reward of 0 for all other states.

#### Graphical Representation
![](gr.png)


## POLICY EVALUATION FUNCTION
![](pef.png)

## PROGRAM:

```
Name: Pavan Kishore.M
Reg No: 212221230076
```

```
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)
# code  to evaluate the given policy
    while True:
      V=np.zeros(len(P),dtype=np.float64)
      for s in range(len(P)):
        for prob, next_state, reward, done in P[s][pi(s)]:
          V[s]+=prob*(reward+gamma+prev_V[next_state]*(not done))
      if np.max(np.abs(prev_V-V))<theta:
        break
      prev_V=V.copy()
      return V

# Code to evaluate the first policy
V1 = policy_evaluation(pi_1, P,gamma=0.99)
print_state_value_function(V1, P, n_cols=7, prec=5)

# Code to evaluate the second policy
V2 = policy_evaluation(pi_2, P)
print_state_value_function(V2, P, n_cols=7, prec=5)

# Comparing the two policies
if(np.sum(V1>=V2)==7):
  print("The first policy is the better policy")
elif(np.sum(V2>=V1)==7):
  print("The second policy is the better policy")
else:
  print("Both policies have their merits.")

```

## OUTPUT:
### POLICY 1
![o1 (1)](https://github.com/pavankishore-AIDS/rl-policy-evaluation/assets/94154941/187d0a5f-5046-402c-8803-bb91571d2459)


![o2 (1)](https://github.com/pavankishore-AIDS/rl-policy-evaluation/assets/94154941/6f8e307a-a337-42e7-bea6-e07bae78ad2c)


### POLICY 2
![o3 (1)](https://github.com/pavankishore-AIDS/rl-policy-evaluation/assets/94154941/1dbcd3ce-7e9f-4609-9274-f989fa13182b)

![o4](https://github.com/pavankishore-AIDS/rl-policy-evaluation/assets/94154941/e13b71a8-96c9-4e38-967c-352f90dfc88e)

### COMPARISON:
![o5](https://github.com/pavankishore-AIDS/rl-policy-evaluation/assets/94154941/4c3cbbba-2460-4ed4-82f9-b92b7a6a9da1)


### CONCLUSION:
![Uploading o6.png…]()


## RESULT:
Thus, a Python program is developed to evaluate the given policy.
