# Broad Agency Announcements (BAA) U.S. Army Research Project
# "Collective Planning and Leadership for the U.S. Army"
#
# Agent-based simulation model
# Shun Cao (scao12@binghamton.edu)
#
# Copyright (c) 2019 by Shun Cao


from pylab import *
import operator
from sklearn.neighbors import KNeighborsRegressor


problem_dim=2  # dimensions of problem space
l = 5  # number of coexisting frequencies
number_of_iteration = 300
alpha = 1.3
search_radius = 0.005
beta = 0.05
norm = 2
penalty = 0.2
bonus = 0.4
cauchy = 3
partition_a = -0.045
partition_b = 0.56
initial_tolerance = 0.5

number_of_agent = 4
intelligence_4 = [0.50, 0.94, 0.87, 0.23]
credibility_4 = [0.81, 0.34, 0.68, 0.15]
talkativeness_4 = [0.51, 0.33, 0.62, 0.72]


#  the true utility function
w = [7.57706733, 0.0689787, 3.2692918, 3.264955, 6.2233866, 2.63043152, 9.20947695, 2.42213212, 6.47712788,7.04956148]
coef = [0.9914310467996582, 0.08719498081379773, 0.08193242073323459, 0.7966082310872893, 0.24413385080004923, 0.8485074577156798, 0.3170434875061522, 0.47060980793378127, 0.9099052134009824, 0.6702420558463249, 0.27795303163433804, 0.8413839340475718, 0.29767624953648153, 0.6792877160064543, 0.9379655509348522, 0.4208920003070794, 0.4378265136768884, 0.2503626910983201, 0.7469642559557864, 0.8294142282473057, 0.335666596264552, 0.7977542966245004, 0.23843358324769437, 0.751816628306705, 0.32048603357245775, 0.8723786221094905, 0.35434346424920693, 0.022747425444053504, 0.6557598232110744, 0.826144072020447]
def true_U(plan):
    tem = 0
    tem_list = []
    for i in range(problem_dim):
        Tem = []
        for j in range(l):
            tem += sin(w[j + i * l] * plan[i])*coef[j+l*i]
            Tem.append(sin(w[j + i * l] * plan[i]))
        tem_list.append(Tem)
    for i in range(len(tem_list)-1):
        for j in range(len(Tem)):
            tem += tem_list[i][j]*tem_list[i+1][j]*coef[2*l*problem_dim + j + l*i]
    return tem



def min_distance(agent, plan):
    dist_list = []
    op = list(agent.plan_dictionary.keys())
    for i in range(len(op)):
        dist_list.append(sqrt(sum([abs((op[i][j] - plan[j])) ** norm for j in range(problem_dim)])))
    return min(dist_list)

def individual_U(agent, plan):
    X = list(agent.plan_dictionary.keys())
    Y = list(agent.plan_dictionary.values())
    neigh = KNeighborsRegressor(n_neighbors=2)
    neigh.fit(X, Y)
    return neigh.predict([plan])[0]


def local_search(agent):
    new_plan =[]
    for i in range(problem_dim):
        new_plan.append(uniform(agent.best_plan[i]-search_radius, agent.best_plan[i]+search_radius))
    #  update plan_dictionary
    agent.plan_dictionary[tuple(new_plan)] = individual_U(agent, new_plan)
    if individual_U(agent, new_plan) > agent.plan_dictionary[agent.best_plan]:
        #  update best plan
        agent.best_plan = tuple(new_plan)


def aspect_selection(speaker):
    opinion = {}
    for i in range(problem_dim):
        tem = list(group_plan)
        tem[i] = speaker.best_plan[i]
        opinion[tuple(tem)] = individual_U(speaker, tem)
        Max = max(opinion.items(), key=lambda kv: kv[1])
    return Max[0], Max[1]



###  creat agent
class agent:
    pass

### observiation
group_plan_score = []
group_plan_list = []
agent_0_plan = []
agent_1_plan = []
agent_2_plan = []
agent_3_plan = []


perceived_leadership = {}

agent_acceptance = {}
agent_rejection = {}


### initialize
group_plan = tuple(uniform(0,1,problem_dim))
#  for observation
group_plan_score.append(true_U(group_plan))
group_plan_list.append(group_plan)

agents = []

for i in range(number_of_agent):
    ag = agent()

    #  add leadership skills
    ag.intelligence = intelligence_4[i]
    ag.acceptance = 0
    ag.rejection = 0
    ag.credibility = credibility_4[i]
    #  add initial opinions
    initial_number_of_opinions = int(ag.intelligence * 20) + 1
    ag.initial_opinions = [uniform(0, 1, problem_dim) for i in range(initial_number_of_opinions)]
    #  add extroversion
    ag.talkativeness = talkativeness_4[i]
    #  leadership evaluation is initially a 0-list, except self evaluation which is a random value
    ag.perceived_leadership = [0.]*number_of_agent

    #  add plans
    ag.plan_dictionary = {tuple(i): (true_U(i) + true_U(i) * uniform(ag.intelligence - 1, 1 - ag.intelligence)) for i in ag.initial_opinions}
    ag.best_plan = tuple(max(ag.plan_dictionary.items(), key=operator.itemgetter(1))[0])

    #  add agent into network
    agents.append(ag)

    #  for observation
    agent_acceptance[i] = ag.acceptance
    agent_rejection[i] = ag.rejection
    perceived_leadership[i] = [0.]



### update
for iter in range(number_of_iteration):
    if random() > number_of_agent*partition_a + partition_b: # silence
        #  local search
        for i in range(number_of_agent):
            local_search(agents[i])

        #  for observation
        agent_0_plan.append(agents[0].best_plan)
        agent_1_plan.append(agents[1].best_plan)
        agent_2_plan.append(agents[2].best_plan)
        agent_3_plan.append(agents[3].best_plan)
        #agent_4_plan.append(agents[4].best_plan)
        #agent_5_plan.append(agents[5].best_plan)
        #agent_6_plan.append(agents[6].best_plan)
        #agent_7_plan.append(agents[7].best_plan)

        group_plan_list.append(group_plan_list[-1])
        group_plan_score.append(true_U(group_plan_list[-1]))

        for i in range(number_of_agent):
            perceived_leadership[i].append(perceived_leadership[i][-1])

    else: # speaking
        time_rate = exp(-alpha * iter / number_of_iteration)
        #  select speaker
        P = [i.talkativeness ** cauchy / sum([ag.talkativeness ** cauchy for ag in agents]) for i in agents]
        speaker = choice(agents, p=P)
        #  speaker select aspect and propose its suggestion for the group plan
        tem_group_plan = aspect_selection(speaker)[0]
        tem_group_plan_score = aspect_selection(speaker)[1]
        #  for observation
        agent_0_plan.append(agents[0].best_plan)
        agent_1_plan.append(agents[1].best_plan)
        agent_2_plan.append(agents[2].best_plan)
        agent_3_plan.append(agents[3].best_plan)

        #  other agents' responses to the proposed plan
        sigma = 0
        for i in range(number_of_agent):
            if agents[i] != speaker:
                min_dist = min_distance(agents[i], tem_group_plan)
                ratio_diff = abs((tem_group_plan_score - individual_U(agents[i], tem_group_plan)) / individual_U(agents[i], tem_group_plan))
                tolerance = min_dist + initial_tolerance * time_rate
                if ratio_diff < tolerance:
                    sigma += 1
                    speaker.perceived_leadership[i] += (time_rate * beta * number_of_agent * speaker.credibility + bonus)
                    if min_dist != 0:
                        agents[i].plan_dictionary[tuple(tem_group_plan)] = individual_U(agents[i], tem_group_plan)
                        if individual_U(agents[i], tem_group_plan) > agents[i].plan_dictionary[agents[i].best_plan]:
                            agents[i].best_plan = tuple(tem_group_plan)

                else:  # not accepted firstly, but could be supported as well
                    speaker.perceived_leadership[i] += time_rate * (beta * number_of_agent * speaker.credibility - penalty)
                    if random() < min_dist * speaker.credibility * time_rate:
                        sigma += 1

        #  group decision of the suggested plan
        if (sigma + 1) / number_of_agent > 0.5:
            group_plan = tuple(tem_group_plan)
            group_plan_list.append(group_plan)
            speaker.acceptance += 1
        else:
            speaker.rejection += 1
            group_plan_list.append(group_plan)

        #  for observation
        group_plan_score.append(true_U(group_plan))

        #  for observation of acceptance and rejection
        for i in range(number_of_agent):
            agent_acceptance[i] = agents[i].acceptance
            agent_rejection[i] = agents[i].rejection
            perceived_leadership[i].append(sum(agents[i].perceived_leadership) / (number_of_agent - 1))



fig = plt.figure(figsize=(10,7))

plt.subplot(2, 2, 1)
#  plot the perceived leadership
plt.plot(list(perceived_leadership[0]), label='agent 0', color='darkblue', linestyle=':', linewidth=1.6)
plt.plot(list(perceived_leadership[1]), label='agent 1', color='darkorange', linestyle='-.', linewidth=1.6)
plt.plot(list(perceived_leadership[2]), label='agent 2', color='darkgreen', linestyle='--', linewidth=1.6)
plt.plot(list(perceived_leadership[3]), label='agent 3', color='darkcyan', linewidth=1.6)
plt.xlabel('Number of iterations')
plt.ylabel('Average of perceived leadership')
plt.legend()


plt.subplot(2, 2, 2)
#  plot bar plot of acceptance and rejection
barWidth = 0.25
bars1 = list(agent_acceptance.values())
bars2 = list(agent_rejection.values())
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
plt.bar(r1, bars1, color='seagreen', width=barWidth, edgecolor='white', label='accepted')
plt.bar(r2, bars2, color='darksalmon', width=barWidth, edgecolor='white', label='rejected')
plt.xticks([r + barWidth for r in range(len(bars1))], [str(item) for item in list(perceived_leadership.keys())])
plt.ylabel('Quantity')
plt.xlabel('Agents')
plt.legend()


plt.subplot(2, 2, 3)
for i in range(len(group_plan_list)):
    if i < len(group_plan_list)-1:
        plt.plot([group_plan_list[i][0], group_plan_list[i+1][0]], [group_plan_list[i][1], group_plan_list[i+1][1]], alpha=0.5, color='b')
plt.scatter(group_plan_list[0][0], group_plan_list[0][1], color='dimgray',alpha=0.8, marker='s', label='initial group plan')
plt.scatter(group_plan_list[-1][0], group_plan_list[-1][1], color='black', marker='s', linewidths=0.5, label='final group plan')
plt.xlabel('Problem aspect 1')
plt.ylabel('Problem aspect 2')
plt.xlim(0,1.05)
plt.ylim(0,1.05)
plt.legend()


plt.subplot(2, 2, 4)
plt.plot(group_plan_score, color='black',alpha=0.8)
plt.ylim(-1.5,6)
plt.xlabel('Number of iterations')
plt.ylabel('True utility of group plan')
plt.rcParams['svg.fonttype']
plt.savefig(fname='fig3_2', format='svg')
plt.show()


