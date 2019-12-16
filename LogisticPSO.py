import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import csv
import random
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# config
trails = 100

def compute_disruption_weeks(average_interval, duration):
    disruption_weeks = []
    i = 0
    while i < duration:
        i = i + np.random.poisson(average_interval)
        if i < duration:
            disruption_weeks.append(i)
    return disruption_weeks


def best_path_PSO(delivery, G):
    print('===delivery===')
    print(delivery)
    print('===Graph===')
    print(G.nodes())
    ans = [[['Origin', 'A', 'C', 'Destination'], 200, 10],
           [['Origin', 'A', 'D', 'Destination'], 150, 5],
           [['Origin', 'D', 'E', 'Destination'], 120, 15]]
    return random.choice(ans)

# create directed graph
G = nx.MultiDiGraph()

# read delivery info
delivery = pd.read_csv('delivery.csv')
simulation_duration = delivery['time'].max() + 2

# read nodes info
with open('nodes.csv') as csv_file:
    headers = []
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            headers = row[:]
            line_count += 1
        else:
            attributes = {headers[i]: row[i] for i in range(1, len(headers))}
            attributes['availability'] = True
            G.add_node(row[0], **attributes)
            line_count += 1

# read edges info
with open('edges.csv') as csv_file:
    headers = []
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            headers = row[:]
            line_count += 1
        else:
            attributes = {headers[i]: row[i] for i in range(2, len(headers))}
            attributes['availability'] = True
            G.add_edge(row[0], row[1], **attributes)
            line_count += 1

# compute disruption weeks for nodes and edges
disruption_weeks_dict = {}
for node in G:
    disruption_periods = []
    average_interval = G.nodes[node]['dlambda']
    disruption_weeks = compute_disruption_weeks(
        int(average_interval), simulation_duration)
    for start in disruption_weeks:
        duration = int(np.random.normal(float(G.nodes[node]['rmean']), 0.1))
        end = start + duration
        disruption_periods.append((start, end))
    disruption_weeks_dict[node] = disruption_periods
nx.set_node_attributes(G, disruption_weeks_dict, 'disruptions')

disruption_weeks_dict = {}
for edge in G.edges():
    for i in G[edge[0]][edge[1]]:
        disruption_periods = []
        average_interval = G[edge[0]][edge[1]][i]['dlambda']
        disruption_weeks = compute_disruption_weeks(
            int(average_interval), simulation_duration)
        for start in disruption_weeks:
            duration = int(np.random.normal(
                float(G[edge[0]][edge[1]][i]['rmean']), 0.1))
            end = start+duration
            disruption_periods.append((start, end))
        disruption_weeks_dict[(*edge, i)] = disruption_periods

nx.set_edge_attributes(G, disruption_weeks_dict, 'disruptions')

# simulate 100 trails
all_edges = []
for edge in G.edges():
    for i in G[edge[0]][edge[1]]:
        all_edges.append((*edge, i))

result = []
for run in range(trails):
    # simulate one trail
    trail_result = []
    for index, row in delivery.iterrows():
        time = row['time']
        # set availability to True
        nx.set_node_attributes(G, {node: True for node in G}, 'availability')
        nx.set_edge_attributes(
            G, {edge: True for edge in all_edges}, 'availability')
        for node in G:
            for start, end in G.nodes[node]['disruptions']:
                if start < time and end > time:
                    nx.set_node_attributes(G, {node: False}, 'availability')

        for edge in G.edges():
            for i in G[edge[0]][edge[1]]:
                for start, end in G[edge[0]][edge[1]][i]['disruptions']:
                    if start < time and end > time:
                        nx.set_edge_attributes(
                            G, {(*edge, i): False}, 'availability')

        # run PSO algo
        trail_result.append(best_path_PSO(row, G))
    result.append(trail_result)
# analysis
average_costs = [np.mean([trail[i][1] for trail in result]) for i in range(len(result[1]))]
average_time = [np.mean([trail[i][2] for trail in result]) for i in range(len(result[1]))]

# visuallisation

fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(2, 2)

# plot graph diagram
logistic_map = fig.add_subplot(gs[0, :])

# coloring

def adjust_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

color_map = []
for node in G:
    if node == "Origin" or node == 'Destination':
        color_map.append('red')
    else:
        color_map.append(adjust_color('b', 1-100/int(G.nodes[node]['dlambda'])))

# handle positions
positions = {node_data[0]: (float(node_data[1]['lon']), float(
    node_data[1]['lat'])) for node_data in G.nodes(data=True)}
pos = nx.spring_layout(G, pos=positions, fixed=positions.keys())
nx.draw(G, pos, node_color=color_map, with_labels=True)
logistic_map.plot()

# plot graph
cost_ax = fig.add_subplot(gs[1, 0])
cost_ax.plot(average_costs)
cost_ax.set_ylabel('cost')
cost_ax.set_xlabel('order')
time_ax = fig.add_subplot(gs[1, 1])
time_ax.plot(average_time)
time_ax.set_ylabel('time')
time_ax.set_xlabel('order')
fig.align_labels()
plt.show()