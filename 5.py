import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from typing import Dict, List, Tuple
import math
import heapq
import time

def haversine(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    lon1, lat1 = coord1
    lon2, lat2 = coord2
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def dijkstra(graph: Dict[Tuple[float, float], List[Tuple[Tuple[float, float], float]]],
             origin: Tuple[float, float],
             goal: Tuple[float, float],
             raw_edges: List[Tuple[Tuple[float, float], Tuple[float, float], str]]) -> Tuple[
    List[Tuple[float, float]], float, List[str]]:
    nodes_to_visit = [(0.0, origin)]
    min_costs = {origin: 0.0}
    backtrack_map = {origin: None}
    processed = set()

    while nodes_to_visit:
        current_w, u = heapq.heappop(nodes_to_visit)
        if u in processed:
            continue
        if u == goal:
            break
        processed.add(u)
        for v, weight in graph.get(u, []):
            new_cost = current_w + weight
            if new_cost < min_costs.get(v, float('inf')):
                min_costs[v] = new_cost
                backtrack_map[v] = u
                heapq.heappush(nodes_to_visit, (new_cost, v))
    full_path = []
    if goal in min_costs:
        step = goal
        while step is not None:
            full_path.append(step)
            step = backtrack_map[step]
        full_path.reverse()

    road_names = []
    if full_path:
        link_to_name = {(e[0], e[1]): e[2] for e in raw_edges}
        link_to_name.update({(e[1], e[0]): e[2] for e in raw_edges})
        for i in range(len(full_path) - 1):
            segment = (full_path[i], full_path[i + 1])
            name = link_to_name.get(segment)

            if name and (not road_names or road_names[-1] != name):
                road_names.append(name)

    return full_path, min_costs.get(goal, 0.0), road_names

def build_graph(edges: List[Tuple[Tuple[float, float], Tuple[float, float], str]]) -> Dict[
    Tuple[float, float], List[Tuple[Tuple[float, float], float]]]:
    graph = {}
    for start, end, _ in edges:
        dist = haversine(start, end)
        graph.setdefault(start, []).append((end, dist))
        graph.setdefault(end, []).append((start, dist))
    return graph


def read_graphml(file_path: str) -> Tuple[
    Dict[str, Tuple[float, float]], List[Tuple[Tuple[float, float], Tuple[float, float], str]]]:
    tree = ET.parse(file_path)
    root = tree.getroot()
    ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}
    nodes = {}
    for node in root.findall('.//g:node', ns):
        node_id = node.get('id')
        x, y = None, None
        for data in node.findall('.//g:data', ns):
            if data.get('key') == 'd4':
                x = float(data.text)
            elif data.get('key') == 'd5':
                y = float(data.text)
        if x is not None and y is not None:
            nodes[node_id] = (x, y)
    edges = []
    for edge in root.findall('.//g:edge', ns):
        source, target = edge.get('source'), edge.get('target')
        street_name = None
        for data in edge.findall('.//g:data', ns):
            if data.get('key') == 'd13': street_name = data.text
        if source in nodes and target in nodes:
            edges.append((nodes[source], nodes[target], street_name))
    return nodes, edges


def find_street_index(edges: List[Tuple[Tuple[float, float], Tuple[float, float], str]], query: str) -> int:
    for i, (_, _, name) in enumerate(edges):
        if name and query.lower() in name.lower():
            return i
    return -1


def visualize_path_with_network(nodes, edges, path, figsize=(15, 15)):
    plt.figure(figsize=figsize)
    ax = plt.gca()
    all_lines = [(start, end) for start, end, _ in edges]
    lc = LineCollection(all_lines, linewidths=0.3, colors='gray', alpha=0.4)
    ax.add_collection(lc)
    if path and len(path) > 1:
        path_lines = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        lc_path = LineCollection(path_lines, linewidths=2.5, colors='red', alpha=1.0)
        ax.add_collection(lc_path)
    ax.autoscale()
    plt.axis('equal')
    plt.title('Дорожная сеть и кратчайший маршрут')
    plt.xlabel('Долгота')
    plt.ylabel('Широта')
    plt.show()


if __name__ == "__main__":
    nodes_data, edges_list = read_graphml("minsk_road_network.graphml")
    print(f"количество вершин: {len(nodes_data)}")
    print(f"количество рёбер: {len(edges_list)}")
    start_query = "Лінейны завулак"
    end_query = "Дубравінская вуліца"
    start_idx = find_street_index(edges_list, start_query)
    end_idx = find_street_index(edges_list, end_query)
    if start_idx == -1 or end_idx == -1:
        print("Улицы не найдены.")
    else:
        start_node = edges_list[start_idx][0]
        end_node = edges_list[end_idx][1]
        graph = build_graph(edges_list)
        start_time = time.perf_counter()
        path, dist, streets = dijkstra(graph, start_node, end_node, edges_list)
        end_time = time.perf_counter()
        if not path:
            print("Маршрут не найден.")
        else:
            print(f"время выполнения алгоритма Дейкстры: {(end_time - start_time)} сек")
            print(f"Дистанция: {dist} км")
            print("Улицы на пути:", ", ".join(streets))
            visualize_path_with_network(nodes_data, edges_list, path)