# A Star algorithm for shortest paths with heuristic
# Omar Hossain, based on David Eppstein's code, UC Irvine, 29 December 2022

from priodict import priorityDictionary


def AStar(G, H, start, end=None):

    # final distances is the distance between a given verticy and the destination
    D = {}  # dictionary of final distances
    P = {}  # dictionary of predecessors
    Q = priorityDictionary()  # estimated distances of non-final vertices
    Q[start] = 0

    for v in Q:
        D[v] = Q[v]
        if v == end:
            break
        
        # We are checking all the verticies that connect to v that are closer to the final point (final distances)
        for w in G[v]:
            # final distance of given vertice (Distance from beginning to current latest node) + one of the connected verticies + Heuristic cost
            vwLength = D[v] + G[v][w] + H[w]
            if w in D:
                if vwLength < D[w]:
                    raise ValueError("Dijkstra: found better path to already-final vertex")
            elif w not in Q or vwLength < Q[w]:
                Q[w] = vwLength - H[w]
                P[w] = v

    return (D, P)


def shortestPath(G, H, start, end):
    """
    Find a single shortest path from the given start vertex to the given
    end vertex. The input has the same conventions as AStar(). The
    output is a list of the vertices in order along the shortest path.
    """

    D, P = AStar(G, H, start, end)
    Path = []
    while 1:
        Path.append(end)
        if end == start:
            break
        end = P[end]
    Path.reverse()
    return Path

# example, CLR p.528
G = {'s': {'u':10, 'x':5},
    'u': {'v':1, 'x':2},
    'v': {'y':4},
    'x':{'u':3,'v':9,'y':2},
    'y':{'s':7,'v':6}}

H = {'s': 1,
    'u': 2,
    'v': 0,
    'x': 10,
    'y': 1}

if __name__ == "__main__":
    print(AStar(G, H, 's'))
    print(shortestPath(G, H, 's', 'v'))