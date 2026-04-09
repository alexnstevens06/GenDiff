def myers_diff(a, b):
    """
    An implementation of the Myers diff algorithm.
    Returns a list of opcodes: (tag, i1, i2, j1, j2)
    where tag is one of 'replace', 'delete', 'insert', 'equal'.
    """
    n = len(a)
    m = len(b)
    
    # We'll use a standard BFS approach to find the shortest edit script.
    # v[k] stores the furthest reached x-coordinate for a given k-line.
    v = {1: 0}
    trace = []
    
    for d in range(n + m + 1):
        v_copy = v.copy()
        trace.append(v_copy)
        for k in range(-d, d + 1, 2):
            if k == -d or (k != d and v[k-1] < v[k+1]):
                x = v[k+1]
            else:
                x = v[k-1] + 1
            
            y = x - k
            
            while x < n and y < m and a[x] == b[y]:
                x, y = x + 1, y + 1
            
            v[k] = x
            
            if x >= n and y >= m:
                # Found the end! Now backtrack to get the path.
                return _backtrack(trace, a, b)

def _backtrack(trace, a, b):
    x, y = len(a), len(b)
    path = []
    
    for d in range(len(trace) - 1, -1, -1):
        v = trace[d]
        k = x - y
        
        if k == -d or (k != d and v[k-1] < v[k+1]):
            prev_k = k + 1
        else:
            prev_k = k - 1
            
        prev_x = v[prev_k]
        prev_y = prev_x - prev_k
        
        while x > prev_x and y > prev_y:
            path.append(('equal', x-1, x, y-1, y))
            x, y = x-1, y-1
            
        if d > 0:
            if x == prev_x:
                path.append(('insert', x, x, prev_y, y))
            else:
                path.append(('delete', prev_x, x, y, y))
            x, y = prev_x, prev_y
            
    path.reverse()
    
    # Consolidate path into opcodes
    if not path:
        return []
        
    consolidated = []
    curr_tag, curr_i1, curr_i2, curr_j1, curr_j2 = path[0]
    
    for next_tag, next_i1, next_i2, next_j1, next_j2 in path[1:]:
        if next_tag == curr_tag and next_i1 == curr_i2 and next_j1 == curr_j2:
            curr_i2 = next_i2
            curr_j2 = next_j2
        else:
            consolidated.append((curr_tag, curr_i1, curr_i2, curr_j1, curr_j2))
            curr_tag, curr_i1, curr_i2, curr_j1, curr_j2 = next_tag, next_i1, next_i2, next_j1, next_j2
    
    consolidated.append((curr_tag, curr_i1, curr_i2, curr_j1, curr_j2))
    
    # One more pass to turn delete+insert into replace
    final = []
    i = 0
    while i < len(consolidated):
        if (i < len(consolidated) - 1 and 
            consolidated[i][0] == 'delete' and 
            consolidated[i+1][0] == 'insert' and 
            consolidated[i][2] == consolidated[i+1][1]):
            final.append(('replace', consolidated[i][1], consolidated[i][2], consolidated[i+1][3], consolidated[i+1][4]))
            i += 2
        elif (i < len(consolidated) - 1 and 
              consolidated[i][0] == 'insert' and 
              consolidated[i+1][0] == 'delete' and 
              consolidated[i][4] == consolidated[i+1][3]):
             # This order shouldn't really happen with Myers but handled for safety
            final.append(('replace', consolidated[i+1][1], consolidated[i+1][2], consolidated[i][3], consolidated[i][4]))
            i += 2
        else:
            final.append(consolidated[i])
            i += 1
            
    return final
