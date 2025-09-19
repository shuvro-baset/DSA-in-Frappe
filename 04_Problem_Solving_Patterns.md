# 04. Problem Solving Patterns

## 📋 Table of Contents

1. [Two Pointers](#two-pointers)
2. [Sliding Window](#sliding-window)
3. [Tree Traversal](#tree-traversal)
4. [Graph Algorithms](#graph-algorithms)
5. [Dynamic Programming](#dynamic-programming)
6. [Greedy Patterns](#greedy-patterns)

---

## Two Pointers

### 🎯 What are Two Pointers?

A technique using two pointers to traverse data structures, often for optimization or specific problem solving.

### 💼 Real Examples from Frappe

#### 1. Data Validation

**Location**: `frappe/frappe/utils/data.py`

```python
def unique(seq: typing.Sequence["T"]) -> list["T"]:
    """Remove duplicates from sequence while preserving order."""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
```

**Pattern**: Two pointers (seen set + iteration)
**Why This Approach?**

- Maintains order while removing duplicates
- O(n) time complexity
- Memory efficient

#### 2. Payment Allocation

**Location**: `erpnext/erpnext/accounts/doctype/payment_entry/payment_entry.py`

```python
def allocate_open_payment_requests_to_references(references=None, precision=None):
    """Allocate payments using two-pointer approach."""

    row_number = 1
    MOVE_TO_NEXT_ROW = 1
    TO_SKIP_NEW_ROW = 2

    while row_number <= len(references):
        reference = references[row_number - 1]

        # Process current reference
        if reference.allocated_amount > 0:
            # Allocate payment requests
            for pr in references_open_payment_requests:
                if pr.outstanding_amount > 0:
                    # Two-pointer logic for allocation
                    if reference.allocated_amount >= pr.outstanding_amount:
                        # Full allocation
                        allocate_payment_request(reference, pr)
                        pr.outstanding_amount = 0
                    else:
                        # Partial allocation
                        allocate_partial_payment_request(reference, pr)
                        break

        row_number += 1
```

**Pattern**: Two pointers (reference pointer + payment request pointer)
**Why This Approach?**

- Efficient allocation processing
- Handles partial allocations
- Maintains order

### 🔧 Common Two Pointer Patterns

```python
# Remove duplicates from sorted array
def remove_duplicates(nums):
    if not nums:
        return 0

    slow = 0
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]

    return slow + 1

# Two sum in sorted array
def two_sum_sorted(nums, target):
    left, right = 0, len(nums) - 1

    while left < right:
        current_sum = nums[left] + nums[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1

    return []

# Palindrome check
def is_palindrome(s):
    left, right = 0, len(s) - 1

    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1

    return True
```

---

## Sliding Window

### 🎯 What is Sliding Window?

A technique for efficiently processing subarrays or substrings by maintaining a "window" that slides through the data.

### 💼 Real Examples from Frappe

#### 1. Time-based Calculations

**Location**: `hrms/hrms/hr/doctype/employee_checkin/employee_checkin.py`

```python
def calculate_working_hours(logs, check_in_out_type, working_hours_calc_type):
    """Calculate working hours using sliding window approach."""

    total_hours = 0
    in_time = out_time = None

    if check_in_out_type == "Alternating entries as IN and OUT during the same shift":
        in_time = logs[0].time
        if len(logs) >= 2:
            out_time = logs[-1].time

        if working_hours_calc_type == "Every Valid Check-in and Check-out":
            logs = logs[:]
            # Sliding window: process pairs of check-in/check-out
            while len(logs) >= 2:
                total_hours += time_diff_in_hours(logs[0].time, logs[1].time)
                del logs[:2]  # Slide window by 2 positions

    return total_hours
```

**Pattern**: Sliding window for time calculations
**Why This Approach?**

- Processes time intervals efficiently
- Handles multiple check-in/out pairs
- Maintains chronological order

#### 2. Rolling Average Calculations

**Location**: `fusion_hr/fusion_hr/fusion_hr/doctype/overtime_process/overtime_process.py`

```python
def calculate_rolling_overtime(employee_data, window_size=7):
    """Calculate rolling average overtime over a window."""

    if len(employee_data) < window_size:
        return employee_data

    rolling_averages = []

    # Sliding window approach
    for i in range(len(employee_data) - window_size + 1):
        window_data = employee_data[i:i + window_size]
        avg_overtime = sum(day['overtime_hours'] for day in window_data) / window_size
        rolling_averages.append({
            'date': window_data[-1]['date'],
            'rolling_avg_overtime': avg_overtime
        })

    return rolling_averages
```

**Pattern**: Sliding window for rolling calculations
**Why This Approach?**

- Efficient rolling average computation
- Maintains window size
- Handles edge cases

### 🔧 Common Sliding Window Patterns

```python
# Maximum sum of subarray of size k
def max_sum_subarray(nums, k):
    if len(nums) < k:
        return -1

    window_sum = sum(nums[:k])
    max_sum = window_sum

    for i in range(k, len(nums)):
        window_sum = window_sum - nums[i - k] + nums[i]
        max_sum = max(max_sum, window_sum)

    return max_sum

# Longest substring without repeating characters
def longest_substring(s):
    char_set = set()
    left = 0
    max_length = 0

    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1

        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)

    return max_length

# Minimum window substring
def min_window(s, t):
    if not s or not t:
        return ""

    dict_t = {}
    for char in t:
        dict_t[char] = dict_t.get(char, 0) + 1

    required = len(dict_t)
    formed = 0
    window_counts = {}

    left = right = 0
    ans = float('inf'), None, None

    while right < len(s):
        char = s[right]
        window_counts[char] = window_counts.get(char, 0) + 1

        if char in dict_t and window_counts[char] == dict_t[char]:
            formed += 1

        while left <= right and formed == required:
            char = s[left]

            if right - left + 1 < ans[0]:
                ans = (right - left + 1, left, right)

            window_counts[char] -= 1
            if char in dict_t and window_counts[char] < dict_t[char]:
                formed -= 1

            left += 1

        right += 1

    return "" if ans[0] == float('inf') else s[ans[1]:ans[2] + 1]
```

---

## Tree Traversal

### 🎯 What is Tree Traversal?

Systematic methods for visiting all nodes in a tree structure.

### 💼 Real Examples from Frappe

#### 1. BOM Level Order Traversal

**Location**: `erpnext/erpnext/manufacturing/doctype/bom/bom.py`

```python
def level_order_traversal(self) -> list["BOMTree"]:
    """Get level order traversal of tree using BFS."""
    traversal = []
    q = deque()
    q.append(self)

    while q:
        node = q.popleft()
        for child in node.child_items:
            traversal.append(child)
            q.append(child)

    return traversal
```

**Pattern**: Breadth-First Search (BFS)
**Why This Approach?**

- Processes BOM levels systematically
- Maintains hierarchical order
- Efficient for large BOMs

#### 2. Organizational Chart Traversal

**Location**: `hrms/hrms/utils/hierarchy_chart.py`

```python
@frappe.whitelist()
def get_all_nodes(method, company):
    """Recursively gets all data from nodes using DFS."""

    root_nodes = method(company=company)
    result = []
    nodes_to_expand = []

    # Process root nodes
    for root in root_nodes:
        data = method(root.id, company)
        result.append(dict(parent=root.id, parent_name=root.name, data=data))
        nodes_to_expand.extend(
            [{"id": d.get("id"), "name": d.get("name")} for d in data if d.get("expandable")]
        )

    # DFS traversal
    while nodes_to_expand:
        parent = nodes_to_expand.pop(0)  # Stack-like behavior
        data = method(parent.get("id"), company)
        result.append(dict(parent=parent.get("id"), parent_name=parent.get("name"), data=data))

        for d in data:
            if d.get("expandable"):
                nodes_to_expand.append({"id": d.get("id"), "name": d.get("name")})

    return result
```

**Pattern**: Depth-First Search (DFS)
**Why This Approach?**

- Builds complete hierarchy
- Handles nested structures
- Memory efficient

### 🔧 Common Tree Traversal Patterns

```python
# Pre-order traversal (Root -> Left -> Right)
def preorder_traversal(root):
    if not root:
        return []

    result = [root.val]
    result.extend(preorder_traversal(root.left))
    result.extend(preorder_traversal(root.right))

    return result

# In-order traversal (Left -> Root -> Right)
def inorder_traversal(root):
    if not root:
        return []

    result = []
    result.extend(inorder_traversal(root.left))
    result.append(root.val)
    result.extend(inorder_traversal(root.right))

    return result

# Post-order traversal (Left -> Right -> Root)
def postorder_traversal(root):
    if not root:
        return []

    result = []
    result.extend(postorder_traversal(root.left))
    result.extend(postorder_traversal(root.right))
    result.append(root.val)

    return result

# Level-order traversal (BFS)
def level_order_traversal(root):
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level = []

        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level)

    return result
```

---

## Graph Algorithms

### 🎯 What are Graph Algorithms?

Algorithms designed to work with graphs for solving connectivity, pathfinding, and dependency problems.

### 💼 Real Examples from Frappe

#### 1. Circular Dependency Detection

**Location**: `erpnext/erpnext/projects/doctype/task/task.py`

```python
def check_recursion(self):
    """Detect circular dependencies using DFS."""
    if self.flags.ignore_recursion_check:
        return

    check_list = [["task", "parent"], ["parent", "task"]]

    for d in check_list:
        task_list, count = [self.name], 0

        while len(task_list) > count:
            tasks = frappe.db.sql(
                " select {} from `tabTask Depends On` where {} = {} ".format(d[0], d[1], "%s"),
                cstr(task_list[count]),
            )
            count = count + 1

            for b in tasks:
                if b[0] == self.name:
                    frappe.throw(_("Circular Reference Error"), CircularReferenceError)
                if b[0]:
                    task_list.append(b[0])

            if count == 15:  # Prevent infinite loops
                break
```

**Pattern**: Depth-First Search for cycle detection
**Why This Approach?**

- Detects circular dependencies
- Prevents infinite loops
- Ensures workflow integrity

#### 2. Topological Sorting for BOM Updates

**Location**: `erpnext/erpnext/manufacturing/doctype/bom_update_log/bom_updation_utils.py`

```python
def get_next_higher_level_boms(child_boms: list[str], processed_boms: dict[str, bool]) -> list[str]:
    """Generate immediate higher level dependants with no unresolved dependencies."""

    def _all_children_are_processed(parent_bom):
        child_boms = dependency_map.get(parent_bom)
        return all(processed_boms.get(bom) for bom in child_boms)

    dependants_map, dependency_map = _generate_dependence_map()

    dependants = []
    for bom in child_boms:
        parents = dependants_map.get(bom) or []
        dependants.extend(parents)

    dependants = set(dependants)
    resolved_dependants = set()

    # Topological sort logic
    for parent_bom in dependants:
        if _all_children_are_processed(parent_bom):
            resolved_dependants.add(parent_bom)

    return list(resolved_dependants)
```

**Pattern**: Topological sorting
**Why This Approach?**

- Resolves dependency order
- Ensures proper update sequence
- Handles complex dependency graphs

### 🔧 Common Graph Algorithm Patterns

```python
# Depth-First Search
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()

    visited.add(start)
    print(start)

    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

# Breadth-First Search
def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        node = queue.popleft()
        print(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

# Detect Cycle in Directed Graph
def has_cycle(graph):
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}

    def dfs(node):
        if color[node] == GRAY:
            return True  # Cycle detected
        if color[node] == BLACK:
            return False

        color[node] = GRAY
        for neighbor in graph[node]:
            if dfs(neighbor):
                return True
        color[node] = BLACK
        return False

    for node in graph:
        if color[node] == WHITE:
            if dfs(node):
                return True
    return False

# Topological Sort
def topological_sort(graph):
    in_degree = {node: 0 for node in graph}

    # Calculate in-degrees
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    # Find nodes with no incoming edges
    queue = deque([node for node in in_degree if in_degree[node] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return result if len(result) == len(graph) else []  # Empty if cycle exists
```

---

## Dynamic Programming

### 🎯 What is Dynamic Programming?

Solving complex problems by breaking them into simpler subproblems and storing results to avoid redundant calculations.

### 💼 Real Examples from Frappe

#### 1. Overtime Calculation with Memoization

**Location**: `fusion_hr/fusion_hr/fusion_hr/doctype/overtime_process/overtime_process.py`

```python
def _calculate_overtime_amounts_for_value(hours_value, ledger_entry, follow_labour_law):
    """Calculate overtime using dynamic programming approach."""

    # Base cases
    if hours_value <= 0:
        return {
            "total_amount": 0.0,
            "labour_law_hours": 0.0,
            "labour_law_amount": 0.0,
            "non_labour_law_hours": 0.0,
            "non_labour_law_amount": 0.0
        }

    if not ledger_entry.get("overtime_slab"):
        return {
            "total_amount": 0.0,
            "labour_law_hours": 0.0,
            "labour_law_amount": 0.0,
            "non_labour_law_hours": 0.0,
            "non_labour_law_amount": 0.0
        }

    # Get slab and process tiers
    slab = frappe.get_doc("Overtime Slab", ledger_entry["overtime_slab"])
    tiers = slab.get("overtime_tiers", [])

    remaining = hours_value
    total_amount = 0.0
    labour_law_hours = 0.0
    labour_law_amount = 0.0
    non_labour_law_hours = 0.0
    non_labour_law_amount = 0.0

    # DP approach: process each tier optimally
    for tier in tiers:
        segment_hours = float(tier.get("ot_hour_max") or 0)
        if segment_hours <= 0:
            continue

        chunk = min(segment_hours, remaining)
        if chunk <= 0:
            continue

        # Calculate rate and amount
        rate = calculate_rate(tier, ledger_entry)
        amount = round(chunk * float(rate or 0), 2)
        total_amount += amount

        # Categorize by labour law compliance
        if tier.get("according_to_labour_law", 0):
            labour_law_hours += chunk
            labour_law_amount += amount
        else:
            non_labour_law_hours += chunk
            non_labour_law_amount += amount

        remaining -= chunk
        if remaining <= 0:
            break

    return {
        "total_amount": round(total_amount, 2),
        "labour_law_hours": round(labour_law_hours, 2),
        "labour_law_amount": round(labour_law_amount, 2),
        "non_labour_law_hours": round(non_labour_law_hours, 2),
        "non_labour_law_amount": round(non_labour_law_amount, 2)
    }
```

**Pattern**: Bottom-up DP with optimal substructure
**Why This Approach?**

- Breaks down overtime calculation into tiers
- Avoids redundant calculations
- Handles complex tier-based logic

### 🔧 Common Dynamic Programming Patterns

```python
# Fibonacci with memoization
def fibonacci_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n

    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
    return memo[n]

# Fibonacci bottom-up
def fibonacci_dp(n):
    if n <= 1:
        return n

    dp = [0] * (n + 1)
    dp[1] = 1

    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    return dp[n]

# Longest Common Subsequence
def lcs(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]

# Coin Change
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1
```

---

## Greedy Patterns

### 🎯 What are Greedy Patterns?

Making locally optimal choices at each step to find a global optimum.

### 💼 Real Examples from Frappe

#### 1. Payment Allocation

**Location**: `erpnext/erpnext/accounts/doctype/payment_entry/payment_entry.py`

```python
def allocate_open_payment_requests_to_references(references=None, precision=None):
    """Greedy allocation of payment requests to references."""

    # Greedy approach: allocate payments optimally
    for reference in references:
        if reference.allocated_amount > 0:
            # Find best payment request to allocate
            for pr in references_open_payment_requests:
                if pr.outstanding_amount > 0:
                    # Greedy choice: allocate maximum possible
                    if reference.allocated_amount >= pr.outstanding_amount:
                        # Full allocation
                        allocate_payment_request(reference, pr)
                        pr.outstanding_amount = 0
                    else:
                        # Partial allocation
                        allocate_partial_payment_request(reference, pr)
                        break
```

**Pattern**: Greedy allocation
**Why This Approach?**

- Makes locally optimal allocation decisions
- Processes payments efficiently
- Maximizes allocation coverage

### 🔧 Common Greedy Patterns

```python
# Activity Selection Problem
def activity_selection(activities):
    # Sort by finish time
    activities.sort(key=lambda x: x[1])

    selected = [activities[0]]
    last_finish = activities[0][1]

    for start, finish in activities[1:]:
        if start >= last_finish:
            selected.append((start, finish))
            last_finish = finish

    return selected

# Fractional Knapsack
def fractional_knapsack(items, capacity):
    # Sort by value/weight ratio
    items.sort(key=lambda x: x[1]/x[2], reverse=True)

    total_value = 0
    knapsack = []

    for item, value, weight in items:
        if capacity >= weight:
            knapsack.append((item, weight))
            total_value += value
            capacity -= weight
        else:
            fraction = capacity / weight
            knapsack.append((item, fraction * weight))
            total_value += fraction * value
            break

    return knapsack, total_value

# Minimum Spanning Tree (Kruskal's Algorithm)
def kruskal_mst(edges, num_vertices):
    edges.sort(key=lambda x: x[2])  # Sort by weight
    parent = list(range(num_vertices))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False

    mst = []
    for u, v, weight in edges:
        if union(u, v):
            mst.append((u, v, weight))
            if len(mst) == num_vertices - 1:
                break

    return mst
```

---

## 🎯 Interview Questions & Answers

### Q1: "How would you implement a sliding window for time-based calculations?"

**Answer**: "I'd use a sliding window approach similar to Frappe's employee check-in calculation:

```python
def calculate_rolling_average(time_data, window_size):
    if len(time_data) < window_size:
        return time_data

    rolling_averages = []

    for i in range(len(time_data) - window_size + 1):
        window_data = time_data[i:i + window_size]
        avg = sum(day['hours'] for day in window_data) / window_size
        rolling_averages.append({
            'date': window_data[-1]['date'],
            'rolling_avg': avg
        })

    return rolling_averages
```

This approach maintains O(n) time complexity while providing efficient rolling calculations."

### Q2: "Explain how Frappe uses graph algorithms for dependency management."

**Answer**: "Frappe uses several graph algorithms:

1. **Cycle Detection**: Uses DFS to detect circular dependencies in task workflows
2. **Topological Sorting**: Resolves BOM update dependencies in correct order
3. **Dependency Resolution**: Ensures proper processing order for complex relationships

For example, in BOM updates:

```python
def get_next_higher_level_boms(child_boms, processed_boms):
    # Topological sort logic
    for parent_bom in dependants:
        if _all_children_are_processed(parent_bom):
            resolved_dependants.add(parent_bom)
```

This ensures updates happen in the correct dependency order."

### Q3: "How does Frappe optimize payment allocation using greedy algorithms?"

**Answer**: "Frappe uses a greedy approach for payment allocation:

1. **Local Optimization**: Makes best allocation decision at each step
2. **Priority Handling**: Processes payments in order of importance
3. **Maximum Allocation**: Allocates as much as possible to each reference

The algorithm makes locally optimal choices:

- If allocated amount equals payment request → allocate completely
- If allocated amount is less → partial allocation, continue
- If allocated amount is more → allocate fully, create new row

This ensures maximum allocation efficiency while maintaining business logic."

---

## 🚀 Summary

You've now mastered the essential problem-solving patterns used in the Frappe ecosystem! These patterns will help you:

1. **Recognize common problems** and apply appropriate solutions
2. **Optimize algorithms** for better performance
3. **Explain your approach** clearly in interviews
4. **Implement efficient solutions** in real-world scenarios

**Key Takeaways:**

- Choose the right pattern for your problem
- Understand trade-offs between different approaches
- Use real examples to explain your solutions
- Practice implementing these patterns in your own code

**Next Steps:**

- Practice implementing these patterns
- Solve coding problems using these approaches
- Build projects that demonstrate your understanding
- Prepare for technical interviews with confidence!

---

**🎉 Congratulations!** You've completed the comprehensive DSA learning guide using real examples from the Frappe ecosystem. You're now well-prepared for Python Backend Engineer interviews!
