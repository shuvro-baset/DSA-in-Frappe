# 07. Quick Reference Guide

## 📋 Table of Contents

1. [Data Structures Cheat Sheet](#data-structures-cheat-sheet)
2. [Algorithms Quick Reference](#algorithms-quick-reference)
3. [Complexity Summary](#complexity-summary)
4. [Common Patterns](#common-patterns)
5. [Interview Tips](#interview-tips)

---

## Data Structures Cheat Sheet

### 🏗️ Arrays & Lists

```python
# Basic Operations
arr = [1, 2, 3, 4, 5]
arr.append(6)        # O(1)
arr.insert(0, 0)     # O(n)
arr.pop()           # O(1)
arr.remove(3)       # O(n)
arr.index(2)        # O(n)

# Frappe Example: Email Queue Recipients
def set_recipients(self, recipients):
    self.set("recipients", [])
    for r in recipients:
        self.append("recipients", {"recipient": r.strip(), "status": "Not Sent"})
```

### 🗂️ Hash Tables & Dictionaries

```python
# Basic Operations
cache = {}
cache['key'] = 'value'    # O(1)
value = cache.get('key')  # O(1)
del cache['key']          # O(1)
'key' in cache            # O(1)

# Frappe Example: Site Cache
_SITE_CACHE = defaultdict(lambda: defaultdict(dict))
```

### 📚 Stacks & Queues

```python
# Stack (LIFO)
stack = []
stack.append(item)  # Push
item = stack.pop()  # Pop

# Queue (FIFO)
from collections import deque
queue = deque()
queue.append(item)      # Enqueue
item = queue.popleft()  # Dequeue

# Frappe Example: Background Jobs
def enqueue(method, queue="default", **kwargs):
    q = get_queue(queue)
    return q.enqueue_call(execute_job, kwargs=queue_args)
```

### 🌳 Trees

```python
# Tree Node
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.children = []

# Traversal Patterns
def preorder(root):
    if not root: return []
    return [root.val] + preorder(root.left) + preorder(root.right)

def inorder(root):
    if not root: return []
    return inorder(root.left) + [root.val] + inorder(root.right)

def postorder(root):
    if not root: return []
    return postorder(root.left) + postorder(root.right) + [root.val]

# Frappe Example: BOM Tree
class BOMTree:
    def __init__(self, name, is_bom=True):
        self.name = name
        self.child_items = []
```

### 🕸️ Graphs

```python
# Graph Representation
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A', 'D'],
    'D': ['B', 'C']
}

# DFS
def dfs(graph, start, visited=None):
    if visited is None: visited = set()
    visited.add(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

# BFS
def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

# Frappe Example: Task Dependencies
def check_recursion(self):
    task_list, count = [self.name], 0
    while len(task_list) > count:
        # DFS for cycle detection
```

---

## Algorithms Quick Reference

### 🔄 Sorting Algorithms

```python
# Merge Sort - O(n log n)
def merge_sort(arr):
    if len(arr) <= 1: return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Frappe Example: Employee Search Sorting
order by
    (case when locate(%(_txt)s, name) > 0 then locate(%(_txt)s, name) else 99999 end),
    name, employee_name
```

### 🔍 Searching Algorithms

```python
# Binary Search - O(log n)
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Frappe Example: Global Search
rank = Match(global_search.content).Against(word)
query = (
    frappe.qb.from_(global_search)
    .select(global_search.doctype, global_search.name, rank.as_("rank"))
    .where(rank)
    .orderby("rank", order=frappe.qb.desc)
)
```

### 🔄 Recursion Patterns

```python
# Base Case + Recursive Case
def recursive_function(n):
    # Base case
    if n <= 1:
        return 1
    # Recursive case
    return n * recursive_function(n - 1)

# Frappe Example: BOM Tree Creation
def __create_tree(self):
    bom = frappe.get_cached_doc("BOM", self.name)
    for item in bom.get("items", []):
        if item.bom_no:
            # Recursive call
            child = BOMTree(item.bom_no, exploded_qty=exploded_qty)
            self.child_items.append(child)
```

### 💾 Dynamic Programming

```python
# Memoization
def fibonacci_memo(n, memo={}):
    if n in memo: return memo[n]
    if n <= 1: return n
    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]

# Tabulation
def fibonacci_dp(n):
    if n <= 1: return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

# Frappe Example: Overtime Calculation
def _calculate_overtime_amounts_for_value(hours_value, ledger_entry):
    memo = {}  # Memoization for sub-calculations
    # Process tiers optimally
```

### 🎯 Greedy Algorithms

```python
# Activity Selection
def activity_selection(activities):
    activities.sort(key=lambda x: x[1])  # Sort by finish time
    selected = [activities[0]]
    last_finish = activities[0][1]

    for start, finish in activities[1:]:
        if start >= last_finish:
            selected.append((start, finish))
            last_finish = finish
    return selected

# Frappe Example: Payment Allocation
def allocate_payments_optimally(payments, invoices):
    sorted_payments = sorted(payments, reverse=True)
    # Greedy allocation strategy
```

---

## Complexity Summary

### ⏱️ Time Complexity

| Algorithm         | Best Case  | Average Case | Worst Case | Space    |
| ----------------- | ---------- | ------------ | ---------- | -------- |
| **Array Access**  | O(1)       | O(1)         | O(1)       | O(n)     |
| **Array Search**  | O(1)       | O(n)         | O(n)       | O(n)     |
| **Hash Table**    | O(1)       | O(1)         | O(n)       | O(n)     |
| **Binary Search** | O(1)       | O(log n)     | O(log n)   | O(1)     |
| **Merge Sort**    | O(n log n) | O(n log n)   | O(n log n) | O(n)     |
| **Quick Sort**    | O(n log n) | O(n log n)   | O(n²)      | O(log n) |
| **DFS/BFS**       | O(V+E)     | O(V+E)       | O(V+E)     | O(V)     |

### 📊 Space Complexity

- **O(1)**: Constant space (simple variables)
- **O(n)**: Linear space (arrays, hash tables)
- **O(log n)**: Logarithmic space (recursive calls)
- **O(n²)**: Quadratic space (2D arrays)

### 🎯 Frappe Performance Examples

```python
# O(1) - Cache Lookup
if cache_key in _SITE_CACHE[site][func.__name__]:
    return _SITE_CACHE[site][func.__name__][cache_key]

# O(n) - Employee Search
return frappe.db.sql("SELECT * FROM `tabEmployee` WHERE employee_name LIKE %s")

# O(n log n) - Report Sorting
data.sort((a, b) => aIndex - bIndex)

# O(n²) - BOM Explosion
while count < len(bom_list):
    for child_bom in _get_children(bom_list[count]):
        bom_list.append(child_bom)
```

---

## Common Patterns

### 👆 Two Pointers

```python
# Remove duplicates from sorted array
def remove_duplicates(nums):
    if not nums: return 0
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
```

### 🪟 Sliding Window

```python
# Maximum sum of subarray of size k
def max_sum_subarray(nums, k):
    if len(nums) < k: return -1
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
```

### 🌳 Tree Traversal

```python
# Level order traversal (BFS)
def level_order_traversal(root):
    if not root: return []
    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level = []
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        result.append(level)
    return result
```

### 🕸️ Graph Algorithms

```python
# Detect cycle in directed graph
def has_cycle(graph):
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}

    def dfs(node):
        if color[node] == GRAY: return True
        if color[node] == BLACK: return False
        color[node] = GRAY
        for neighbor in graph[node]:
            if dfs(neighbor): return True
        color[node] = BLACK
        return False

    for node in graph:
        if color[node] == WHITE:
            if dfs(node): return True
    return False

# Topological sort
def topological_sort(graph):
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    queue = deque([node for node in in_degree if in_degree[node] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return result if len(result) == len(graph) else []
```

---

## Interview Tips

### 🎯 Problem-Solving Approach

1. **Understand the Problem**

   - Ask clarifying questions
   - Identify input/output format
   - Consider edge cases

2. **Think Out Loud**

   - Explain your thought process
   - Discuss different approaches
   - Consider trade-offs

3. **Start Simple**

   - Begin with brute force solution
   - Then optimize step by step
   - Explain complexity improvements

4. **Code Cleanly**
   - Use meaningful variable names
   - Add comments for complex logic
   - Handle edge cases

### 💡 Common Interview Questions

#### **"Explain how Frappe uses hash tables for caching"**

```python
# Answer: Frappe uses hash tables for O(1) cache lookups
_SITE_CACHE = defaultdict(lambda: defaultdict(dict))

@site_cache(ttl=3600)
def expensive_calculation(data):
    # Function result cached in hash table
    return perform_calculation(data)
```

#### **"How would you optimize a slow database query?"**

```python
# Answer: Multiple optimization strategies
# 1. Add indexes
frappe.db.sql("CREATE INDEX idx_employee_name ON `tabEmployee` (employee_name)")

# 2. Use LIMIT
frappe.db.sql("SELECT * FROM `tabEmployee` LIMIT 20")

# 3. Cache results
@frappe.cache(ttl=300)
def get_employees():
    return frappe.get_all("Employee")
```

#### **"Design a system to handle 1M concurrent users"**

```python
# Answer: Multi-layered scalable architecture
class ScalableSystem:
    def __init__(self):
        self.cache_cluster = RedisCluster()  # Distributed cache
        self.db_pool = DatabasePool()        # Connection pooling
        self.queue_system = JobQueue()       # Background processing
        self.load_balancer = LoadBalancer()  # Traffic distribution
```

### 🚀 Last-Minute Review Checklist

- [ ] **Data Structures**: Know when to use each one
- [ ] **Algorithms**: Understand time/space complexity
- [ ] **Patterns**: Recognize common problem types
- [ ] **Frappe Examples**: Be ready to discuss real implementations
- [ ] **System Design**: Understand scalability principles
- [ ] **Code Quality**: Write clean, efficient code

### 🎯 Success Strategies

1. **Practice Daily**: Consistent practice builds confidence
2. **Explain Concepts**: Teaching others solidifies understanding
3. **Real Examples**: Use Frappe examples in explanations
4. **Stay Calm**: Take time to think before coding
5. **Ask Questions**: Clarify requirements before starting

---

## 🎉 Final Reminders

### ✅ You're Ready When You Can:

- Explain any data structure with real examples
- Implement algorithms from memory
- Analyze time/space complexity
- Discuss trade-offs between approaches
- Solve problems using multiple patterns
- Design scalable systems
- Write clean, efficient code

### 🚀 Interview Day Tips:

- **Arrive Early**: Give yourself time to relax
- **Bring Examples**: Have Frappe examples ready
- **Think Out Loud**: Explain your process
- **Ask Questions**: Clarify requirements
- **Stay Positive**: Show enthusiasm for learning

**You've got this!** 🎯 Your comprehensive preparation with real Frappe examples will set you apart from other candidates. Good luck with your Python Backend Engineer interview!
