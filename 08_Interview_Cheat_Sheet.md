# 08. Interview Cheat Sheet

## 📋 Table of Contents

1. [Essential Concepts](#essential-concepts)
2. [Common Questions & Answers](#common-questions--answers)
3. [Code Templates](#code-templates)
4. [System Design Patterns](#system-design-patterns)
5. [Last-Minute Checklist](#last-minute-checklist)

---

## Essential Concepts

### 🏗️ Data Structure Selection Guide

| Use Case              | Best Data Structure | Time Complexity | Frappe Example    |
| --------------------- | ------------------- | --------------- | ----------------- |
| **Fast Lookups**      | Hash Table          | O(1)            | Cache system      |
| **Ordered Data**      | Array/List          | O(1) access     | Email queue       |
| **LIFO Operations**   | Stack               | O(1)            | Function calls    |
| **FIFO Operations**   | Queue               | O(1)            | Background jobs   |
| **Hierarchical Data** | Tree                | O(log n)        | BOM structure     |
| **Relationships**     | Graph               | O(V+E)          | Task dependencies |

### ⚡ Algorithm Complexity Quick Reference

| Algorithm         | Time           | Space    | When to Use     |
| ----------------- | -------------- | -------- | --------------- |
| **Linear Search** | O(n)           | O(1)     | Unsorted data   |
| **Binary Search** | O(log n)       | O(1)     | Sorted data     |
| **Merge Sort**    | O(n log n)     | O(n)     | Stable sorting  |
| **Quick Sort**    | O(n log n)     | O(log n) | General sorting |
| **DFS**           | O(V+E)         | O(V)     | Path finding    |
| **BFS**           | O(V+E)         | O(V)     | Shortest path   |
| **Dijkstra**      | O((V+E) log V) | O(V)     | Weighted graphs |

### 🎯 Problem Pattern Recognition

| Problem Type          | Pattern        | Template                          |
| --------------------- | -------------- | --------------------------------- |
| **Array Problems**    | Two Pointers   | `left, right = 0, len(arr)-1`     |
| **Subarray Problems** | Sliding Window | `window_start, window_sum = 0, 0` |
| **Tree Problems**     | DFS/BFS        | `def traverse(node): ...`         |
| **Graph Problems**    | DFS/BFS        | `visited = set()`                 |
| **Optimization**      | DP/Greedy      | `memo = {}` or `sort()`           |
| **Search Problems**   | Binary Search  | `left, right = 0, len(arr)-1`     |

---

## Common Questions & Answers

### 🎯 Q1: "Explain the difference between a stack and a queue"

**Answer**:

- **Stack (LIFO)**: Last In, First Out - like a stack of plates
- **Queue (FIFO)**: First In, First Out - like a line of people

**Frappe Examples**:

```python
# Stack: Function call stack
def recursive_function(n):
    if n <= 1: return 1
    return n * recursive_function(n - 1)  # Stack frames

# Queue: Background job processing
def enqueue(method, queue="default"):
    q = get_queue(queue)
    return q.enqueue_call(execute_job, kwargs=queue_args)
```

### 🎯 Q2: "When would you use a hash table vs a binary search tree?"

**Answer**:

- **Hash Table**: O(1) average lookup, no ordering, good for caching
- **BST**: O(log n) lookup, maintains order, good for range queries

**Frappe Examples**:

```python
# Hash Table: Cache system
_SITE_CACHE = defaultdict(lambda: defaultdict(dict))
# O(1) cache lookups

# BST: Employee search with ordering
employees.sort(key=lambda x: x['name'])
# O(log n) search with alphabetical order
```

### 🎯 Q3: "Explain time complexity of different sorting algorithms"

**Answer**:

- **Bubble Sort**: O(n²) - simple but inefficient
- **Merge Sort**: O(n log n) - stable, consistent performance
- **Quick Sort**: O(n log n) average, O(n²) worst case
- **Heap Sort**: O(n log n) - in-place sorting

**Frappe Example**:

```python
# JavaScript sorting in Frappe
data.sort((a, b) => {
    const aIndex = departmentOrders.indexOf(a.sub_department);
    const bIndex = departmentOrders.indexOf(b.sub_department);
    return aIndex - bIndex;
});
// O(n log n) complexity
```

### 🎯 Q4: "How do you detect a cycle in a directed graph?"

**Answer**: Use DFS with three colors (WHITE, GRAY, BLACK)

```python
def has_cycle(graph):
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}

    def dfs(node):
        if color[node] == GRAY: return True  # Back edge = cycle
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

# Frappe Example: Task dependency cycle detection
def check_recursion(self):
    task_list, count = [self.name], 0
    while len(task_list) > count:
        # DFS-based cycle detection
```

### 🎯 Q5: "Explain dynamic programming vs greedy algorithms"

**Answer**:

- **Dynamic Programming**: Solves subproblems optimally, uses memoization
- **Greedy**: Makes locally optimal choices, doesn't reconsider

**Frappe Examples**:

```python
# DP: Overtime calculation with memoization
def _calculate_overtime_amounts_for_value(hours_value, ledger_entry):
    memo = {}  # Memoization
    # Break down into optimal subproblems

# Greedy: Payment allocation
def allocate_payments_optimally(payments, invoices):
    sorted_payments = sorted(payments, reverse=True)
    # Make locally optimal allocation decisions
```

---

## Code Templates

### 🔄 Two Pointers Template

```python
def two_pointers_problem(arr):
    left, right = 0, len(arr) - 1

    while left < right:
        # Process current pair
        if condition_met(arr[left], arr[right]):
            # Found solution
            return [left, right]
        elif arr[left] + arr[right] < target:
            left += 1
        else:
            right -= 1

    return []  # No solution found

# Example: Two Sum in Sorted Array
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

### 🪟 Sliding Window Template

```python
def sliding_window_problem(s, k):
    window_start = 0
    window_sum = 0
    max_sum = 0

    for window_end in range(len(s)):
        # Expand window
        window_sum += s[window_end]

        # Shrink window if needed
        while window_end - window_start + 1 > k:
            window_sum -= s[window_start]
            window_start += 1

        # Update result
        max_sum = max(max_sum, window_sum)

    return max_sum

# Example: Maximum Sum Subarray of Size K
def max_sum_subarray(nums, k):
    if len(nums) < k: return -1
    window_sum = sum(nums[:k])
    max_sum = window_sum

    for i in range(k, len(nums)):
        window_sum = window_sum - nums[i - k] + nums[i]
        max_sum = max(max_sum, window_sum)
    return max_sum
```

### 🌳 Tree Traversal Template

```python
# DFS Template
def dfs_traversal(root):
    if not root: return []

    result = []
    stack = [root]

    while stack:
        node = stack.pop()
        result.append(node.val)

        # Add children to stack
        if node.right: stack.append(node.right)
        if node.left: stack.append(node.left)

    return result

# BFS Template
def bfs_traversal(root):
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

### 🕸️ Graph Algorithm Template

```python
# DFS Template
def dfs_graph(graph, start):
    visited = set()
    result = []

    def dfs(node):
        visited.add(node)
        result.append(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)

    dfs(start)
    return result

# BFS Template
def bfs_graph(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return result
```

### 💾 Dynamic Programming Template

```python
# Memoization Template
def dp_memoization(n, memo={}):
    if n in memo: return memo[n]
    if base_case(n): return base_value

    memo[n] = dp_memoization(n-1, memo) + dp_memoization(n-2, memo)
    return memo[n]

# Tabulation Template
def dp_tabulation(n):
    if n <= 1: return n

    dp = [0] * (n + 1)
    dp[1] = 1

    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]

    return dp[n]

# Example: Fibonacci
def fibonacci(n):
    if n <= 1: return n
    return fibonacci(n-1) + fibonacci(n-2)  # Recursive
    # With memoization: O(n) time, O(n) space
```

---

## System Design Patterns

### 🏗️ Scalable Architecture Template

```python
class ScalableSystem:
    def __init__(self):
        self.load_balancer = LoadBalancer()
        self.cache_cluster = RedisCluster()
        self.db_pool = DatabasePool()
        self.queue_system = JobQueue()
        self.cdn = CDN()

    def handle_request(self, request):
        # 1. Load balancing
        server = self.load_balancer.get_server()

        # 2. Check cache
        cached_result = self.cache_cluster.get(request.key)
        if cached_result:
            return cached_result

        # 3. Process with database
        result = self.db_pool.execute(request.query)

        # 4. Cache result
        self.cache_cluster.set(request.key, result, ttl=300)

        # 5. Background processing
        self.queue_system.enqueue(process_result, result)

        return result
```

### 💾 Caching Strategy Template

```python
class CacheStrategy:
    def __init__(self, max_size=1000, default_ttl=3600):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.expiry_times = {}

    def get(self, key):
        if key not in self.cache:
            return None

        # Check TTL
        if time.time() > self.expiry_times[key]:
            self._remove(key)
            return None

        # Move to end (LRU)
        self.cache.move_to_end(key)
        return self.cache[key]

    def set(self, key, value, ttl=None):
        if len(self.cache) >= self.max_size:
            self._evict_lru()

        self.cache[key] = value
        self.expiry_times[key] = time.time() + (ttl or self.default_ttl)

    def _evict_lru(self):
        if self.cache:
            self.cache.popitem(last=False)
```

### 🔄 Background Job Template

```python
class BackgroundJobSystem:
    def __init__(self):
        self.queues = {
            "high": deque(),
            "normal": deque(),
            "low": deque()
        }
        self.job_status = {}

    def enqueue(self, job_func, args=None, kwargs=None, priority="normal"):
        job_id = str(uuid.uuid4())
        job = {
            "id": job_id,
            "func": job_func,
            "args": args or [],
            "kwargs": kwargs or {},
            "priority": priority,
            "status": "queued"
        }

        self.queues[priority].append(job)
        self.job_status[job_id] = job
        return job_id

    def process_jobs(self):
        while True:
            job = self._get_next_job()
            if not job:
                time.sleep(1)
                continue

            try:
                self._execute_job(job)
            except Exception as e:
                self._handle_job_failure(job, e)
```

---

## Last-Minute Checklist

### ✅ Before the Interview

- [ ] **Review Data Structures**: Know when to use each one
- [ ] **Practice Algorithms**: Implement from memory
- [ ] **Study Complexity**: Understand Big O notation
- [ ] **Prepare Examples**: Have Frappe examples ready
- [ ] **Mock Interview**: Practice explaining solutions
- [ ] **Rest Well**: Get good sleep the night before

### ✅ During the Interview

- [ ] **Listen Carefully**: Understand the problem fully
- [ ] **Ask Questions**: Clarify requirements
- [ ] **Think Out Loud**: Explain your thought process
- [ ] **Start Simple**: Begin with brute force solution
- [ ] **Optimize Gradually**: Improve step by step
- [ ] **Test Your Code**: Check edge cases
- [ ] **Stay Calm**: Take time to think

### ✅ Key Points to Remember

- **Hash Tables**: O(1) average lookup, good for caching
- **Trees**: O(log n) operations, good for hierarchical data
- **Graphs**: O(V+E) traversal, good for relationships
- **Dynamic Programming**: Memoization for optimization
- **Greedy Algorithms**: Local optimization for global solution
- **Two Pointers**: Efficient array processing
- **Sliding Window**: Subarray/substring problems

### 🎯 Frappe Examples to Mention

- **Caching**: `_SITE_CACHE` for O(1) lookups
- **Queues**: Background job processing
- **Trees**: BOM structure and organizational charts
- **Graphs**: Task dependency management
- **Search**: Global search with relevance scoring
- **Sorting**: Employee search with custom comparators

### 💡 Success Tips

1. **Be Confident**: You've prepared well with real examples
2. **Use Examples**: Frappe examples show practical knowledge
3. **Explain Trade-offs**: Discuss pros and cons of approaches
4. **Write Clean Code**: Use meaningful variable names
5. **Handle Edge Cases**: Consider empty inputs, single elements
6. **Optimize Thoughtfully**: Explain why you're optimizing

---

## 🎉 Final Words

### 🚀 You're Ready When You Can:

- Explain any data structure with real-world examples
- Implement algorithms from memory
- Analyze and optimize time/space complexity
- Design scalable systems
- Write clean, efficient code
- Discuss trade-offs between different approaches

### 🎯 Remember:

- **Frappe Examples**: Your secret weapon for standing out
- **Real Experience**: 3 years of practical knowledge
- **Comprehensive Prep**: You've covered everything
- **Stay Calm**: Take your time and think clearly

**You've got this!** 🎯 Your preparation with real Frappe ecosystem examples will give you a significant advantage. Good luck with your Python Backend Engineer interview!

---

**Quick Reference**: Keep this cheat sheet handy for last-minute review before your interview! 📚✨
