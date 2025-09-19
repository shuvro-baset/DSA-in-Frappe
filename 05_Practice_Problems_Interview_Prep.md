# 05. Practice Problems & Interview Prep

## 📋 Table of Contents

1. [Coding Challenges](#coding-challenges)
2. [System Design Questions](#system-design-questions)
3. [Interview Scenarios](#interview-scenarios)
4. [Code Review Examples](#code-review-examples)
5. [Performance Optimization](#performance-optimization)

---

## Coding Challenges

### 🎯 Problem 1: Employee Hierarchy Traversal

**Problem**: Implement a function to find all subordinates of a given employee in an organizational hierarchy.

**Frappe Context**: Similar to `hrms/hrms/hr/page/organizational_chart/organizational_chart.py`

```python
def find_all_subordinates(employee_id, company=None):
    """
    Find all subordinates of an employee using BFS traversal.

    Args:
        employee_id (str): ID of the employee
        company (str): Company filter (optional)

    Returns:
        list: List of all subordinate employees
    """
    subordinates = []
    queue = deque([employee_id])
    visited = set()

    while queue:
        current_employee = queue.popleft()
        if current_employee in visited:
            continue

        visited.add(current_employee)

        # Get direct reports
        direct_reports = frappe.get_all(
            "Employee",
            filters={
                "reports_to": current_employee,
                "status": "Active",
                "company": company or ("!=", "")
            },
            fields=["name", "employee_name", "designation"]
        )

        for report in direct_reports:
            subordinates.append(report)
            queue.append(report.name)

    return subordinates

# Time Complexity: O(V + E) where V = employees, E = reporting relationships
# Space Complexity: O(V) for queue and visited set
```

**Interview Questions**:

- "How would you optimize this for a company with 100,000 employees?"
- "What if the hierarchy has cycles? How would you detect them?"
- "How would you implement this using recursion instead of iteration?"

### 🎯 Problem 2: BOM Cost Calculation

**Problem**: Calculate the total cost of a BOM (Bill of Materials) including all sub-BOMs.

**Frappe Context**: Similar to `erpnext/erpnext/manufacturing/doctype/bom/bom.py`

```python
def calculate_bom_cost(bom_name, quantity=1):
    """
    Calculate total cost of BOM using dynamic programming approach.

    Args:
        bom_name (str): BOM name
        quantity (float): Quantity to calculate for

    Returns:
        dict: Cost breakdown
    """
    memo = {}  # Memoization for sub-BOM costs

    def _calculate_cost(bom_id, qty):
        if bom_id in memo:
            return memo[bom_id] * qty

        bom_doc = frappe.get_doc("BOM", bom_id)
        total_cost = 0
        item_costs = []

        for item in bom_doc.items:
            item_cost = 0

            if item.bom_no:
                # Recursive call for sub-BOM
                sub_bom_cost = _calculate_cost(item.bom_no, item.stock_qty)
                item_cost = sub_bom_cost
            else:
                # Direct item cost
                item_cost = item.stock_qty * item.rate

            total_cost += item_cost
            item_costs.append({
                "item_code": item.item_code,
                "quantity": item.stock_qty,
                "rate": item.rate,
                "amount": item_cost,
                "is_bom": bool(item.bom_no)
            })

        memo[bom_id] = total_cost / bom_doc.quantity
        return total_cost * qty

    total_cost = _calculate_cost(bom_name, quantity)

    return {
        "bom_name": bom_name,
        "quantity": quantity,
        "total_cost": total_cost,
        "unit_cost": total_cost / quantity,
        "memoized_sub_boms": len(memo)
    }

# Time Complexity: O(V + E) where V = BOMs, E = BOM items
# Space Complexity: O(V) for memoization
```

**Interview Questions**:

- "How would you handle circular dependencies in BOMs?"
- "What's the space-time trade-off of using memoization here?"
- "How would you parallelize this calculation for large BOMs?"

### 🎯 Problem 3: Payment Allocation Algorithm

**Problem**: Allocate payments to multiple invoices optimally.

**Frappe Context**: Similar to `erpnext/erpnext/accounts/doctype/payment_entry/payment_entry.py`

```python
def allocate_payments_optimally(payments, invoices):
    """
    Allocate payments to invoices using greedy algorithm.

    Args:
        payments (list): List of payment amounts
        invoices (list): List of invoice amounts

    Returns:
        list: Allocation matrix
    """
    # Sort payments by amount (descending) for optimal allocation
    sorted_payments = sorted(payments, reverse=True)

    # Create invoice tracking
    invoice_status = [{"amount": inv, "remaining": inv, "allocated": 0} for inv in invoices]

    allocations = []

    for payment in sorted_payments:
        remaining_payment = payment

        for i, invoice in enumerate(invoice_status):
            if remaining_payment <= 0:
                break

            if invoice["remaining"] > 0:
                # Allocate as much as possible
                allocation_amount = min(remaining_payment, invoice["remaining"])

                invoice["remaining"] -= allocation_amount
                invoice["allocated"] += allocation_amount
                remaining_payment -= allocation_amount

                allocations.append({
                    "payment": payment,
                    "invoice_index": i,
                    "amount": allocation_amount
                })

    return {
        "allocations": allocations,
        "invoice_status": invoice_status,
        "total_allocated": sum(inv["allocated"] for inv in invoice_status)
    }

# Time Complexity: O(P * I) where P = payments, I = invoices
# Space Complexity: O(P + I) for tracking
```

**Interview Questions**:

- "What if invoices have different priorities? How would you modify this?"
- "How would you handle partial payments and refunds?"
- "What's the optimal strategy for minimizing outstanding amounts?"

---

## System Design Questions

### 🎯 Question 1: Design a Caching System

**Problem**: Design a caching system for Frappe's document metadata.

**Requirements**:

- Cache document metadata (fields, permissions, etc.)
- Support TTL (Time To Live)
- Handle cache invalidation
- Support multiple sites
- Memory efficient

**Solution**:

```python
class DocumentMetadataCache:
    """Multi-level caching system for document metadata."""

    def __init__(self, max_size=10000, default_ttl=3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = {}
        self.access_times = {}
        self.expiry_times = {}

    def get(self, key):
        """Get cached metadata with TTL check."""
        if key not in self.cache:
            return None

        # Check TTL
        if time.time() > self.expiry_times[key]:
            self._remove(key)
            return None

        # Update access time for LRU
        self.access_times[key] = time.time()
        return self.cache[key]

    def set(self, key, value, ttl=None):
        """Set cached metadata with TTL."""
        if len(self.cache) >= self.max_size:
            self._evict_lru()

        self.cache[key] = value
        self.access_times[key] = time.time()
        self.expiry_times[key] = time.time() + (ttl or self.default_ttl)

    def invalidate(self, pattern):
        """Invalidate cache entries matching pattern."""
        keys_to_remove = [k for k in self.cache.keys() if pattern in k]
        for key in keys_to_remove:
            self._remove(key)

    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self.access_times:
            return

        lru_key = min(self.access_times.keys(), key=self.access_times.get)
        self._remove(lru_key)

    def _remove(self, key):
        """Remove entry from all tracking structures."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.expiry_times.pop(key, None)

# Usage in Frappe context
metadata_cache = DocumentMetadataCache()

@metadata_cache.cached(ttl=3600)
def get_doctype_meta(doctype):
    return frappe.get_meta(doctype)
```

**Interview Questions**:

- "How would you implement distributed caching across multiple servers?"
- "What happens when cache memory is full?"
- "How would you handle cache consistency in a multi-user environment?"

### 🎯 Question 2: Design a Background Job System

**Problem**: Design a system for processing background jobs in Frappe.

**Requirements**:

- Queue management (priority queues)
- Job retry mechanism
- Dead letter queue
- Job monitoring
- Scalable to multiple workers

**Solution**:

```python
class BackgroundJobSystem:
    """Background job processing system."""

    def __init__(self):
        self.queues = {
            "high": deque(),
            "normal": deque(),
            "low": deque()
        }
        self.failed_jobs = deque()
        self.job_status = {}
        self.workers = []

    def enqueue(self, job_func, args=None, kwargs=None, priority="normal", retries=3):
        """Enqueue a job with priority."""
        job_id = str(uuid.uuid4())
        job = {
            "id": job_id,
            "func": job_func,
            "args": args or [],
            "kwargs": kwargs or {},
            "priority": priority,
            "retries": retries,
            "created_at": time.time(),
            "status": "queued"
        }

        self.queues[priority].append(job)
        self.job_status[job_id] = job

        return job_id

    def process_jobs(self):
        """Process jobs from queues."""
        while True:
            job = self._get_next_job()
            if not job:
                time.sleep(1)
                continue

            try:
                self._execute_job(job)
            except Exception as e:
                self._handle_job_failure(job, e)

    def _get_next_job(self):
        """Get next job based on priority."""
        for priority in ["high", "normal", "low"]:
            if self.queues[priority]:
                return self.queues[priority].popleft()
        return None

    def _execute_job(self, job):
        """Execute a job."""
        job["status"] = "running"
        job["started_at"] = time.time()

        try:
            result = job["func"](*job["args"], **job["kwargs"])
            job["status"] = "completed"
            job["result"] = result
            job["completed_at"] = time.time()
        except Exception as e:
            raise e

    def _handle_job_failure(self, job, error):
        """Handle job failure with retry logic."""
        job["retries"] -= 1
        job["last_error"] = str(error)

        if job["retries"] > 0:
            # Retry with exponential backoff
            delay = 2 ** (3 - job["retries"])
            time.sleep(delay)
            self.queues[job["priority"]].append(job)
        else:
            # Move to dead letter queue
            job["status"] = "failed"
            self.failed_jobs.append(job)

    def get_job_status(self, job_id):
        """Get status of a specific job."""
        return self.job_status.get(job_id)

    def get_queue_stats(self):
        """Get statistics about job queues."""
        return {
            "queues": {priority: len(queue) for priority, queue in self.queues.items()},
            "failed_jobs": len(self.failed_jobs),
            "total_jobs": len(self.job_status)
        }
```

**Interview Questions**:

- "How would you ensure job ordering within the same priority?"
- "What happens if a worker crashes while processing a job?"
- "How would you implement job scheduling (run at specific time)?"

---

## Interview Scenarios

### 🎯 Scenario 1: Performance Optimization

**Situation**: "Our employee search is taking 5 seconds for 100,000 employees. How would you optimize it?"

**Analysis**:

1. **Current Implementation**: O(n) linear scan
2. **Bottlenecks**: No indexing, full table scan
3. **Optimization Strategies**:

```python
# Current slow implementation
def slow_employee_search(query):
    return frappe.db.sql(
        f"SELECT * FROM `tabEmployee` WHERE employee_name LIKE '%{query}%'",
        as_dict=True
    )

# Optimized implementation
def optimized_employee_search(query):
    # 1. Use database indexes
    return frappe.db.sql(
        """
        SELECT name, employee_name, designation
        FROM `tabEmployee`
        WHERE employee_name LIKE %s
        AND status = 'Active'
        ORDER BY
            CASE WHEN employee_name LIKE %s THEN 0 ELSE 1 END,
            employee_name
        LIMIT 20
        """,
        (f"{query}%", f"{query}%"),
        as_dict=True
    )

# 2. Implement caching
@frappe.cache(ttl=300)
def cached_employee_search(query):
    return optimized_employee_search(query)

# 3. Use full-text search for better relevance
def fulltext_employee_search(query):
    return frappe.db.sql(
        """
        SELECT name, employee_name, designation,
               MATCH(employee_name) AGAINST(%s IN NATURAL LANGUAGE MODE) as relevance
        FROM `tabEmployee`
        WHERE MATCH(employee_name) AGAINST(%s IN NATURAL LANGUAGE MODE)
        ORDER BY relevance DESC
        LIMIT 20
        """,
        (query, query),
        as_dict=True
    )
```

**Optimization Results**:

- **Database Indexing**: O(log n) instead of O(n)
- **Caching**: O(1) for repeated queries
- **Full-text Search**: Better relevance scoring
- **Pagination**: Reduced memory usage

### 🎯 Scenario 2: Data Structure Choice

**Situation**: "We need to store user permissions. What data structure would you use and why?"

**Analysis**:

```python
# Option 1: Dictionary (Hash Table)
class PermissionManager:
    def __init__(self):
        self.permissions = {}  # {user_id: {doctype: [permissions]}}

    def get_permissions(self, user_id, doctype):
        return self.permissions.get(user_id, {}).get(doctype, [])

    def set_permissions(self, user_id, doctype, permissions):
        if user_id not in self.permissions:
            self.permissions[user_id] = {}
        self.permissions[user_id][doctype] = permissions

# Option 2: Trie for hierarchical permissions
class PermissionTrie:
    def __init__(self):
        self.root = {}

    def add_permission(self, path, permission):
        node = self.root
        for part in path.split('.'):
            if part not in node:
                node[part] = {}
            node = node[part]
        node['permission'] = permission

    def get_permission(self, path):
        node = self.root
        for part in path.split('.'):
            if part not in node:
                return None
            node = node[part]
        return node.get('permission')

# Option 3: Bitmask for efficient storage
class BitmaskPermissions:
    PERMISSIONS = {
        'read': 1,      # 001
        'write': 2,     # 010
        'delete': 4,    # 100
        'admin': 7      # 111
    }

    def __init__(self):
        self.user_permissions = {}  # {user_id: {doctype: bitmask}}

    def has_permission(self, user_id, doctype, permission):
        user_mask = self.user_permissions.get(user_id, {}).get(doctype, 0)
        perm_mask = self.PERMISSIONS.get(permission, 0)
        return bool(user_mask & perm_mask)
```

**Trade-offs**:

- **Dictionary**: O(1) lookup, easy to implement
- **Trie**: Good for hierarchical permissions, O(k) where k = path length
- **Bitmask**: Memory efficient, O(1) operations, limited to predefined permissions

---

## Code Review Examples

### 🎯 Example 1: Inefficient BOM Calculation

**Original Code** (Inefficient):

```python
def calculate_bom_cost_inefficient(bom_name):
    """Inefficient BOM cost calculation."""
    bom_doc = frappe.get_doc("BOM", bom_name)
    total_cost = 0

    for item in bom_doc.items:
        if item.bom_no:
            # Recursive call without memoization
            sub_cost = calculate_bom_cost_inefficient(item.bom_no)
            total_cost += sub_cost * item.stock_qty
        else:
            total_cost += item.stock_qty * item.rate

    return total_cost
```

**Issues**:

- No memoization (recalculates same BOMs multiple times)
- No error handling
- No validation
- Time Complexity: O(2^n) in worst case

**Improved Code**:

```python
def calculate_bom_cost_optimized(bom_name, quantity=1, memo=None):
    """Optimized BOM cost calculation with memoization."""
    if memo is None:
        memo = {}

    if bom_name in memo:
        return memo[bom_name] * quantity

    try:
        bom_doc = frappe.get_doc("BOM", bom_name)
        if not bom_doc.is_active:
            raise ValueError(f"BOM {bom_name} is not active")

        total_cost = 0
        item_costs = []

        for item in bom_doc.items:
            if not item.item_code:
                continue

            item_cost = 0

            if item.bom_no:
                # Recursive call with memoization
                sub_cost = calculate_bom_cost_optimized(item.bom_no, item.stock_qty, memo)
                item_cost = sub_cost
            else:
                # Direct item cost
                item_cost = item.stock_qty * (item.rate or 0)

            total_cost += item_cost
            item_costs.append({
                "item_code": item.item_code,
                "quantity": item.stock_qty,
                "rate": item.rate,
                "amount": item_cost
            })

        # Store in memo
        memo[bom_name] = total_cost / bom_doc.quantity

        return total_cost * quantity

    except Exception as e:
        frappe.log_error(f"Error calculating BOM cost for {bom_name}: {str(e)}")
        raise

# Time Complexity: O(V + E) with memoization
# Space Complexity: O(V) for memoization
```

### 🎯 Example 2: Memory Leak in Cache

**Original Code** (Memory Leak):

```python
class BadCache:
    def __init__(self):
        self.cache = {}

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value):
        self.cache[key] = value  # No size limit, no TTL
```

**Issues**:

- No size limit (memory leak)
- No TTL (stale data)
- No eviction policy
- No thread safety

**Improved Code**:

```python
import threading
import time
from collections import OrderedDict

class OptimizedCache:
    def __init__(self, max_size=1000, default_ttl=3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = OrderedDict()
        self.expiry_times = {}
        self.lock = threading.RLock()

    def get(self, key):
        with self.lock:
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
        with self.lock:
            # Remove if exists
            if key in self.cache:
                self._remove(key)

            # Evict if at capacity
            if len(self.cache) >= self.max_size:
                self._evict_lru()

            # Add new entry
            self.cache[key] = value
            self.expiry_times[key] = time.time() + (ttl or self.default_ttl)

    def _remove(self, key):
        self.cache.pop(key, None)
        self.expiry_times.pop(key, None)

    def _evict_lru(self):
        if self.cache:
            self.cache.popitem(last=False)
```

---

## Performance Optimization

### 🎯 Optimization Techniques

#### 1. Database Query Optimization

```python
# Bad: N+1 Query Problem
def get_employees_with_departments_bad():
    employees = frappe.get_all("Employee", fields=["name", "employee_name"])
    for emp in employees:
        emp.department = frappe.db.get_value("Employee", emp.name, "department")
    return employees

# Good: Single Query with JOIN
def get_employees_with_departments_good():
    return frappe.db.sql(
        """
        SELECT e.name, e.employee_name, e.department, d.department_name
        FROM `tabEmployee` e
        LEFT JOIN `tabDepartment` d ON e.department = d.name
        WHERE e.status = 'Active'
        """,
        as_dict=True
    )
```

#### 2. Algorithm Optimization

```python
# Bad: O(n²) nested loops
def find_duplicate_employees_bad(employees):
    duplicates = []
    for i in range(len(employees)):
        for j in range(i + 1, len(employees)):
            if employees[i].employee_name == employees[j].employee_name:
                duplicates.append((employees[i], employees[j]))
    return duplicates

# Good: O(n) using hash table
def find_duplicate_employees_good(employees):
    seen = {}
    duplicates = []

    for emp in employees:
        if emp.employee_name in seen:
            duplicates.append((seen[emp.employee_name], emp))
        else:
            seen[emp.employee_name] = emp

    return duplicates
```

#### 3. Memory Optimization

```python
# Bad: Loading all data at once
def process_all_employees_bad():
    all_employees = frappe.get_all("Employee", fields=["*"])
    results = []

    for emp in all_employees:
        # Process each employee
        result = process_employee(emp)
        results.append(result)

    return results

# Good: Batch processing
def process_all_employees_good(batch_size=1000):
    results = []
    offset = 0

    while True:
        batch = frappe.get_all(
            "Employee",
            fields=["name", "employee_name", "department"],
            limit=batch_size,
            limit_start=offset
        )

        if not batch:
            break

        for emp in batch:
            result = process_employee(emp)
            results.append(result)

        offset += batch_size

        # Clear memory
        del batch

    return results
```

---

## 🎯 Interview Questions & Answers

### Q1: "How would you design a system to handle 1 million concurrent users?"

**Answer**: "For handling 1 million concurrent users, I'd implement a multi-layered architecture:

1. **Load Balancing**: Use multiple application servers with round-robin or least-connections
2. **Caching**: Redis cluster for session data and frequently accessed data
3. **Database**: Read replicas for queries, connection pooling
4. **CDN**: For static assets
5. **Queue System**: Background job processing
6. **Monitoring**: Real-time performance metrics

````python
# Example architecture
class ScalableSystem:
    def __init__(self):
        self.cache_cluster = RedisCluster()
        self.db_pool = DatabasePool()
        self.queue_system = JobQueue()

    def handle_request(self, request):
        # Check cache first
        cached_result = self.cache_cluster.get(request.key)
        if cached_result:
            return cached_result

        # Process with database
        result = self.db_pool.execute(request.query)

        # Cache result
        self.cache_cluster.set(request.key, result, ttl=300)

        return result
```"

### Q2: "Explain how you'd optimize a slow database query."

**Answer**: "I'd use a systematic approach:

1. **Analyze**: Use EXPLAIN to understand query execution plan
2. **Index**: Add appropriate indexes on WHERE and JOIN columns
3. **Optimize**: Rewrite query to use indexes effectively
4. **Cache**: Cache results for repeated queries
5. **Partition**: Consider table partitioning for large datasets

```python
# Example optimization
def optimize_employee_search():
    # Add index
    frappe.db.sql("CREATE INDEX idx_employee_name ON `tabEmployee` (employee_name)")

    # Optimized query
    return frappe.db.sql(
        """
        SELECT name, employee_name, department
        FROM `tabEmployee`
        WHERE employee_name LIKE %s
        AND status = 'Active'
        ORDER BY employee_name
        LIMIT 20
        """,
        (f"{search_term}%",)
    )
```"

---

## 🚀 Next Steps

Ready for the final piece? Move on to [Study Guide & Roadmap](./06_Study_Guide_Roadmap.md) for a structured learning plan!

**Key Takeaways:**
- Practice coding problems with real-world context
- Understand system design principles
- Learn to optimize performance
- Prepare for common interview scenarios
````
