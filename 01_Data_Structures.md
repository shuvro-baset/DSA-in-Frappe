# 01. Data Structures in Frappe Ecosystem

## 📋 Table of Contents

1. [Arrays & Lists](#arrays--lists)
2. [Dictionaries & Hash Tables](#dictionaries--hash-tables)
3. [Stacks & Queues](#stacks--queues)
4. [Trees](#trees)
5. [Graphs](#graphs)

---

## Arrays & Lists

### 🎯 What are Arrays/Lists?

Arrays and lists are ordered collections of elements that can be accessed by index.

### 💼 Real Examples from Frappe

#### 1. Email Queue Recipients

**Location**: `frappe/frappe/email/doctype/email_queue/email_queue.py`

```python
def set_recipients(self, recipients):
    self.set("recipients", [])
    for r in recipients:
        self.append("recipients", {"recipient": r.strip(), "status": "Not Sent"})
```

**Why Arrays?**

- Ordered collection of email recipients
- Easy to iterate through all recipients
- Simple to add/remove recipients

**Interview Question**: "How would you handle a list of 10,000 email recipients efficiently?"

#### 2. BOM Tree Child Items

**Location**: `erpnext/erpnext/manufacturing/doctype/bom/bom.py`

```python
class BOMTree:
    def __init__(self, name: str, is_bom: bool = True, exploded_qty: float = 1.0, qty: float = 1):
        self.name = name
        self.child_items: list["BOMTree"] = []  # List of child BOM items
        self.is_bom = is_bom
        self.item_code: str = None
        self.qty = qty
        self.exploded_qty = exploded_qty
```

**Why Lists?**

- Dynamic size - can add/remove BOM items
- Maintains order of components
- Easy to traverse for calculations

#### 3. Festival Bonus Slabs Processing

**Location**: `fusion_hr/fusion_hr/fusion_hr/doctype/festival_bonus_entry/festival_bonus_entry.py`

```python
def find_matching_slab(service_days, festival_slabs):
    matching_slabs = []

    for slab in festival_slabs:
        min_days = slab.get('minimum_days', 0) or 0
        max_days = slab.get('maximum_days', 0) or 0

        if service_days >= min_days and (max_days == 0 or service_days <= max_days):
            matching_slabs.append(slab)

    # Return slab with highest bonus percentage
    if matching_slabs:
        return max(matching_slabs, key=lambda x: x.get('bonus_percentage', 0) or 0)

    return None
```

**Why Lists?**

- Process multiple bonus slabs
- Filter and find matching criteria
- Sort by bonus percentage

### 🔧 Common Operations

```python
# Adding elements
recipients.append("user@example.com")

# Removing elements
recipients.remove("user@example.com")

# Finding elements
if "user@example.com" in recipients:
    print("User found")

# Sorting
recipients.sort()  # Alphabetical order
```

---

## Dictionaries & Hash Tables

### 🎯 What are Dictionaries/Hash Tables?

Key-value pairs that provide O(1) average time complexity for lookups, insertions, and deletions.

### 💼 Real Examples from Frappe

#### 1. Cache Management

**Location**: `frappe/frappe/utils/caching.py`

```python
_SITE_CACHE = defaultdict(lambda: defaultdict(dict))

def site_cache(ttl: int | None = None, maxsize: int | None = None):
    def time_cache_wrapper(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = __generate_request_cache_key(args, kwargs)

            if cache_key in _SITE_CACHE[frappe.local.site][func.__name__]:
                return _SITE_CACHE[frappe.local.site][func.__name__][cache_key]

            result = func(*args, **kwargs)
            _SITE_CACHE[frappe.local.site][func.__name__][cache_key] = result
            return result
        return wrapper
    return time_cache_wrapper
```

**Why Hash Tables?**

- Fast O(1) cache lookups
- Efficient storage of function results
- Easy cache invalidation

#### 2. User Settings Storage

**Location**: `frappe/frappe/model/utils/user_settings.py`

```python
filter_dict = {"doctype": 0, "docfield": 1, "operator": 2, "value": 3}

def get_user_settings(doctype, for_update=False):
    user_settings = frappe.cache.hget("_user_settings", f"{doctype}::{frappe.session.user}")

    if user_settings is None:
        user_settings = frappe.db.sql(
            """select data from `__UserSettings`
            where `user`=%s and `doctype`=%s""",
            (frappe.session.user, doctype),
        )
        user_settings = user_settings and user_settings[0][0] or "{}"
```

**Why Hash Tables?**

- Quick user-specific settings lookup
- Efficient key-based access
- Easy to update individual settings

#### 3. BOM Dependency Mapping

**Location**: `erpnext/erpnext/manufacturing/doctype/bom_update_log/bom_updation_utils.py`

```python
def _generate_dependence_map() -> defaultdict:
    bom_items = (
        frappe.qb.from_(bom_item)
        .join(bom)
        .on(bom_item.parent == bom.name)
        .select(bom_item.bom_no, bom_item.parent)
        .where(
            (bom_item.bom_no.isnotnull())
            & (bom_item.bom_no != "")
            & (bom.docstatus == 1)
            & (bom.is_active == 1)
            & (bom_item.parenttype == "BOM")
        )
    ).run(as_dict=True)

    child_parent_map = defaultdict(list)
    parent_child_map = defaultdict(list)
    for row in bom_items:
        child_parent_map[row.bom_no].append(row.parent)
        parent_child_map[row.parent].append(row.bom_no)

    return child_parent_map, parent_child_map
```

**Why Hash Tables?**

- Fast parent-child relationship lookups
- Efficient dependency resolution
- Easy to build dependency graphs

### 🔧 Common Operations

```python
# Creating dictionaries
cache = {}
settings = {"theme": "dark", "language": "en"}

# Accessing values
theme = settings["theme"]
theme = settings.get("theme", "light")  # Safe access

# Updating values
settings["theme"] = "light"
settings.update({"language": "es", "timezone": "UTC"})

# Checking existence
if "theme" in settings:
    print("Theme setting exists")
```

---

## Stacks & Queues

### 🎯 What are Stacks & Queues?

- **Stack**: LIFO (Last In, First Out) - like a stack of plates
- **Queue**: FIFO (First In, First Out) - like a line of people

### 💼 Real Examples from Frappe

#### 1. Background Job Queue

**Location**: `frappe/frappe/utils/background_jobs.py`

```python
def enqueue(
    method: str | Callable,
    queue: str = "default",
    timeout: int | None = None,
    event=None,
    is_async: bool = True,
    job_name: str | None = None,
    now: bool = False,
    **kwargs,
) -> Job | Any:

    q = get_queue(queue, is_async=is_async)

    queue_args = {
        "site": frappe.local.site,
        "user": frappe.session.user,
        "method": method,
        "event": event,
        "job_name": job_name or method_name,
        "is_async": is_async,
        "kwargs": kwargs,
    }

    return q.enqueue_call(
        execute_job,
        timeout=timeout,
        kwargs=queue_args,
        at_front=at_front,
        job_id=job_id,
    )
```

**Why Queues?**

- Process jobs in order (FIFO)
- Handle background tasks efficiently
- Prevent system overload

#### 2. Email Queue Processing

**Location**: `frappe/frappe/email/doctype/email_queue/email_queue.py`

```python
class QueueBuilder:
    def __init__(self, recipients=None, sender=None, subject=None, message=None, **kwargs):
        self._recipients = recipients
        self._sender = sender
        self._message = message
        # ... other initialization

    def _get_emails_list(self, emails=None):
        emails = split_emails(emails) if isinstance(emails, str) else (emails or [])
        return [each for each in set(emails) if each]
```

**Why Queues?**

- Process emails in order
- Handle email sending failures gracefully
- Maintain email delivery order

#### 3. Deferred Insert Queue

**Location**: `frappe/frappe/deferred_insert.py`

```python
def deferred_insert(doctype: str, records: list[Union[dict, "Document"]] | str):
    if isinstance(records, dict | list):
        _records = json.dumps(records)
    else:
        _records = records

    try:
        frappe.cache.rpush(f"{queue_prefix}{doctype}", _records)
    except redis.exceptions.ConnectionError:
        for record in records:
            insert_record(record, doctype)
```

**Why Queues?**

- Batch database operations
- Improve performance
- Handle connection failures

### 🔧 Common Operations

```python
# Stack operations (using list)
stack = []
stack.append(item)  # Push
item = stack.pop()  # Pop
top = stack[-1]     # Peek

# Queue operations (using deque for efficiency)
from collections import deque
queue = deque()
queue.append(item)  # Enqueue
item = queue.popleft()  # Dequeue
```

---

## Trees

### 🎯 What are Trees?

Hierarchical data structures with nodes connected by edges, having one root and no cycles.

### 💼 Real Examples from Frappe

#### 1. Organizational Chart (HRMS)

**Location**: `hrms/hrms/hr/page/organizational_chart/organizational_chart.py`

```python
@frappe.whitelist()
def get_children(parent=None, company=None, exclude_node=None):
    filters = [["status", "=", "Active"]]
    if company and company != "All Companies":
        filters.append(["company", "=", company])

    if parent and company and parent != company:
        filters.append(["reports_to", "=", parent])
    else:
        filters.append(["reports_to", "=", ""])

    employees = frappe.get_all(
        "Employee",
        fields=[
            "employee_name as name",
            "name as id",
            "lft",
            "rgt",
            "reports_to",
            "image",
            "designation as title",
        ],
        filters=filters,
        order_by="name",
    )

    for employee in employees:
        employee.connections = get_connections(employee.id, employee.lft, employee.rgt)
        employee.expandable = bool(employee.connections)

    return employees
```

**Why Trees?**

- Natural hierarchy representation
- Easy to find parent-child relationships
- Efficient traversal algorithms

#### 2. BOM Tree Structure

**Location**: `erpnext/erpnext/manufacturing/doctype/bom/bom.py`

```python
class BOMTree:
    def __init__(self, name: str, is_bom: bool = True, exploded_qty: float = 1.0, qty: float = 1):
        self.name = name
        self.child_items: list["BOMTree"] = []
        self.is_bom = is_bom
        self.item_code: str = None
        self.qty = qty
        self.exploded_qty = exploded_qty

    def level_order_traversal(self) -> list["BOMTree"]:
        """Get level order traversal of tree."""
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

**Why Trees?**

- Represent product structure hierarchy
- Calculate material requirements efficiently
- Handle multi-level BOMs

#### 3. Goal Hierarchy (HRMS)

**Location**: `hrms/hrms/hr/doctype/goal/goal.py`

```python
class Goal(NestedSet):
    nsm_parent_field = "parent_goal"

    def on_update(self):
        NestedSet.on_update(self)
        doc_before_save = self.get_doc_before_save()

        if doc_before_save:
            self.update_kra_in_child_goals(doc_before_save)
            if doc_before_save.parent_goal != self.parent_goal:
                self.update_parent_progress(doc_before_save.parent_goal)

        self.update_parent_progress()
        self.update_goal_progress_in_appraisal()
```

**Why Trees?**

- Represent goal hierarchies
- Calculate progress up the tree
- Handle goal dependencies

### 🔧 Tree Traversal Patterns

```python
# Pre-order traversal (Root -> Left -> Right)
def preorder_traversal(node):
    if node:
        print(node.value)
        preorder_traversal(node.left)
        preorder_traversal(node.right)

# In-order traversal (Left -> Root -> Right)
def inorder_traversal(node):
    if node:
        inorder_traversal(node.left)
        print(node.value)
        inorder_traversal(node.right)

# Post-order traversal (Left -> Right -> Root)
def postorder_traversal(node):
    if node:
        postorder_traversal(node.left)
        postorder_traversal(node.right)
        print(node.value)

# Level-order traversal (BFS)
def level_order_traversal(root):
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        node = queue.popleft()
        result.append(node.value)

        for child in node.children:
            queue.append(child)

    return result
```

---

## Graphs

### 🎯 What are Graphs?

Collections of nodes (vertices) connected by edges, used to represent relationships and dependencies.

### 💼 Real Examples from Frappe

#### 1. Workflow Dependencies

**Location**: `erpnext/erpnext/projects/doctype/task/task.py`

```python
def check_recursion(self):
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

            if count == 15:
                break
```

**Why Graphs?**

- Detect circular dependencies
- Validate task relationships
- Ensure workflow integrity

#### 2. BOM Dependency Graph

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

    for parent_bom in dependants:
        if _all_children_are_processed(parent_bom):
            resolved_dependants.add(parent_bom)

    return list(resolved_dependants)
```

**Why Graphs?**

- Model BOM dependencies
- Resolve update order
- Prevent circular references

#### 3. Hierarchy Chart Processing

**Location**: `hrms/hrms/utils/hierarchy_chart.py`

```python
@frappe.whitelist()
def get_all_nodes(method, company):
    """Recursively gets all data from nodes"""
    method = frappe.get_attr(method)

    root_nodes = method(company=company)
    result = []
    nodes_to_expand = []

    for root in root_nodes:
        data = method(root.id, company)
        result.append(dict(parent=root.id, parent_name=root.name, data=data))
        nodes_to_expand.extend(
            [{"id": d.get("id"), "name": d.get("name")} for d in data if d.get("expandable")]
        )

    while nodes_to_expand:
        parent = nodes_to_expand.pop(0)
        data = method(parent.get("id"), company)
        result.append(dict(parent=parent.get("id"), parent_name=parent.get("name"), data=data))
        for d in data:
            if d.get("expandable"):
                nodes_to_expand.append({"id": d.get("id"), "name": d.get("name")})

    return result
```

**Why Graphs?**

- Build organizational hierarchies
- Handle complex relationships
- Efficient traversal algorithms

### 🔧 Graph Algorithms

```python
# Depth-First Search (DFS)
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()

    visited.add(start)
    print(start)

    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

# Breadth-First Search (BFS)
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
```

---

## 🎯 Interview Questions & Answers

### Q1: "How would you implement a cache system using hash tables?"

**Answer**: "In Frappe, we use hash tables for caching because they provide O(1) average lookup time. Here's how it works:

```python
# Cache implementation from Frappe
_SITE_CACHE = defaultdict(lambda: defaultdict(dict))

def site_cache(ttl=None, maxsize=None):
    def wrapper(func):
        @wraps(func)
        def cached_func(*args, **kwargs):
            cache_key = generate_cache_key(args, kwargs)
            if cache_key in _SITE_CACHE[site][func.__name__]:
                return _SITE_CACHE[site][func.__name__][cache_key]

            result = func(*args, **kwargs)
            _SITE_CACHE[site][func.__name__][cache_key] = result
            return result
        return cached_func
    return wrapper
```

The hash table allows us to quickly check if a function result is already cached and retrieve it in constant time."

### Q2: "Explain how Frappe uses trees for organizational charts."

**Answer**: "Frappe uses trees to represent organizational hierarchies because they naturally model reporting relationships. Each employee has a `reports_to` field that creates parent-child relationships. The system uses nested set model (NSM) with `lft` and `rgt` values for efficient tree operations:

```python
# Getting children in organizational chart
def get_children(parent=None, company=None):
    filters = [["status", "=", "Active"]]
    if parent:
        filters.append(["reports_to", "=", parent])
    else:
        filters.append(["reports_to", "=", ""])

    employees = frappe.get_all("Employee", filters=filters)
    return employees
```

This allows efficient queries like 'get all subordinates' or 'get management chain'."

### Q3: "How does Frappe handle circular dependencies in BOMs?"

**Answer**: "Frappe uses graph algorithms to detect circular dependencies in BOMs. The system performs a depth-first search to check if a BOM references itself through its components:

```python
def check_recursion(self):
    bom_list = self.traverse_tree()
    child_items = frappe.get_all("BOM Item",
        filters={"parent": ("in", bom_list), "parenttype": "BOM"})

    for item in child_items:
        if self.name == item.bom_no:
            frappe.throw("BOM recursion detected")
```

This prevents infinite loops and ensures data integrity."

---

## 🚀 Next Steps

Ready to dive deeper? Move on to [Algorithms](./02_Algorithms.md) to learn how these data structures are manipulated efficiently!

**Key Takeaways:**

- Choose the right data structure for your use case
- Understand time/space complexity trade-offs
- Use real examples to explain concepts in interviews
- Practice implementing these patterns in your own code
