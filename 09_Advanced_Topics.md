# 09. Advanced Topics

## 📋 Table of Contents

1. [Advanced Data Structures](#advanced-data-structures)
2. [Complex Algorithms](#complex-algorithms)
3. [System Design Patterns](#system-design-patterns)
4. [Performance Optimization](#performance-optimization)
5. [Concurrency & Threading](#concurrency--threading)

---

## Advanced Data Structures

### 🔗 Trie (Prefix Tree)

**Use Case**: Efficient string operations, autocomplete, IP routing

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.word_count = 0

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.word_count += 1
        node.is_end_of_word = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

    def get_word_count(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return 0
            node = node.children[char]
        return node.word_count

# Frappe Application: Global Search Autocomplete
class GlobalSearchTrie:
    def __init__(self):
        self.trie = Trie()
        self.populate_from_doctypes()

    def populate_from_doctypes(self):
        # Populate trie with common search terms
        common_terms = frappe.get_all("__global_search",
                                   fields=["content"],
                                   limit=10000)
        for term in common_terms:
            self.trie.insert(term.content.lower())

    def autocomplete(self, query, limit=10):
        if not self.trie.starts_with(query.lower()):
            return []

        # Get suggestions based on prefix
        suggestions = []
        self._dfs_suggestions(self.trie.root, query.lower(), "", suggestions, limit)
        return suggestions

    def _dfs_suggestions(self, node, prefix, current, suggestions, limit):
        if len(suggestions) >= limit:
            return

        if node.is_end_of_word:
            suggestions.append(current)

        for char, child_node in node.children.items():
            if prefix.startswith(current + char):
                self._dfs_suggestions(child_node, prefix, current + char,
                                   suggestions, limit)
```

### 🌳 Segment Tree

**Use Case**: Range queries, interval updates, competitive programming

```python
class SegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.build(arr, 0, 0, self.n - 1)

    def build(self, arr, node, start, end):
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self.build(arr, 2 * node + 1, start, mid)
            self.build(arr, 2 * node + 2, mid + 1, end)
            self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]

    def update(self, idx, val):
        self._update(0, 0, self.n - 1, idx, val)

    def _update(self, node, start, end, idx, val):
        if start == end:
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            if idx <= mid:
                self._update(2 * node + 1, start, mid, idx, val)
            else:
                self._update(2 * node + 2, mid + 1, end, idx, val)
            self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]

    def query(self, l, r):
        return self._query(0, 0, self.n - 1, l, r)

    def _query(self, node, start, end, l, r):
        if r < start or end < l:
            return 0
        if l <= start and end <= r:
            return self.tree[node]

        mid = (start + end) // 2
        left_sum = self._query(2 * node + 1, start, mid, l, r)
        right_sum = self._query(2 * node + 2, mid + 1, end, l, r)
        return left_sum + right_sum

# Frappe Application: Stock Level Monitoring
class StockLevelMonitor:
    def __init__(self, items):
        self.stock_levels = [item.stock_qty for item in items]
        self.segment_tree = SegmentTree(self.stock_levels)

    def update_stock(self, item_idx, new_qty):
        self.stock_levels[item_idx] = new_qty
        self.segment_tree.update(item_idx, new_qty)

    def get_total_stock(self, start_idx, end_idx):
        return self.segment_tree.query(start_idx, end_idx)

    def find_low_stock_items(self, threshold):
        # Use segment tree for efficient range queries
        low_stock_items = []
        for i in range(len(self.stock_levels)):
            if self.stock_levels[i] < threshold:
                low_stock_items.append(i)
        return low_stock_items
```

### 🔄 Disjoint Set Union (Union-Find)

**Use Case**: Connected components, Kruskal's MST, network connectivity

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False

        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px

        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

        self.components -= 1
        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)

    def get_components(self):
        return self.components

# Frappe Application: Employee Network Analysis
class EmployeeNetwork:
    def __init__(self, employees):
        self.employees = employees
        self.uf = UnionFind(len(employees))
        self.build_connections()

    def build_connections(self):
        # Connect employees based on department, project, etc.
        for i, emp1 in enumerate(self.employees):
            for j, emp2 in enumerate(self.employees[i+1:], i+1):
                if self.should_connect(emp1, emp2):
                    self.uf.union(i, j)

    def should_connect(self, emp1, emp2):
        # Connect if same department, project, or reporting structure
        return (emp1.department == emp2.department or
                emp1.project == emp2.project or
                emp1.reports_to == emp2.name or
                emp2.reports_to == emp1.name)

    def get_connected_employees(self, employee_id):
        emp_idx = next(i for i, emp in enumerate(self.employees)
                      if emp.name == employee_id)

        connected = []
        for i, emp in enumerate(self.employees):
            if self.uf.connected(emp_idx, i):
                connected.append(emp)

        return connected
```

---

## Complex Algorithms

### 🎯 Advanced Dynamic Programming

**Longest Common Subsequence with Space Optimization**

```python
def lcs_optimized(text1, text2):
    m, n = len(text1), len(text2)

    # Space-optimized DP
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev, curr = curr, prev

    return prev[n]

# Frappe Application: Document Version Comparison
class DocumentVersionComparator:
    def __init__(self):
        self.lcs_cache = {}

    def compare_versions(self, doc1, doc2):
        # Extract text content from documents
        text1 = self.extract_text(doc1)
        text2 = self.extract_text(doc2)

        # Find common subsequence
        lcs_length = lcs_optimized(text1, text2)

        # Calculate similarity percentage
        max_len = max(len(text1), len(text2))
        similarity = lcs_length / max_len if max_len > 0 else 0

        return {
            "similarity": similarity,
            "common_length": lcs_length,
            "total_length": max_len
        }

    def extract_text(self, doc):
        # Extract meaningful text from document
        text_fields = []
        for field in doc.meta.fields:
            if field.fieldtype in ['Text', 'Small Text', 'Long Text']:
                text_fields.append(str(doc.get(field.fieldname) or ''))
        return ' '.join(text_fields)
```

### 🕸️ Advanced Graph Algorithms

**Dijkstra's Algorithm with Priority Queue**

```python
import heapq

def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    previous = {node: None for node in graph}

    pq = [(0, start)]
    visited = set()

    while pq:
        current_dist, current = heapq.heappop(pq)

        if current in visited:
            continue

        visited.add(current)

        for neighbor, weight in graph[current].items():
            if neighbor in visited:
                continue

            new_dist = current_dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                previous[neighbor] = current
                heapq.heappush(pq, (new_dist, neighbor))

    return distances, previous

def get_shortest_path(previous, start, end):
    path = []
    current = end

    while current is not None:
        path.append(current)
        current = previous[current]

    path.reverse()
    return path if path[0] == start else []

# Frappe Application: Delivery Route Optimization
class DeliveryRouteOptimizer:
    def __init__(self, warehouses, customers):
        self.warehouses = warehouses
        self.customers = customers
        self.graph = self.build_graph()

    def build_graph(self):
        graph = {}

        # Add warehouse nodes
        for warehouse in self.warehouses:
            graph[warehouse.name] = {}

        # Add customer nodes
        for customer in self.customers:
            graph[customer.name] = {}

        # Add edges with distances
        for warehouse in self.warehouses:
            for customer in self.customers:
                distance = self.calculate_distance(warehouse, customer)
                graph[warehouse.name][customer.name] = distance
                graph[customer.name][warehouse.name] = distance

        return graph

    def calculate_distance(self, point1, point2):
        # Calculate Euclidean distance
        return ((point1.lat - point2.lat) ** 2 +
                (point1.lng - point2.lng) ** 2) ** 0.5

    def find_optimal_route(self, start_warehouse, target_customer):
        distances, previous = dijkstra(self.graph, start_warehouse)
        path = get_shortest_path(previous, start_warehouse, target_customer)

        return {
            "path": path,
            "total_distance": distances[target_customer],
            "is_reachable": distances[target_customer] != float('inf')
        }
```

### 🔄 Advanced Sorting: Radix Sort

**For sorting integers with linear time complexity**

```python
def radix_sort(arr):
    if not arr:
        return arr

    # Find maximum number to determine number of digits
    max_num = max(arr)

    # Do counting sort for every digit
    exp = 1
    while max_num // exp > 0:
        counting_sort_by_digit(arr, exp)
        exp *= 10

    return arr

def counting_sort_by_digit(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10

    # Store count of occurrences
    for i in range(n):
        index = (arr[i] // exp) % 10
        count[index] += 1

    # Change count to actual position
    for i in range(1, 10):
        count[i] += count[i - 1]

    # Build output array
    for i in range(n - 1, -1, -1):
        index = (arr[i] // exp) % 10
        output[count[index] - 1] = arr[i]
        count[index] -= 1

    # Copy output to original array
    for i in range(n):
        arr[i] = output[i]

# Frappe Application: Employee ID Sorting
class EmployeeIDManager:
    def __init__(self):
        self.employee_ids = []

    def add_employee_ids(self, ids):
        self.employee_ids.extend(ids)

    def sort_employee_ids(self):
        # Convert to integers for radix sort
        int_ids = [int(id.replace('EMP-', '')) for id in self.employee_ids]
        sorted_int_ids = radix_sort(int_ids)

        # Convert back to string format
        return [f'EMP-{id:04d}' for id in sorted_int_ids]

    def get_sorted_employees(self):
        sorted_ids = self.sort_employee_ids()
        return sorted_ids
```

---

## System Design Patterns

### 🏗️ Microservices Architecture

**Service-oriented design for scalable applications**

```python
class MicroserviceArchitecture:
    def __init__(self):
        self.services = {
            'user_service': UserService(),
            'order_service': OrderService(),
            'payment_service': PaymentService(),
            'notification_service': NotificationService()
        }
        self.service_registry = ServiceRegistry()
        self.api_gateway = APIGateway()

    def process_order(self, order_data):
        # 1. Validate user through user service
        user = self.services['user_service'].validate_user(order_data['user_id'])
        if not user:
            return {"error": "Invalid user"}

        # 2. Create order through order service
        order = self.services['order_service'].create_order(order_data)

        # 3. Process payment through payment service
        payment_result = self.services['payment_service'].process_payment(
            order['id'], order_data['payment_info']
        )

        # 4. Send notification through notification service
        if payment_result['success']:
            self.services['notification_service'].send_confirmation(
                user['email'], order['id']
            )

        return {
            "order_id": order['id'],
            "payment_status": payment_result['status'],
            "notification_sent": payment_result['success']
        }

class ServiceRegistry:
    def __init__(self):
        self.services = {}

    def register_service(self, name, host, port):
        self.services[name] = {
            'host': host,
            'port': port,
            'status': 'healthy',
            'last_heartbeat': time.time()
        }

    def discover_service(self, name):
        return self.services.get(name)

    def health_check(self, name):
        service = self.services.get(name)
        if not service:
            return False

        # Check if service is responsive
        try:
            response = requests.get(f"http://{service['host']}:{service['port']}/health")
            return response.status_code == 200
        except:
            return False

# Frappe Application: Modular ERP Design
class ModularERPSystem:
    def __init__(self):
        self.modules = {
            'hr': HRModule(),
            'finance': FinanceModule(),
            'inventory': InventoryModule(),
            'sales': SalesModule()
        }
        self.module_communicator = ModuleCommunicator()

    def process_employee_payroll(self, employee_id, month):
        # HR module processes employee data
        employee_data = self.modules['hr'].get_employee_data(employee_id)

        # Finance module calculates payroll
        payroll_data = self.modules['finance'].calculate_payroll(
            employee_data, month
        )

        # Inventory module updates employee benefits
        benefits = self.modules['inventory'].get_employee_benefits(employee_id)

        # Sales module tracks commission if applicable
        commission = self.modules['sales'].calculate_commission(
            employee_id, month
        )

        return {
            "employee": employee_data,
            "payroll": payroll_data,
            "benefits": benefits,
            "commission": commission
        }
```

### 🔄 Event-Driven Architecture

**Asynchronous processing with event streams**

```python
class EventDrivenSystem:
    def __init__(self):
        self.event_bus = EventBus()
        self.event_handlers = {}
        self.event_store = EventStore()

    def publish_event(self, event_type, event_data):
        event = Event(
            id=str(uuid.uuid4()),
            type=event_type,
            data=event_data,
            timestamp=time.time()
        )

        # Store event
        self.event_store.store(event)

        # Publish to event bus
        self.event_bus.publish(event)

    def subscribe(self, event_type, handler):
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)

    def handle_event(self, event):
        handlers = self.event_handlers.get(event.type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                self.handle_error(event, handler, e)

    def handle_error(self, event, handler, error):
        # Log error and potentially retry
        print(f"Error handling event {event.id}: {error}")

class EventBus:
    def __init__(self):
        self.subscribers = {}

    def publish(self, event):
        subscribers = self.subscribers.get(event.type, [])
        for subscriber in subscribers:
            subscriber.handle(event)

    def subscribe(self, event_type, subscriber):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(subscriber)

# Frappe Application: Document Workflow Events
class DocumentWorkflowSystem:
    def __init__(self):
        self.event_system = EventDrivenSystem()
        self.setup_event_handlers()

    def setup_event_handlers(self):
        self.event_system.subscribe('document_created', self.handle_document_created)
        self.event_system.subscribe('document_updated', self.handle_document_updated)
        self.event_system.subscribe('document_submitted', self.handle_document_submitted)

    def handle_document_created(self, event):
        # Send notification to relevant users
        self.send_notification(event.data['doctype'], event.data['name'])

        # Update audit trail
        self.update_audit_trail(event.data['doctype'], event.data['name'], 'created')

    def handle_document_updated(self, event):
        # Check for workflow transitions
        self.check_workflow_transitions(event.data['doctype'], event.data['name'])

        # Update related documents
        self.update_related_documents(event.data['doctype'], event.data['name'])

    def handle_document_submitted(self, event):
        # Trigger approval workflow
        self.trigger_approval_workflow(event.data['doctype'], event.data['name'])

        # Update document status
        self.update_document_status(event.data['doctype'], event.data['name'], 'submitted')
```

---

## Performance Optimization

### ⚡ Database Query Optimization

**Advanced techniques for database performance**

```python
class DatabaseOptimizer:
    def __init__(self):
        self.query_cache = {}
        self.index_recommendations = {}

    def optimize_query(self, query):
        # 1. Analyze query execution plan
        execution_plan = self.analyze_execution_plan(query)

        # 2. Suggest indexes
        indexes = self.suggest_indexes(query)

        # 3. Rewrite query if needed
        optimized_query = self.rewrite_query(query)

        # 4. Add query hints
        hinted_query = self.add_query_hints(optimized_query)

        return {
            "original_query": query,
            "optimized_query": hinted_query,
            "suggested_indexes": indexes,
            "execution_plan": execution_plan
        }

    def analyze_execution_plan(self, query):
        # Simulate EXPLAIN ANALYZE
        return {
            "cost": self.estimate_query_cost(query),
            "rows": self.estimate_row_count(query),
            "operations": self.identify_operations(query)
        }

    def suggest_indexes(self, query):
        indexes = []

        # Analyze WHERE clauses
        where_columns = self.extract_where_columns(query)
        for column in where_columns:
            indexes.append(f"CREATE INDEX idx_{column} ON table_name ({column})")

        # Analyze JOIN clauses
        join_columns = self.extract_join_columns(query)
        for column in join_columns:
            indexes.append(f"CREATE INDEX idx_join_{column} ON table_name ({column})")

        return indexes

    def rewrite_query(self, query):
        # Convert subqueries to JOINs where possible
        query = self.convert_subqueries_to_joins(query)

        # Add LIMIT clauses where appropriate
        query = self.add_limit_clauses(query)

        # Optimize ORDER BY clauses
        query = self.optimize_order_by(query)

        return query

# Frappe Application: Query Performance Monitor
class FrappeQueryMonitor:
    def __init__(self):
        self.slow_queries = []
        self.query_stats = {}

    def monitor_query(self, query, execution_time):
        if execution_time > 1.0:  # Queries taking more than 1 second
            self.slow_queries.append({
                "query": query,
                "execution_time": execution_time,
                "timestamp": time.time()
            })

            # Analyze and suggest optimizations
            optimizer = DatabaseOptimizer()
            optimization = optimizer.optimize_query(query)

            self.query_stats[query] = {
                "execution_time": execution_time,
                "optimization": optimization,
                "frequency": self.query_stats.get(query, {}).get("frequency", 0) + 1
            }

    def get_performance_report(self):
        return {
            "slow_queries": self.slow_queries,
            "query_stats": self.query_stats,
            "recommendations": self.generate_recommendations()
        }

    def generate_recommendations(self):
        recommendations = []

        for query, stats in self.query_stats.items():
            if stats["frequency"] > 10:  # Frequently executed queries
                recommendations.append({
                    "query": query,
                    "priority": "high",
                    "suggestions": stats["optimization"]["suggested_indexes"]
                })

        return recommendations
```

### 🚀 Memory Optimization

**Advanced memory management techniques**

```python
class MemoryOptimizer:
    def __init__(self):
        self.memory_pool = {}
        self.object_cache = {}
        self.gc_threshold = 1000

    def optimize_memory_usage(self, objects):
        # 1. Use generators for large datasets
        optimized_objects = self.convert_to_generators(objects)

        # 2. Implement object pooling
        pooled_objects = self.implement_object_pooling(optimized_objects)

        # 3. Use weak references where appropriate
        weak_objects = self.use_weak_references(pooled_objects)

        return weak_objects

    def convert_to_generators(self, objects):
        # Convert large lists to generators
        if isinstance(objects, list) and len(objects) > 1000:
            return (obj for obj in objects)
        return objects

    def implement_object_pooling(self, objects):
        # Reuse objects instead of creating new ones
        object_type = type(objects[0]) if objects else None
        if object_type not in self.memory_pool:
            self.memory_pool[object_type] = []

        # Return objects from pool
        pooled = []
        for obj in objects:
            if self.memory_pool[object_type]:
                pooled.append(self.memory_pool[object_type].pop())
            else:
                pooled.append(obj)

        return pooled

    def use_weak_references(self, objects):
        import weakref

        # Use weak references for objects that can be garbage collected
        weak_objects = []
        for obj in objects:
            weak_objects.append(weakref.ref(obj))

        return weak_objects

# Frappe Application: Large Dataset Processing
class LargeDatasetProcessor:
    def __init__(self):
        self.memory_optimizer = MemoryOptimizer()
        self.batch_size = 1000

    def process_large_dataset(self, dataset):
        # Process in batches to avoid memory issues
        results = []

        for i in range(0, len(dataset), self.batch_size):
            batch = dataset[i:i + self.batch_size]

            # Optimize memory usage for this batch
            optimized_batch = self.memory_optimizer.optimize_memory_usage(batch)

            # Process batch
            batch_results = self.process_batch(optimized_batch)
            results.extend(batch_results)

            # Clear memory
            del optimized_batch
            del batch_results

        return results

    def process_batch(self, batch):
        # Process individual batch
        return [self.process_item(item) for item in batch]

    def process_item(self, item):
        # Process individual item
        return item
```

---

## Concurrency & Threading

### 🔄 Thread Pool Management

**Efficient thread management for concurrent operations**

```python
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class ThreadPoolManager:
    def __init__(self, max_workers=10):
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.active_tasks = {}

    def submit_task(self, func, *args, **kwargs):
        future = self.thread_pool.submit(func, *args, **kwargs)
        task_id = id(future)
        self.active_tasks[task_id] = {
            'future': future,
            'func': func.__name__,
            'submitted_at': time.time()
        }
        return future

    def submit_batch_tasks(self, tasks):
        futures = []
        for task in tasks:
            future = self.submit_task(task['func'], *task.get('args', []),
                                    **task.get('kwargs', {}))
            futures.append(future)
        return futures

    def wait_for_completion(self, futures, timeout=None):
        results = []
        for future in as_completed(futures, timeout=timeout):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append({'error': str(e)})
        return results

    def get_pool_status(self):
        return {
            'max_workers': self.max_workers,
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len([t for t in self.active_tasks.values()
                                  if t['future'].done()])
        }

# Frappe Application: Concurrent Document Processing
class ConcurrentDocumentProcessor:
    def __init__(self):
        self.thread_pool = ThreadPoolManager(max_workers=5)
        self.processing_results = {}

    def process_documents_concurrently(self, documents):
        # Submit all documents for concurrent processing
        futures = []
        for doc in documents:
            future = self.thread_pool.submit_task(
                self.process_single_document, doc
            )
            futures.append(future)

        # Wait for all to complete
        results = self.thread_pool.wait_for_completion(futures)

        return results

    def process_single_document(self, document):
        # Simulate document processing
        time.sleep(0.1)  # Simulate processing time

        # Process document
        processed_doc = {
            'id': document['id'],
            'status': 'processed',
            'processed_at': time.time()
        }

        return processed_doc

    def process_documents_in_batches(self, documents, batch_size=10):
        all_results = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_results = self.process_documents_concurrently(batch)
            all_results.extend(batch_results)

        return all_results
```

### 🔒 Thread-Safe Data Structures

**Concurrent access patterns for shared data**

```python
import threading
from collections import defaultdict

class ThreadSafeCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.lock = threading.RLock()
        self.access_times = {}

    def get(self, key):
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key]
            return None

    def set(self, key, value):
        with self.lock:
            if len(self.cache) >= self.max_size:
                self._evict_lru()

            self.cache[key] = value
            self.access_times[key] = time.time()

    def _evict_lru(self):
        if not self.access_times:
            return

        lru_key = min(self.access_times.keys(),
                     key=self.access_times.get)
        del self.cache[lru_key]
        del self.access_times[lru_key]

    def clear(self):
        with self.lock:
            self.cache.clear()
            self.access_times.clear()

class ThreadSafeCounter:
    def __init__(self):
        self.counters = defaultdict(int)
        self.lock = threading.RLock()

    def increment(self, key):
        with self.lock:
            self.counters[key] += 1

    def decrement(self, key):
        with self.lock:
            self.counters[key] -= 1

    def get(self, key):
        with self.lock:
            return self.counters[key]

    def get_all(self):
        with self.lock:
            return dict(self.counters)

# Frappe Application: Thread-Safe Session Management
class ThreadSafeSessionManager:
    def __init__(self):
        self.sessions = {}
        self.session_locks = defaultdict(threading.RLock)
        self.global_lock = threading.RLock()

    def get_session(self, user_id):
        with self.global_lock:
            if user_id not in self.sessions:
                self.sessions[user_id] = {
                    'user_id': user_id,
                    'created_at': time.time(),
                    'last_activity': time.time(),
                    'data': {}
                }
            return self.sessions[user_id]

    def update_session(self, user_id, data):
        with self.session_locks[user_id]:
            session = self.get_session(user_id)
            session['data'].update(data)
            session['last_activity'] = time.time()

    def cleanup_expired_sessions(self, timeout=3600):
        with self.global_lock:
            current_time = time.time()
            expired_users = []

            for user_id, session in self.sessions.items():
                if current_time - session['last_activity'] > timeout:
                    expired_users.append(user_id)

            for user_id in expired_users:
                del self.sessions[user_id]
                if user_id in self.session_locks:
                    del self.session_locks[user_id]
```

---

## 🎯 Advanced Interview Questions

### Q1: "How would you implement a distributed cache system?"

**Answer**: "I'd implement a distributed cache using consistent hashing and replication:

```python
class DistributedCache:
    def __init__(self, nodes):
        self.nodes = nodes
        self.hash_ring = ConsistentHashRing(nodes)
        self.replication_factor = 3

    def get(self, key):
        # Find primary node
        primary_node = self.hash_ring.get_node(key)

        # Try primary node first
        value = primary_node.get(key)
        if value:
            return value

        # Try replica nodes
        replica_nodes = self.get_replica_nodes(key)
        for node in replica_nodes:
            value = node.get(key)
            if value:
                # Repair primary node
                primary_node.set(key, value)
                return value

        return None

    def set(self, key, value, ttl=None):
        # Set on primary node
        primary_node = self.hash_ring.get_node(key)
        primary_node.set(key, value, ttl)

        # Replicate to replica nodes
        replica_nodes = self.get_replica_nodes(key)
        for node in replica_nodes:
            node.set(key, value, ttl)
```

This ensures high availability and data consistency across the distributed system."

### Q2: "Explain how you'd optimize a system handling 10 million requests per day"

**Answer**: "For 10M requests/day (~115 requests/second), I'd implement:

1. **Load Balancing**: Distribute traffic across multiple servers
2. **Caching**: Redis cluster for frequently accessed data
3. **Database Optimization**: Read replicas, connection pooling
4. **CDN**: For static assets
5. **Background Processing**: Queue system for heavy operations

```python
class HighTrafficSystem:
    def __init__(self):
        self.load_balancer = LoadBalancer()
        self.cache_cluster = RedisCluster()
        self.db_pool = DatabasePool()
        self.queue_system = JobQueue()

    def handle_request(self, request):
        # Check cache first
        cached = self.cache_cluster.get(request.key)
        if cached:
            return cached

        # Process with database
        result = self.db_pool.execute(request.query)

        # Cache result
        self.cache_cluster.set(request.key, result, ttl=300)

        # Queue background tasks
        self.queue_system.enqueue(process_result, result)

        return result
```

This architecture can handle the load efficiently while maintaining performance."

---

## 🚀 Summary

### ✅ Advanced Topics Covered:

- **Trie**: Efficient string operations and autocomplete
- **Segment Tree**: Range queries and interval updates
- **Union-Find**: Connected components and network analysis
- **Advanced DP**: Space-optimized algorithms
- **Dijkstra**: Shortest path algorithms
- **Radix Sort**: Linear time sorting
- **Microservices**: Scalable architecture patterns
- **Event-Driven**: Asynchronous processing
- **Thread Safety**: Concurrent programming
- **Performance**: Database and memory optimization

### 🎯 Key Takeaways:

- **Advanced Data Structures**: Know when to use specialized structures
- **Complex Algorithms**: Understand advanced algorithmic techniques
- **System Design**: Design scalable, maintainable systems
- **Performance**: Optimize for both speed and memory usage
- **Concurrency**: Handle multiple operations safely

**You're now equipped with advanced DSA knowledge!** 🎉 These topics will help you tackle complex interview questions and demonstrate deep understanding of computer science fundamentals.
