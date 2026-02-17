- [3147. Taking Maximum Energy From the Mystic Dungeon](https://leetcode.com/problems/taking-maximum-energy-from-the-mystic-dungeon/description/?envType=daily-question&envId=2026-02-15)
 ```python
 class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        dp = [0] * n
        result = float('-inf')
        for i in range(n - 1, -1, -1):
            dp[i] = energy[i] + (dp[i + k] if i + k < n else 0)
            result = max(result, dp[i])
        return result
 ```

[86. Partition List](https://leetcode.com/problems/partition-list/description/)
```python
class Solution:
    def partition(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
        before, after = ListNode(0), ListNode(0)
        before_curr, after_curr = before, after
        while head:
            if head.val < x:
                before_curr.next = head
                before_curr = before_curr.next
            else:
                after_curr.next = head
                after_curr = after_curr.next
            head = head.next

        after_curr.next = None
        before_curr.next = after.next
        
        return before.next
```