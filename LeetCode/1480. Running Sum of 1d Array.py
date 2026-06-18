"""
LeetCode 1480: Running Sum of 1d Array
Given an array nums, return an array runningSum where runningSum[i] is the sum of all elements nums[0]... nums[i].
"""
class Solution:
    def runningSum(self, nums: List[int]) -> List[int]:
        """
        Calculate the running sum of an array.
        
        Args:
            nums: List of integers
            
        Returns:
            List of integers representing the running sum
        """
        for i in range(1,len(nums)):
            nums[i]+=nums[i-1]
        return nums