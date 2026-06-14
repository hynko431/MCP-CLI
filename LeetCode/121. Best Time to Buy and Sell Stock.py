# 83 ms time
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        l , r = 0, 1
        maxP = 0

        while r < len(prices):
            # profitable?
            if prices[l] < prices[r]:
                profit = prices[r] - prices[l]
                maxP = max(maxP, profit)
            else:
                # l += 1
                l = r
            r += 1
        return maxP

# 0ms time
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        low=high=prices[0]
        res=0
        for price in prices:
            if price<low:
                low=price
                high=price
            elif price>high:
                high=price
                res=max(res,high-low)
        return res