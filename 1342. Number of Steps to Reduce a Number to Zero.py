class Solution:
    def numberOfSteps(self, num: int) -> int:
        if num == 0:
            return 0
    
        bits = len(bin(num)) - 2      # remove '0b'
        ones = bin(num).count('1')
        
        return (bits - 1) + ones